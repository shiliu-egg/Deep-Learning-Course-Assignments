from copy import deepcopy
from config.config import Config
from model.TranslationModel import TranslationModel
from utils.data_helpers import LoadEnglishGermanDataset, my_tokenizer
from nltk.translate.bleu_score import corpus_bleu
import torch
import time
import os
import logging
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

def accuracy(logits, y_true, PAD_IDX):
    y_pred = logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    y_true = y_true.transpose(0, 1).reshape(-1)
    acc = y_pred.eq(y_true)
    mask = torch.logical_not(y_true.eq(PAD_IDX))
    acc = acc.logical_and(mask)
    correct = acc.sum().item()
    total = mask.sum().item()
    return float(correct) / total, correct, total

def evaluate(config, valid_iter, model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    hypotheses = []  # 存储所有预测翻译
    references = []  # 存储所有参考翻译
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(valid_iter):
            src = src.to(config.device)
            tgt = tgt.to(config.device)  # [tgt_len, batch_size]
            tgt_input = tgt[:-1, :]  # 解码部分的输入

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                data_loader.create_mask(src, tgt_input, device=config.device)

            logits = model(src=src, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
                           memory_key_padding_mask=src_padding_mask)

            tgt_out = tgt[1:, :]  # 解码部分的真实值
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()

            # 将预测和参考翻译转换为单词列表
            y_pred = logits.argmax(-1).transpose(0, 1)
            for i in range(y_pred.shape[0]):
                hypotheses.append([data_loader.en_vocab.itos[word] for word in y_pred[i].cpu().numpy()])
                tgt_list = tgt[1:, i].cpu().numpy()
                references.append([[data_loader.en_vocab.itos[word] for word in tgt_list]])

    bleu_score = corpus_bleu(references, hypotheses)
    model.train()
    return total_loss / len(valid_iter), bleu_score

def train_model(config):
    logging.info("############载入数据集############")
    data_loader = LoadEnglishGermanDataset(config.train_corpus_file_paths,
                                           batch_size=config.batch_size,
                                           tokenizer=my_tokenizer,
                                           min_freq=config.min_freq)
    logging.info("############划分数据集############")
    train_iter, valid_iter, test_iter = \
        data_loader.load_train_val_test_data(config.train_corpus_file_paths,
                                             config.val_corpus_file_paths,
                                             config.test_corpus_file_paths)
    logging.info("############初始化模型############")
    translation_model = TranslationModel(src_vocab_size=len(data_loader.de_vocab),
                                         tgt_vocab_size=len(data_loader.en_vocab),
                                         d_model=config.d_model,
                                         nhead=config.num_head,
                                         num_encoder_layers=config.num_encoder_layers,
                                         num_decoder_layers=config.num_decoder_layers,
                                         dim_feedforward=config.dim_feedforward,
                                         dropout=config.dropout)
    model_save_path = os.path.join(config.model_save_dir, 'model.pkl')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        translation_model.load_state_dict(loaded_paras)
        logging.info("#### 成功载入已有模型，进行追加训练...")
    translation_model = translation_model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_loader.PAD_IDX)

    optimizer = torch.optim.Adam(translation_model.parameters(),
                                 lr=0.,
                                 betas=(config.beta1, config.beta2), eps=config.epsilon)
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer)
    translation_model.train()
    
    train_losses = []
    valid_losses = []

    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (src, tgt) in enumerate(train_iter):
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask \
                = data_loader.create_mask(src, tgt_input, config.device)
            logits = translation_model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask)
            optimizer.zero_grad()
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            losses += loss.item()
            acc, _, _ = accuracy(logits, tgt_out, data_loader.PAD_IDX)
            msg = f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], Train loss :{loss.item():.3f}, Train acc: {acc}"
            logging.info(msg)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        train_losses.append(train_loss)
        valid_loss, valid_bleu = evaluate(config, valid_iter, translation_model, data_loader, loss_fn)
        valid_losses.append(valid_loss)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}, Valid BLEU: {valid_bleu:.3f}, Epoch time = {(end_time - start_time):.3f}s"
        logging.info(msg)
        if epoch % 2 == 0:
            state_dict = deepcopy(translation_model.state_dict())
            torch.save(state_dict, model_save_path)

    # 绘制损失曲线
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('fis.png')

    # 在测试集上测试模型
    test_loss, test_bleu = evaluate(config, test_iter, translation_model, data_loader, loss_fn)
    print(f"Test Loss: {test_loss}, Test BLEU: {test_bleu}")

if __name__ == '__main__':
    config = Config()
    train_model(config)
