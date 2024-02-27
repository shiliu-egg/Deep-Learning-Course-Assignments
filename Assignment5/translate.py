from config.config import Config
from model.TranslationModel import TranslationModel
from utils.data_helpers import LoadEnglishGermanDataset, my_tokenizer
import torch


def greedy_decode(model, src, max_len, start_symbol, config, data_loader):
    src = src.to(config.device)
    memory = model.encoder(src)  # 对输入的Token序列进行解码翻译
    ys = torch.ones(1, 1).fill_(start_symbol). \
        type(torch.long).to(config.device)  # 解码的第一个输入，起始符号
    for i in range(max_len - 1):
        memory = memory.to(config.device)
        out = model.decoder(ys, memory)  # [tgt_len,1,embed_dim]
        out = out.transpose(0, 1)  # [1,tgt_len, embed_dim]
        prob = model.classification(out[:, -1])  # 只对对预测的下一个词进行分类
        # out[:,1] shape : [1,embed_dim],  prob shape:  [1,tgt_vocab_size]
        _, next_word = torch.max(prob, dim=1)  # 选择概率最大者
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # 将当前时刻解码的预测输出结果，同之前所有的结果堆叠作为输入再去预测下一个词。
        if next_word == data_loader.EOS_IDX:  # 如果当前时刻的预测输出为结束标志，则跳出循环结束预测。
            break
    return ys


def translate(model, src, data_loader, config):
    src_vocab = data_loader.de_vocab
    tgt_vocab = data_loader.en_vocab
    src_tokenizer = data_loader.tokenizer['de']
    model.eval()
    tokens = [src_vocab.stoi[tok] for tok in src_tokenizer(src)]  # 构造一个样本
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))  # 将src_len 作为第一个维度
    with torch.no_grad():
        tgt_tokens = greedy_decode(model, src, max_len=num_tokens + 5,
                                   start_symbol=data_loader.BOS_IDX, config=config,
                                   data_loader=data_loader).flatten()  # 解码的预测结果
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")


def translate_german_to_english(srcs, config):
    data_loader = LoadEnglishGermanDataset(config.train_corpus_file_paths,
                                           batch_size=config.batch_size,
                                           tokenizer=my_tokenizer,
                                           min_freq=config.min_freq)
    translation_model = TranslationModel(src_vocab_size=len(data_loader.de_vocab),
                                         tgt_vocab_size=len(data_loader.en_vocab),
                                         d_model=config.d_model,
                                         nhead=config.num_head,
                                         num_encoder_layers=config.num_encoder_layers,
                                         num_decoder_layers=config.num_decoder_layers,
                                         dim_feedforward=config.dim_feedforward,
                                         dropout=config.dropout)
    translation_model = translation_model.to(config.device)
    loaded_paras = torch.load(config.model_save_dir + '/model.pkl')
    translation_model.load_state_dict(loaded_paras)
    results = []
    for src in srcs:
        r = translate(translation_model, src, data_loader, config)
        results.append(r)
    return results


if __name__ == '__main__':
    # srcs = ["Eine Gruppe von Menschen steht vor einem Iglu.",
    #         "Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster."]
    # tgts = ["A group of people are facing an igloo.",
    #         "A man in a blue shirt is standing on a ladder cleaning a window."]
    srcs = [
        "我们所做的一切是残骸。",
        "这种家务对扁桃体肌肉和肩部肌肉是很好的。",
        "他们闻到火葬场里烧焦的人肉的气味。",
        "Conwy 县坠机事件中的2名妇女死亡。",
        "IIR",
        "基克允许用户保持匿名。",
        "自行车盗窃是开放式校园政策的头号牺牲品。",
        "Rushcliffe",
        "他是71岁。",
        "上面显示的是渲染它可能是什么样的东西。",
        "这种松露不能被分享。",
        "他在实地的才干应进一步探讨。",
        "唯一的选择",
        "联合王国国防部",
        "作为支持球队这个赛季最积极的方面是什么?",
        "医疗保险不包括救护车。",
        "这幅画高于预期的售价3500万至5000万美元。",
        "房地产市场在所有年龄段都有优势。",
        "凯特·温斯林在金球奖和巴弗斯的比赛中赢得了最佳的支持。",
        "关于 Beyan 视频的争议",
        "什叶派的拉博夫是一个以艺术的名义进入电梯的人。",
        "他们的研究提供了关于独立公投的一些令人着迷的信息和观察。",
        "该协会将仇恨团体定义为攻击基于种族、性取向和宗教等核心特征的组织的组织。",
        "Bernat Mosgue / AP 照片",
        "巴拉克·奥巴马和家人在复活节",
        "这给我们造成了极大的不安。",
        "20世纪80年代通过了一个由电话运营商和国家运营商控制的连接节点的封闭网络。",
        "我可以打子弹。",
        "此举使消费者受益于标准的国内天然气关税。",
        "她是邪恶的化身!",
        "失去子女补助金的人数可能在五年内上升50%。",
        "前妻子保罗·麦卡特尼的妻子希瑟·米尔斯和王菲的歌手汤姆·帕克被任命为顶替者。",
        "我的厨房画廊。",
        "有些人在压力过大或过度兴奋时往往会打嗝。",
        "凯特·厄普顿热气球升空",
        "上次比赛",
        "她们有自己的背景和历史上的女性和女权运动。",
        "她还在德黑兰度过了童年的一部分。",
        "这一次没有男朋友。",
        "我们采取疟疾预防措施。",
        "388g 箱",
        "塞德里克·福特在工厂里杀害了三人",
        "你什么时候开始意识到人们觉得你很有趣?",
        "按新闻协会",
        "由此产生的与美国的争吵已被莫斯科利用。",
        "Coacin 4",
        "我们把这个家带到了加拿大。",
        "DNA 只是我们身体的另一个可能出错。",
        "没有人能相信。",
        "虾髻"]
    tgts = [
        "All we did locate was wreckage.",
        "This chore is great for toning arms and shoulder muscles.",
        "They smelled the... burning human flesh coming from the crematoria.",
        "Woman dies, two injured in Conwy county crash",
        "IR.",
        "Kik allows users to remain anonymous.",
        "Bike thefts are the number one casualty of open campus policy.",
        "Rushcliffe",
        "He was 71.",
        "A rendering of what it may have looked like is shown above.",
        "The truffles could not be shared.",
        "His talent for the field should be explored further.",
        "The only choice",
        "UK Ministry of Defence",
        "As a supporter what has been the most positive aspect about your team this season?",
        "Ambulances are not covered by Medicare.",
        "The painting sold above its expected sale price of $35 million to $50 million.",
        "The real estate mogul also has a commanding advantage across all age groups.",
        "Kate Winslet stands a strong chance of taking home best supporting actress after winning in the category both at the Golden Globes and the Baftas.",
        "Controversy over Beyonce video",
        "Shia LaBeouf is trapping himself inside an elevator in the name of art.",
        "Their study offers some fascinating nuggets of information and observation about the independence referendum.",
        "The SPLC defines hate groups as organizations that attack people based on central characteristics such as race, sexual orientation and religion.",
        "Bernat Armangue / AP Photo",
        "Barack Obama and Family on Easter Sunday",
        "It caused us a huge amount of concern.",
        "Adopted in the 1980s as a closed network with connecting nodes controlled by phone carriers and national operators, SS7 directs mobile traffic from cellphone towers to the Internet.",
        "I can hit shots.",
        "The moves benefit customers on a standard domestic gas tariff.",
        "She is evil personified!",
        "The number losing child benefit might rise by 50% within five years.",
        "Heather Mills, the former wife of Sir Paul McCartney, and The Wanted singer Tom Parker have been drafted in as replacements.",
        "My kitchen gallery: Georgia Levy",
        "Some people tend to hiccup when stressed or overexcited.",
        "Kate Upton Takes a Hot Air Balloon",
        "Last contest",
        "They have some kind of background and history of women and feminism.",
        "She also spent part of her childhood in Tehran.",
        "No boyfriend this time around...",
        "We take malaria prophylaxis.",
        "388g box",
        "Cedric Ford killed three people at factory",
        "When did you begin to realise that people found you funny?",
        "By Press Association",
        "The resulting row with the US has been exploited by Moscow.",
        "Costac4",
        "We brought this home for Canada.",
        "DNA is just another bit of our body that might go wrong.",
        "Nobody could believe it.",
        "Prawn bun cha"
    ]

    config = Config()
    results = translate_german_to_english(srcs, config)
    for src, tgt, r in zip(srcs, tgts, results):
        print(f"中文：{src}")
        print(f"翻译：{r}")
        print(f"英语：{tgt}")
        print("\n")

