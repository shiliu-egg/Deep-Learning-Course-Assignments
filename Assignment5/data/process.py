import os
from sklearn.model_selection import train_test_split

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        en_sentences = []
        zh_sentences = []
        for line in lines:
            # 跳过空行或不含制表符的行（即不包含中英文对的行）
            if '\t' not in line.strip():
                continue
            en, zh = line.strip().split('\t', 1)  # 最多分割成两部分
            if not en or not zh:  # 如果英文或中文部分为空，跳过
                continue
            en_sentences.append(en + '\n')  # 加入换行符，方便后续文件写入
            zh_sentences.append(zh + '\n')
    return en_sentences, zh_sentences

def split_dataset(en_sentences, zh_sentences, test_size=0.2):
    # 先分出80%的训练集，20%的临时测试集
    en_train, en_temp, zh_train, zh_temp = train_test_split(en_sentences, zh_sentences, test_size=test_size, random_state=42)
    # 再将临时测试集一分为二，得到10%的验证集和10%的测试集
    en_val, en_test, zh_val, zh_test = train_test_split(en_temp, zh_temp, test_size=0.5, random_state=42)
    return en_train, zh_train, en_val, zh_val, en_test, zh_test

def save_datasets(en_train, zh_train, en_val, zh_val, en_test, zh_test, dataset_dir=''):
    # if not os.path.exists(dataset_dir):
    #     os.makedirs(dataset_dir)
    
    with open(os.path.join(dataset_dir, 'train.en'), 'w', encoding='utf-8') as file:
        file.writelines(en_train)
    with open(os.path.join(dataset_dir, 'train.zh'), 'w', encoding='utf-8') as file:
        file.writelines(zh_train)
    
    with open(os.path.join(dataset_dir, 'val.en'), 'w', encoding='utf-8') as file:
        file.writelines(en_val)
    with open(os.path.join(dataset_dir, 'val.zh'), 'w', encoding='utf-8') as file:
        file.writelines(zh_val)
    
    with open(os.path.join(dataset_dir, 'test.en'), 'w', encoding='utf-8') as file:
        file.writelines(en_test)
    with open(os.path.join(dataset_dir, 'test.zh'), 'w', encoding='utf-8') as file:
        file.writelines(zh_test)

# 加载数据集
file_path = 'news-commentary-v15.en-zh.tsv'
en_sentences, zh_sentences = load_dataset(file_path)

# 划分数据集
en_train, zh_train, en_val, zh_val, en_test, zh_test = split_dataset(en_sentences, zh_sentences)

# 保存数据集
save_datasets(en_train, zh_train, en_val, zh_val, en_test, zh_test)
