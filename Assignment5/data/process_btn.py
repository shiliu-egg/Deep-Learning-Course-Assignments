import random

# 假设文件路径
en_file_path = 'news.en'
zh_file_path = 'news.translatedto.zh'

# 读取数据
with open(en_file_path, 'r', encoding='utf-8') as en_file, open(zh_file_path, 'r', encoding='utf-8') as zh_file:
    en_lines = en_file.readlines()
    zh_lines = zh_file.readlines()

# 检查数据长度是否相同
assert len(en_lines) == len(zh_lines), "The number of sentences in both files should be the same."

# 将数据合并为元组列表
combined_data = list(zip(en_lines, zh_lines))

# 随机打乱数据
# random.shuffle(combined_data)

# 定义数据集的大小
train_size = 2000000
valid_test_size = 400000  # 验证集和测试集的大小相同

# 分割数据集
train_data = combined_data[:train_size]
valid_data = combined_data[train_size:train_size + valid_test_size]
test_data = combined_data[train_size + valid_test_size:train_size + 2 * valid_test_size]

# 定义保存数据的函数
def save_data(dataset, en_filename, zh_filename):
    with open(en_filename, 'w', encoding='utf-8') as en_file, open(zh_filename, 'w', encoding='utf-8') as zh_file:
        for en_line, zh_line in dataset:
            en_file.write(en_line)
            zh_file.write(zh_line)

# 保存数据集
save_data(train_data, '2train.en', '2train.zh')
save_data(valid_data, '2valid.en', '2valid.zh')
save_data(test_data, '2test.en', '2test.zh')

print("Datasets have been successfully split and saved.")
