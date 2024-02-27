def is_garbage(line):
    # 假设乱码由重复的引号字符组成，我们可以检查是否有大量连续重复的字符
    return '"' in line or "'" in line or "’" in line or "“" in line or "-" in line or "—" in line or "”" in line or "-" in line or "，" in line or "：" in line

def clean_data_sync(en_lines, zh_lines, is_garbage_func):
    # 确保两个列表的长度相同
    assert len(en_lines) == len(zh_lines), "Files do not have the same number of lines."
    
    # 清除两个列表中的空行和乱码行，确保它们的索引保持同步
    cleaned_en_lines = []
    cleaned_zh_lines = []
    for en_line, zh_line in zip(en_lines, zh_lines):
        if en_line.strip() and zh_line.strip() and not is_garbage_func(en_line) and not is_garbage_func(zh_line):
            cleaned_en_lines.append(en_line)
            cleaned_zh_lines.append(zh_line)
            
    return cleaned_en_lines, cleaned_zh_lines

# 文件路径
en_file_path = '2test.en'
zh_file_path = '2test.zh'

# 读取文件
with open(en_file_path, 'r', encoding='utf-8') as f_en:
    en_lines = f_en.readlines()
    
with open(zh_file_path, 'r', encoding='utf-8') as f_zh:
    zh_lines = f_zh.readlines()

# 清洗数据
clean_en, clean_zh = clean_data_sync(en_lines, zh_lines, is_garbage)

# 检查清洗后的数据行数是否相同，此时它们应该是相同的
assert len(clean_en) == len(clean_zh), "The number of cleaned lines does not match after cleaning."

# 写入清洗后的数据到新文件
cleaned_en_file_path = f'cleaned_{en_file_path}'
cleaned_zh_file_path = f'cleaned_{zh_file_path}'

with open(cleaned_en_file_path, 'w', encoding='utf-8') as f_clean_en:
    f_clean_en.writelines(clean_en)
    
with open(cleaned_zh_file_path, 'w', encoding='utf-8') as f_clean_zh:
    f_clean_zh.writelines(clean_zh)

# 输出清洗后的数据行数和新文件的路径
print(len(clean_en), cleaned_en_file_path, cleaned_zh_file_path)
