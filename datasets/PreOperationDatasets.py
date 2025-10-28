import pandas as pd
import os

# 读取数据
input_file = '/data/gtm/textclassify/datasets/summary_label.csv'
output_file = '/data/gtm/textclassify/datasets/summary_label_cleaned.csv'

# 读取CSV文件
df = pd.read_csv(input_file)

# 查看原始数据的前几行
print("原始数据示例：")
print(df.head())
print(f"\n原始数据总行数: {len(df)}")

# 清洗label列，只保留第一个分类
# 如果label包含逗号，则只保留逗号前的第一个类别
df['label'] = df['label'].astype(str).apply(lambda x: x.split(',')[0].strip())

# 查看清洗后的数据
print("\n清洗后数据示例：")
print(df.head())

# 统计各个label的数量
print("\n各类别数量统计：")
print(df['label'].value_counts())

# 保存到新文件
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n数据已保存到: {output_file}")
print(f"清洗后数据总行数: {len(df)}")