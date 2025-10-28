import pandas as pd
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# 读取数据
data_path = '/data/gtm/textclassify/datasets/summary_label_cleaned.csv'
df = pd.read_csv(data_path)

print("加载tokenizer...")
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

# 计算每个summary的token长度
print("正在计算token长度...")
token_lengths = []
for text in tqdm(df['summary'].astype(str), desc="Tokenizing"):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_lengths.append(len(tokens))

df['token_length'] = token_lengths

# 基本统计信息
print("\n" + "=" * 60)
print("Summary Token长度统计:")
print("=" * 60)
print(f"总样本数: {len(df)}")
print(f"最小token长度: {df['token_length'].min()}")
print(f"最大token长度: {df['token_length'].max()}")
print(f"平均token长度: {df['token_length'].mean():.2f}")
print(f"中位数token长度: {df['token_length'].median():.2f}")
print(f"标准差: {df['token_length'].std():.2f}")

# 百分位数统计
print("\n百分位数统计:")
print("-" * 60)
percentiles = [50, 75, 80, 85, 90, 95, 99]
for p in percentiles:
    value = np.percentile(df['token_length'], p)
    print(f"{p}%的数据token长度 <= {value:.0f}")

# token长度区间分布
print("\nToken长度区间分布:")
print("-" * 60)
bins = [0, 32, 64, 128, 192, 256, 384, 512, float('inf')]
labels = ['0-32', '32-64', '64-128', '128-192', '192-256', '256-384', '384-512', '512+']
df['token_range'] = pd.cut(df['token_length'], bins=bins, labels=labels)
distribution = df['token_range'].value_counts().sort_index()
for label, count in distribution.items():
    percentage = (count / len(df)) * 100
    print(f"{label:12s}: {count:5d} 样本 ({percentage:5.2f}%)")

# 显示一些示例
print("\n最短的3个样本:")
print("-" * 60)
shortest = df.nsmallest(3, 'token_length')
for idx, row in shortest.iterrows():
    tokens = tokenizer.encode(row['summary'], add_special_tokens=True)
    print(f"Token长度: {row['token_length']}")
    print(f"内容: {row['summary'][:100]}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)[:20]}")
    print()

print("最长的3个样本:")
print("-" * 60)
longest = df.nlargest(3, 'token_length')
for idx, row in longest.iterrows():
    tokens = tokenizer.encode(row['summary'], add_special_tokens=True)
    print(f"Token长度: {row['token_length']}")
    print(f"内容: {row['summary'][:100]}...")
    print(f"前20个Tokens: {tokenizer.convert_ids_to_tokens(tokens)[:20]}")
    print()

# 推荐MAX_LENGTH
print("=" * 60)
print("推荐的MAX_LENGTH设置:")
print("=" * 60)
p90 = int(np.percentile(df['token_length'], 90))
p95 = int(np.percentile(df['token_length'], 95))
p99 = int(np.percentile(df['token_length'], 99))

print(f"如果设置MAX_LENGTH={p90}, 可以覆盖90%的数据")
print(f"如果设置MAX_LENGTH={p95}, 可以覆盖95%的数据")
print(f"如果设置MAX_LENGTH={p99}, 可以覆盖99%的数据")

# 考虑到BERT的常见长度限制和效率
common_lengths = [64, 128, 192, 256, 384, 512]
print("\n常用MAX_LENGTH的覆盖率:")
print("-" * 60)
for length in common_lengths:
    coverage = (df['token_length'] <= length).sum() / len(df) * 100
    truncated = (df['token_length'] > length).sum()
    print(f"MAX_LENGTH={length:3d}: 覆盖 {coverage:5.2f}% 的数据, {truncated:4d} 个样本会被截断")

# 计算会被截断的平均token数
print("\n截断损失分析:")
print("-" * 60)
for length in common_lengths:
    truncated_samples = df[df['token_length'] > length]
    if len(truncated_samples) > 0:
        avg_loss = (truncated_samples['token_length'] - length).mean()
        max_loss = (truncated_samples['token_length'] - length).max()
        print(f"MAX_LENGTH={length:3d}: 平均截断 {avg_loss:.1f} tokens, 最大截断 {max_loss:.0f} tokens")
    else:
        print(f"MAX_LENGTH={length:3d}: 无样本被截断")

# 绘制token长度分布图
try:
    plt.figure(figsize=(14, 6))
    
    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(df['token_length'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Token长度')
    plt.ylabel('样本数量')
    plt.title('Summary Token长度分布直方图')
    plt.axvline(df['token_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均值: {df["token_length"].mean():.0f}')
    plt.axvline(df['token_length'].median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {df["token_length"].median():.0f}')
    
    # 添加常用MAX_LENGTH参考线
    for length in [128, 256, 512]:
        coverage = (df['token_length'] <= length).sum() / len(df) * 100
        plt.axvline(length, color='orange', linestyle=':', alpha=0.6, label=f'{length} ({coverage:.1f}%)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 累积分布图
    plt.subplot(1, 2, 2)
    sorted_lengths = np.sort(df['token_length'])
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, cumulative, linewidth=2)
    plt.xlabel('Token长度')
    plt.ylabel('累积百分比 (%)')
    plt.title('Token长度累积分布')
    plt.grid(True, alpha=0.3)
    
    # 添加常用MAX_LENGTH参考线
    for length in [128, 256, 512]:
        coverage = (df['token_length'] <= length).sum() / len(df) * 100
        plt.axvline(length, color='red', linestyle='--', alpha=0.5)
        plt.axhline(coverage, color='red', linestyle='--', alpha=0.5)
        plt.text(length, 5, f'{length}', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('/data/gtm/textclassify/datasets/token_length_distribution.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存至: /data/gtm/textclassify/datasets/token_length_distribution.png")
except Exception as e:
    print(f"\n注意: 无法生成图表 - {str(e)}")

# 保存详细统计结果
stats_output = '/data/gtm/textclassify/datasets/token_length_stats.txt'
with open(stats_output, 'w', encoding='utf-8') as f:
    f.write(f"Summary Token长度统计\n")
    f.write(f"{'=' * 60}\n")
    f.write(f"总样本数: {len(df)}\n")
    f.write(f"最小token长度: {df['token_length'].min()}\n")
    f.write(f"最大token长度: {df['token_length'].max()}\n")
    f.write(f"平均token长度: {df['token_length'].mean():.2f}\n")
    f.write(f"中位数token长度: {df['token_length'].median():.2f}\n")
    f.write(f"标准差: {df['token_length'].std():.2f}\n\n")
    
    for length in common_lengths:
        coverage = (df['token_length'] <= length).sum() / len(df) * 100
        f.write(f"MAX_LENGTH={length}: 覆盖 {coverage:.2f}% 的数据\n")

print(f"\n统计结果已保存至: {stats_output}")

print("\n" + "=" * 60)
print("建议:")
print("=" * 60)
print("根据token长度统计结果选择MAX_LENGTH:")
print("- 如果追求训练速度，选择能覆盖90%数据的长度")
print("- 如果追求准确性，选择能覆盖95-99%数据的长度")
print("- 平衡考虑，推荐使用128或256")
print("=" * 60)