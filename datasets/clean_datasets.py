import pandas as pd
import os

def clean_dataset(input_path, output_path, min_samples=100):
    """
    清洗数据集，删除样本数少于min_samples的类别
    
    Args:
        input_path: 原始数据集路径
        output_path: 清洗后数据集保存路径
        min_samples: 每个类别最少需要的样本数
    """
    print('='*60)
    print('数据集清洗工具')
    print('='*60)
    
    # 读取数据
    print(f'\n读取数据: {input_path}')
    df = pd.read_csv(input_path)
    print(f'原始样本数: {len(df)}')
    
    # 删除缺失值
    original_size = len(df)
    df = df.dropna(subset=['summary', 'label'])
    if len(df) < original_size:
        print(f'删除了 {original_size - len(df)} 条缺失数据')
        print(f'剩余样本数: {len(df)}')
    
    # 统计每个类别的样本数
    print('\n原始类别分布:')
    label_counts = df['label'].value_counts().sort_values(ascending=False)
    print(f'总类别数: {len(label_counts)}')
    print('-'*60)
    for label, count in label_counts.items():
        status = '✓ 保留' if count >= min_samples else '✗ 删除'
        print(f'{status} | {label}: {count} 样本')
    
    # 找出样本数少于min_samples的类别
    small_classes = label_counts[label_counts < min_samples].index.tolist()
    large_classes = label_counts[label_counts >= min_samples].index.tolist()
    
    print('\n'+'='*60)
    print(f'筛选标准: 每个类别至少需要 {min_samples} 个样本')
    print('='*60)
    print(f'需要删除的类别数: {len(small_classes)}')
    print(f'保留的类别数: {len(large_classes)}')
    
    if len(small_classes) > 0:
        print(f'\n将删除以下类别:')
        for cls in small_classes:
            print(f'  - {cls}: {label_counts[cls]} 样本')
        
        # 过滤数据
        df_cleaned = df[df['label'].isin(large_classes)].copy()
        removed_samples = len(df) - len(df_cleaned)
        
        print(f'\n删除了 {removed_samples} 个样本')
        print(f'剩余样本数: {len(df_cleaned)}')
        
        # 显示清洗后的类别分布
        print('\n清洗后的类别分布:')
        cleaned_label_counts = df_cleaned['label'].value_counts().sort_values(ascending=False)
        print('-'*60)
        for label, count in cleaned_label_counts.items():
            print(f'{label}: {count} 样本')
        
        # 保存清洗后的数据
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        print(f'\n清洗后的数据已保存到: {output_path}')
        
        # 保存统计信息
        stats_path = output_path.replace('.csv', '_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write('='*60 + '\n')
            f.write('数据清洗统计信息\n')
            f.write('='*60 + '\n')
            f.write(f'原始样本数: {len(df)}\n')
            f.write(f'清洗后样本数: {len(df_cleaned)}\n')
            f.write(f'删除样本数: {removed_samples}\n')
            f.write(f'原始类别数: {len(label_counts)}\n')
            f.write(f'清洗后类别数: {len(cleaned_label_counts)}\n')
            f.write(f'删除类别数: {len(small_classes)}\n')
            f.write(f'最小样本数标准: {min_samples}\n\n')
            
            f.write('删除的类别:\n')
            f.write('-'*60 + '\n')
            for cls in small_classes:
                f.write(f'{cls}: {label_counts[cls]} 样本\n')
            
            f.write('\n保留的类别分布:\n')
            f.write('-'*60 + '\n')
            for label, count in cleaned_label_counts.items():
                f.write(f'{label}: {count} 样本\n')
        
        print(f'统计信息已保存到: {stats_path}')
        
    else:
        print('\n所有类别的样本数都满足要求，无需清洗')
        df_cleaned = df.copy()
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        print(f'数据已保存到: {output_path}')
    
    print('\n'+'='*60)
    print('清洗完成!')
    print('='*60)
    
    return df_cleaned

if __name__ == '__main__':
    # 配置参数
    INPUT_PATH = '/data/gtm/textclassify/datasets/summary_label_cleaned.csv'
    OUTPUT_PATH = '/data/gtm/textclassify/datasets/summary_label_cleaned_min100.csv'
    MIN_SAMPLES = 100  # 每个类别最少需要的样本数
    
    # 执行清洗
    df_cleaned = clean_dataset(INPUT_PATH, OUTPUT_PATH, MIN_SAMPLES)
    
    print(f'\n最终数据集信息:')
    print(f'样本总数: {len(df_cleaned)}')
    print(f'类别总数: {df_cleaned["label"].nunique()}')
    print(f'\n可以在 train.py 中将数据路径修改为:')
    print(f'DATA_PATH = "{OUTPUT_PATH}"')