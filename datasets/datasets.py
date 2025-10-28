import pandas as pd
import os

def process_dataset(input_path, output_path, min_samples=100):
    """
    处理数据集：
    1. 清洗label列，只保留第一个分类
    2. 删除缺失值
    3. 删除样本数少于min_samples的类别
    
    Args:
        input_path: 原始数据集路径
        output_path: 处理后数据集保存路径
        min_samples: 每个类别最少需要的样本数
    """
    print('='*60)
    print('数据集处理工具')
    print('='*60)
    
    # ========== 步骤1: 读取数据 ==========
    print(f'\n【步骤1】读取数据: {input_path}')
    df = pd.read_csv(input_path)
    print(f'原始数据总行数: {len(df)}')
    
    # 显示原始数据示例
    print('\n原始数据示例：')
    print(df.head())
    
    # ========== 步骤2: 清洗label列 ==========
    print('\n'+'='*60)
    print('【步骤2】清洗label列，只保留第一个分类')
    print('='*60)
    
    # 只保留第一个分类（支持中文逗号和英文逗号）
    def extract_first_label(label):
        label_str = str(label).strip()
        # 先尝试中文逗号分割
        if '，' in label_str:
            return label_str.split('，')[0].strip()
        # 再尝试英文逗号分割
        elif ',' in label_str:
            return label_str.split(',')[0].strip()
        else:
            return label_str
    
    df['label'] = df['label'].apply(extract_first_label)
    
    print('\n清洗后数据示例：')
    print(df.head())
    
    # ========== 步骤3: 删除缺失值 ==========
    print('\n'+'='*60)
    print('【步骤3】删除缺失值')
    print('='*60)
    
    original_size = len(df)
    df = df.dropna(subset=['summary', 'label'])
    if len(df) < original_size:
        print(f'删除了 {original_size - len(df)} 条缺失数据')
    else:
        print('没有发现缺失数据')
    print(f'剩余样本数: {len(df)}')
    
    # ========== 步骤4: 统计类别分布 ==========
    print('\n'+'='*60)
    print('【步骤4】统计类别分布')
    print('='*60)
    
    label_counts = df['label'].value_counts().sort_values(ascending=False)
    print(f'总类别数: {len(label_counts)}')
    print('\n原始类别分布:')
    print('-'*60)
    for label, count in label_counts.items():
        status = '✓ 保留' if count >= min_samples else '✗ 删除'
        print(f'{status} | {label}: {count} 样本')
    
    # ========== 步骤5: 过滤小样本类别 ==========
    print('\n'+'='*60)
    print('【步骤5】过滤样本数不足的类别')
    print('='*60)
    print(f'筛选标准: 每个类别至少需要 {min_samples} 个样本')
    
    # 找出样本数少于min_samples的类别
    small_classes = label_counts[label_counts < min_samples].index.tolist()
    large_classes = label_counts[label_counts >= min_samples].index.tolist()
    
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
        
    else:
        print('\n所有类别的样本数都满足要求，无需过滤')
        df_cleaned = df.copy()
        cleaned_label_counts = label_counts
        removed_samples = 0
    
    # ========== 步骤6: 保存结果 ==========
    print('\n'+'='*60)
    print('【步骤6】保存处理结果')
    print('='*60)
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存处理后的数据
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
    print(f'处理后的数据已保存到: {output_path}')
    
    # 保存统计信息
    stats_path = output_path.replace('.csv', '_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('='*60 + '\n')
        f.write('数据处理统计信息\n')
        f.write('='*60 + '\n')
        f.write(f'原始样本数: {original_size}\n')
        f.write(f'处理后样本数: {len(df_cleaned)}\n')
        f.write(f'删除样本数: {removed_samples}\n')
        f.write(f'原始类别数: {len(label_counts)}\n')
        f.write(f'处理后类别数: {len(cleaned_label_counts)}\n')
        f.write(f'删除类别数: {len(small_classes)}\n')
        f.write(f'最小样本数标准: {min_samples}\n\n')
        
        if len(small_classes) > 0:
            f.write('删除的类别:\n')
            f.write('-'*60 + '\n')
            for cls in small_classes:
                f.write(f'{cls}: {label_counts[cls]} 样本\n')
        
        f.write('\n保留的类别分布:\n')
        f.write('-'*60 + '\n')
        for label, count in cleaned_label_counts.items():
            f.write(f'{label}: {count} 样本\n')
    
    print(f'统计信息已保存到: {stats_path}')
    
    # ========== 完成 ==========
    print('\n'+'='*60)
    print('处理完成!')
    print('='*60)
    print(f'\n最终数据集信息:')
    print(f'样本总数: {len(df_cleaned)}')
    print(f'类别总数: {df_cleaned["label"].nunique()}')
    print(f'\n可以在 train.py 中将数据路径修改为:')
    print(f'DATA_PATH = "{output_path}"')
    
    return df_cleaned

if __name__ == '__main__':
    # 配置参数
    INPUT_PATH = '/data/gtm/textclassify/datasets/third_summary_keyword/third_summary_keyword.csv'
    OUTPUT_PATH = '/data/gtm/textclassify/datasets/third_summary_keyword/third_summary_keyword_processed.csv'
    MIN_SAMPLES = 100  # 每个类别最少需要的样本数
    
    # 执行处理
    df_processed = process_dataset(INPUT_PATH, OUTPUT_PATH, MIN_SAMPLES)