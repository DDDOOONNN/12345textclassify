import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import warnings
from transformers import logging as transformers_logging

# ==================== 自定义Dataset ====================
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss

        if loss.dim() > 0:
            loss = loss.mean()

        logits = outputs.logits
        
        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # 每accumulation_steps步更新一次参数
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        
        current_acc = correct_predictions / total_samples
        progress_bar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    # 处理最后不足一个accumulation_steps的部分
    if (step + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

# ==================== 评估函数 ====================
def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss

            if loss.dim() > 0:
                loss = loss.mean()
            
            logits = outputs.logits
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

# ==================== 主函数 ====================
def main():
    # ==================== 配置参数 ====================
    MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
    DATA_PATH = '/data/gtm/textclassify/datasets/traindatasets100up.csv'
    OUTPUT_DIR = '/data/gtm/textclassify/models1/roberta_classifier'
    MAX_LENGTH = 128
    
    # ====== 优化GPU利用率的配置 ======
    BATCH_SIZE = 64
    GRADIENT_ACCUMULATION_STEPS = 2
    # 有效batch size = 64 x 4 GPUs x 2 accumulation = 512
    
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # DataLoader优化
    NUM_WORKERS = 16
    PREFETCH_FACTOR = 4
    PERSISTENT_WORKERS = True
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    
    # ==================== 加载和处理数据 ====================
    print('\n' + '='*60)
    print('Loading and preprocessing data...')
    print('='*60)
    
    df = pd.read_csv(DATA_PATH)
    print(f'Total samples: {len(df)}')
    print(f'Columns: {df.columns.tolist()}')
    
    # 编码标签
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    num_labels = len(label_encoder.classes_)
    print(f'Number of labels: {num_labels}')
    print(f'Label distribution:\n{df["label"].value_counts()}')
    
    # ==================== 划分数据集 ====================
    print('\n' + '='*60)
    print('Splitting dataset...')
    print('='*60)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['summary'].values,
        df['label_encoded'].values,
        test_size=0.1,  # 修改为10%验证集
        random_state=42,
        stratify=df['label_encoded'].values
    )
    
    print(f'Training samples: {len(train_texts)}')
    print(f'Validation samples: {len(val_texts)}')
    
    # ==================== 加载模型和tokenizer ====================
    print('\n' + '='*60)
    print('Loading tokenizer and model...')
    print('='*60)
    
    # 消除警告信息
    transformers_logging.set_verbosity_error()
    warnings.filterwarnings('ignore')
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    
    # 使用DataParallel进行多GPU训练
    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    print(f'Model loaded: {MODEL_NAME}')
    
    # ==================== 创建数据集和数据加载器 ====================
    print('\n' + '='*60)
    print('Creating dataloaders...')
    print('='*60)
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS
    )
    
    print(f'Train batches: {len(train_dataloader)}')
    print(f'Validation batches: {len(val_dataloader)}')
    print(f'Batch size per GPU: {BATCH_SIZE}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print(f'Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}')
    print(f'Effective batch size: {BATCH_SIZE} x {torch.cuda.device_count()} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * torch.cuda.device_count() * GRADIENT_ACCUMULATION_STEPS}')
    
    # ==================== 优化器和调度器 ====================
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_dataloader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f'\nTotal training steps: {total_steps}')
    print(f'Warmup steps: {warmup_steps}')
    
    # ==================== 训练循环 ====================
    print('\n' + '='*60)
    print('Starting training...')
    print('='*60)
    
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 60)
        
        train_loss, train_acc = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            scheduler, 
            device,
            GRADIENT_ACCUMULATION_STEPS
        )
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        
        val_loss, val_acc = eval_model(model, val_dataloader, device)
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            
            # 如果使用了DataParallel，需要保存module
            model_to_save = model.module if hasattr(model, 'module') else model
            
            model_to_save.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            
            # 保存标签编码器
            import pickle
            with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(label_encoder, f)
            
            print(f'✓ Best model saved with accuracy: {best_accuracy:.4f}')
    
    print('\n' + '='*60)
    print('Training completed!')
    print(f'Best validation accuracy: {best_accuracy:.4f}')
    print('='*60)

if __name__ == '__main__':
    main()