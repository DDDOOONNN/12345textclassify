import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import os
import pickle
import json

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

set_seed(42)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
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
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions.double() / total_predictions:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_predictions
    
    return avg_loss, accuracy

def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
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
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.sum(np.array(predictions) == np.array(true_labels)) / len(true_labels)
    
    return avg_loss, accuracy, predictions, true_labels

def main():
    # ==================== 配置参数 ====================
    MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
    DATA_PATH = '/data/gtm/textclassify/datasets/traindatasets100up.csv'
    OUTPUT_DIR = '/data/gtm/textclassify/models/roberta_classifier'
    MAX_LENGTH = 128
    BATCH_SIZE = 16  # 每张卡的batch size，总batch size = 16 * 4 = 64
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # 指定使用的GPU
    GPU_IDS = [4, 5, 6, 7]  # 使用cuda:4, 5, 6, 7
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, GPU_IDS))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置主设备
    device = torch.device('cuda:0')  # 因为设置了CUDA_VISIBLE_DEVICES，所以这里是第一张可见的卡
    print(f'Using {len(GPU_IDS)} GPUs: {GPU_IDS}')
    print(f'Primary device: {device}')
    
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
    
    # ==================== 加载和处理数据 ====================
    print('\n' + '='*60)
    print('Loading data...')
    print('='*60)
    df = pd.read_csv(DATA_PATH)
    print(f'Total samples: {len(df)}')
    
    df = df.dropna(subset=['summary', 'label'])
    print(f'Samples after removing missing values: {len(df)}')
    print('\nLabel distribution:')
    print(df['label'].value_counts())
    
    # ==================== 编码标签 ====================
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    num_labels = len(label_encoder.classes_)
    print(f'\nNumber of classes: {num_labels}')
    print(f'Classes: {list(label_encoder.classes_)}')
    
    label_encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    
    # ==================== 划分数据集 ====================
    print('\n' + '='*60)
    print('Splitting dataset...')
    print('='*60)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['summary'].values,
        df['label_encoded'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label_encoded'].values
    )
    
    print(f'Training samples: {len(train_texts)}')
    print(f'Validation samples: {len(val_texts)}')
    
    # ==================== 加载模型和tokenizer ====================
    print('\n' + '='*60)
    print('Loading tokenizer and model...')
    print('='*60)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    
    # ========== 关键：使用DataParallel包装模型 ==========
    if torch.cuda.device_count() > 1:
        print(f'\nUsing DataParallel with {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    
    model.to(device)
    
    print(f'Model loaded: {MODEL_NAME}')
    
    # ==================== 创建数据加载器 ====================
    print('\n' + '='*60)
    print('Creating dataloaders...')
    print('='*60)
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # 增加worker数量
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True
    )
    
    print(f'Train batches: {len(train_dataloader)}')
    print(f'Validation batches: {len(val_dataloader)}')
    print(f'Effective batch size: {BATCH_SIZE} x {torch.cuda.device_count()} = {BATCH_SIZE * torch.cuda.device_count()}')
    
    # ==================== 设置优化器和学习率调度器 ====================
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_dataloader) * EPOCHS
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
    training_stats = []
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 60)
        
        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        
        val_loss, val_acc, val_preds, val_true = eval_model(
            model, val_dataloader, device
        )
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': float(train_acc),
            'val_loss': val_loss,
            'val_acc': float(val_acc)
        })
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            
            # 保存模型时需要unwrap DataParallel
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f'✓ Best model saved with accuracy: {best_accuracy:.4f}')
            
            print('\nClassification Report:')
            report = classification_report(
                val_true,
                val_preds,
                target_names=label_encoder.classes_,
                digits=4
            )
            print(report)
            
            with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
                f.write(report)
            
            cm = confusion_matrix(val_true, val_preds)
            np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), cm)
    
    # ==================== 保存训练历史 ====================
    with open(os.path.join(OUTPUT_DIR, 'training_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(training_stats, f, indent=2)
    
    config = {
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'num_gpus': torch.cuda.device_count(),
        'effective_batch_size': BATCH_SIZE * torch.cuda.device_count(),
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'warmup_ratio': WARMUP_RATIO,
        'weight_decay': WEIGHT_DECAY,
        'num_labels': num_labels,
        'best_accuracy': float(best_accuracy),
        'train_samples': len(train_texts),
        'val_samples': len(val_texts)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print('\n' + '='*60)
    print('Training completed!')
    print('='*60)
    print(f'Best validation accuracy: {best_accuracy:.4f}')
    print(f'Model saved to: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()