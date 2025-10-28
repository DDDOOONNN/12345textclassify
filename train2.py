import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_logging(output_dir):
    """设置日志系统"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f'Log file created: {log_file}')
    
    return logger

class TextClassificationDataset(Dataset):
    """文本分类数据集"""
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
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    
    for batch_idx, batch in enumerate(progress_bar):
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
        
        current_acc = correct_predictions.double() / total_predictions
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f'Epoch {epoch} - Batch {batch_idx + 1}/{len(dataloader)} - '
                       f'Loss: {loss.item():.4f}, Acc: {current_acc:.4f}, '
                       f'LR: {scheduler.get_last_lr()[0]:.2e}')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_predictions
    
    return avg_loss, accuracy

def validate(model, dataloader, device, logger):
    """验证模型"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validating')
        
        for batch in progress_bar:
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
            
            current_acc = np.mean(np.array(predictions) == np.array(true_labels))
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    return avg_loss, accuracy, predictions, true_labels

class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience=3, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False

def print_detailed_metrics(val_true, val_preds, label_encoder, logger, output_dir):
    """打印详细的分类指标"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # 计算每个类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        val_true, val_preds, average=None
    )
    
    logger.info('\n' + '='*80)
    logger.info('DETAILED CLASSIFICATION METRICS')
    logger.info('='*80)
    
    # 打印表头
    logger.info(f"\n{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    logger.info('-' * 80)
    
    # 打印每个类别的指标
    for i, label in enumerate(label_encoder.classes_):
        logger.info(f"{label:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                   f"{f1[i]:<12.4f} {support[i]:<10}")
    
    # 打印总体指标
    logger.info('-' * 80)
    overall_acc = accuracy_score(val_true, val_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        val_true, val_preds, average='macro'
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        val_true, val_preds, average='weighted'
    )
    
    logger.info(f"{'Overall Accuracy':<20} {overall_acc:.4f}")
    logger.info(f"{'Macro Avg':<20} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}")
    logger.info(f"{'Weighted Avg':<20} {weighted_p:<12.4f} {weighted_r:<12.4f} {weighted_f1:<12.4f}")
    
    # 保存详细指标到JSON
    detailed_metrics = {
        'per_class': {
            label_encoder.classes_[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(label_encoder.classes_))
        },
        'overall': {
            'accuracy': float(overall_acc),
            'macro_avg': {
                'precision': float(macro_p),
                'recall': float(macro_r),
                'f1_score': float(macro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_p),
                'recall': float(weighted_r),
                'f1_score': float(weighted_f1)
            }
        }
    }
    
    with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed_metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f'\n✓ Detailed metrics saved to: {output_dir}/detailed_metrics.json')
    
    return detailed_metrics

def main():
    # ==================== 配置参数 ====================
    MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
    DATA_PATH = '/data/gtm/textclassify/datasets/forth_summary_keyword/forth_summary_keyword_processed.csv'
    OUTPUT_DIR = '/data/gtm/textclassify/models42/roberta_classifier'
    MAX_LENGTH = 128
    
    # 🔥 优化后的配置
    BATCH_SIZE = 64  # 每卡64，8卡总batch = 512
    NUM_GPUS = 8
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * NUM_GPUS  # 512

    EPOCHS = 15  # 从5增加到15
    LEARNING_RATE = 5e-5  # 根据更大的batch size调整
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 3  # 连续3个epoch验证集没提升就停止
    MIN_DELTA = 0.001  # 最小提升阈值
    
    # 新增：从checkpoint恢复训练
    RESUME_FROM_CHECKPOINT = False
    CHECKPOINT_PATH = None
    
    # 新增：每N个epoch保存一次checkpoint
    SAVE_CHECKPOINT_EVERY = 1
    
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 使用8张卡
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, GPU_IDS))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==================== 设置日志系统 ====================
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info('='*80)
    logger.info('TRAINING CONFIGURATION')
    logger.info('='*80)
    logger.info(f'Model: {MODEL_NAME}')
    logger.info(f'Data path: {DATA_PATH}')
    logger.info(f'Output directory: {OUTPUT_DIR}')
    logger.info(f'Max length: {MAX_LENGTH}')
    logger.info(f'Batch size per GPU: {BATCH_SIZE}')
    logger.info(f'Number of GPUs: {NUM_GPUS}')
    logger.info(f'Effective batch size: {EFFECTIVE_BATCH_SIZE}')
    logger.info(f'Epochs: {EPOCHS}')
    logger.info(f'Learning rate: {LEARNING_RATE:.2e}')
    logger.info(f'Warmup ratio: {WARMUP_RATIO}')
    logger.info(f'Weight decay: {WEIGHT_DECAY}')
    logger.info(f'Early stopping patience: {EARLY_STOPPING_PATIENCE}')
    logger.info(f'Min delta: {MIN_DELTA}')
    
    device = torch.device('cuda:0')
    logger.info(f'\nUsing {NUM_GPUS} GPUs: {GPU_IDS}')
    logger.info(f'Primary device: {device}')
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f'GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)')
    
    # ==================== 加载和预处理数据 ====================
    logger.info('\n' + '='*80)
    logger.info('LOADING AND PREPROCESSING DATA')
    logger.info('='*80)
    
    df = pd.read_csv(DATA_PATH)
    logger.info(f'Total samples: {len(df)}')
    logger.info(f'Columns: {df.columns.tolist()}')
    
    # 数据清洗
    df = df.dropna(subset=['summary', 'label'])
    df['summary'] = df['summary'].astype(str).str.strip()
    df = df[df['summary'].str.len() > 0]
    
    logger.info(f'Samples after cleaning: {len(df)}')
    logger.info(f'\nLabel distribution:')
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        logger.info(f'  {label}: {count} ({count/len(df)*100:.2f}%)')
    
    # 编码标签
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])
    
    num_classes = len(label_encoder.classes_)
    logger.info(f'\nNumber of classes: {num_classes}')
    logger.info(f'Class names: {label_encoder.classes_.tolist()}')
    
    # 保存label encoder
    label_encoder_info = {
        'classes': label_encoder.classes_.tolist(),
        'num_classes': num_classes
    }
    with open(os.path.join(OUTPUT_DIR, 'label_encoder.json'), 'w', encoding='utf-8') as f:
        json.dump(label_encoder_info, f, ensure_ascii=False, indent=2)
    logger.info(f'✓ Label encoder saved to: {OUTPUT_DIR}/label_encoder.json')
    
    # 分层划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['summary'].values,
        df['encoded_label'].values,
        test_size=0.1,
        random_state=41,
        stratify=df['encoded_label'].values
    )
    
    logger.info(f'\nTrain samples: {len(train_texts)}')
    logger.info(f'Validation samples: {len(val_texts)}')
    
    # ==================== 初始化模型和分词器 ====================
    logger.info('\n' + '='*80)
    logger.info('INITIALIZING MODEL AND TOKENIZER')
    logger.info('='*80)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    if RESUME_FROM_CHECKPOINT and CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        logger.info(f'Loading model from checkpoint: {CHECKPOINT_PATH}')
        model = BertForSequenceClassification.from_pretrained(
            CHECKPOINT_PATH,
            num_labels=num_classes
        )
    else:
        logger.info(f'Loading pretrained model: {MODEL_NAME}')
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes
        )
    
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # ==================== 创建数据集和数据加载器 ====================
    logger.info('\n' + '='*80)
    logger.info('CREATING DATASETS AND DATALOADERS')
    logger.info('='*80)
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,  # 8卡使用16个worker
        pin_memory=True,
        prefetch_factor=2  # 预取数据
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )
    
    logger.info(f'Train batches: {len(train_dataloader)}')
    logger.info(f'Validation batches: {len(val_dataloader)}')
    logger.info(f'Samples per batch: {EFFECTIVE_BATCH_SIZE}')
    
    # ==================== 设置优化器和学习率调度器 ====================
    logger.info('\n' + '='*80)
    logger.info('SETTING UP OPTIMIZER AND SCHEDULER')
    logger.info('='*80)
    
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )
    
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f'Total training steps: {total_steps}')
    logger.info(f'Warmup steps: {warmup_steps}')
    
    # ==================== 初始化早停机制 ====================
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        mode='max'
    )
    
    # ==================== 训练历史记录 ====================
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # ==================== 训练循环 ====================
    logger.info('\n' + '='*80)
    logger.info('STARTING TRAINING')
    logger.info('='*80)
    
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        logger.info(f'\n{"="*80}')
        logger.info(f'Epoch {epoch + 1}/{EPOCHS}')
        logger.info(f'{"="*80}')
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, epoch + 1, logger
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_true = validate(
            model, val_dataloader, device, logger
        )
        
        # 记录训练历史
        training_history['train_loss'].append(float(train_loss))
        training_history['train_acc'].append(float(train_acc))
        training_history['val_loss'].append(float(val_loss))
        training_history['val_acc'].append(float(val_acc))
        training_history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        logger.info(f'\n📊 Epoch {epoch + 1} Summary:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        logger.info(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info(f'✅ Best model saved with accuracy: {best_accuracy:.4f}')
            
            # 保存分类报告
            logger.info('\nClassification Report:')
            report = classification_report(
                val_true,
                val_preds,
                target_names=label_encoder.classes_,
                digits=4
            )
            logger.info('\n' + report)
            
            with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
                f.write(f'Best Epoch: {epoch + 1}\n')
                f.write(f'Validation Accuracy: {val_acc:.4f}\n\n')
                f.write(report)
            
            # 保存详细指标
            print_detailed_metrics(val_true, val_preds, label_encoder, logger, OUTPUT_DIR)
            
            # 保存混淆矩阵
            cm = confusion_matrix(val_true, val_preds)
            np.save(os.path.join(OUTPUT_DIR, 'confusion_matrix.npy'), cm)
        
        # 定期保存checkpoint
        if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_dir = os.path.join(OUTPUT_DIR, f'checkpoint-epoch-{epoch + 1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f'💾 Checkpoint saved: {checkpoint_dir}')
        
        # 早停检查
        if early_stopping(val_acc, epoch + 1):
            logger.info(f'\n⏹️  Early stopping triggered!')
            logger.info(f'   Best epoch: {early_stopping.best_epoch}')
            logger.info(f'   Best validation accuracy: {early_stopping.best_score:.4f}')
            logger.info(f'   No improvement for {EARLY_STOPPING_PATIENCE} epochs')
            break
    
    # 保存训练历史
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info('\n' + '='*80)
    logger.info('TRAINING COMPLETED')
    logger.info('='*80)
    logger.info(f'Best validation accuracy: {best_accuracy:.4f}')
    logger.info(f'Model saved to: {OUTPUT_DIR}')
    logger.info(f'Training history saved to: {OUTPUT_DIR}/training_history.json')

if __name__ == '__main__':
    main()

# nohup python your_script.py > training.log 2>&1 &