import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os

def load_model(model_path):
    """加载模型和tokenizer"""
    print(f'📂 Loading model from: {model_path}')
    
    # 加载label encoder
    label_encoder_path = os.path.join(model_path, 'label_encoder.json')
    with open(label_encoder_path, 'r', encoding='utf-8') as f:
        label_info = json.load(f)
    
    class_names = label_info['classes']
    
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f'✅ Model loaded! Device: {device}\n')
    
    return model, tokenizer, class_names, device

def predict_top5(text, model, tokenizer, class_names, device, max_length=128):
    """预测并返回Top-5结果"""
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]  # 获取概率
    
    # 获取Top-5
    top5_probs, top5_indices = torch.topk(probs, k=min(5, len(class_names)))
    
    # 构建结果
    results = []
    for i in range(len(top5_probs)):
        results.append({
            'rank': i + 1,
            'class': class_names[top5_indices[i]],
            'probability': float(top5_probs[i]),
            'percentage': float(top5_probs[i]) * 100
        })
    
    return results

def print_top5_results(text, results):
    """美观地打印Top-5结果"""
    print('='*80)
    print('📝 INPUT TEXT:')
    print('='*80)
    print(f'{text}\n')
    
    print('='*80)
    print('🏆 TOP-5 PREDICTIONS:')
    print('='*80)
    
    for result in results:
        rank = result['rank']
        class_name = result['class']
        prob = result['probability']
        percent = result['percentage']
        
        # 生成进度条
        bar_length = int(prob * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        # 添加奖牌emoji
        medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][rank - 1]
        
        print(f'{medal} Rank {rank}: {class_name:<20} {percent:6.2f}% {bar} ({prob:.4f})')
    
    print('='*80 + '\n')

def interactive_mode(model, tokenizer, class_names, device):
    """交互式测试模式"""
    print('\n' + '='*80)
    print('🎮 INTERACTIVE TOP-5 PREDICTION MODE')
    print('='*80)
    print('Enter text to see Top-5 predictions (type "quit" or "exit" to stop)\n')
    
    while True:
        text = input('📝 Input text: ').strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print('👋 Goodbye!')
            break
        
        if not text:
            print('⚠️  Empty text, please try again.\n')
            continue
        
        results = predict_top5(text, model, tokenizer, class_names, device)
        print_top5_results(text, results)

def main():
    # ==================== 配置 ====================
    MODEL_PATH = '/data/gtm/textclassify/models3/roberta_classifier/checkpoint-epoch-4'
    
    # 模式选择: 'single' 或 'interactive'
    MODE = 'single'  # 改为 'single' 可以测试单条文本
    
    # 单条文本测试（仅在MODE='single'时使用）
    TEST_TEXT = "潮州市湘桥区社会保险基金管理局在核查失业保险待遇时，发现一位人员在潮州和厦门思明区同时参保失业保险，需协查核实，要求提供厦门思明区失业保险科的具体联系方式及地址。"
    
    # ==================== 加载模型 ====================
    model, tokenizer, class_names, device = load_model(MODEL_PATH)
    
    print(f'📊 Total classes: {len(class_names)}')
    print(f'📋 Classes: {", ".join(class_names[:10])}{"..." if len(class_names) > 10 else ""}\n')
    
    # ==================== 执行预测 ====================
    if MODE == 'single':
        # 单条文本测试
        results = predict_top5(TEST_TEXT, model, tokenizer, class_names, device)
        print_top5_results(TEST_TEXT, results)
        
    elif MODE == 'interactive':
        # 交互式模式
        interactive_mode(model, tokenizer, class_names, device)
    
    else:
        print(f'❌ Unknown mode: {MODE}')
        print('   Available modes: single, interactive')

if __name__ == '__main__':
    main()