import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from unified_models import get_model_by_number, get_model_info
import itertools
import os
import json
from datetime import datetime
os.makedirs('results', exist_ok=True)

# 1. 读取数据
DATA_FILE = 'btcusdt.json'
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 2. 数据预处理（与common_utils一致，预测下一小时涨跌）
def load_and_preprocess_data(file_path=DATA_FILE, sequence_length=30):
    with open(file_path, 'r') as f:
        data = pd.DataFrame(pd.read_json(f)) if file_path.endswith('.json') else pd.read_csv(f)
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    feature_columns = [col for col in data.columns if col not in ['target', 'open_time', 'close_time'] and data[col].dtype in ['float64', 'int64']]
    X = data[feature_columns].values[:-1]
    y = data['target'].values[:-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 构造序列
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)
    return X_seq, y_seq

# 4. 训练与验证
BATCH_SIZE = 1024
EPOCHS = 100
PATIENCE = 15
TARGET_ACC = 0.7

# 统一序列长度
SEQUENCE_LENGTH = 30

# 3. 超参数搜索空间（每个模型独立，batch size固定）
search_spaces = {
    1: dict(
        hidden_size=[64, 128, 256],
        num_layers=[1, 2, 3],
        dropout=[0.1, 0.2, 0.3],
        bidirectional=[True, False],
        attention_type=['multihead', 'self_attention', 'bahdanau', None],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]
    ),
    2: dict(
        hidden_size=[64, 128, 256],
        num_layers=[1, 2, 3],
        dropout=[0.1, 0.2, 0.3],
        bidirectional=[True, False],
        attention_type=['multihead', 'bahdanau', None],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]
    ),
    3: dict(
        d_model=[64, 128, 256],
        nhead=[4, 8],
        num_layers=[1, 2, 3],
        dropout=[0.1, 0.2, 0.3],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]
    ),
    4: dict(
        cnn_channels=[16, 32, 64],
        cnn_kernel=[3, 5],
        lstm_hidden=[32, 64, 128],
        lstm_layers=[1, 2],
        dropout=[0.1, 0.2, 0.3],
        bidirectional=[True, False],
        attention_heads=[2, 4, 8],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]
    ),
    5: dict(
        num_channels=[[16, 32], [32, 64], [64, 128]],
        kernel_size=[3, 5],
        dropout=[0.1, 0.2, 0.3],
        attention_heads=[2, 4, 8],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]
    ),
    6: dict(
        model_types=[['lstm', 'gru'], ['lstm', 'transformer'], ['gru', 'transformer']],
        hidden_sizes=[[64, 64], [128, 128]],
        num_layers=[[1, 1], [2, 2]],
        d_models=[[64, 64], [128, 128]],
        nheads=[[4, 4], [8, 8]],
        dropout=[0.1, 0.2],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2],
        patience=[10, 15],
        epochs=[50, 100]
    ),
    7: dict(
        d_model=[64, 128, 256],
        nhead=[4, 8],
        num_layers=[1, 2, 3],
        num_scales=[2, 3],
        dropout=[0.1, 0.2, 0.3],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]
    ),
    8: dict(
        n_d=[8, 16, 32],
        n_a=[8, 16, 32],
        n_steps=[1, 2, 3],
        gamma=[1.0, 1.3, 1.5],
        n_ind=[1, 2],
        n_shared=[1, 2],
        virtual_batch_size=[128, 256],
        learning_rate=[1e-4, 5e-4, 1e-3, 1e-2],
        weight_decay=[1e-4, 1e-3, 1e-2]
    )
}

# 5. 主流程
def save_exp_result(model_name, params, best_acc, best_epoch, extra_metrics=None):
    fname = f"results/{model_name}_valacc{best_acc:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result = {
        'model': model_name,
        'params': params,
        'best_val_acc': best_acc,
        'best_epoch': best_epoch,
        'timestamp': datetime.now().isoformat()
    }
    if extra_metrics:
        result.update(extra_metrics)
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)

# 全局最优追踪
GLOBAL_BEST = {
    'best_val_acc': 0,
    'model': None,
    'params': None,
    'best_epoch': 0
}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"检测到GPU数量: {n_gpus}")
    print(f"使用设备: {device}")
    
    # 加载数据
    X, y = load_and_preprocess_data(sequence_length=SEQUENCE_LENGTH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    for model_num in range(1, 9):
        model_name = get_model_info()[model_num]
        # 生成所有参数组合
        param_grid = search_spaces[model_num]
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for params in param_combinations:
            params['sequence_length'] = SEQUENCE_LENGTH
            print(f"\n==== 测试模型: {model_name} 参数: {params} ====")
            # 构造模型参数
            model_class = get_model_by_number(model_num)
            model_kwargs = params.copy()
            # 统一参数
            if model_num in [1,2]:
                model_kwargs['input_size'] = X_train.shape[2]
            elif model_num == 3:
                model_kwargs['input_size'] = X_train.shape[2]
            elif model_num == 4:
                model_kwargs['input_size'] = X_train.shape[2]
            elif model_num == 5:
                model_kwargs['input_size'] = X_train.shape[2]
            elif model_num == 6:
                model_kwargs['input_size'] = X_train.shape[2]
            elif model_num == 7:
                model_kwargs['input_size'] = X_train.shape[2]
            elif model_num == 8:
                model_kwargs['input_dim'] = X_train.shape[2]
                model_kwargs['output_dim'] = 1
            model_kwargs.pop('sequence_length', None)  # 新增，去除多余参数
            # 新增，去除优化器和训练相关参数
            for k in ['learning_rate', 'weight_decay', 'patience', 'epochs']:
                model_kwargs.pop(k, None)
            # 实例化模型
            if model_num == 6:
                model = model_class(
                    input_size=model_kwargs['input_size'],
                    sequence_length=params['sequence_length'],
                    config=params['config'] if 'config' in params else params
                )
            else:
                model = model_class(**model_kwargs)
            if n_gpus > 1:
                model = torch.nn.DataParallel(model)
            model = model.to(device)
            # 构造dataloader
            Xtr = torch.tensor(X_train, dtype=torch.float32)
            ytr = torch.tensor(y_train, dtype=torch.float32)
            Xva = torch.tensor(X_val, dtype=torch.float32)
            yva = torch.tensor(y_val, dtype=torch.float32)
            train_ds = torch.utils.data.TensorDataset(Xtr, ytr)
            val_ds = torch.utils.data.TensorDataset(Xva, yva)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
            # 优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001), weight_decay=params.get('weight_decay', 0.01))
            criterion = torch.nn.BCEWithLogitsLoss()
            best_acc = 0
            best_epoch = 0
            patience_counter = 0
            for epoch in range(EPOCHS):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    if model_num == 8:
                        out, _ = model(batch_X[:, -1, :])
                        logits = out.squeeze()
                    else:
                        logits = model(batch_X).squeeze()
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                # 验证
                model.eval()
                val_logits = []
                val_targets = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        if model_num == 8:
                            out, _ = model(batch_X[:, -1, :])
                            logits = out.squeeze()
                        else:
                            logits = model(batch_X).squeeze()
                        val_logits.append(logits.cpu().numpy())
                        val_targets.append(batch_y.cpu().numpy())
                val_logits = np.concatenate(val_logits)
                val_targets = np.concatenate(val_targets)
                val_probs = 1 / (1 + np.exp(-val_logits))
                val_preds = (val_probs > 0.5).astype(int)
                acc = accuracy_score(val_targets, val_preds)
                print(f"Epoch {epoch+1}: val_acc={acc:.4f} | best_val_acc={max(best_acc, acc):.4f} | global_best_val_acc={GLOBAL_BEST['best_val_acc']:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                if acc >= TARGET_ACC:
                    print(f"\nð 找到准确率>0.7的模型: {model_name} 参数: {params} val_acc={acc:.4f}")
                    save_exp_result(model_name, params, best_acc, best_epoch)
                    exit(0)
                if patience_counter >= PATIENCE:
                    print(f"早停: val_acc未提升 {PATIENCE} 次，停止训练")
                    break
            # 每组参数实验后自动保存结果
            save_exp_result(model_name, params, best_acc, best_epoch)
            # 全局最优追踪
            if best_acc > GLOBAL_BEST['best_val_acc']:
                GLOBAL_BEST['best_val_acc'] = best_acc
                GLOBAL_BEST['model'] = model_name
                GLOBAL_BEST['params'] = params.copy()
                GLOBAL_BEST['best_epoch'] = best_epoch
    print("未找到准确率>0.7的模型，请调整参数或增加训练轮数。") 
    # 输出并保存全局最优
    print("\n==== 全局最优结果 ====")
    print(f"模型: {GLOBAL_BEST['model']}")
    print(f"参数: {GLOBAL_BEST['params']}")
    print(f"best_val_acc: {GLOBAL_BEST['best_val_acc']:.4f} @ epoch {GLOBAL_BEST['best_epoch']}")
    with open('results/global_best_result.json', 'w') as f:
        json.dump(GLOBAL_BEST, f, indent=2) 
