import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import itertools
import os
import json
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

# 导入增强的特征工程模块
from unified_models_temp import (
    add_technical_indicators, add_time_features, add_price_features, add_volume_features,
    add_volatility_features, add_momentum_features, add_trend_features, add_oscillator_features,
    add_support_resistance_features, add_trade_activity_features, add_market_microstructure_features,
    add_order_flow_features, add_time_based_features, add_advanced_technical_indicators,
    add_volume_price_relationship_features
)

# 1. 在文件顶部导入load_and_preprocess_data_no_leak
# main 直接调用本文件内定义的 load_and_preprocess_data_no_leak，无需import

def add_all_features(df):
    df = add_technical_indicators(df)
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_volatility_features(df)
    df = add_momentum_features(df)
    df = add_trend_features(df)
    df = add_oscillator_features(df)
    df = add_support_resistance_features(df)
    df = add_trade_activity_features(df)
    df = add_market_microstructure_features(df)
    df = add_order_flow_features(df)
    df = add_time_based_features(df)
    df = add_advanced_technical_indicators(df)
    df = add_volume_price_relationship_features(df)
    return df

def load_and_preprocess_data_no_leak(file_path='btcusdt.json', sequence_length=30, dropna=True):
    """安全的数据加载和特征工程，避免数据泄露"""
    # 1. 加载原始数据
    with open(file_path, 'r') as f:
        data = pd.DataFrame(pd.read_json(f)) if file_path.endswith('.json') else pd.read_csv(f)
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data = data.sort_values('open_time').reset_index(drop=True)
    # 2. 生成标签
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    # 3. 按时间顺序划分训练集和验证集
    train_size = int(len(data) * 0.8)
    df_train = data.iloc[:train_size].reset_index(drop=True)
    df_val = data.iloc[train_size:].reset_index(drop=True)
    # 4. 分别做特征工程
    df_train_feat = add_all_features(df_train.copy())
    df_val_feat = add_all_features(df_val.copy())
    # 5. 选择特征列
    feature_columns = [col for col in df_train_feat.columns if col not in ['target', 'open_time', 'close_time', 'timestamp'] and df_train_feat[col].dtype in ['float64', 'int64']]
    # 6. 去掉最后一行（没有目标值）
    X_train = df_train_feat[feature_columns].values[:-1]
    y_train = df_train_feat['target'].values[:-1]
    X_val = df_val_feat[feature_columns].values[:-1]
    y_val = df_val_feat['target'].values[:-1]
    # 7. 标准化（fit在train，transform在train和val）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    # 8. 构造序列
    def create_sequences(X, y, sequence_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    print(f"安全特征工程完成，train特征数: {len(feature_columns)}，train样本: {X_train_seq.shape[0]}，val样本: {X_val_seq.shape[0]}")
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, feature_columns

os.makedirs('results', exist_ok=True)

# 1. 数据加载和预处理
DATA_FILE = 'btcusdt.json'
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_and_preprocess_data(file_path=DATA_FILE, sequence_length=30, use_enhanced_features=True):
    """加载和预处理BTC价格数据，支持增强特征工程"""
    if use_enhanced_features:
        # 使用增强的特征工程
        from unified_models_temp import load_and_preprocess_data_advanced, create_sequences
        print("使用增强特征工程...")
        X, y, feature_columns = load_and_preprocess_data_advanced(file_path, sequence_length)
        
        # 构造序列
        X_seq, y_seq = create_sequences(X, y, sequence_length)
        
        print(f"增强特征工程完成，特征数量: {len(feature_columns)}")
        print(f"最终数据形状: X={X_seq.shape}, y={y_seq.shape}")
        return X_seq, y_seq
    else:
        # 使用基础特征工程
        print("使用基础特征工程...")
        with open(file_path, 'r') as f:
            data = pd.DataFrame(pd.read_json(f)) if file_path.endswith('.json') else pd.read_csv(f)
        
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # 确保数值列
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data = data.dropna()
        
        # 预测下一小时涨跌（分类任务）
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # 基础特征
        data['price_change'] = data['close'].pct_change()
        data['high_low_ratio'] = data['high'] / data['low']
        data['volume_price_ratio'] = data['volume'] / data['close']
        data['price_range'] = (data['high'] - data['low']) / data['close']
        
        # 移动平均线
        for window in [3, 5, 7, 10, 15, 20, 30, 50]:
            data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
        
        # 时间特征
        if 'open_time' in data.columns:
            data['hour'] = pd.to_datetime(data['open_time']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['open_time']).dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # 特征列
        feature_columns = [col for col in data.columns 
                          if col not in ['target', 'open_time', 'close_time'] 
                          and data[col].dtype in ['float64', 'int64']]
        
        X = data[feature_columns].values[:-1]
        y = data['target'].values[:-1]
        
        # 处理无穷大和NaN值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 构造序列
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.int64)
        
        print(f"基础特征工程完成，特征数量: {len(feature_columns)}")
        print(f"最终数据形状: X={X_seq.shape}, y={y_seq.shape}")
        return X_seq, y_seq

# 2. xPatch核心组件
class EMA(nn.Module):
    """指数移动平均，用于趋势分解"""
    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        # x: [Batch, Input, Channel]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)

class DECOMP(nn.Module):
    """时间序列分解模块"""
    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        else:
            raise ValueError(f"Unsupported ma_type: {ma_type}")

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average

class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    def __init__(self, num_features, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"Mode {mode} not supported")
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:, -1:].detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-5).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-8)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class xPatchNetwork(nn.Module):
    """xPatch双流网络"""
    def __init__(self, seq_len, patch_len, stride, padding_patch, hidden_size=128):
        super(xPatchNetwork, self).__init__()

        # 参数
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # 非线性流（季节性）
        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # 分类头
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, hidden_size)
        self.gelu4 = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(hidden_size, 1)

        # 线性流（趋势）
        self.fc5 = nn.Linear(seq_len, hidden_size)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(hidden_size // 2)

        self.fc6 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(hidden_size // 8)

        self.fc7 = nn.Linear(hidden_size // 8, 1)

        # 流融合
        self.fc8 = nn.Linear(2, 1)

    def forward(self, s, t):
        # s - 季节性, t - 趋势
        s = s.permute(0, 2, 1)  # [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # [Batch, Channel, Input]
        
        # 通道分离处理
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B*C, I))  # [Batch*Channel, Input]
        t = torch.reshape(t, (B*C, I))  # [Batch*Channel, Input]

        # 非线性流（季节性）
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # 分类头
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.dropout(s)
        s = self.fc4(s)

        # 线性流（趋势）
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # 流融合
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # 通道合并
        x = torch.reshape(x, (B, C, 1))  # [Batch, Channel, 1]
        x = x.permute(0, 2, 1)  # [Batch, 1, Channel]
        
        # 全局平均池化得到最终预测
        x = torch.mean(x, dim=-1)  # [Batch, 1]
        
        return x.squeeze()

# 3. xPatch模型（适配二分类）
class xPatch(nn.Module):
    def __init__(self, input_size, seq_len, patch_len=4, stride=2, padding_patch='end',
                 ma_type='ema', alpha=0.1, revin=True, hidden_size=128):
        super(xPatch, self).__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.revin = revin
        
        # 归一化
        if self.revin:
            self.revin_layer = RevIN(input_size, affine=True, subtract_last=False)

        # 分解
        self.decomp = DECOMP(ma_type, alpha, None)
        
        # 网络
        self.net = xPatchNetwork(seq_len, patch_len, stride, padding_patch, hidden_size)

    def forward(self, x):
        # x: [Batch, Input, Channel]
        
        # 归一化
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # 分解
        seasonal_init, trend_init = self.decomp(x)
        
        # 网络处理
        x = self.net(seasonal_init, trend_init)

        # 反归一化
        if self.revin:
            # 对于分类任务，我们不需要反归一化
            pass

        return x

# 4. PatchMixer模型实现（保留原有实现）
class PatchMixer(nn.Module):
    def __init__(self, input_size, patch_size=4, num_patches=8, hidden_size=128, 
                 num_layers=3, dropout=0.1, num_classes=1):
        super(PatchMixer, self).__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        
        # 确保序列长度能被patch_size整除
        self.sequence_length = patch_size * num_patches
        
        # Patch embedding
        self.patch_embedding = nn.Linear(input_size * patch_size, hidden_size)
        
        # Patch mixing layers
        self.mixing_layers = nn.ModuleList([
            PatchMixingLayer(hidden_size, dropout) 
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * num_patches, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # 确保序列长度正确
        if x.size(1) != self.sequence_length:
            # 如果序列长度不匹配，进行padding或truncation
            if x.size(1) > self.sequence_length:
                x = x[:, :self.sequence_length, :]
            else:
                # padding
                pad_size = self.sequence_length - x.size(1)
                x = F.pad(x, (0, 0, 0, pad_size))
        
        # 分割成patches
        # (batch_size, num_patches, patch_size, input_size)
        patches = x.view(batch_size, self.num_patches, self.patch_size, self.input_size)
        
        # Flatten patches
        # (batch_size, num_patches, patch_size * input_size)
        patches = patches.view(batch_size, self.num_patches, -1)
        
        # Patch embedding
        # (batch_size, num_patches, hidden_size)
        patches = self.patch_embedding(patches)
        
        # Apply mixing layers
        for mixing_layer in self.mixing_layers:
            patches = mixing_layer(patches)
        
        # Global pooling and classification
        # (batch_size, hidden_size * num_patches)
        patches = patches.view(batch_size, -1)
        
        # Classification
        output = self.classifier(patches)
        
        return output.squeeze()

class PatchMixingLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PatchMixingLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Patch mixing MLP
        self.patch_mixing = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Channel mixing MLP
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, num_patches, hidden_size)
        
        # Patch mixing (across patches)
        residual = x
        x = self.norm1(x)
        x = self.patch_mixing(x)
        x = x + residual
        
        # Channel mixing (across channels)
        residual = x
        x = self.norm2(x)
        x = self.channel_mixing(x)
        x = x + residual
        
        return x

# 5. 训练配置
BATCH_SIZE = 512
EPOCHS = 100
PATIENCE = 15


def save_exp_result(model_name, params, best_acc, best_epoch, extra_metrics=None):
    """保存实验结果"""
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

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience, target_acc):
    """通用训练函数"""
    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
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
                logits = model(batch_X).squeeze()
                val_logits.append(logits.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        val_logits = np.concatenate(val_logits)
        val_targets = np.concatenate(val_targets)
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_preds = (val_probs > 0.5).astype(int)
        acc = accuracy_score(val_targets, val_preds)
        
        print(f"Epoch {epoch+1}: val_acc={acc:.4f} | best_val_acc={max(best_acc, acc):.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if acc >= target_acc:
            print(f"\nð 找到准确率>0.7的模型! val_acc={acc:.4f}")
            return best_acc, best_epoch, True
        
        if patience_counter >= patience:
            print(f"早停: val_acc未提升 {patience} 次，停止训练")
            break
    
    return best_acc, best_epoch, False

# 7. PatchMixer模型（适配二分类）
class PatchMixerLayer(nn.Module):
    def __init__(self, dim, a, kernel_size=8):
        super().__init__()
        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, a, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )
    def forward(self, x):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x

class PatchMixerBinary(nn.Module):
    def __init__(self, input_size, seq_len, patch_size=8, stride=4, d_model=64, e_layers=2, mixer_kernel_size=8, dropout=0.1, head_dropout=0.1, revin=True):
        super().__init__()
        self.nvals = input_size
        self.lookback = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.kernel_size = mixer_kernel_size
        self.d_model = d_model
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.depth = e_layers
        self.revin = revin
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.a = self.patch_num
        self.PatchMixer_blocks = nn.ModuleList([
            PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size)
            for _ in range(self.depth)
        ])
        self.W_P = nn.Linear(self.patch_size, self.d_model)
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, 32),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(32, 1)
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, 32),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(32, 1)
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        if self.revin:
            self.revin_layer = RevIN(self.nvals, affine=True, subtract_last=False)
    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)  # [batch, n_val, seq_len]
        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [batch, n_val, patch_num, patch_size]
        x = self.W_P(x)  # [batch, n_val, patch_num, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [batch * n_val, patch_num, d_model]
        x = self.dropout_layer(x)
        u = self.head0(x)
        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)
        x = self.head1(x)
        x = u + x
        x = torch.reshape(x, (bs, nvars, -1))  # [batch, n_val, 1]
        x = x.permute(0, 2, 1)  # [batch, 1, n_val]
        x = torch.mean(x, dim=-1)  # [batch, 1]
        return x.squeeze()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"检测到GPU数量: {n_gpus}")
    print(f"使用设备: {device}")
    
    # 配置参数
    USE_ENHANCED_FEATURES = True  # 是否使用增强特征工程
    SEQUENCE_LENGTH = 30
    BATCH_SIZE = 512
    EPOCHS = 300
    PATIENCE = 30
    TARGET_ACC = 0.75
    
    print(f"使用增强特征: {USE_ENHANCED_FEATURES}")
    print(f"序列长度: {SEQUENCE_LENGTH}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"早停耐心: {PATIENCE}")
    print(f"目标准确率: {TARGET_ACC}")
    
    # 定义搜索空间
    """
    xpatch_search_space = {
        'patch_len': [2, 4],
        'stride': [1, 2, 4],
        'alpha': [0.05, 0.1, 0.2, 0.3],  # EMA平滑因子
        'hidden_size': [64, 128, 256],
        'learning_rate': [1e-2],
        'weight_decay': [1e-3, 1e-4, 1e-2]
    }
    """
    xpatch_search_space = {
        'patch_len': [4, 8],
        'stride': [1, 2], 
        'alpha': [0.05, 0.1, 0.2, 0.3],  # EMA平滑因子
        'hidden_size': [64, 128, 256, 512],
        'learning_rate': [1e-2],
        'weight_decay': [1e-3]
    }
    
    patchmixer_search_space = {
        'patch_size': [4, 8],
        'num_patches': [4, 8],
        'hidden_size': [64, 128],
        'num_layers': [2, 3],
        'dropout': [0.0, 0.1],
        'learning_rate': [1e-4, 5e-4, 1e-3, 1e-2],
        'weight_decay': [1e-4, 1e-3, 1e-2]
    }
    
    patchmixer_binary_search_space = {
        'patch_size': [4, 8],
        'stride': [2, 4],
        'd_model': [64, 128],
        'e_layers': [2, 3],
        'mixer_kernel_size': [4, 8],
        'dropout': [0.0, 0.1],
        'head_dropout': [0.0, 0.1],
        'learning_rate': [1e-4, 1e-3],
        'weight_decay': [1e-4, 1e-3]
    }
    
    # 全局最优追踪
    GLOBAL_BEST = {
        'best_val_acc': 0,
        'model': None,
        'params': None,
        'best_epoch': 0
    }
    
    # 加载数据
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, feature_columns = load_and_preprocess_data_no_leak(
        file_path=DATA_FILE, sequence_length=SEQUENCE_LENGTH, dropna=True)
    X, y = load_and_preprocess_data(sequence_length=SEQUENCE_LENGTH, use_enhanced_features=USE_ENHANCED_FEATURES)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    # 构造dataloader
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    
    train_ds = torch.utils.data.TensorDataset(Xtr, ytr)
    val_ds = torch.utils.data.TensorDataset(Xva, yva)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 测试xPatch模型
    print("\n" + "="*50)
    print("开始测试xPatch模型")
    print("="*50)
    
    keys, values = zip(*xpatch_search_space.items())
    xpatch_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"xPatch总共需要测试 {len(xpatch_param_combinations)} 种参数组合")
    
    for i, params in enumerate(xpatch_param_combinations):
        print(f"\n==== 测试xPatch [{i+1}/{len(xpatch_param_combinations)}] 参数: {params} ====")
        
        # 构造xPatch模型
        model = xPatch(
            input_size=X_train.shape[2],
            seq_len=SEQUENCE_LENGTH,
            patch_len=params['patch_len'],
            stride=params['stride'],
            alpha=params['alpha'],
            hidden_size=params['hidden_size']
        )
        
        if n_gpus > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=params.get('learning_rate', 0.001), 
            weight_decay=params.get('weight_decay', 0.01)
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # 训练
        best_acc, best_epoch, found_target = train_model(
            model, train_loader, val_loader, optimizer, criterion, 
            device, EPOCHS, PATIENCE, TARGET_ACC
        )
        
        # 保存结果
        save_exp_result("xPatch", params, best_acc, best_epoch)
        
        # 更新全局最优
        if best_acc > GLOBAL_BEST['best_val_acc']:
            GLOBAL_BEST['best_val_acc'] = best_acc
            GLOBAL_BEST['model'] = "xPatch"
            GLOBAL_BEST['params'] = params.copy()
            GLOBAL_BEST['best_epoch'] = best_epoch
        
        if found_target:
            break
