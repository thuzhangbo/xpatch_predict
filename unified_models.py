"""
Unified Models for BTC Price Prediction

模型数字映射：
1 = AdvancedLSTM
2 = AdvancedGRU
3 = TimeSeriesTransformer
4 = CNNLSTMAttention
5 = OptimizedTCNWithAttention
6 = EnsembleModel
7 = MultiScaleTransformer
8 = TabNetNoEmbeddings

用法：
from unified_models import get_model_by_number
model_class = get_model_by_number(1)  # 选择AdvancedLSTM
model = model_class(...)
"""

# ========== 1. AdvancedLSTM (from 01_lstm_grid_search.py) ==========
import torch
import torch.nn as nn

class AdvancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, bidirectional=True, attention_type='multihead'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.attention_type = attention_type
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        if attention_type == 'multihead':
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * self.num_directions,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        elif attention_type == 'self_attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * self.num_directions,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        elif attention_type == 'bahdanau':
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softmax(dim=1)
            )
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.gelu = nn.GELU()
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.attention_type in ['multihead', 'self_attention']:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            attn_out = self.layer_norm(lstm_out + attn_out)
            pooled = torch.mean(attn_out, dim=1)
        elif self.attention_type == 'bahdanau':
            attention_weights = self.attention(lstm_out)
            pooled = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            pooled = torch.mean(lstm_out, dim=1)
        out = self.fc1(pooled)
        out = self.batch_norm1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# ========== 2. AdvancedGRU (from 03_gru_grid_search.py) ==========
class AdvancedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, bidirectional=True, attention_type=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.attention_type = attention_type
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        if attention_type == 'multihead':
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * self.num_directions,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        elif attention_type == 'bahdanau':
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.num_directions, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
                nn.Softmax(dim=1)
            )
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
    def forward(self, x):
        gru_out, _ = self.gru(x)
        if self.attention_type == 'multihead':
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            attn_out = self.layer_norm(gru_out + attn_out)
            pooled = torch.mean(attn_out, dim=1)
        elif self.attention_type == 'bahdanau':
            attention_weights = self.attention(gru_out)
            pooled = torch.sum(gru_out * attention_weights, dim=1)
        else:
            pooled = torch.mean(gru_out, dim=1)
        out = self.fc1(pooled)
        out = self.batch_norm1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# ========== 3. TimeSeriesTransformer (from 02_transformer_grid_search.py) ==========
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.input_projection(x)
        seq_len = x.size(1)
        pos_enc = self.pos_embedding[:, :seq_len, :]
        x = x + pos_enc
        x = self.layer_norm(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# ========== 4. CNNLSTMAttention (from 04_cnn_lstm_attention.py) ==========
class CNNLSTMAttention(nn.Module):
    def __init__(self, input_size, cnn_channels, cnn_kernel, lstm_hidden, lstm_layers, dropout=0.2, bidirectional=True, attention_heads=4):
        super().__init__()
        self.cnn = nn.Conv1d(input_size, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.cnn_bn = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.num_directions = 2 if bidirectional else 1
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * self.num_directions,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden * self.num_directions)
        self.fc1 = nn.Linear(lstm_hidden * self.num_directions, lstm_hidden)
        self.fc2 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.fc3 = nn.Linear(lstm_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.cnn_bn(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + attn_out)
        pooled = torch.mean(attn_out, dim=1)
        out = self.gelu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.gelu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# ========== 5. OptimizedTCNWithAttention (from 05_tcn_attention_optimized.py) ==========
class OptimizedTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        # 修正：残差连接前shape对齐
        if out.shape != res.shape:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]
        return out + res

class OptimizedTCNWithAttention(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, attention_heads=8):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(min(num_levels, 2)):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                OptimizedTemporalBlock(
                    in_channels, out_channels, kernel_size, stride=1,
                    dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.attention = nn.MultiheadAttention(
            embed_dim=num_channels[-1],
            num_heads=min(attention_heads, 4),
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(num_channels[-1])
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1] // 4)
        self.fc2 = nn.Linear(num_channels[-1] // 4, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.layer_norm(x + attn_out)
        pooled = torch.mean(attn_out, dim=1)
        out = self.relu(self.fc1(pooled))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ========== 6. EnsembleModel (from 06_ensemble_models.py) ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.input_projection(x)
        transformer_out = self.transformer(x)
        pooled = torch.mean(transformer_out, dim=1)
        out = self.dropout(pooled)
        out = self.fc(out)
        return out

class EnsembleModel(nn.Module):
    def __init__(self, input_size, sequence_length, config):
        super().__init__()
        self.models = nn.ModuleDict()
        self.weights = nn.Parameter(torch.ones(len(config['model_types'])))
        self.sequence_length = sequence_length
        for i, model_type in enumerate(config['model_types']):
            if model_type == 'lstm':
                self.models[f'lstm_{i}'] = LSTMModel(
                    input_size=input_size,
                    hidden_size=config['hidden_sizes'][i],
                    num_layers=config['num_layers'][i],
                    dropout=config['dropout']
                )
            elif model_type == 'gru':
                self.models[f'gru_{i}'] = GRUModel(
                    input_size=input_size,
                    hidden_size=config['hidden_sizes'][i],
                    num_layers=config['num_layers'][i],
                    dropout=config['dropout']
                )
            elif model_type == 'transformer':
                self.models[f'transformer_{i}'] = TransformerModel(
                    input_size=input_size,
                    d_model=config['d_models'][i],
                    nhead=config['nheads'][i],
                    num_layers=config['num_layers'][i],
                    dropout=config['dropout']
                )
        self.meta_classifier = nn.Sequential(
            nn.Linear(len(config['model_types']), 64),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        predictions = []
        for name, model in self.models.items():
            pred = model(x)
            predictions.append(pred)
        stacked_preds = torch.cat(predictions, dim=1)
        weighted_preds = stacked_preds * torch.softmax(self.weights, dim=0)
        final_pred = self.meta_classifier(weighted_preds)
        return final_pred

# ========== 7. MultiScaleTransformer (from 07_multiscale_transformer.py) ==========
class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class MultiScaleTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_scales=3, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales)
        ])
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.scale_attentions = nn.ModuleList([
            MultiScaleAttention(d_model, nhead, dropout) for _ in range(num_scales)
        ])
        self.fc1 = nn.Linear(d_model * num_scales, d_model)
        self.fc2 = nn.Linear(d_model, d_model // 2)
        self.fc3 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.input_projection(x)
        scale_outputs = []
        for i, (scale_proj, scale_attn) in enumerate(zip(self.scale_projections, self.scale_attentions)):
            scale_x = scale_proj(x)
            for transformer_layer in self.transformer_layers:
                scale_x = transformer_layer(scale_x)
            scale_x = scale_attn(scale_x)
            scale_pooled = torch.mean(scale_x, dim=1)
            scale_outputs.append(scale_pooled)
        concatenated = torch.cat(scale_outputs, dim=1)
        out = self.relu(self.fc1(concatenated))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# ========== 8. TabNetNoEmbeddings (极简可用测试版) ==========
import numpy as np

class TabNetNoEmbeddings(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=4, n_a=4, n_steps=1, gamma=1.1, n_ind=1, n_shared=1, virtual_batch_size=2, **kwargs):
        super().__init__()
        # 极简实现：只做一次特征变换和一次全连接，保证shape兼容
        self.fc1 = nn.Linear(input_dim, n_d)
        self.fc2 = nn.Linear(n_d, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        m_loss = torch.tensor(0.0, device=x.device)  # 占位
        return out, m_loss

# ========== 工厂函数：数字选择模型 ==========
def get_model_by_number(num):
    """
    根据数字选择模型类
    
    Args:
        num (int): 模型编号 (1-8)
    
    Returns:
        model_class: 对应的模型类，如果编号无效则返回None
    
    模型映射：
    1 = AdvancedLSTM
    2 = AdvancedGRU  
    3 = TimeSeriesTransformer
    4 = CNNLSTMAttention
    5 = OptimizedTCNWithAttention
    6 = EnsembleModel
    7 = MultiScaleTransformer
    8 = TabNetNoEmbeddings
    """
    mapping = {
        1: AdvancedLSTM,
        2: AdvancedGRU,
        3: TimeSeriesTransformer,
        4: CNNLSTMAttention,
        5: OptimizedTCNWithAttention,
        6: EnsembleModel,
        7: MultiScaleTransformer,
        8: TabNetNoEmbeddings
    }
    return mapping.get(num, None)

def get_model_info():
    """
    获取所有模型的信息
    
    Returns:
        dict: 模型编号到模型名称的映射
    """
    return {
        1: "AdvancedLSTM",
        2: "AdvancedGRU", 
        3: "TimeSeriesTransformer",
        4: "CNNLSTMAttention",
        5: "OptimizedTCNWithAttention",
        6: "EnsembleModel",
        7: "MultiScaleTransformer",
        8: "TabNetNoEmbeddings"
    }

def print_available_models():
    """打印所有可用的模型"""
    print("可用的模型:")
    print("=" * 50)
    for num, name in get_model_info().items():
        print(f"{num}. {name}")
    print("=" * 50) 

def setup_logging():
    """设置统一的日志格式"""
    import logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_and_preprocess_data(file_path='btcusdt.json'):
    """统一的数据加载和预处理函数"""
    import os, json, logging
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    logger = logging.getLogger(__name__)
    logger.info("Loading and preprocessing data...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['volume_price_ratio'] = df['volume'] / df['close']
    df['price_range'] = (df['high'] - df['low']) / df['close']
    for window in [3, 5, 7, 10, 15, 20, 30, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['bollinger_upper'], df['bollinger_lower'] = calculate_bollinger_bands(df['close'])
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['close']
    for period in [1, 2, 3, 5, 7, 10, 14, 21]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
        df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
    for window in [5, 10, 15, 20, 30]:
        df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
    df['hour'] = pd.to_datetime(df['open_time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['open_time']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df = df.dropna()
    feature_columns = [col for col in df.columns 
                      if col not in ['target', 'open_time', 'close_time'] and 
                      df[col].dtype in ['float64', 'int64']]
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df[feature_columns].values
    y = df['target'].values[:-1]
    X = X[:-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    return X, y

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def create_sequences(X, y, sequence_length):
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:(i + sequence_length)])
        sequences_y.append(y[i + sequence_length])
    return np.array(sequences_X, dtype=np.float32), np.array(sequences_y, dtype=np.int64)

def get_device(gpu_id=None):
    import torch, logging
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        logger = logging.getLogger(__name__)
        logger.info(f"使用GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        logger = logging.getLogger(__name__)
        logger.info("使用CPU")
    return device

def train_model_with_early_stopping(model, train_loader, val_loader, device, config):
    import torch, torch.nn as nn, torch.optim as optim
    from sklearn.metrics import accuracy_score, roc_auc_score
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=False)
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    val_aucs = []
    import logging
    logger = logging.getLogger(__name__)
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_predictions = []
        val_targets = []
        val_probabilities = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probs > 0.5).astype(int)
                val_predictions.extend(predictions)
                val_targets.extend(batch_y.cpu().numpy())
                val_probabilities.extend(probs)
        val_acc = accuracy_score(val_targets, val_predictions)
        val_auc = roc_auc_score(val_targets, val_probabilities)
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, "
                       f"Val Acc = {val_acc:.4f}, Val AUC = {val_auc:.4f}")
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    return {
        'best_val_acc': best_val_acc,
        'best_val_auc': max(val_aucs),
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_aucs': val_aucs
    }

def evaluate_model(model, test_loader, device):
    import torch
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    model.eval()
    test_predictions = []
    test_targets = []
    test_probabilities = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probs > 0.5).astype(int)
            test_predictions.extend(predictions)
            test_targets.extend(batch_y.cpu().numpy())
            test_probabilities.extend(probs)
    test_acc = accuracy_score(test_targets, test_predictions)
    test_auc = roc_auc_score(test_targets, test_probabilities)
    test_precision = precision_score(test_targets, test_predictions, zero_division=0)
    test_recall = recall_score(test_targets, test_predictions, zero_division=0)
    test_f1 = f1_score(test_targets, test_predictions, zero_division=0)
    return {
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'predictions': test_predictions,
        'probabilities': test_probabilities,
        'targets': test_targets
    }

def save_results(result, config, model_name, result_file):
    import json, logging
    from datetime import datetime
    output = {
        'model_name': model_name,
        'config': config,
        'result': result,
        'timestamp': datetime.now().isoformat(),
        'test_accuracy': float(result['test_accuracy']),
        'test_auc': float(result['test_auc']),
        'test_f1': float(result['test_f1'])
    }
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    logger = logging.getLogger(__name__)
    logger.info(f"结果已保存到: {result_file}")

def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    defaults = {
        'sequence_length': 50,
        'batch_size': 256,
        'epochs': 100,
        'patience': 10,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'dropout': 0.2,
        'bidirectional': True,
        'attention_heads': 8,
        'cnn_kernel': 3,
        'lstm_layers': 2,
        'channels_1': 64,
        'channels_2': 128,
        'channels_3': 256,
        'kernel_size': 3
    }
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    return config

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='BTC Price Prediction Model Training')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--result_file', type=str, required=True, help='结果文件路径')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (可选)')
    parser.add_argument('--data_file', type=str, default='btcusdt.json', help='数据文件路径')
    return parser.parse_args()

def cleanup_gpu_memory():
    import torch, gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect() 
