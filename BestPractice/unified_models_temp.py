import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 在文件顶部补充所有特征工程函数的import/export声明，确保add_trade_activity_features、add_market_microstructure_features、add_order_flow_features、add_time_based_features等均被正确定义。

def calculate_rsi(prices, window=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """计算布林带"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band, sma

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """计算随机指标"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_atr(high, low, close, window=14):
    """计算平均真实波幅"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_cci(high, low, close, window=20):
    """计算商品通道指数"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci

def calculate_williams_r(high, low, close, window=14):
    """计算威廉指标"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_mfi(high, low, close, volume, window=14):
    """计算资金流量指标"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()
    
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi

def add_technical_indicators(data):
    """添加技术指标"""
    # RSI
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['rsi_21'] = calculate_rsi(data['close'], 21)
    
    # MACD
    data['macd'], data['macd_signal'], data['macd_histogram'] = calculate_macd(data['close'])
    
    # 布林带
    data['bb_upper'], data['bb_lower'], data['bb_middle'] = calculate_bollinger_bands(data['close'])
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # 随机指标
    data['stoch_k'], data['stoch_d'] = calculate_stochastic(data['high'], data['low'], data['close'])
    
    # ATR
    data['atr_14'] = calculate_atr(data['high'], data['low'], data['close'], 14)
    data['atr_21'] = calculate_atr(data['high'], data['low'], data['close'], 21)
    
    # CCI
    data['cci_20'] = calculate_cci(data['high'], data['low'], data['close'], 20)
    
    # 威廉指标
    data['williams_r_14'] = calculate_williams_r(data['high'], data['low'], data['close'], 14)
    
    # 资金流量指标
    data['mfi_14'] = calculate_mfi(data['high'], data['low'], data['close'], data['volume'], 14)
    
    return data

def add_time_features(data):
    """添加时间特征"""
    # 转换时间列
    if 'open_time' in data.columns:
        data['timestamp'] = pd.to_datetime(data['open_time'])
    elif 'close_time' in data.columns:
        data['timestamp'] = pd.to_datetime(data['close_time'])
    else:
        # 如果没有时间列，创建索引作为时间
        data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='H')
    
    # 提取时间特征
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['weekday'] = data['timestamp'].dt.weekday
    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
    data['is_month_start'] = data['timestamp'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['timestamp'].dt.is_month_end.astype(int)
    
    # 周期性特征
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
    data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
    
    return data

def add_price_features(data):
    """添加价格特征"""
    # 基础价格特征
    data['price_change'] = data['close'].pct_change()
    data['price_change_abs'] = data['price_change'].abs()
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))
    
    # 价格比率
    data['high_low_ratio'] = data['high'] / data['low']
    data['open_close_ratio'] = data['open'] / data['close']
    data['close_open_ratio'] = data['close'] / data['open']
    
    # 价格范围
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['price_range_abs'] = data['high'] - data['low']
    
    # 价格位置
    data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # 多期收益率
    for period in [1, 2, 3, 5, 7, 10, 14, 21]:
        data[f'return_{period}'] = data['close'].pct_change(period)
        data[f'log_return_{period}'] = np.log(data['close'] / data['close'].shift(period))
    
    return data

def add_volume_features(data):
    """添加成交量特征"""
    # 基础成交量特征
    data['volume_change'] = data['volume'].pct_change()
    data['volume_sma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_sma_10'] = data['volume'].rolling(window=10).mean()
    data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
    
    # 成交量比率
    data['volume_ratio_5'] = data['volume'] / data['volume_sma_5']
    data['volume_ratio_10'] = data['volume'] / data['volume_sma_10']
    data['volume_ratio_20'] = data['volume'] / data['volume_sma_20']
    
    # 价量关系
    data['volume_price_ratio'] = data['volume'] / data['close']
    data['volume_price_change'] = data['volume'] * data['price_change']
    
    # VWAP (成交量加权平均价格)
    data['vwap'] = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    data['price_vwap_ratio'] = data['close'] / data['vwap']
    
    # 成交量波动率
    data['volume_volatility_10'] = data['volume'].rolling(window=10).std()
    data['volume_volatility_20'] = data['volume'].rolling(window=20).std()
    
    # 基于taker_buy_base_asset_volume的特征
    if 'taker_buy_base_asset_volume' in data.columns:
        # 主动买入比例
        data['taker_buy_ratio'] = data['taker_buy_base_asset_volume'] / data['volume']
        
        # 主动卖出比例
        data['taker_sell_ratio'] = 1 - data['taker_buy_ratio']
        
        # 买卖压力指标
        data['buy_sell_pressure'] = data['taker_buy_ratio'] - data['taker_sell_ratio']
        
        # 主动买入变化率
        data['taker_buy_change'] = data['taker_buy_base_asset_volume'].pct_change()
        
        # 主动买入移动平均
        data['taker_buy_sma_5'] = data['taker_buy_base_asset_volume'].rolling(window=5).mean()
        data['taker_buy_sma_10'] = data['taker_buy_base_asset_volume'].rolling(window=10).mean()
        data['taker_buy_sma_20'] = data['taker_buy_base_asset_volume'].rolling(window=20).mean()
        
        # 主动买入比率
        data['taker_buy_ratio_5'] = data['taker_buy_base_asset_volume'] / data['taker_buy_sma_5']
        data['taker_buy_ratio_10'] = data['taker_buy_base_asset_volume'] / data['taker_buy_sma_10']
        data['taker_buy_ratio_20'] = data['taker_buy_base_asset_volume'] / data['taker_buy_sma_20']
        
        # 主动买入强度 (相对于价格变化)
        data['taker_buy_intensity'] = data['taker_buy_ratio'] * data['price_change'].abs()
        
        # 主动买入趋势
        data['taker_buy_trend_5'] = np.where(data['taker_buy_sma_5'] > data['taker_buy_sma_5'].shift(1), 1, -1)
        data['taker_buy_trend_10'] = np.where(data['taker_buy_sma_10'] > data['taker_buy_sma_10'].shift(1), 1, -1)
        
        # 主动买入波动率
        data['taker_buy_volatility_10'] = data['taker_buy_base_asset_volume'].rolling(window=10).std()
        data['taker_buy_volatility_20'] = data['taker_buy_base_asset_volume'].rolling(window=20).std()
        
        # 主动买入与总成交量关系
        data['taker_buy_volume_ratio'] = data['taker_buy_base_asset_volume'] / data['volume']
        data['taker_buy_volume_change'] = data['taker_buy_volume_ratio'].pct_change()
        
        # 主动买入价格影响
        data['taker_buy_price_impact'] = data['taker_buy_ratio'] * data['price_change']
        
        # 主动买入异常值检测
        data['taker_buy_zscore'] = (data['taker_buy_base_asset_volume'] - data['taker_buy_sma_20']) / data['taker_buy_volatility_20']
        data['taker_buy_anomaly'] = np.where(data['taker_buy_zscore'].abs() > 2, 1, 0)
    
    # 基于taker_buy_quote_asset_volume的特征 (美元金额)
    if 'taker_buy_quote_asset_volume' in data.columns:
        # 主动买入金额变化率
        data['taker_buy_amount_change'] = data['taker_buy_quote_asset_volume'].pct_change()
        
        # 主动买入金额移动平均
        data['taker_buy_amount_sma_5'] = data['taker_buy_quote_asset_volume'].rolling(window=5).mean()
        data['taker_buy_amount_sma_10'] = data['taker_buy_quote_asset_volume'].rolling(window=10).mean()
        data['taker_buy_amount_sma_20'] = data['taker_buy_quote_asset_volume'].rolling(window=20).mean()
        
        # 主动买入金额比率
        data['taker_buy_amount_ratio_5'] = data['taker_buy_quote_asset_volume'] / data['taker_buy_amount_sma_5']
        data['taker_buy_amount_ratio_10'] = data['taker_buy_quote_asset_volume'] / data['taker_buy_amount_sma_10']
        
        # 平均买入价格
        data['avg_buy_price'] = data['taker_buy_quote_asset_volume'] / data['taker_buy_base_asset_volume']
        data['avg_buy_price_ratio'] = data['avg_buy_price'] / data['close']
        
        # 买入金额强度
        data['taker_buy_amount_intensity'] = data['taker_buy_quote_asset_volume'] / data['volume']
    
    return data

def add_volatility_features(data):
    """添加波动率特征"""
    # 价格波动率
    for window in [5, 10, 15, 20, 30]:
        data[f'volatility_{window}'] = data['close'].rolling(window=window).std()
        data[f'volatility_ratio_{window}'] = data[f'volatility_{window}'] / data['close']
    
    # 真实波动率 (基于ATR)
    data['true_volatility'] = data['atr_14'] / data['close']
    
    # 波动率变化
    data['volatility_change_10'] = data['volatility_10'].pct_change()
    data['volatility_change_20'] = data['volatility_20'].pct_change()
    
    # 波动率比率
    data['volatility_ratio_short_long'] = data['volatility_10'] / data['volatility_20']
    
    return data

def add_momentum_features(data):
    """添加动量特征"""
    # 价格动量
    for period in [1, 2, 3, 5, 7, 10, 14, 21]:
        data[f'momentum_{period}'] = data['close'].pct_change(period)
        data[f'roc_{period}'] = (data['close'] / data['close'].shift(period) - 1) * 100
    
    # 相对强弱
    data['relative_strength_5'] = data['close'] / data['close'].rolling(window=5).mean()
    data['relative_strength_10'] = data['close'] / data['close'].rolling(window=10).mean()
    data['relative_strength_20'] = data['close'] / data['close'].rolling(window=20).mean()
    
    # 动量振荡器
    data['momentum_oscillator_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_oscillator_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    return data

def add_trend_features(data):
    """添加趋势特征"""
    # 移动平均线
    for window in [3, 5, 7, 10, 15, 20, 30, 50]:
        data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
        data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
        
        # 价格与移动平均线的关系
        data[f'price_sma_ratio_{window}'] = data['close'] / data[f'sma_{window}']
        data[f'price_ema_ratio_{window}'] = data['close'] / data[f'ema_{window}']
    
    # 趋势强度
    data['trend_strength_5'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
    data['trend_strength_10'] = (data['sma_10'] - data['sma_30']) / data['sma_30']
    
    # 趋势方向
    data['trend_direction_5'] = np.where(data['sma_5'] > data['sma_5'].shift(1), 1, -1)
    data['trend_direction_10'] = np.where(data['sma_10'] > data['sma_10'].shift(1), 1, -1)
    data['trend_direction_20'] = np.where(data['sma_20'] > data['sma_20'].shift(1), 1, -1)
    
    # 移动平均线交叉
    data['ma_cross_5_20'] = np.where(data['sma_5'] > data['sma_20'], 1, 0)
    data['ma_cross_10_30'] = np.where(data['sma_10'] > data['sma_30'], 1, 0)
    
    return data

def add_oscillator_features(data):
    """添加振荡器特征"""
    # CCI
    data['cci_10'] = calculate_cci(data['high'], data['low'], data['close'], 10)
    data['cci_20'] = calculate_cci(data['high'], data['low'], data['close'], 20)
    
    # 威廉指标
    data['williams_r_10'] = calculate_williams_r(data['high'], data['low'], data['close'], 10)
    data['williams_r_14'] = calculate_williams_r(data['high'], data['low'], data['close'], 14)
    
    # 资金流量指标
    data['mfi_10'] = calculate_mfi(data['high'], data['low'], data['close'], data['volume'], 10)
    data['mfi_14'] = calculate_mfi(data['high'], data['low'], data['close'], data['volume'], 14)
    
    # 振荡器信号
    data['rsi_signal'] = np.where(data['rsi_14'] > 70, -1, np.where(data['rsi_14'] < 30, 1, 0))
    data['stoch_signal'] = np.where(data['stoch_k'] > 80, -1, np.where(data['stoch_k'] < 20, 1, 0))
    data['williams_signal'] = np.where(data['williams_r_14'] > -20, -1, np.where(data['williams_r_14'] < -80, 1, 0))
    
    return data

def add_support_resistance_features(data):
    """添加支撑阻力特征"""
    # 局部最高点和最低点
    data['local_high_5'] = data['high'].rolling(window=5, center=True).max()
    data['local_low_5'] = data['low'].rolling(window=5, center=True).min()
    data['local_high_10'] = data['high'].rolling(window=10, center=True).max()
    data['local_low_10'] = data['low'].rolling(window=10, center=True).min()
    
    # 支撑阻力位置
    data['resistance_distance_5'] = (data['local_high_5'] - data['close']) / data['close']
    data['support_distance_5'] = (data['close'] - data['local_low_5']) / data['close']
    data['resistance_distance_10'] = (data['local_high_10'] - data['close']) / data['close']
    data['support_distance_10'] = (data['close'] - data['local_low_10']) / data['close']
    
    # 价格通道
    data['price_channel_5'] = (data['local_high_5'] - data['local_low_5']) / data['close']
    data['price_channel_10'] = (data['local_high_10'] - data['local_low_10']) / data['close']
    
    # 价格在通道中的位置
    data['channel_position_5'] = (data['close'] - data['local_low_5']) / (data['local_high_5'] - data['local_low_5'])
    data['channel_position_10'] = (data['close'] - data['local_low_10']) / (data['local_high_10'] - data['local_low_10'])
    
    # 枢轴点
    data['pivot_point'] = (data['high'] + data['low'] + data['close']) / 3
    data['r1'] = 2 * data['pivot_point'] - data['low']
    data['s1'] = 2 * data['pivot_point'] - data['high']
    
    return data

def add_trade_activity_features(data):
    """添加交易活跃度特征"""
    if 'number_of_trades' in data.columns:
        # 基础交易活跃度特征
        data['trades_change'] = data['number_of_trades'].pct_change()
        data['trades_sma_5'] = data['number_of_trades'].rolling(window=5).mean()
        data['trades_sma_10'] = data['number_of_trades'].rolling(window=10).mean()
        data['trades_sma_20'] = data['number_of_trades'].rolling(window=20).mean()
        
        # 交易活跃度比率
        data['trades_ratio_5'] = data['number_of_trades'] / data['trades_sma_5']
        data['trades_ratio_10'] = data['number_of_trades'] / data['trades_sma_10']
        data['trades_ratio_20'] = data['number_of_trades'] / data['trades_sma_20']
        
        # 交易活跃度波动率
        data['trades_volatility_10'] = data['number_of_trades'].rolling(window=10).std()
        data['trades_volatility_20'] = data['number_of_trades'].rolling(window=20).std()
        
        # 交易活跃度趋势
        data['trades_trend_5'] = np.where(data['trades_sma_5'] > data['trades_sma_5'].shift(1), 1, -1)
        data['trades_trend_10'] = np.where(data['trades_sma_10'] > data['trades_sma_10'].shift(1), 1, -1)
        
        # 平均每笔交易量
        data['avg_trade_volume'] = data['volume'] / data['number_of_trades']
        data['avg_trade_volume_sma_5'] = data['avg_trade_volume'].rolling(window=5).mean()
        data['avg_trade_volume_sma_10'] = data['avg_trade_volume'].rolling(window=10).mean()
        
        # 平均每笔交易金额
        if 'quote_asset_volume' in data.columns:
            data['avg_trade_amount'] = data['quote_asset_volume'] / data['number_of_trades']
            data['avg_trade_amount_sma_5'] = data['avg_trade_amount'].rolling(window=5).mean()
            data['avg_trade_amount_sma_10'] = data['avg_trade_amount'].rolling(window=10).mean()
            
            # 大单交易指标
            data['large_trade_ratio'] = data['avg_trade_amount'] / data['avg_trade_amount_sma_10']
            data['large_trade_signal'] = np.where(data['large_trade_ratio'] > 1.5, 1, 0)
        
        # 交易活跃度异常值检测
        data['trades_zscore'] = (data['number_of_trades'] - data['trades_sma_20']) / data['trades_volatility_20']
        data['trades_anomaly'] = np.where(data['trades_zscore'].abs() > 2, 1, 0)
        
        # 交易活跃度与价格关系
        data['trades_price_correlation_10'] = data['number_of_trades'].rolling(window=10).corr(data['close'])
        data['trades_volume_correlation_10'] = data['number_of_trades'].rolling(window=10).corr(data['volume'])
    
    return data

def add_market_microstructure_features(data):
    """添加市场微观结构特征"""
    # 基于quote_asset_volume的特征
    if 'quote_asset_volume' in data.columns:
        # 总成交金额特征
        data['quote_volume_change'] = data['quote_asset_volume'].pct_change()
        data['quote_volume_sma_5'] = data['quote_asset_volume'].rolling(window=5).mean()
        data['quote_volume_sma_10'] = data['quote_asset_volume'].rolling(window=10).mean()
        data['quote_volume_sma_20'] = data['quote_asset_volume'].rolling(window=20).mean()
        
        # 成交金额比率
        data['quote_volume_ratio_5'] = data['quote_asset_volume'] / data['quote_volume_sma_5']
        data['quote_volume_ratio_10'] = data['quote_asset_volume'] / data['quote_volume_sma_10']
        data['quote_volume_ratio_20'] = data['quote_asset_volume'] / data['quote_volume_sma_20']
        
        # 成交金额波动率
        data['quote_volume_volatility_10'] = data['quote_asset_volume'].rolling(window=10).std()
        data['quote_volume_volatility_20'] = data['quote_asset_volume'].rolling(window=20).std()
        
        # 平均成交价格
        data['avg_trade_price'] = data['quote_asset_volume'] / data['volume']
        data['avg_trade_price_ratio'] = data['avg_trade_price'] / data['close']
        data['price_efficiency'] = 1 - abs(data['avg_trade_price_ratio'] - 1)
        
        # 成交金额趋势
        data['quote_volume_trend_5'] = np.where(data['quote_volume_sma_5'] > data['quote_volume_sma_5'].shift(1), 1, -1)
        data['quote_volume_trend_10'] = np.where(data['quote_volume_sma_10'] > data['quote_volume_sma_10'].shift(1), 1, -1)
        
        # 成交金额异常值
        data['quote_volume_zscore'] = (data['quote_asset_volume'] - data['quote_volume_sma_20']) / data['quote_volume_volatility_20']
        data['quote_volume_anomaly'] = np.where(data['quote_volume_zscore'].abs() > 2, 1, 0)
    
    # 市场深度指标
    if all(col in data.columns for col in ['volume', 'quote_asset_volume', 'number_of_trades']):
        # 市场深度
        data['market_depth'] = data['quote_asset_volume'] / data['number_of_trades']
        data['market_depth_sma_10'] = data['market_depth'].rolling(window=10).mean()
        data['market_depth_ratio'] = data['market_depth'] / data['market_depth_sma_10']
        
        # 流动性指标
        data['liquidity_ratio'] = data['volume'] / data['number_of_trades']
        data['liquidity_sma_10'] = data['liquidity_ratio'].rolling(window=10).mean()
        data['liquidity_trend'] = np.where(data['liquidity_sma_10'] > data['liquidity_sma_10'].shift(1), 1, -1)
        
        # 交易效率指标
        data['trade_efficiency'] = data['quote_asset_volume'] / (data['volume'] * data['close'])
        data['trade_efficiency_sma_10'] = data['trade_efficiency'].rolling(window=10).mean()
        
        # 市场活跃度综合指标
        data['market_activity_score'] = (
            data['trades_ratio_10'] * 0.4 + 
            data['quote_volume_ratio_10'] * 0.4 + 
            data['volume_ratio_10'] * 0.2
        )
    
    return data

def add_order_flow_features(data):
    """添加订单流特征"""
    if all(col in data.columns for col in ['taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'volume', 'quote_asset_volume']):
        # 订单流不平衡
        data['order_flow_imbalance'] = (data['taker_buy_base_asset_volume'] / data['volume']) - 0.5
        data['order_flow_imbalance_abs'] = abs(data['order_flow_imbalance'])
        
        # 订单流强度
        data['order_flow_strength'] = data['order_flow_imbalance'] * data['volume']
        data['order_flow_strength_sma_10'] = data['order_flow_strength'].rolling(window=10).mean()
        
        # 订单流趋势
        data['order_flow_trend_5'] = data['order_flow_imbalance'].rolling(window=5).mean()
        data['order_flow_trend_10'] = data['order_flow_imbalance'].rolling(window=10).mean()
        data['order_flow_trend_20'] = data['order_flow_imbalance'].rolling(window=20).mean()
        
        # 订单流波动率
        data['order_flow_volatility_10'] = data['order_flow_imbalance'].rolling(window=10).std()
        data['order_flow_volatility_20'] = data['order_flow_imbalance'].rolling(window=20).std()
        
        # 订单流异常值
        data['order_flow_zscore'] = (data['order_flow_imbalance'] - data['order_flow_trend_20']) / data['order_flow_volatility_20']
        data['order_flow_anomaly'] = np.where(data['order_flow_zscore'].abs() > 2, 1, 0)
        
        # 订单流与价格关系
        data['order_flow_price_correlation_10'] = data['order_flow_imbalance'].rolling(window=10).corr(data['price_change'])
        
        # 订单流持续性
        data['order_flow_persistence'] = data['order_flow_imbalance'].rolling(window=5).apply(
            lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x)
        )
        
        # 订单流反转信号
        data['order_flow_reversal'] = np.where(
            (data['order_flow_imbalance'] > 0.1) & (data['order_flow_imbalance'].shift(1) < -0.1), 1,
            np.where((data['order_flow_imbalance'] < -0.1) & (data['order_flow_imbalance'].shift(1) > 0.1), -1, 0)
        )
    
    return data

def add_time_based_features(data):
    """添加基于时间的特征"""
    # 转换时间列
    if 'open_time' in data.columns:
        data['timestamp'] = pd.to_datetime(data['open_time'])
    elif 'close_time' in data.columns:
        data['timestamp'] = pd.to_datetime(data['close_time'])
    else:
        data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='H')
    
    # 基础时间特征
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['weekday'] = data['timestamp'].dt.weekday
    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
    data['is_month_start'] = data['timestamp'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['timestamp'].dt.is_month_end.astype(int)
    data['is_quarter_start'] = data['timestamp'].dt.is_quarter_start.astype(int)
    data['is_quarter_end'] = data['timestamp'].dt.is_quarter_end.astype(int)
    
    # 周期性特征
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
    data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
    
    # 交易时段特征
    data['is_asia_session'] = ((data['hour'] >= 0) & (data['hour'] < 8)).astype(int)
    data['is_europe_session'] = ((data['hour'] >= 8) & (data['hour'] < 16)).astype(int)
    data['is_america_session'] = ((data['hour'] >= 16) & (data['hour'] < 24)).astype(int)
    
    # 时间间隔特征
    data['time_since_market_open'] = (data['hour'] + data['weekday'] * 24) % 168  # 一周168小时
    data['time_since_market_open_sin'] = np.sin(2 * np.pi * data['time_since_market_open'] / 168)
    data['time_since_market_open_cos'] = np.cos(2 * np.pi * data['time_since_market_open'] / 168)
    
    # 季节性特征
    data['season'] = pd.cut(data['month'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3])
    data['season'] = data['season'].cat.codes  # 转换为数值
    data['season_sin'] = np.sin(2 * np.pi * data['season'] / 4)
    data['season_cos'] = np.cos(2 * np.pi * data['season'] / 4)
    
    return data

def add_advanced_technical_indicators(data):
    """添加高级技术指标"""
    # 价格位置指标
    data['price_position_5'] = (data['close'] - data['low'].rolling(window=5).min()) / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    data['price_position_10'] = (data['close'] - data['low'].rolling(window=10).min()) / (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min())
    data['price_position_20'] = (data['close'] - data['low'].rolling(window=20).min()) / (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min())
    
    # 价格效率指标
    data['price_efficiency_5'] = abs(data['close'] - data['close'].shift(5)) / data['close'].rolling(window=5).apply(lambda x: np.sum(np.abs(x.diff().dropna())))
    data['price_efficiency_10'] = abs(data['close'] - data['close'].shift(10)) / data['close'].rolling(window=10).apply(lambda x: np.sum(np.abs(x.diff().dropna())))
    
    # 价格动量指标
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # 价格加速度
    data['price_acceleration_5'] = data['price_momentum_5'] - data['price_momentum_5'].shift(5)
    data['price_acceleration_10'] = data['price_momentum_10'] - data['price_momentum_10'].shift(10)
    
    # 价格波动率指标
    data['price_volatility_5'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    data['price_volatility_10'] = data['close'].rolling(window=10).std() / data['close'].rolling(window=10).mean()
    data['price_volatility_20'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
    
    # 价格趋势强度
    data['trend_strength_5'] = abs(data['sma_5'] - data['sma_20']) / data['sma_20']
    data['trend_strength_10'] = abs(data['sma_10'] - data['sma_30']) / data['sma_30']
    
    # 价格反转信号
    data['price_reversal_5'] = np.where(
        (data['close'] > data['high'].shift(1)) & (data['close'].shift(1) < data['low'].shift(2)), 1,
        np.where((data['close'] < data['low'].shift(1)) & (data['close'].shift(1) > data['high'].shift(2)), -1, 0)
    )
    
    return data

def add_volume_price_relationship_features(data):
    """添加价量关系特征"""
    # 价量相关性
    data['volume_price_correlation_5'] = data['volume'].rolling(window=5).corr(data['close'])
    data['volume_price_correlation_10'] = data['volume'].rolling(window=10).corr(data['close'])
    data['volume_price_correlation_20'] = data['volume'].rolling(window=20).corr(data['close'])
    
    # 价量背离指标
    data['volume_price_divergence_5'] = np.where(
        (data['close'] > data['close'].shift(1)) & (data['volume'] < data['volume'].shift(1)), 1,
        np.where((data['close'] < data['close'].shift(1)) & (data['volume'] > data['volume'].shift(1)), -1, 0)
    )
    
    # 价量确认指标
    data['volume_price_confirmation_5'] = np.where(
        (data['close'] > data['close'].shift(1)) & (data['volume'] > data['volume'].shift(1)), 1,
        np.where((data['close'] < data['close'].shift(1)) & (data['volume'] < data['volume'].shift(1)), -1, 0)
    )
    
    # 价量比率
    data['volume_price_ratio'] = data['volume'] / data['close']
    data['volume_price_ratio_sma_10'] = data['volume_price_ratio'].rolling(window=10).mean()
    data['volume_price_ratio_std_10'] = data['volume_price_ratio'].rolling(window=10).std()
    
    # 价量效率
    data['volume_price_efficiency'] = abs(data['price_change']) / data['volume_price_ratio']
    data['volume_price_efficiency_sma_10'] = data['volume_price_efficiency'].rolling(window=10).mean()
    
    return data

def load_and_preprocess_data_advanced(file_path='btcusdt.json', sequence_length=30, dropna=True):
    """增强版数据加载和特征工程，适配auto_temp.py"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    print(f"正在加载数据文件: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"原始数据形状: {df.shape}")
    
    # 确保数值列
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 添加所有特征
    print("正在添加技术指标...")
    df = add_technical_indicators(df)
    
    print("正在添加时间特征...")
    df = add_time_features(df)
    
    print("正在添加价格特征...")
    df = add_price_features(df)
    
    print("正在添加成交量特征...")
    df = add_volume_features(df)
    
    print("正在添加波动率特征...")
    df = add_volatility_features(df)
    
    print("正在添加动量特征...")
    df = add_momentum_features(df)
    
    print("正在添加趋势特征...")
    df = add_trend_features(df)
    
    print("正在添加振荡器特征...")
    df = add_oscillator_features(df)
    
    print("正在添加支撑阻力特征...")
    df = add_support_resistance_features(df)
    
    print("正在添加交易活跃度特征...")
    df = add_trade_activity_features(df)
    
    print("正在添加市场微观结构特征...")
    df = add_market_microstructure_features(df)
    
    print("正在添加订单流特征...")
    df = add_order_flow_features(df)
    
    print("正在添加基于时间的特征...")
    df = add_time_based_features(df)
    
    print("正在添加高级技术指标...")
    df = add_advanced_technical_indicators(df)
    
    print("正在添加价量关系特征...")
    df = add_volume_price_relationship_features(df)
    
    # 处理NaN值
    if dropna:
        df = df.dropna()
        print(f"删除NaN值后数据形状: {df.shape}")
    
    # 创建目标变量
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 选择特征列
    feature_columns = [col for col in df.columns 
                      if col not in ['target', 'open_time', 'close_time', 'timestamp'] and 
                      df[col].dtype in ['float64', 'int64']]
    
    print(f"特征数量: {len(feature_columns)}")
    
    # 准备数据
    X = df[feature_columns].values[:-1]  # 去掉最后一行，因为没有目标值
    y = df['target'].values[:-1]
    
    # 处理无穷大和NaN值
    X = np.array(X, dtype=np.float32)
    X = np.where(np.isnan(X), 0.0, X)
    X = np.where(np.isinf(X), 0.0, X)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"最终数据形状: X={X.shape}, y={y.shape}")
    
    return X, y, feature_columns

def create_sequences(X, y, sequence_length):
    """创建时间序列序列"""
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:(i + sequence_length)])
        sequences_y.append(y[i + sequence_length])
    return np.array(sequences_X, dtype=np.float32), np.array(sequences_y, dtype=np.int64) 