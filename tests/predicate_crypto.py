import sys
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体（Windows 使用 SimHei 或 Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import Kronos, KronosTokenizer, KronosPredictor

import gate_api

# API 密钥配置（请设置环境变量或直接修改此处）
# Windows: set GATE_API_KEY=your_key
# Linux/Mac: export GATE_API_KEY=your_key
GATE_API_KEY = os.environ.get('GATE_API_KEY', 'your_api_key_here')
GATE_API_SECRET = os.environ.get('GATE_API_SECRET', 'your_api_secret_here')

configuration = gate_api.Configuration(
    host="https://api.gateio.ws/api/v4",
    key=GATE_API_KEY,
    secret=GATE_API_SECRET
)

api_client = gate_api.ApiClient(configuration)


# 获取K线数据
def generate_future_timestamps(last_timestamp, interval, pred_len):
    """生成未来时间戳"""
    interval_map = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
    }
    delta = interval_map.get(interval, timedelta(hours=1))
    future_times = [last_timestamp + delta * (i + 1) for i in range(pred_len)]
    return pd.Series(future_times)


def get_candles(contract='BTC_USDT', interval='1h'):
    # Create an instance of the API class
    api_instance = gate_api.FuturesApi(api_client)
    data = api_instance.list_futures_candlesticks(settle='usdt', contract=contract, interval=interval, limit=1000)

    df = pd.DataFrame([{'close': float(d.c), 'high': float(d.h), 'low': float(d.l), 'open': float(d.o),
                        'time': d.t * 1000, 'volume': float(d.v)} for d in data])
    df['time'] = pd.to_datetime(df['time'], utc=True, unit='ms', origin='unix')
    df['time'] = df['time'] + timedelta(hours=8)
    df.set_index("time", inplace=True)
    return df


def plot_prediction(kline_df, pred_df, title='Kronos Crypto Prediction'):
    """绘制加密货币预测结果（pred_df.index 已经是未来时间戳）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # 收盘价对比
    ax1.plot(kline_df['close'], label='真实值', color='blue', linewidth=1.5)
    ax1.plot(pred_df['close'], label='预测值', color='red', linewidth=1.5)
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.legend(loc='lower left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title, fontsize=16)

    # 成交量对比
    ax2.bar(kline_df.index[-pred_df.shape[0]:], kline_df['volume'].iloc[-pred_df.shape[0]:],
            label='真实成交量', color='blue', alpha=0.5)
    ax2.bar(pred_df.index, pred_df['volume'], label='预测成交量', color='red', alpha=0.5)
    ax2.set_ylabel('成交量', fontsize=12)
    ax2.set_xlabel('时间', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('crypto_prediction_result.png', dpi=150, bbox_inches='tight')
    print('✅ 图表已保存为 crypto_prediction_result.png')
    plt.show()


def run_prediction(contract='BTC_USDT', interval='1h', lookback=400, pred_len=120):
    """
    使用 Kronos 对加密货币进行预测

    Args:
        contract: 交易对，如 'BTC_USDT', 'ETH_USDT'
        interval: K线周期，如 '1h', '15m', '5m'
        lookback: 历史窗口大小
        pred_len: 预测长度
    """
    print('=' * 60)
    print(f'🚀 Kronos 加密货币预测 - {contract} ({interval})')
    print('=' * 60)

    # 1. 下载K线数据
    print('\n📊 步骤 1: 从 Gate.io 下载K线数据...')
    df = get_candles(contract=contract, interval=interval)
    df = df.sort_index()  # 确保时间升序
    print(f'   数据形状: {df.shape}')
    print(f'   时间范围: {df.index[0]} ~ {df.index[-1]}')
    print(f'   列名: {list(df.columns)}')

    # 补充 amount 列（成交量 * 均价）
    df['amount'] = df['volume'] * (df['open'] + df['close']) / 2

    # 确保列顺序符合 Kronos 要求
    cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in cols:
        if col not in df.columns:
            raise ValueError(f'缺少必要列: {col}')

    print(f'   最新收盘价: {df["close"].iloc[-1]:.2f}')

    # 2. 加载模型
    print('\n🤖 步骤 2: 加载 Kronos 模型...')
    tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
    model = Kronos.from_pretrained('NeoQuasar/Kronos-small')
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print('   ✅ 模型加载完成')

    # 3. 准备输入
    print(f'\n📝 步骤 3: 准备预测输入 (lookback={lookback}, pred_len={pred_len})...')

    if len(df) < lookback:
        print(f'   ⚠️ 数据不足! 仅有 {len(df)} 条，需要 {lookback} 条')
        lookback = len(df)
        print(f'   自动调整 lookback={lookback}')

    x_df = df.iloc[-lookback:][cols].copy()
    x_timestamp = pd.Series(df.index[-lookback:]).reset_index(drop=True)
    y_timestamp = generate_future_timestamps(df.index[-1], interval, pred_len)

    print(f'   历史数据: {len(x_df)} 条')
    print(f'   预测长度: {pred_len} 条')
    print(f'   历史区间: {x_timestamp.iloc[0]} ~ {x_timestamp.iloc[-1]}')
    print(f'   预测区间: {y_timestamp.iloc[0]} ~ {y_timestamp.iloc[-1]}')

    # 4. 执行预测
    print('\n🔮 步骤 4: 执行预测（可能需要几分钟）...')
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )
    print('   ✅ 预测完成!')

    # 5. 显示结果
    print('\n📈 步骤 5: 预测结果')
    print('   预测数据前 10 行:')
    print(pred_df.head(10).to_string())

    # 计算涨跌预测
    last_close = x_df['close'].iloc[-1]
    pred_close = pred_df['close'].iloc[-1]
    change_pct = (pred_close - last_close) / last_close * 100
    trend = '📈 上涨' if change_pct > 0 else '📉 下跌'
    print(f'\n   当前价格: {last_close:.2f}')
    print(f'   预测最终价格: {pred_close:.2f} ({trend} {change_pct:+.2f}%)')

    # 6. 可视化
    print('\n📊 步骤 6: 可视化结果...')
    kline_df = df.iloc[-lookback:].copy()
    kline_df.index = x_timestamp.values
    pred_df.index = y_timestamp.values
    plot_prediction(kline_df, pred_df, title=f'{contract} ({interval}) 预测')

    # 7. 保存预测结果
    pred_df.to_csv(f'prediction_{contract}_{interval}.csv')
    print(f'\n💾 预测结果已保存为 prediction_{contract}_{interval}.csv')

    print('\n' + '=' * 60)
    print('🎉 预测完成！')
    print('=' * 60)

    return pred_df


if __name__ == '__main__':
    # 预测 BTC/USDT 小时线
    run_prediction(
        contract='BTC_USDT',
        interval='1h',
        lookback=400,
        pred_len=120
    )
