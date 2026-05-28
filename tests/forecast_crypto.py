"""
Kronos 加密货币未来走势预测与分析工具

功能：
- 使用最新市场数据预测未来价格走势
- 提供详细的技术分析报告
- 计算支撑位、阻力位、趋势强度等指标
- 生成可视化分析图表
"""

import sys
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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


def get_candles(contract='BTC_USDT', interval='1h'):
    """从 Gate.io 获取K线数据"""
    api_instance = gate_api.FuturesApi(api_client)
    data = api_instance.list_futures_candlesticks(
        settle='usdt', contract=contract, interval=interval, limit=1000
    )
    df = pd.DataFrame([{
        'close': float(d.c), 'high': float(d.h), 'low': float(d.l),
        'open': float(d.o), 'time': d.t * 1000, 'volume': float(d.v)
    } for d in data])
    df['time'] = pd.to_datetime(df['time'], utc=True, unit='ms', origin='unix')
    df['time'] = df['time'] + timedelta(hours=8)
    df.set_index("time", inplace=True)
    return df


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


def calculate_technical_indicators(pred_df, current_price):
    """计算技术指标"""
    indicators = {}

    # 基础统计
    indicators['current_price'] = current_price
    indicators['pred_high'] = pred_df['high'].max()
    indicators['pred_low'] = pred_df['low'].min()
    indicators['pred_close_avg'] = pred_df['close'].mean()
    indicators['pred_close_final'] = pred_df['close'].iloc[-1]

    # 价格变动
    indicators['price_change'] = indicators['pred_close_final'] - current_price
    indicators['price_change_pct'] = (indicators['price_change'] / current_price) * 100

    # 波动率
    returns = pred_df['close'].pct_change().dropna()
    indicators['volatility'] = returns.std() * 100 if len(returns) > 0 else 0

    # 趋势方向
    if indicators['price_change_pct'] > 2:
        indicators['trend'] = '强势上涨'
        indicators['trend_emoji'] = '🚀'
    elif indicators['price_change_pct'] > 0.5:
        indicators['trend'] = '温和上涨'
        indicators['trend_emoji'] = '📈'
    elif indicators['price_change_pct'] > -0.5:
        indicators['trend'] = '横盘震荡'
        indicators['trend_emoji'] = '➡️'
    elif indicators['price_change_pct'] > -2:
        indicators['trend'] = '温和下跌'
        indicators['trend_emoji'] = '📉'
    else:
        indicators['trend'] = '强势下跌'
        indicators['trend_emoji'] = '🔻'

    # 支撑位和阻力位
    recent_lows = pred_df['low'].nsmallest(5).values
    recent_highs = pred_df['high'].nlargest(5).values
    indicators['support_1'] = np.mean(recent_lows[:2])
    indicators['support_2'] = np.mean(recent_lows)
    indicators['resistance_1'] = np.mean(recent_highs[:2])
    indicators['resistance_2'] = np.mean(recent_highs)

    # 预测区间
    indicators['expected_range'] = indicators['pred_high'] - indicators['pred_low']
    indicators['expected_range_pct'] = (indicators['expected_range'] / current_price) * 100

    # 成交量趋势
    indicators['avg_volume'] = pred_df['volume'].mean()
    indicators['volume_trend'] = '放量' if pred_df['volume'].iloc[-10:].mean() > pred_df['volume'].iloc[:10].mean() else '缩量'

    return indicators


def generate_trading_signals(indicators):
    """生成交易信号"""
    signals = []

    # 趋势信号
    if indicators['price_change_pct'] > 3:
        signals.append(('强烈看多', '🟢🟢🟢', '模型预测价格将大幅上涨'))
    elif indicators['price_change_pct'] > 1:
        signals.append(('看多', '🟢🟢', '模型预测价格将上涨'))
    elif indicators['price_change_pct'] < -3:
        signals.append(('强烈看空', '🔴🔴🔴', '模型预测价格将大幅下跌'))
    elif indicators['price_change_pct'] < -1:
        signals.append(('看空', '🔴🔴', '模型预测价格将下跌'))
    else:
        signals.append(('中性', '⚪⚪', '模型预测价格横盘震荡'))

    # 波动率信号
    if indicators['volatility'] > 2:
        signals.append(('高波动', '⚠️', '市场波动较大，注意风险控制'))
    elif indicators['volatility'] < 0.5:
        signals.append(('低波动', '😴', '市场波动较小，可能酝酿突破'))

    # 成交量信号
    if indicators['volume_trend'] == '放量':
        signals.append(('放量', '📊', '成交量放大，趋势可能延续'))
    else:
        signals.append(('缩量', '📉', '成交量萎缩，趋势可能减弱'))

    # 价格位置信号
    if indicators['current_price'] < indicators['support_1']:
        signals.append(('接近支撑', '🛡️', f'当前价格接近支撑位 {indicators["support_1"]:.2f}'))
    elif indicators['current_price'] > indicators['resistance_1']:
        signals.append(('接近阻力', '🚧', f'当前价格接近阻力位 {indicators["resistance_1"]:.2f}'))

    return signals


def print_analysis_report(contract, interval, indicators, signals, pred_df):
    """打印分析报告"""
    print('\n' + '=' * 70)
    print(f'📊 {contract} ({interval}) 未来走势分析报告')
    print('=' * 70)

    # 价格预测
    print(f'\n【价格预测】')
    print(f'   当前价格:      {indicators["current_price"]:,.2f} USDT')
    print(f'   预测最终价格:  {indicators["pred_close_final"]:,.2f} USDT')
    print(f'   价格变动:      {indicators["price_change"]:+,.2f} USDT ({indicators["price_change_pct"]:+.2f}%)')
    print(f'   趋势判断:      {indicators["trend_emoji"]} {indicators["trend"]}')

    # 价格区间
    print(f'\n【价格区间预测】')
    print(f'   预测最高价:    {indicators["pred_high"]:,.2f} USDT')
    print(f'   预测最低价:    {indicators["pred_low"]:,.2f} USDT')
    print(f'   预期波动范围:  {indicators["expected_range"]:,.2f} USDT ({indicators["expected_range_pct"]:.2f}%)')
    print(f'   预测均价:      {indicators["pred_close_avg"]:,.2f} USDT')

    # 支撑位和阻力位
    print(f'\n【支撑位与阻力位】')
    print(f'   强阻力位 (R2): {indicators["resistance_2"]:,.2f} USDT')
    print(f'   弱阻力位 (R1): {indicators["resistance_1"]:,.2f} USDT')
    print(f'   当前价格:      {indicators["current_price"]:,.2f} USDT  ◀')
    print(f'   弱支撑位 (S1): {indicators["support_1"]:,.2f} USDT')
    print(f'   强支撑位 (S2): {indicators["support_2"]:,.2f} USDT')

    # 成交量分析
    print(f'\n【成交量分析】')
    print(f'   平均成交量:    {indicators["avg_volume"]:,.0f}')
    print(f'   成交量趋势:    {indicators["volume_trend"]}')

    # 风险指标
    print(f'\n【风险指标】')
    print(f'   预测波动率:    {indicators["volatility"]:.2f}%')
    risk_level = '低' if indicators['volatility'] < 1 else '中' if indicators['volatility'] < 2 else '高'
    print(f'   风险等级:      {risk_level}')

    # 交易信号
    print(f'\n【交易信号】')
    for signal_name, emoji, description in signals:
        print(f'   {emoji} {signal_name}: {description}')

    # 风险提示
    print(f'\n【⚠️ 风险提示】')
    print('   1. 以上预测仅供参考，不构成投资建议')
    print('   2. 加密货币市场波动剧烈，请严格控制仓位')
    print('   3. 建议结合其他技术指标和基本面分析')
    print('   4. 设置好止损点，保护本金安全')

    print('\n' + '=' * 70)


def plot_forecast_analysis(hist_df, pred_df, indicators, contract, interval):
    """绘制预测分析图表"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

    # 1. 价格走势预测 (占据整行)
    ax1 = fig.add_subplot(gs[0, :])

    # 历史数据
    hist_recent = hist_df.tail(100)
    ax1.plot(hist_recent.index, hist_recent['close'], label='历史价格',
             color='#2196F3', linewidth=2)

    # 预测数据
    ax1.plot(pred_df.index, pred_df['close'], label='预测价格',
             color='#FF5722', linewidth=2, linestyle='--')

    # 预测区间
    ax1.fill_between(pred_df.index, pred_df['low'], pred_df['high'],
                     alpha=0.2, color='#FF5722', label='预测区间 (高/低)')

    # 支撑位和阻力位
    ax1.axhline(y=indicators['resistance_1'], color='#F44336', linestyle=':',
                alpha=0.7, label=f'阻力位 R1 ({indicators["resistance_1"]:.2f})')
    ax1.axhline(y=indicators['support_1'], color='#4CAF50', linestyle=':',
                alpha=0.7, label=f'支撑位 S1 ({indicators["support_1"]:.2f})')
    ax1.axhline(y=indicators['current_price'], color='#9C27B0', linestyle='-',
                alpha=0.5, label=f'当前价格 ({indicators["current_price"]:.2f})')

    ax1.set_title(f'{contract} ({interval}) - Kronos 未来走势预测',
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. K线图 (预测部分)
    ax2 = fig.add_subplot(gs[1, 0])
    candle_colors = ['#F44336' if pred_df['close'].iloc[i] < pred_df['open'].iloc[i]
                     else '#4CAF50' for i in range(len(pred_df))]

    for i in range(len(pred_df)):
        color = candle_colors[i]
        # 实体
        ax2.bar(pred_df.index[i], abs(pred_df['close'].iloc[i] - pred_df['open'].iloc[i]),
                bottom=min(pred_df['open'].iloc[i], pred_df['close'].iloc[i]),
                color=color, alpha=0.8, width=timedelta(minutes=30) if interval == '1h' else timedelta(hours=2))
        # 影线
        ax2.vlines(pred_df.index[i], pred_df['low'].iloc[i], pred_df['high'].iloc[i],
                   color=color, linewidth=1)

    ax2.set_title('预测K线图', fontsize=12)
    ax2.set_ylabel('价格 (USDT)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 3. 成交量预测
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['#4CAF50' if pred_df['close'].iloc[i] >= pred_df['open'].iloc[i]
              else '#F44336' for i in range(len(pred_df))]
    ax3.bar(pred_df.index, pred_df['volume'], color=colors, alpha=0.7)
    ax3.set_title('预测成交量', fontsize=12)
    ax3.set_ylabel('成交量', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # 4. 价格分布
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(pred_df['close'], bins=30, color='#2196F3', alpha=0.7, edgecolor='black')
    ax4.axvline(indicators['pred_close_avg'], color='red', linestyle='--',
                label=f'均值: {indicators["pred_close_avg"]:.2f}')
    ax4.axvline(indicators['current_price'], color='green', linestyle='-',
                label=f'当前: {indicators["current_price"]:.2f}')
    ax4.set_title('预测价格分布', fontsize=12)
    ax4.set_xlabel('价格 (USDT)', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 收益率分布
    ax5 = fig.add_subplot(gs[2, 1])
    returns = pred_df['close'].pct_change().dropna() * 100
    ax5.hist(returns, bins=30, color='#FF9800', alpha=0.7, edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--')
    ax5.axvline(returns.mean(), color='blue', linestyle='--',
                label=f'均值: {returns.mean():.2f}%')
    ax5.set_title('预测收益率分布', fontsize=12)
    ax5.set_xlabel('收益率 (%)', fontsize=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.savefig(f'forecast_analysis_{contract}_{interval}.png', dpi=150, bbox_inches='tight')
    print(f'\n✅ 分析图表已保存为 forecast_analysis_{contract}_{interval}.png')
    plt.show()


def forecast_future(contract='BTC_USDT', interval='1h', lookback=400, pred_len=120):
    """
    预测加密货币未来走势

    Args:
        contract: 交易对，如 'BTC_USDT', 'ETH_USDT'
        interval: K线周期，如 '1h', '15m', '5m'
        lookback: 历史窗口大小
        pred_len: 预测长度
    """
    print('=' * 70)
    print(f'🔮 Kronos 加密货币未来走势预测 - {contract} ({interval})')
    print('=' * 70)

    # 1. 下载最新数据
    print('\n📊 步骤 1: 下载最新市场数据...')
    df = get_candles(contract=contract, interval=interval)
    df = df.sort_index()
    df['amount'] = df['volume'] * (df['open'] + df['close']) / 2

    cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    print(f'   数据形状: {df.shape}')
    print(f'   数据时间范围: {df.index[0]} ~ {df.index[-1]}')
    print(f'   最新收盘价: {df["close"].iloc[-1]:,.2f} USDT')

    # 2. 加载模型
    print('\n🤖 步骤 2: 加载 Kronos 预测模型...')
    tokenizer = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
    model = Kronos.from_pretrained('NeoQuasar/Kronos-small')
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print('   ✅ 模型加载完成')

    # 3. 准备预测输入（使用最新数据作为历史）
    print(f'\n📝 步骤 3: 准备预测输入...')
    actual_lookback = min(lookback, len(df))
    x_df = df.iloc[-actual_lookback:][cols].copy()
    x_timestamp = pd.Series(df.index[-actual_lookback:]).reset_index(drop=True)

    # 生成未来时间戳
    y_timestamp = generate_future_timestamps(df.index[-1], interval, pred_len)

    print(f'   历史数据: {len(x_df)} 条')
    print(f'   预测长度: {pred_len} 条')
    print(f'   历史区间: {x_timestamp.iloc[0]} ~ {x_timestamp.iloc[-1]}')
    print(f'   预测区间: {y_timestamp.iloc[0]} ~ {y_timestamp.iloc[-1]}')

    # 4. 执行预测
    # ==================== 采样参数说明 ====================
    # T (Temperature - 温度参数):
    #   控制预测的随机性，影响概率分布的平滑度
    #   - T = 1.0: 默认行为，平衡随机性
    #   - T < 1.0: 更保守，倾向选择高概率 token，预测更稳定
    #   - T > 1.0: 更激进，增加随机性，探索更多可能性
    #   - T → 0:   几乎只选概率最高的（贪婪解码）
    #
    # top_p (Nucleus Sampling - 核采样):
    #   限制采样范围，只保留累积概率达到 top_p 的最小 token 集合
    #   - top_p = 0.9: 只从累积概率达 90% 的 token 中采样（推荐）
    #   - top_p = 0.5: 更保守，只从累积概率达 50% 的 token 中采样
    #   - top_p = 1.0: 使用所有可能的 token（完全随机）
    #
    # sample_count (采样次数):
    #   执行多次采样并取平均值，可以减少单次随机性带来的波动
    #   - sample_count = 1:  单次采样，速度最快
    #   - sample_count > 1:  多次采样取平均，结果更稳定
    #
    # 推荐配置:
    #   - 保守预测: T=0.8, top_p=0.8, sample_count=5
    #   - 标准预测: T=1.0, top_p=0.9, sample_count=1 (当前设置)
    #   - 激进预测: T=1.2, top_p=0.95, sample_count=1
    # ======================================================
    print('\n🔮 步骤 4: 执行未来走势预测（请耐心等待）...')
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=0.8,              # 温度参数：1.0 为标准预测
        top_p=0.8,          # 核采样：保留累积概率 90% 的 token
        sample_count=5,     # 采样次数：1 为单次采样
        verbose=True
    )
    pred_df.index = y_timestamp.values  # 设置正确的时间索引
    print('   ✅ 预测完成!')

    # 5. 技术分析
    print('\n📈 步骤 5: 生成技术分析报告...')
    current_price = df['close'].iloc[-1]
    indicators = calculate_technical_indicators(pred_df, current_price)
    signals = generate_trading_signals(indicators)

    # 打印报告
    print_analysis_report(contract, interval, indicators, signals, pred_df)

    # 6. 可视化
    print('\n📊 步骤 6: 生成分析图表...')
    plot_forecast_analysis(df, pred_df, indicators, contract, interval)

    # 7. 保存结果
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    pred_df.to_csv(f'forecast_{contract}_{interval}_{timestamp_str}.csv')
    print(f'\n💾 预测数据已保存为 forecast_{contract}_{interval}_{timestamp_str}.csv')

    # 保存分析报告摘要
    report_file = f'forecast_report_{contract}_{interval}_{timestamp_str}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f'{contract} ({interval}) 未来走势预测报告\n')
        f.write(f'生成时间: {pd.Timestamp.now()}\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'当前价格: {current_price:,.2f} USDT\n')
        f.write(f'预测最终价格: {indicators["pred_close_final"]:,.2f} USDT\n')
        f.write(f'价格变动: {indicators["price_change_pct"]:+.2f}%\n')
        f.write(f'趋势判断: {indicators["trend"]}\n\n')
        f.write(f'预测最高价: {indicators["pred_high"]:,.2f} USDT\n')
        f.write(f'预测最低价: {indicators["pred_low"]:,.2f} USDT\n')
        f.write(f'阻力位 R1: {indicators["resistance_1"]:,.2f} USDT\n')
        f.write(f'支撑位 S1: {indicators["support_1"]:,.2f} USDT\n\n')
        f.write('交易信号:\n')
        for signal_name, emoji, description in signals:
            f.write(f'  {emoji} {signal_name}: {description}\n')
    print(f'📄 分析报告已保存为 {report_file}')

    print('\n' + '=' * 70)
    print('🎉 预测分析完成！')
    print('=' * 70)

    return pred_df, indicators, signals


if __name__ == '__main__':
    # 预测 BTC/USDT 未来走势
    # 可修改参数:
    #   contract: 交易对 (如 'ETH_USDT', 'DOGE_USDT')
    #   interval: 周期 (如 '15m', '1h', '4h')
    #   lookback: 使用的历史数据长度
    #   pred_len: 预测未来数据点数

    pred_df, indicators, signals = forecast_future(
        contract='DOGE_USDT',
        interval='15m',
        lookback=400,
        pred_len=120  # 预测未来120根K线
    )
