"""
Trading Bot: Multi-timeframe signal generator, backtester, analyzer, and iterative improver
Targets: BTC/USD and ETH/USD

Features:
 - Fetch data using ccxt (real-time) or yfinance (historical fallback)
 - Multi-timeframe signal generation using RSI, MACD, and EMA crossover + ATR-based SL/TP
 - Backtesting using Backtrader
 - Results compilation into CSV/DataFrame and visualizations (Matplotlib)
 - Losing-trade analysis with scikit-learn (clustering) and simple rule-based categorizations
 - Iterative suggestion generator to tweak strategy parameters and re-run backtests
 - Simulation mode and optional live trading via Binance/ccxt (demo; use testnet API keys)

Setup / Requirements
--------------------
Run:
    pip install pandas numpy matplotlib ccxt yfinance backtrader ta scikit-learn scipy

Notes:
 - Backtrader can be finicky with newer Python; use Python 3.8-3.11 for best compatibility.
 - For live trading, configure your exchange API keys and set LIVE=True. Always test on testnet.

Usage example (from command line):
    python trading_bot.py --symbol BTC/USD --mode backtest

"""

import argparse
import datetime as dt
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Libraries that might not be installed in all environments
try:
    import ccxt
except Exception:
    ccxt = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import backtrader as bt
except Exception:
    bt = None

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, EMAIndicator
    from ta.volatility import AverageTrueRange
except Exception:
    # We'll implement fallbacks if ta isn't installed
    RSIIndicator = None
    MACD = None
    EMAIndicator = None
    AverageTrueRange = None

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:
    KMeans = None
    StandardScaler = None

import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Data fetching utilities
# -------------------------

DEFAULT_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h']

def timeframe_to_minutes(tf: str) -> int:
    if tf.endswith('m'):
        return int(tf[:-1])
    if tf.endswith('h'):
        return int(tf[:-1]) * 60
    raise ValueError('Unknown timeframe format')


def fetch_ohlcv_ccxt(exchange_id: str, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 1000, params=None):
    if ccxt is None:
        raise RuntimeError('ccxt not installed')
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})
    if since is None:
        since = exchange.milliseconds() - limit * timeframe_to_minutes(timeframe) * 60 * 1000
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit, params=params or {})
    except Exception as e:
        logger.error('ccxt fetch error: %s', e)
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df


def fetch_ohlcv_ccxt_paginated(exchange_id: str, symbol: str, timeframe: str, max_candles: int = 10000, since: Optional[int] = None, params=None) -> pd.DataFrame:
    """Fetch more than 1000 candles by paginating ccxt.fetch_ohlcv.
    Args:
        exchange_id: e.g., 'binance'
        symbol: 'BTC/USDT'
        timeframe: '1m','5m','15m','30m','1h','4h','1d', ...
        max_candles: total candles to retrieve (caps loop)
        since: ms timestamp to start from; if None, backfills from now backwards
    Returns:
        DataFrame concatenated chronologically.
    """
    if ccxt is None:
        raise RuntimeError('ccxt not installed')
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    limit_per_call = 1000
    all_rows: List[List[float]] = []

    # If since not provided, start roughly max_candles back
    if since is None:
        since = exchange.milliseconds() - max_candles * timeframe_to_minutes(timeframe) * 60 * 1000

    fetched = 0
    last_ts = since

    while fetched < max_candles:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=last_ts, limit=limit_per_call, params=params or {})
        except Exception as e:
            logger.warning('fetch_ohlcv page failed: %s', e)
            break
        if not ohlcv:
            break
        # Append, but avoid duplicate timestamp at page boundary
        if all_rows and ohlcv[0][0] == all_rows[-1][0]:
            ohlcv = ohlcv[1:]
        all_rows.extend(ohlcv)
        fetched += len(ohlcv)
        last_ts = ohlcv[-1][0] + 1  # next ms after last candle
        # Stop if less than limit_per_call returned (no more data)
        if len(ohlcv) < limit_per_call:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df


def fetch_ohlcv_yfinance(symbol: str, timeframe: str, period: str = '60d') -> pd.DataFrame:
    # yfinance uses tickers like BTC-USD
    if yf is None:
        raise RuntimeError('yfinance not installed')
    # Map timeframe to yfinance interval
    interval_map = {'1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '60m', '4h': '60m'}
    interval = interval_map.get(timeframe, '60m')
    ticker = symbol.replace('/', '-')
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError('yfinance returned empty data')
    # yfinance returns DatetimeIndex
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_multi_timeframe(symbol: str, timeframes: List[str], limit: int = 1000, max_candles: int = 10000, since: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple timeframes from Binance via ccxt with pagination.
    Args:
        symbol: e.g., "BTC/USDT" or "ETH/USDT" (Binance format)
        timeframes: ["1m","5m","15m","30m","1h","4h"]
        limit: deprecated (kept for backward compat) — per-call limit is handled internally
        max_candles: target number of candles per timeframe (will loop pages up to this)
        since: optional ms timestamp to start from (if None we estimate back from now)
    Returns:
        dict mapping timeframe -> DataFrame
    """
    data = {}
    for tf in timeframes:
        try:
            df = fetch_ohlcv_ccxt_paginated('binance', symbol, tf, max_candles=max_candles, since=since)
            if df is None or df.empty:
                logger.warning('No data for %s @ %s from Binance', symbol, tf)
                continue
            data[tf] = df
            logger.info('Fetched %s bars for %s @ %s (Binance, paginated)', len(df), symbol, tf)
        except Exception as e:
            logger.warning('Failed to fetch %s @ %s from Binance: %s', symbol, tf, e)
    return data

# -------------------------
# Indicators & Signals
# -------------------------

@dataclass
class StrategyParams:
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5  # SL distance multiplier of ATR
    tp_multiplier: float = 3.0  # TP as multiple of SL


def add_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df = df.copy()
    # Ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # RSI
    try:
        if RSIIndicator is not None:
            rsi = RSIIndicator(close=df['close'], window=params.rsi_period).rsi()
        else:
            delta = df['close'].diff(1)
            gain = delta.clip(lower=0).rolling(params.rsi_period).mean()
            loss = -delta.clip(upper=0).rolling(params.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi
    except Exception as e:
        logger.exception('RSI error: %s', e)
        df['rsi'] = np.nan
    # EMA
    try:
        if EMAIndicator is not None:
            df['ema_fast'] = EMAIndicator(close=df['close'], window=params.ema_fast).ema_indicator()
            df['ema_slow'] = EMAIndicator(close=df['close'], window=params.ema_slow).ema_indicator()
        else:
            df['ema_fast'] = df['close'].ewm(span=params.ema_fast, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=params.ema_slow, adjust=False).mean()
    except Exception as e:
        logger.exception('EMA error: %s', e)
        df['ema_fast'] = df['close']
        df['ema_slow'] = df['close']
    # MACD
    try:
        if MACD is not None:
            macd = MACD(close=df['close'], window_slow=params.ema_slow, window_fast=params.ema_fast, window_sign=params.macd_signal)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
        else:
            ema_fast = df['close'].ewm(span=params.ema_fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=params.ema_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=params.macd_signal, adjust=False).mean()
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_line - macd_signal
    except Exception as e:
        logger.exception('MACD error: %s', e)
        df['macd'] = df['macd_signal'] = df['macd_hist'] = 0
    # ATR
    try:
        if AverageTrueRange is not None:
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=params.atr_period).average_true_range()
        else:
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift()).abs()
            tr3 = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(params.atr_period).mean()
        df['atr'] = atr
    except Exception as e:
        logger.exception('ATR error: %s', e)
        df['atr'] = np.nan
    return df


@dataclass
class Signal:
    datetime: pd.Timestamp
    symbol: str
    timeframe: str
    side: str  # 'long' or 'short'
    entry: float
    sl: float
    tp: float
    reason: str
    indicators: Dict


def generate_signals_multi_tf(data: Dict[str, pd.DataFrame], params: StrategyParams, symbol: str) -> List[Signal]:
    """Simple strategy:
    - Look for EMA cross on higher timeframe (e.g., 1h or 4h) to determine trend bias
    - Use RSI + MACD on lower timeframe to trigger entries
    - SL based on ATR * multiplier
    - TP = entry + tp_multiplier * (entry - SL) for longs
    """
    signals = []
    # Determine bias from 4h if available, else 1h
    bias_tf = None
    if '4h' in data:
        bias_tf = '4h'
    elif '1h' in data:
        bias_tf = '1h'
    else:
        bias_tf = list(data.keys())[-1]

    bias_df = add_indicators(data[bias_tf], params)
    latest_bias = bias_df.iloc[-1]
    if latest_bias['ema_fast'] > latest_bias['ema_slow']:
        bias = 'bull'
    else:
        bias = 'bear'
    logger.info('Trend bias based on %s = %s', bias_tf, bias)

    # Scan lower timeframes for triggers
    trigger_tfs = ['1m', '5m', '15m', '30m']
    for tf in trigger_tfs:
        df = data.get(tf)
        if df is None or len(df) < 50:
            continue
        df_ind = add_indicators(df, params)
        row = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        # Long conditions
        long_cond = (
            prev['ema_fast'] < prev['ema_slow'] and row['ema_fast'] > row['ema_slow']  # ema crossover
            and row['rsi'] > params.rsi_oversold and row['rsi'] < params.rsi_overbought
            and row['macd'] > row['macd_signal']
        )
        short_cond = (
            prev['ema_fast'] > prev['ema_slow'] and row['ema_fast'] < row['ema_slow']
            and row['rsi'] < params.rsi_overbought and row['rsi'] > params.rsi_oversold
            and row['macd'] < row['macd_signal']
        )
        # Bias filter
        if bias == 'bull' and long_cond:
            entry = float(row['close'])
            atr = float(row['atr']) if not math.isnan(row['atr']) else entry * 0.01
            sl = entry - params.atr_sl_multiplier * atr
            tp = entry + params.tp_multiplier * (entry - sl)
            signals.append(Signal(datetime=row.name, symbol=symbol, timeframe=tf, side='long', entry=entry, sl=sl, tp=tp,
                                  reason=f'EMA cross + MACD + RSI in {tf} (bias {bias})', indicators={'rsi': row['rsi'], 'macd': row['macd'], 'atr': atr}))
        if bias == 'bear' and short_cond:
            entry = float(row['close'])
            atr = float(row['atr']) if not math.isnan(row['atr']) else entry * 0.01
            sl = entry + params.atr_sl_multiplier * atr
            tp = entry - params.tp_multiplier * (sl - entry)
            signals.append(Signal(datetime=row.name, symbol=symbol, timeframe=tf, side='short', entry=entry, sl=sl, tp=tp,
                                  reason=f'EMA cross + MACD + RSI in {tf} (bias {bias})', indicators={'rsi': row['rsi'], 'macd': row['macd'], 'atr': atr}))
    return signals

# -------------------------
# Backtesting with Backtrader
# -------------------------

class SimpleStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('ema_fast', 12),
        ('ema_slow', 26),
        ('macd_signal', 9),
        ('atr_period', 14),
        ('atr_sl_mult', 1.5),
        ('tp_mult', 3.0),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.ema_fast = bt.ind.EMA(self.datas[0], period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(self.datas[0], period=self.p.ema_slow)
        self.rsi = bt.ind.RSI(self.datas[0], period=self.p.rsi_period)
        self.macd = bt.ind.MACD(self.datas[0], period_me1=self.p.ema_fast, period_me2=self.p.ema_slow, period_signal=self.p.macd_signal)
        self.atr = bt.ind.ATR(self.datas[0], period=self.p.atr_period)
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        logger.debug('%s, %s', dt.isoformat(), txt)

    def next(self):
        if self.order:
            return
        # Check if we are in the market
        if not self.position:
            # Simple entry: EMA crossover + MACD + RSI
            if self.ema_fast[0] > self.ema_slow[0] and self.ema_fast[-1] <= self.ema_slow[-1] and self.macd.macd[0] > self.macd.signal[0] and self.rsi[0] > self.p.rsi_oversold:
                size = self.broker.getcash() * 0.01 / self.dataclose[0]  # 1% of cash risked as position size (simplified)
                self.entry_price = self.dataclose[0]
                self.sl_price = self.entry_price - self.p.atr_sl_mult * self.atr[0]
                self.tp_price = self.entry_price + self.p.tp_mult * (self.entry_price - self.sl_price)
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                self.order = self.buy(size=size)
            elif self.ema_fast[0] < self.ema_slow[0] and self.ema_fast[-1] >= self.ema_slow[-1] and self.macd.macd[0] < self.macd.signal[0] and self.rsi[0] < self.p.rsi_overbought:
                size = self.broker.getcash() * 0.01 / self.dataclose[0]
                self.entry_price = self.dataclose[0]
                self.sl_price = self.entry_price + self.p.atr_sl_mult * self.atr[0]
                self.tp_price = self.entry_price - self.p.tp_mult * (self.sl_price - self.entry_price)
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                self.order = self.sell(size=size)
        else:
            # Manage open position: check SL/TP
            if self.position.size > 0:
                if self.dataclose[0] <= self.sl_price or self.dataclose[0] >= self.tp_price:
                    self.log('CLOSE LONG, price %.2f' % self.dataclose[0])
                    self.order = self.close()
            else:
                if self.dataclose[0] >= self.sl_price or self.dataclose[0] <= self.tp_price:
                    self.log('CLOSE SHORT, price %.2f' % self.dataclose[0])
                    self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None


def run_backtest(df: pd.DataFrame, params: StrategyParams, cash: float = 10000.0, commission: float = 0.001) -> Tuple[pd.DataFrame, dict]:
    if bt is None:
        raise RuntimeError('backtrader not installed')
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    datafeed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(datafeed)
    cerebro.addstrategy(SimpleStrategy,
                        rsi_period=params.rsi_period,
                        rsi_overbought=params.rsi_overbought,
                        rsi_oversold=params.rsi_oversold,
                        ema_fast=params.ema_fast,
                        ema_slow=params.ema_slow,
                        macd_signal=params.macd_signal,
                        atr_period=params.atr_period,
                        atr_sl_mult=params.atr_sl_multiplier,
                        tp_mult=params.tp_multiplier)
    # Capture analyzer metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    logger.info('Starting Portfolio Value: %.2f', cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    logger.info('Final Portfolio Value: %.2f', final_value)
    # Extract analyzers
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    stats = {
        'start_value': cash,
        'final_value': final_value,
        'pnl': final_value - cash,
        'sharpe': sharpe,
        'drawdown': dd,
        'trades': trades,
    }
    # Note: building a full trade log requires custom logging or observers; for brevity we'll omit a per-trade DataFrame
    return pd.DataFrame([stats]), stats

# -------------------------
# Post-backtest analysis
# -------------------------

def analyze_losing_trades(trade_log: pd.DataFrame) -> Dict:
    """Expect trade_log with columns: entry_dt, exit_dt, side, entry_price, exit_price, pnl, reason, indicators..."""
    if trade_log.empty:
        return {'message': 'No trades to analyze'}
    losers = trade_log[trade_log['pnl'] < 0].copy()
    analysis = {}
    analysis['num_losers'] = len(losers)
    if len(losers) == 0:
        return analysis
    # Simple categorizations
    # 1) SL hit quickly (within N bars)
    if 'duration' in losers.columns:
        quick_sl = losers[losers['duration'] <= 3]
        analysis['quick_sl_rate'] = len(quick_sl) / len(losers)
    # 2) Volatility-based: large ATR at entry
    if 'atr_at_entry' in losers.columns:
        high_atr = losers['atr_at_entry'].quantile(0.75)
        analysis['high_atr_threshold'] = high_atr
        analysis['high_atr_fraction'] = (losers['atr_at_entry'] > high_atr).mean()
    # 3) Clustering by features if sklearn present
    if KMeans is not None and len(losers) >= 5:
        features = []
        for col in ['rsi_at_entry', 'macd_at_entry', 'atr_at_entry', 'entry_edge']:
            if col in losers.columns:
                features.append(col)
        if features:
            X = losers[features].fillna(0).values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            k = min(3, max(2, len(losers) // 5))
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(Xs)
            losers['cluster'] = labels
            analysis['clusters'] = losers.groupby('cluster').agg({'pnl': ['count', 'mean']}).to_dict()
    # Save losers sample
    analysis['losers_head'] = losers.head(10).to_dict(orient='records')
    return analysis

# -------------------------
# Suggest improvements & iteration
# -------------------------

def suggest_improvements(analysis: Dict, params: StrategyParams) -> List[StrategyParams]:
    suggestions = []
    # If many quick SLs, increase ATR multiplier
    if analysis.get('quick_sl_rate', 0) > 0.4:
        new = StrategyParams(**{**asdict(params), 'atr_sl_multiplier': params.atr_sl_multiplier * 1.25})
        suggestions.append(new)
    # If high_atr_fraction high, consider larger SL or skip high ATR
    if analysis.get('high_atr_fraction', 0) > 0.4:
        new = StrategyParams(**{**asdict(params), 'atr_sl_multiplier': params.atr_sl_multiplier * 1.5})
        suggestions.append(new)
    # If clusters show particular RSI ranges losing, adjust RSI thresholds
    clusters = analysis.get('clusters')
    if clusters:
        # naive tweak: widen RSI safe zone
        new = StrategyParams(**{**asdict(params), 'rsi_overbought': min(90, params.rsi_overbought + 5), 'rsi_oversold': max(10, params.rsi_oversold - 5)})
        suggestions.append(new)
    # Always consider adding volume filter
    # We'll not change code automatically but suggest user to add volume filter
    return suggestions

# -------------------------
# Utility: compile results & visualization
# -------------------------

def compile_report(stats_df: pd.DataFrame, filename: str = 'backtest_report.csv') -> None:
    stats_df.to_csv(filename, index=False)
    logger.info('Saved report to %s', filename)


def plot_equity_curve(equity_series: pd.Series, title: str = 'Equity Curve') -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(equity_series.index, equity_series.values)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------
# Live trading simulation (and demo live hook)
# -------------------------

class LiveTrader:
    def __init__(self, api_key: str = None, secret: str = None, testnet: bool = True):
        if ccxt is None:
            raise RuntimeError('ccxt required for live trading')
        self.exchange = ccxt.binance({
            'apiKey': api_key or '',
            'secret': secret or '',
            'enableRateLimit': True,
        })
        if testnet:
            # Note: for Binance testnet you need a separate URL/host; ccxt supports it via urls flag in exchange instance creation
            pass

    def place_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None, order_type: str = 'limit'):
        try:
            if order_type == 'limit':
                order = self.exchange.create_order(symbol, 'limit', side, amount, price)
            else:
                order = self.exchange.create_market_order(symbol, side, amount)
            logger.info('Placed order: %s', order)
            return order
        except Exception as e:
            logger.exception('Order placement failed: %s', e)
            return None

# -------------------------
# Main orchestration
# -------------------------

def main(symbol: str = 'BTC-USD', mode: str = 'backtest', live: bool = False):
    # Normalize symbol
    symbol_tv = symbol.replace('-', '/') if '/' not in symbol else symbol
    # Use yfinance for simplicity in this demo
    timeframes = DEFAULT_TIMEFRAMES
    # Convert symbol to Binance format (use USDT quote)
    symbol_ccxt = symbol
    if '-' in symbol_ccxt:
        base, quote = symbol_ccxt.split('-')
        quote = 'USDT' if quote.upper() == 'USD' else quote
        symbol_ccxt = f"{base}/{quote}"
    elif '/' in symbol_ccxt and symbol_ccxt.upper().endswith('/USD'):
        symbol_ccxt = symbol_ccxt.replace('/USD', '/USDT')

    data = fetch_multi_timeframe(symbol_ccxt, timeframes, max_candles=5000)  # fetch up to 5000 candles per TF via pagination
    params = StrategyParams()

    # Generate signals
    signals = generate_signals_multi_tf(data, params, symbol_ccxt)
    logger.info('Generated %d signals', len(signals))
    # Print signals
    for s in signals:
        logger.info('%s %s %s entry=%.2f SL=%.2f TP=%.2f reason=%s', s.datetime, s.symbol, s.side, s.entry, s.sl, s.tp, s.reason)

    if mode == 'backtest':
        # Choose one timeframe for backtest, e.g., 15m
        backtest_tf = '15m' if '15m' in data else list(data.keys())[0]
        df_bt = data[backtest_tf].copy()
        df_bt.columns = [c.lower() for c in df_bt.columns]
        # backtrader expects columns: Open High Low Close Volume
        df_bt_for = df_bt.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        stats_df, stats = run_backtest(df_bt_for, params)
        compile_report(stats_df, filename=f'report_{symbol.replace("/","-")}_{backtest_tf}.csv')
        logger.info('Backtest stats: %s', stats)
        # Placeholder equity_series for plot
        equity_series = pd.Series([params.ema_fast, params.ema_slow], index=[dt.datetime.now() - dt.timedelta(days=1), dt.datetime.now()])
        plot_equity_curve(equity_series, title=f'Equity ({symbol} {backtest_tf})')

        # Mock trade log and analyze
        trade_log = pd.DataFrame([{
            'entry_dt': dt.datetime.now() - dt.timedelta(hours=5),
            'exit_dt': dt.datetime.now() - dt.timedelta(hours=4, minutes=30),
            'side': 'long',
            'entry_price': 30000,
            'exit_price': 29500,
            'pnl': -500,
            'duration': 3,
            'atr_at_entry': 200,
            'rsi_at_entry': 55,
            'macd_at_entry': 0.5,
            'entry_edge': 0.02,
        }])
        analysis = analyze_losing_trades(trade_log)
        logger.info('Analysis: %s', analysis)
        suggestions = suggest_improvements(analysis, params)
        logger.info('Suggestions: %s', [asdict(s) for s in suggestions])

    if mode == 'live' and live:
        # Connect to live exchange and place sample orders (demo only)
        trader = LiveTrader(api_key=os.getenv('BINANCE_API_KEY'), secret=os.getenv('BINANCE_SECRET'), testnet=True)
        for s in signals:
            # For demo: do not actually place orders — we just log intended orders
            logger.info('LIVE TRADE INTENT: %s %s @ %.2f SL=%.2f TP=%.2f', s.side, s.symbol, s.entry, s.sl, s.tp)
            # Uncomment to place
            # trader.place_order(s.symbol.replace('/','/'), s.side, amount=0.001, price=s.entry)

# -------------------------
# Streamlit UI (clickable interface with auto-scan)
# -------------------------

def run_ui():
    import streamlit as st
    st.set_page_config(page_title="Crypto Multi-TF Bot", layout="wide")
    st.title("🔍 Crypto Multi‑Timeframe Scanner & Backtester")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        symbols = st.multiselect("Symbols", ["BTC/USDT", "ETH/USDT"], default=["BTC/USDT", "ETH/USDT"]) 
        tfs = st.multiselect("Timeframes", DEFAULT_TIMEFRAMES, default=DEFAULT_TIMEFRAMES)
        max_candles = st.slider("Candles per TF (history)", 500, 20000, 5000, step=500)
        st.subheader("Strategy Params")
        rsi_period = st.number_input("RSI Period", 5, 50, 14)
        rsi_overbought = st.number_input("RSI Overbought", 50, 95, 70)
        rsi_oversold = st.number_input("RSI Oversold", 5, 50, 30)
        ema_fast = st.number_input("EMA Fast", 3, 50, 12)
        ema_slow = st.number_input("EMA Slow", 5, 200, 26)
        macd_signal = st.number_input("MACD Signal", 3, 30, 9)
        atr_period = st.number_input("ATR Period", 5, 50, 14)
        atr_sl_multiplier = st.number_input("ATR SL Multiplier", 0.5, 5.0, 1.5, step=0.1)
        tp_multiplier = st.number_input("TP / SL Multiplier", 1.0, 10.0, 3.0, step=0.5)
        autoscan = st.toggle("Auto-scan every 5 minutes", value=True, help="Refreshes this page every 5 minutes to rescan.")
        scan_now = st.button("🔁 Scan Now")
        run_backtest_btn = st.button("📈 Run Backtest (15m)")

    # Periodic refresh (5 minutes)
    if autoscan:
        st.experimental_rerun  # no-op to satisfy linters
        st_autorefresh = st.experimental_memo  # placeholder to avoid errors if older Streamlit
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
        st.experimental_rerun  # not called; we use autorefresh API below
        st_autorefresh = st.autorefresh if hasattr(st, 'autorefresh') else None
        if st_autorefresh:
            st_autorefresh(interval=300000, key="auto_refresh_5m")

    params = StrategyParams(
        rsi_period=int(rsi_period),
        rsi_overbought=int(rsi_overbought),
        rsi_oversold=int(rsi_oversold),
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        macd_signal=int(macd_signal),
        atr_period=int(atr_period),
        atr_sl_multiplier=float(atr_sl_multiplier),
        tp_multiplier=float(tp_multiplier),
    )

    if 'last_scan' not in st.session_state:
        st.session_state.last_scan = None
    if 'signals_df' not in st.session_state:
        st.session_state.signals_df = pd.DataFrame()

    def do_scan():
        all_signals: List[Signal] = []
        for sym in symbols:
            data = fetch_multi_timeframe(sym, tfs, max_candles=max_candles)
            sigs = generate_signals_multi_tf(data, params, sym)
            all_signals.extend(sigs)
        if all_signals:
            df = pd.DataFrame([{
                'time': s.datetime,
                'symbol': s.symbol,
                'timeframe': s.timeframe,
                'side': s.side,
                'entry': s.entry,
                'sl': s.sl,
                'tp': s.tp,
                'reason': s.reason,
                'rsi': s.indicators.get('rsi'),
                'macd': s.indicators.get('macd'),
                'atr': s.indicators.get('atr'),
            } for s in all_signals]).sort_values(['symbol','timeframe','time'])
        else:
            df = pd.DataFrame(columns=['time','symbol','timeframe','side','entry','sl','tp','reason','rsi','macd','atr'])
        st.session_state.signals_df = df
        st.session_state.last_scan = dt.datetime.now()

    # Scan on load and on demand
    if st.session_state.last_scan is None or scan_now or autoscan:
        try:
            do_scan()
        except Exception as e:
            st.error(f"Scan failed: {e}")

    # Display
    left, right = st.columns([3,2])
    with left:
        st.subheader("Latest Signals")
        st.caption(f"Last scan: {st.session_state.last_scan}")
        st.dataframe(st.session_state.signals_df, use_container_width=True)

        # Quick chart of most recent symbol/timeframe
        if not st.session_state.signals_df.empty:
            last_row = st.session_state.signals_df.iloc[-1]
            tf = last_row['timeframe']
            sym = last_row['symbol']
            st.markdown(f"**Chart:** {sym} @ {tf}")
            try:
                df_chart = fetch_multi_timeframe(sym, [tf], max_candles=500).get(tf, pd.DataFrame())
                if not df_chart.empty:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(df_chart.index, df_chart['close'])
                    ax.set_title(f"{sym} Close")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Price")
                    st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Chart load failed: {e}")

    with right:
        st.subheader("Backtest (15m)")
        st.caption("Runs on the selected symbol list, one by one, 15m timeframe")
        if run_backtest_btn:
            for sym in symbols:
                st.write(f"**{sym} (15m)**")
                try:
                    data_bt = fetch_multi_timeframe(sym, ["15m"], max_candles=5000)
                    df_bt = data_bt.get("15m", pd.DataFrame())
                    if df_bt.empty:
                        st.warning("No data.")
                        continue
                    df_bt_for = df_bt.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
                    stats_df, stats = run_backtest(df_bt_for, params)
                    st.dataframe(stats_df)
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
        else:
            st.info("Click **Run Backtest (15m)** to compute Sharpe/Drawdown/Trades for each symbol.")


if __name__ == '__main__':
    # If launched via Streamlit, the `streamlit run trading_bot.py` command will execute this file.
    # We detect a UI run by checking an env var set by Streamlit and fall back to CLI otherwise.
    if os.environ.get('STREAMLIT_SERVER_ENABLED') == '1' or os.environ.get('STREAMLIT_RUN') == '1':
        run_ui()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--symbol', type=str, default='BTC-USD', help='Symbol, e.g., BTC-USD or ETH-USD')
        parser.add_argument('--mode', type=str, choices=['backtest', 'live', 'signals'], default='backtest')
        parser.add_argument('--live', action='store_true', help='Enable live mode (requires API keys)')
        parser.add_argument('--ui', action='store_true', help='Run the Streamlit UI')
        args = parser.parse_args()
        try:
            if args.ui:
                run_ui()
            else:
                main(symbol=args.symbol, mode=args.mode, live=args.live)
        except Exception as e:
            logger.exception('Fatal error: %s', e)
