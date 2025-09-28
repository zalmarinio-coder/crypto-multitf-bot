"""
Trading Bot: Multi-timeframe signal generator, backtester, analyzer, and Streamlit UI
Targets: BTC/USDT and ETH/USDT (with multi-exchange fallback)

Features
- Fetch OHLCV using ccxt (paginated), with exchange fallback (okx → kucoin → kraken → coinbase → bybit → binance)
- Multi-timeframe indicators (RSI, EMA, MACD, ATR) and signal generation
- Backtesting via Backtrader with Sharpe/Drawdown/Trade analysis
- Result compilation to CSV and equity curve plot
- Losing-trade analysis + simple improvement suggestions
- Streamlit UI: controls, auto-scan every 5 minutes, signals table, quick chart, one-click backtest
- NEW: Bias timeframe selector, Aggressive mode, MACD confirmation toggle, Exchange order input, Status bar

Setup
-----
Python 3.9–3.11 recommended

pip install -U streamlit pandas numpy matplotlib ccxt backtrader ta scikit-learn scipy

Run UI:
    streamlit run trading_bot.py

Run CLI:
    python trading_bot.py --symbol BTC-USD --mode backtest
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Third-party libs
try:
    import ccxt
except ImportError:
    ccxt = None

import backtrader as bt
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

# Try importing streamlit for caching/guards; keep code import-safe when run as CLI
try:
    import streamlit as st  # used in run_ui() and for cache decorators
except Exception:
    st = None

# -------------------------
# Config & Logging
# -------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h"]

# Global fetch info for status bar
LAST_FETCH_INFO = {"exchange": None, "symbol": None, "timeframes": None, "timestamp": None}


def timeframe_to_minutes(tf: str) -> int:
    if tf.endswith('m'):
        return int(tf[:-1])
    if tf.endswith('h'):
        return int(tf[:-1]) * 60
    if tf.endswith('d'):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unknown timeframe format: {tf}")


# -------------------------
# Exchange fetching (ccxt) with pagination
# -------------------------

def fetch_ohlcv_ccxt_paginated(exchange_id: str, symbol: str, timeframe: str,
                               max_candles: int = 10000,
                               since: Optional[int] = None,
                               params=None) -> pd.DataFrame:
    """Fetch more than 1000 candles by paginating ccxt.fetch_ohlcv."""
    if ccxt is None:
        raise RuntimeError('ccxt not installed')
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    # Load markets once if needed (some exchanges require it before fetch_ohlcv)
    try:
        exchange.load_markets()
    except Exception:
        pass

    limit_per_call = 1000
    all_rows: List[List[float]] = []

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
        if all_rows and ohlcv[0][0] == all_rows[-1][0]:
            ohlcv = ohlcv[1:]
        all_rows.extend(ohlcv)
        fetched += len(ohlcv)
        last_ts = ohlcv[-1][0] + 1
        if len(ohlcv) < limit_per_call:
            break

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # Convert to DatetimeIndex and ensure tz-naive (Backtrader prefers naive)
    dt_series = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = dt_series
    df.set_index('datetime', inplace=True)
    try:
        df.index = df.index.tz_convert(None)
    except Exception:
        pass  # already naive
    return df


# Streamlit cache wrapper (defined after the core fetch function)
if st is not None:
    @st.cache_data(ttl=120, show_spinner=False)
    def _cached_fetch_ohlcv(exchange_id, symbol, timeframe, max_candles, since, params):
        return fetch_ohlcv_ccxt_paginated(exchange_id, symbol, timeframe, max_candles, since, params)


def _candidate_symbols(base_symbol: str) -> List[str]:
    """Produce common quote variations for different exchanges (generic)."""
    base = base_symbol.replace('-', '/').upper()
    if '/' in base:
        b, _ = base.split('/')
    else:
        b = base
    return [f"{b}/USDT", f"{b}/USD", f"{b}/USDC"]


def _candidate_symbols_for_exchange(exid: str, base_symbol: str) -> List[str]:
    """Per-exchange symbol variants (e.g., Kraken uses XBT/USD)."""
    base = base_symbol.replace('-', '/').upper()
    if '/' in base:
        b, _ = base.split('/')
    else:
        b = base
    if exid == 'kraken':
        # Kraken uses XBT for BTC, ETH is ETH
        b_alias = 'XBT' if b == 'BTC' else b
        return [f"{b_alias}/USD", f"{b_alias}/USDT"]
    if exid == 'coinbase':
        return [f"{b}/USD", f"{b}/USDT"]
    if exid == 'kucoin':
        return [f"{b}/USDT", f"{b}/USDC"]
    # default
    return [f"{b}/USDT", f"{b}/USD", f"{b}/USDC"]


def _fetch_first_available(symbols: List[str], timeframes: List[str], max_candles: int,
                           since: Optional[int], exchanges: List[str]) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """Try multiple exchanges/symbol formats until data is obtained. Returns (exchange_id, data_dict)."""
    if ccxt is None:
        raise RuntimeError('ccxt not installed')
    for exid in exchanges:
        try:
            exchange_class = getattr(ccxt, exid)
            _ = exchange_class({'enableRateLimit': True})
        except Exception:
            continue
        # Use per-exchange symbol candidates to improve hit-rate
        ex_symbols = _candidate_symbols_for_exchange(exid, symbols[0]) if symbols else []
        for sym in (ex_symbols or symbols):
            ok = True
            tf_data: Dict[str, pd.DataFrame] = {}
            for tf in timeframes:
                try:
                    if st is not None:
                        df = _cached_fetch_ohlcv(exid, sym, tf, max_candles, since, None)
                    else:
                        df = fetch_ohlcv_ccxt_paginated(exid, sym, tf, max_candles=max_candles, since=since)
                    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
                        ok = False
                        logger.warning('No data on %s for %s @ %s', exid, sym, tf)
                        break
                    tf_data[tf] = df
                except Exception as e:
                    logger.warning('Fetch failed on %s %s %s: %s', exid, sym, tf, e)
                    ok = False
                    break
            if ok and tf_data:
                logger.info('Using %s %s for all timeframes', exid, sym)
                LAST_FETCH_INFO.update({
                    'exchange': exid,
                    'symbol': sym,
                    'timeframes': timeframes,
                    'timestamp': dt.datetime.utcnow().isoformat()
                })
                return exid, tf_data
    return '', {}


def fetch_multi_timeframe(symbol: str, timeframes: List[str],
                          max_candles: int = 10000,
                          since: Optional[int] = None,
                          exchanges: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple timeframes with multi-exchange fallback."""
    if exchanges is None:
        exchanges = ['okx', 'kucoin', 'kraken', 'coinbase', 'bybit', 'binance']
    symbols = _candidate_symbols(symbol)
    _, data = _fetch_first_available(symbols, timeframes, max_candles, since, exchanges)
    if not data:
        logger.error('Failed to fetch data from all exchanges for %s', symbol)
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
    atr_sl_multiplier: float = 1.5
    tp_multiplier: float = 3.0
    require_macd: bool = True  # NEW: toggle MACD confirmation


def add_indicators(df: pd.DataFrame, params: 'StrategyParams') -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['rsi'] = RSIIndicator(close=df['close'], window=params.rsi_period).rsi()
    df['ema_fast'] = EMAIndicator(close=df['close'], window=params.ema_fast).ema_indicator()
    df['ema_slow'] = EMAIndicator(close=df['close'], window=params.ema_slow).ema_indicator()
    macd = MACD(close=df['close'], window_slow=params.ema_slow, window_fast=params.ema_fast, window_sign=params.macd_signal)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=params.atr_period).average_true_range()
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


def generate_signals_multi_tf(
    data: Dict[str, pd.DataFrame],
    params: 'StrategyParams',
    symbol: str,
    prefer_bias: str = 'auto'  # 'auto', '4h', or '1h'
) -> List['Signal']:
    """
    - Trend bias from 4h (fallback 1h, else longest TF) or user-selected
    - Triggers on 1m/5m/15m/30m: EMA crossover + optional MACD + RSI band
    - ATR-based SL, TP = multiple of SL distance
    Robust to missing/empty data.
    """
    signals: List['Signal'] = []
    if not data:
        logger.warning('No data available to generate signals for %s', symbol)
        return signals

    # Bias TF selection
    if prefer_bias in ('1h', '4h') and prefer_bias in data:
        bias_tf = prefer_bias
    else:
        bias_tf = '4h' if '4h' in data else ('1h' if '1h' in data else (sorted(data.keys(), key=lambda x: timeframe_to_minutes(x), reverse=True)[0]))
    bias_df = data.get(bias_tf)
    if bias_df is None or len(bias_df) < 50:
        logger.warning('Insufficient bias data for %s', symbol)
        return signals
    bias_df = add_indicators(bias_df, params)
    latest_bias = bias_df.iloc[-1]
    bias = 'bull' if latest_bias['ema_fast'] > latest_bias['ema_slow'] else 'bear'
    logger.info('Trend bias based on %s = %s', bias_tf, bias)

    # Triggers
    trigger_tfs = [tf for tf in ['1m', '5m', '15m', '30m'] if tf in data]
    for tf in trigger_tfs:
        df = data.get(tf)
        if df is None or len(df) < 50:
            continue
        df_ind = add_indicators(df, params)
        if len(df_ind) < 2:
            continue
        row = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        long_cond = (
            prev['ema_fast'] < prev['ema_slow'] and row['ema_fast'] > row['ema_slow']
            and params.rsi_oversold < row['rsi'] < params.rsi_overbought
            and (row['macd'] > row['macd_signal'] if params.require_macd else True)
        )
        short_cond = (
            prev['ema_fast'] > prev['ema_slow'] and row['ema_fast'] < row['ema_slow']
            and params.rsi_oversold < row['rsi'] < params.rsi_overbought
            and (row['macd'] < row['macd_signal'] if params.require_macd else True)
        )
        if bias == 'bull' and long_cond:
            entry = float(row['close'])
            atr = float(row['atr']) if not math.isnan(row['atr']) else entry * 0.01
            sl = entry - params.atr_sl_multiplier * atr
            tp = entry + params.tp_multiplier * (entry - sl)
            signals.append(Signal(row.name, symbol, tf, 'long', entry, sl, tp,
                                  f'EMA cross + {"MACD + " if params.require_macd else ""}RSI in {tf} (bias {bias})',
                                  {'rsi': row['rsi'], 'macd': row['macd'], 'atr': atr}))
        if bias == 'bear' and short_cond:
            entry = float(row['close'])
            atr = float(row['atr']) if not math.isnan(row['atr']) else entry * 0.01
            sl = entry + params.atr_sl_multiplier * atr
            tp = entry - params.tp_multiplier * (sl - entry)
            signals.append(Signal(row.name, symbol, tf, 'short', entry, sl, tp,
                                  f'EMA cross + {"MACD + " if params.require_macd else ""}RSI in {tf} (bias {bias})',
                                  {'rsi': row['rsi'], 'macd': row['macd'], 'atr': atr}))
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

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.ema_fast[0] > self.ema_slow[0] and self.ema_fast[-1] <= self.ema_slow[-1] and self.macd.macd[0] > self.macd.signal[0] and self.rsi[0] > self.p.rsi_oversold:
                size = max(0.0, self.broker.getcash() * 0.01 / max(1e-8, float(self.dataclose[0])))
                self.entry_price = float(self.dataclose[0])
                self.sl_price = self.entry_price - self.p.atr_sl_mult * float(self.atr[0])
                self.tp_price = self.entry_price + self.p.tp_mult * (self.entry_price - self.sl_price)
                self.order = self.buy(size=size)
            elif self.ema_fast[0] < self.ema_slow[0] and self.ema_fast[-1] >= self.ema_slow[-1] and self.macd.macd[0] < self.macd.signal[0] and self.rsi[0] < self.p.rsi_overbought:
                size = max(0.0, self.broker.getcash() * 0.01 / max(1e-8, float(self.dataclose[0])))
                self.entry_price = float(self.dataclose[0])
                self.sl_price = self.entry_price + self.p.atr_sl_mult * float(self.atr[0])
                self.tp_price = self.entry_price - self.p.tp_mult * (self.sl_price - self.entry_price)
                self.order = self.sell(size=size)
        else:
            if self.position.size > 0:
                if self.dataclose[0] <= self.sl_price or self.dataclose[0] >= self.tp_price:
                    self.order = self.close()
            else:
                if self.dataclose[0] >= self.sl_price or self.dataclose[0] <= self.tp_price:
                    self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None


def run_backtest(df: pd.DataFrame, params: 'StrategyParams', cash: float = 10000.0, commission: float = 0.001) -> Tuple[pd.DataFrame, dict]:
    # Ensure tz-naive index for Backtrader
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)

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
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

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
    return pd.DataFrame([stats]), stats


# -------------------------
# Post-backtest analysis
# -------------------------

def analyze_losing_trades(trade_log: pd.DataFrame) -> Dict:
    if trade_log.empty:
        return {'message': 'No trades to analyze'}
    losers = trade_log[trade_log['pnl'] < 0].copy()
    analysis: Dict = {'num_losers': len(losers)}
    if len(losers) == 0:
        return analysis
    if 'duration' in losers.columns:
        quick_sl = losers[losers['duration'] <= 3]
        analysis['quick_sl_rate'] = len(quick_sl) / len(losers) if len(losers) else 0.0
    if 'atr_at_entry' in losers.columns:
        high_atr = losers['atr_at_entry'].quantile(0.75)
        analysis['high_atr_threshold'] = float(high_atr)
        analysis['high_atr_fraction'] = float((losers['atr_at_entry'] > high_atr).mean())
    analysis['losers_head'] = losers.head(10).to_dict(orient='records')
    return analysis


def suggest_improvements(analysis: Dict, params: 'StrategyParams') -> List['StrategyParams']:
    suggestions: List['StrategyParams'] = []
    if analysis.get('quick_sl_rate', 0) > 0.4:
        suggestions.append(StrategyParams(**{**asdict(params), 'atr_sl_multiplier': params.atr_sl_multiplier * 1.25}))
    if analysis.get('high_atr_fraction', 0) > 0.4:
        suggestions.append(StrategyParams(**{**asdict(params), 'atr_sl_multiplier': params.atr_sl_multiplier * 1.5}))
    if not suggestions:
        suggestions.append(params)
    return suggestions


# -------------------------
# Utilities
# -------------------------

def compile_report(stats_df: pd.DataFrame, filename: str = 'backtest_report.csv') -> None:
    stats_df.to_csv(filename, index=False)
    logger.info('Saved report to %s', filename)


def plot_equity_curve(equity_series: pd.Series, title: str = 'Equity Curve') -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(equity_series.index, equity_series.values)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------
# Streamlit UI
# -------------------------

def run_ui():
    import streamlit as st  # local import to ensure availability only in UI mode

    st.set_page_config(page_title='Crypto Multi‑TF Bot', layout='wide')
    st.title('🔍 Crypto Multi‑Timeframe Scanner & Backtester')

    # Friendly message if ccxt is missing
    if ccxt is None:
        st.error("""ccxt is not installed. Please run:

`pip install -U ccxt`""")
        st.stop()

    # Status bar
    with st.container():
        ex = LAST_FETCH_INFO.get('exchange')
        sym = LAST_FETCH_INFO.get('symbol')
        ts = LAST_FETCH_INFO.get('timestamp')
        tfs = LAST_FETCH_INFO.get('timeframes')
        st.info(f"Exchange: {ex or 'n/a'} | Symbol: {sym or 'n/a'} | Last fetch: {ts or 'n/a'} | TFs: {', '.join(tfs) if tfs else 'n/a'}")

    with st.sidebar:
        st.header('Settings')
        symbols = st.multiselect('Symbols', ['BTC/USDT', 'ETH/USDT'], default=['BTC/USDT', 'ETH/USDT'])
        tfs = st.multiselect('Timeframes', DEFAULT_TIMEFRAMES, default=DEFAULT_TIMEFRAMES)
        max_candles = st.slider('Candles per TF (history)', 500, 20000, 5000, step=500)

        st.subheader('Signal Engine')
        bias_choice = st.selectbox('Bias timeframe', ['auto', '4h', '1h'], index=0, help='Which timeframe sets the trend bias')
        aggressive = st.toggle('Aggressive mode', value=False, help='Wider RSI, faster MACD, optional MACD confirm')
        require_macd = st.toggle('Require MACD confirmation', value=True)

        st.subheader('Strategy Params')
        rsi_period = st.number_input('RSI Period', 5, 50, 14)
        rsi_overbought = st.number_input('RSI Overbought', 50, 95, 70)
        rsi_oversold = st.number_input('RSI Oversold', 5, 50, 30)
        ema_fast = st.number_input('EMA Fast', 3, 50, 12)
        ema_slow = st.number_input('EMA Slow', 5, 200, 26)
        macd_signal = st.number_input('MACD Signal', 3, 30, 9)
        atr_period = st.number_input('ATR Period', 5, 50, 14)
        atr_sl_multiplier = st.number_input('ATR SL Multiplier', 0.5, 5.0, 1.5, step=0.1)
        tp_multiplier = st.number_input('TP / SL Multiplier', 1.0, 10.0, 3.0, step=0.5)

        st.subheader('Data Providers')
        ex_order_default = 'okx,kucoin,kraken,coinbase,bybit,binance'
        ex_order_str = st.text_input('Exchange order (comma-separated)', ex_order_default)

        autoscan = st.toggle('Auto-scan every 5 minutes', value=True, help='Auto refresh page every 5 minutes')
        scan_now = st.button('🔁 Scan Now')
        run_backtest_btn = st.button('📈 Run Backtest (15m)')

    # Only refresh the page (meta) if autoscan is on
    if autoscan:
        st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)

    # Build params
    base_params = StrategyParams(
        rsi_period=int(rsi_period),
        rsi_overbought=int(rsi_overbought),
        rsi_oversold=int(rsi_oversold),
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        macd_signal=int(macd_signal),
        atr_period=int(atr_period),
        atr_sl_multiplier=float(atr_sl_multiplier),
        tp_multiplier=float(tp_multiplier),
        require_macd=bool(require_macd),
    )
    if aggressive:
        base_params.rsi_overbought = max(base_params.rsi_overbought, 75)
        base_params.rsi_oversold = min(base_params.rsi_oversold, 25)
        base_params.macd_signal = min(base_params.macd_signal, 5)
        base_params.require_macd = False
    params = base_params

    if 'last_scan' not in st.session_state:
        st.session_state.last_scan = None
    if 'signals_df' not in st.session_state:
        st.session_state.signals_df = pd.DataFrame()

    def do_scan():
        all_signals = []
        exchanges = [e.strip() for e in ex_order_str.split(',') if e.strip()]
        for sym in symbols:
            data = fetch_multi_timeframe(sym, tfs, max_candles=max_candles, exchanges=exchanges)
            # update status bar state
            LAST_FETCH_INFO['timeframes'] = tfs
            LAST_FETCH_INFO['timestamp'] = dt.datetime.utcnow().isoformat()
            LAST_FETCH_INFO['symbol'] = sym
            sigs = generate_signals_multi_tf(data, params, sym, prefer_bias=bias_choice)
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

    # Scan cadence control (avoid re-fetch on every rerun)
    should_scan = st.session_state.last_scan is None or scan_now
    if st.session_state.last_scan is not None and autoscan:
        elapsed = (dt.datetime.now() - st.session_state.last_scan).total_seconds()
        if elapsed >= 300:  # 5 minutes
            should_scan = True

    if should_scan:
        try:
            do_scan()
        except Exception as e:
            st.error(f'Scan failed: {e}')

    left, right = st.columns([3, 2])
    with left:
        st.subheader('Latest Signals')
        st.caption(f'Last scan: {st.session_state.last_scan}')
        st.dataframe(st.session_state.signals_df, use_container_width=True)
        if not st.session_state.signals_df.empty:
            last_row = st.session_state.signals_df.iloc[-1]
            tf = last_row['timeframe']
            sym = last_row['symbol']
            st.markdown(f'**Chart:** {sym} @ {tf}')
            try:
                df_chart = fetch_multi_timeframe(sym, [tf], max_candles=500).get(tf, pd.DataFrame())
                if not df_chart.empty:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(df_chart.index, df_chart['close'])
                    ax.set_title(f'{sym} Close')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Price')
                    st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.warning(f'Chart load failed: {e}')

    with right:
        st.subheader('Backtest (15m)')
        st.caption('Runs on the selected symbol list, one by one, 15m timeframe')
        if run_backtest_btn:
            for sym in symbols:
                st.write(f'**{sym} (15m)**')
                try:
                    data_bt = fetch_multi_timeframe(sym, ['15m'], max_candles=5000)
                    df_bt = data_bt.get('15m', pd.DataFrame())
                    if df_bt.empty:
                        st.warning('No data.')
                        continue
                    df_bt_for = df_bt.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    stats_df, _ = run_backtest(df_bt_for, params)
                    st.dataframe(stats_df)
                except Exception as e:
                    st.error(f'Backtest failed: {e}')
        else:
            st.info('Click **Run Backtest (15m)** to compute Sharpe/Drawdown/Trades for each symbol.')



# -------------------------
# CLI entry
# -------------------------

def main(symbol: str = 'BTC-USD', mode: str = 'backtest', live: bool = False):
    # Normalize to exchange format
    symbol_ccxt = symbol
    if '-' in symbol_ccxt:
        base, quote = symbol_ccxt.split('-')
        quote = 'USDT' if quote.upper() == 'USD' else quote
        symbol_ccxt = f'{base}/{quote}'
    elif '/' in symbol_ccxt and symbol_ccxt.upper().endswith('/USD'):
        symbol_ccxt = symbol_ccxt.replace('/USD', '/USDT')

    timeframes = DEFAULT_TIMEFRAMES
    data = fetch_multi_timeframe(symbol_ccxt, timeframes, max_candles=5000)

    params = StrategyParams()
    signals = generate_signals_multi_tf(data, params, symbol_ccxt)
    logger.info('Generated %d signals', len(signals))
    for s in signals:
        logger.info('%s %s %s entry=%.2f SL=%.2f TP=%.2f reason=%s', s.datetime, s.symbol, s.side, s.entry, s.sl, s.tp, s.reason)

    if mode == 'backtest':
        backtest_tf = '15m' if '15m' in data else (list(data.keys())[0] if data else '15m')
        df_bt = data.get(backtest_tf, pd.DataFrame()).copy()
        if df_bt.empty:
            logger.error('No data available for backtest')
            return
        df_bt_for = df_bt.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        stats_df, stats = run_backtest(df_bt_for, params)
        compile_report(stats_df, filename=f'report_{symbol_ccxt.replace("/","-")}_{backtest_tf}.csv')
        logger.info('Backtest stats: %s', stats)

        # Minimal equity plot placeholder
        eq = pd.Series([params.ema_fast, params.ema_slow],
                       index=[dt.datetime.now() - dt.timedelta(days=1), dt.datetime.now()])
        plot_equity_curve(eq, title=f'Equity ({symbol_ccxt} {backtest_tf})')

        # Mock losing trade for analysis demo
        trade_log = pd.DataFrame([{
            'entry_dt': dt.datetime.now() - dt.timedelta(hours=5),
            'exit_dt': dt.datetime.now() - dt.timedelta(hours=4, minutes=30),
            'side': 'long',
            'entry_price': 30000, 'exit_price': 29500,
            'pnl': -500, 'duration': 3,
            'atr_at_entry': 200, 'rsi_at_entry': 55, 'macd_at_entry': 0.5, 'entry_edge': 0.02,
        }])
        analysis = analyze_losing_trades(trade_log)
        logger.info('Analysis: %s', analysis)
        suggestions = suggest_improvements(analysis, params)
        logger.info('Suggestions: %s', [asdict(s) for s in suggestions])


if __name__ == '__main__':
    import sys
    # Prefer running the Streamlit UI when invoked via `streamlit run`.
    # Detect streamlit by environment or argv path; otherwise allow CLI.
    is_streamlit = (
        (st is not None) and (
            os.environ.get('STREAMLIT_RUNTIME') or
            os.environ.get('STREAMLIT_SERVER_PORT') or
            'streamlit' in (sys.argv[0].lower() if sys.argv and sys.argv[0] else '')
        )
    )
    if is_streamlit:
        run_ui()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--symbol', type=str, default='BTC-USD', help='Symbol, e.g., BTC-USD or ETH-USD')
        parser.add_argument('--mode', type=str, choices=['backtest', 'live', 'signals'], default='backtest')
        parser.add_argument('--live', action='store_true', help='Enable live mode (requires API keys)')
        parser.add_argument('--ui', action='store_true', help='Run the Streamlit UI')
        parser.add_argument('--no-ui', action='store_true', help='Force CLI even if Streamlit is present')
        args = parser.parse_args()
        try:
            if args.ui and not args.no_ui:
                run_ui()
            else:
                main(symbol=args.symbol, mode=args.mode, live=args.live)
        except Exception as e:
            logger.exception('Fatal error: %s', e)
