'''
市場數據完整測試 Demo
展示 4 種市場數據回調：Depth, Ticker, Trade, IndexPrice

Author: AI Assistant
Date: 2025-11-19
Purpose: 學習路徑階段 3 - 理解市場數據結構
'''

from kungfu.wingchun.constants import *
from pywingchun.constants import Side, InstrumentType, OrderType
import kungfu.yijinjing.time as kft
import pyyjj

# 配置：使用 USDT-Margined Futures
exchange = Exchange.BINANCE
instrument_type = InstrumentType.FFuture

# 全局變量：統計計數器（因為 pre_stop 時 context.get_object() 不可用）
depth_count = 0
ticker_count = 0
trade_count = 0
index_count = 0
start_time = 0

# ========== 策略初始化 ==========
def pre_start(context):
    """
    策略啟動前的初始化
    訂閱 4 種市場數據：Depth, Ticker, Trade, IndexPrice
    """
    config = context.get_config()
    symbol = config["symbol"]  # 例如 "btc_usdt"
    md_source = config["md_source"]  # 例如 "binance_md"

    context.log().info("=" * 80)
    context.log().info(f"🚀 市場數據測試 Demo 啟動")
    context.log().info(f"   交易對: {symbol}")
    context.log().info(f"   交易所: BINANCE")
    context.log().info(f"   市場類型: USDT-Margined Futures")
    context.log().info(f"   數據源: {md_source}")
    context.log().info("=" * 80)

    # 訂閱 1: Depth (訂單簿深度數據，10 檔買賣盤)
    context.log().info("📊 訂閱 Depth (訂單簿深度)")
    context.subscribe(md_source, [symbol], instrument_type, exchange)

    # 訂閱 2: Ticker (24小時統計數據)
    context.log().info("📈 訂閱 Ticker (24小時統計)")
    context.subscribe_ticker(md_source, [symbol], instrument_type, exchange)

    # 訂閱 3: Trade (公開成交數據)
    context.log().info("💰 訂閱 Trade (公開成交流)")
    context.subscribe_trade(md_source, [symbol], instrument_type, exchange)

    # 訂閱 4: IndexPrice (期貨指數價格)
    context.log().info("🔢 訂閱 IndexPrice (指數價格)")
    context.subscribe_index_price(md_source, [symbol], instrument_type, exchange)

    # 初始化計數器（用於統計數據接收頻率）
    global depth_count, ticker_count, trade_count, index_count, start_time
    depth_count = 0
    ticker_count = 0
    trade_count = 0
    index_count = 0
    start_time = pyyjj.now_in_nano()

    context.log().info("\n✅ 訂閱完成，等待市場數據...\n")


# ========== 1. Depth 回調 (訂單簿深度) ==========
def on_depth(context, depth):
    """
    深度數據回調

    數據結構：
    - symbol: 交易對 (如 "btc_usdt")
    - bid_price[0-9]: 買盤價格 (降序，[0] 是最佳買價)
    - bid_volume[0-9]: 買盤數量
    - ask_price[0-9]: 賣盤價格 (升序，[0] 是最佳賣價)
    - ask_volume[0-9]: 賣盤數量
    - data_time: 數據生成時間 (納秒)

    觸發頻率：~100ms (Binance 默認)
    """
    # 更新計數器
    global depth_count
    depth_count += 1

    # 每 10 次打印一次（減少日誌量）
    if depth_count % 10 != 0:
        return

    # 提取關鍵數據
    symbol = depth.symbol
    best_bid = depth.bid_price[0]      # 最佳買價
    best_bid_vol = depth.bid_volume[0]  # 最佳買量
    best_ask = depth.ask_price[0]      # 最佳賣價
    best_ask_vol = depth.ask_volume[0]  # 最佳賣量

    # 計算衍生指標
    spread = best_ask - best_bid  # 價差
    spread_pct = (spread / best_bid) * 100  # 價差百分比
    mid_price = (best_bid + best_ask) / 2  # 中間價

    # 計算訂單簿總量（前 5 檔）
    total_bid_vol = sum(depth.bid_volume[i] for i in range(5))
    total_ask_vol = sum(depth.ask_volume[i] for i in range(5))

    # 格式化輸出
    context.log().info(
        f"\n📊 [{depth_count}] Depth Update - {symbol}\n"
        f"   ├─ 最佳買價: {best_bid:,.2f} USDT (量: {best_bid_vol:.4f})\n"
        f"   ├─ 最佳賣價: {best_ask:,.2f} USDT (量: {best_ask_vol:.4f})\n"
        f"   ├─ 中間價: {mid_price:,.2f} USDT\n"
        f"   ├─ 價差: {spread:.2f} USDT ({spread_pct:.4f}%)\n"
        f"   ├─ 買盤總量 (前5檔): {total_bid_vol:.4f}\n"
        f"   ├─ 賣盤總量 (前5檔): {total_ask_vol:.4f}\n"
        f"   └─ 數據時間: {kft.strftime(depth.data_time, '%H:%M:%S')}"
    )

    # 打印完整訂單簿（可選，僅每 50 次打印一次）
    if depth_count % 50 == 0:
        context.log().info("\n📖 完整訂單簿 (前 5 檔):")
        context.log().info("   檔位 |     買量     |    買價    |    賣價    |     賣量")
        context.log().info("   " + "-" * 65)
        for i in range(5):
            context.log().info(
                f"   [{i+1}]  | {depth.bid_volume[i]:>10.4f}  | "
                f"{depth.bid_price[i]:>10.2f} | {depth.ask_price[i]:>10.2f} | "
                f"{depth.ask_volume[i]:>10.4f}"
            )


# ========== 2. Ticker 回調 (24小時統計) ==========
def on_ticker(context, ticker):
    """
    Ticker 數據回調

    數據結構：
    - symbol: 交易對
    - bid_price: 最佳買價
    - bid_volume: 最佳買量
    - ask_price: 最佳賣價
    - ask_volume: 最佳賣量
    - data_time: 數據時間

    注意：Binance Ticker 可能包含額外的 24h 統計數據 (high, low, volume 等)
          但基礎結構只包含最佳買賣價

    觸發頻率：~1-3秒 (取決於交易所)
    """
    # 更新計數器
    global ticker_count
    ticker_count += 1

    # 提取數據
    symbol = ticker.symbol
    bid = ticker.bid_price
    ask = ticker.ask_price
    bid_vol = ticker.bid_volume
    ask_vol = ticker.ask_volume

    # 格式化輸出
    context.log().info(
        f"\n📈 [{ticker_count}] Ticker Update - {symbol}\n"
        f"   ├─ 買: {bid:,.2f} USDT (量: {bid_vol:.4f})\n"
        f"   ├─ 賣: {ask:,.2f} USDT (量: {ask_vol:.4f})\n"
        f"   └─ 時間: {kft.strftime(ticker.data_time, '%H:%M:%S')}"
    )


# ========== 3. Trade 回調 (公開成交) ==========
def on_trade(context, trade):
    """
    公開成交數據回調 (Market Trade Stream)


    數據結構：
    - symbol: 交易對
    - trade_id: 成交 ID
    - price: 成交價
    - volume: 成交量
    - side: 主動成交方向 (Buy=主動買入，Sell=主動賣出)
    - trade_time: 成交時間
    - bid_id, ask_id: 買賣訂單 ID (如果交易所提供)

    觸發頻率：實時 (每筆成交)
    """
    # 更新計數器
    global trade_count
    trade_count += 1

    # 提取數據
    symbol = trade.symbol
    trade_id = trade.trade_id
    price = trade.price
    volume = trade.volume
    side = trade.side
    trade_time = trade.trade_time

    # 判斷方向
    side_str = "🟢 買入" if side == Side.Buy else "🔴 賣出"

    # 格式化輸出
    context.log().info(
        f"\n💰 [{trade_count}] Trade - {symbol}\n"
        f"   ├─ ID: {trade_id}\n"
        f"   ├─ 方向: {side_str}\n"
        f"   ├─ 價格: {price:,.2f} USDT\n"
        f"   ├─ 數量: {volume:.4f}\n"
        f"   ├─ 金額: {price * volume:,.2f} USDT\n"
        f"   └─ 時間: {kft.strftime(trade_time, '%H:%M:%S.%f')[:-3]}"
    )


# ========== 4. IndexPrice 回調 (指數價格) ==========
def on_index_price(context, index_price):
    """
    指數價格回調 (僅 Futures 有效)

    數據結構：
    - symbol: 交易對
    - price: 指數價格 (多個現貨交易所的加權平均)
    - exchange_id: 交易所 ID
    - instrument_type: 工具類型

    用途：
    - 期貨合約標記價格參考
    - 資金費率計算依據
    - 強平價格參考

    觸發頻率：~1秒 (取決於交易所)
    """
    # 更新計數器
    global index_count
    index_count += 1

    # 提取數據
    symbol = index_price.symbol
    price = index_price.price

    # 格式化輸出
    context.log().info(
        f"\n🔢 [{index_count}] IndexPrice - {symbol}\n"
        f"   ├─ 指數價格: {price:,.2f} USDT\n"
        f"   └─ 說明: 多個現貨交易所的加權平均價格"
    )


# ========== 策略停止 ==========
def pre_stop(context):
    """
    策略停止前的清理與統計
    """
    # 計算運行時長
    global depth_count, ticker_count, trade_count, index_count, start_time
    end_time = pyyjj.now_in_nano()
    duration_sec = (end_time - start_time) / 1e9 if start_time > 0 else 0

    # 計算頻率
    depth_freq = depth_count / duration_sec if duration_sec > 0 else 0
    ticker_freq = ticker_count / duration_sec if duration_sec > 0 else 0
    trade_freq = trade_count / duration_sec if duration_sec > 0 else 0
    index_freq = index_count / duration_sec if duration_sec > 0 else 0

    # 打印統計（pre_stop 階段 context.log() 不可用，使用 print）
    print("\n" + "=" * 80)
    print("📊 市場數據接收統計")
    print("=" * 80)
    print(f"運行時長: {duration_sec:.1f} 秒")
    print("")
    print(f"Depth (訂單簿):     接收 {depth_count:>6} 次 | 平均頻率: {depth_freq:>6.2f} 次/秒")
    print(f"Ticker (統計):      接收 {ticker_count:>6} 次 | 平均頻率: {ticker_freq:>6.2f} 次/秒")
    print(f"Trade (公開成交):   接收 {trade_count:>6} 次 | 平均頻率: {trade_freq:>6.2f} 次/秒")
    print(f"IndexPrice (指數):  接收 {index_count:>6} 次 | 平均頻率: {index_freq:>6.2f} 次/秒")
    print("=" * 80)
    print("✅ 策略已停止")
    print("=" * 80)
