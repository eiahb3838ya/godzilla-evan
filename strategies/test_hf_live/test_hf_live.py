"""
test_hf_live - 端到端測試策略（漸進式驗證）
Phase 6: 全市場數據 + 線性模型測試
測試 Binance → hf-live (15因子) → 線性模型 → on_factor

數據流:
  Binance WebSocket
    ├─ Depth (type=101) → FactorEngine → 5 Depth factors
    ├─ Trade (type=103) → FactorEngine → 5 Trade factors
    ├─ Ticker (type=102) → FactorEngine → 3 Ticker factors
    └─ IndexPrice (type=104) → FactorEngine → 2 IndexPrice factors
                                    ↓
                            15 market factors
                                    ↓
                            LinearModel
                                    ↓
                    [pred_signal, pred_confidence]
                                    ↓
                            on_factor (Python)
"""
from kungfu.wingchun.constants import *
from pywingchun.constants import InstrumentType, OrderType, Side, OrderStatus
import math
from decimal import Decimal, ROUND_DOWN

def pre_start(context):
    """策略初始化"""
    context.log().info("🏁 [Phase 6] Pre-Start - Testing Full Market Data + Linear Model")

    # 訂閱市場數據 - Depth, Trade, Ticker, IndexPrice
    config = context.get_config()

    # 註冊交易帳號（必須在下單前完成）
    context.add_account(config["td_source"], config["account"])

    symbol = config["symbol"]
    md_source = config["md_source"]

    # ✅ Phase 6 Fix: 使用獨立的訂閱方法訂閱多種數據類型
    # 注意：IndexPrice 不支持（marketdata_binance.cpp:340 故意返回 false）

    # 訂閱 1: Depth (默認，必須，10檔買賣盤)
    context.subscribe(md_source, [symbol], InstrumentType.FFuture, Exchange.BINANCE)

    # 訂閱 2: Trade (公開成交數據)
    context.subscribe_trade(md_source, [symbol], InstrumentType.FFuture, Exchange.BINANCE)

    # 訂閱 3: Ticker (24小時統計數據)
    context.subscribe_ticker(md_source, [symbol], InstrumentType.FFuture, Exchange.BINANCE)

    context.log().info(f"📡 Subscribed: {symbol} (Futures) - Market Data")
    context.log().info(f"   ├─ Depth: Order book snapshots → 5 factors ✅")
    context.log().info(f"   ├─ Trade: Market trades → 5 factors ✅")
    context.log().info(f"   └─ Ticker: 24h statistics → 3 factors ✅")
    context.log().info(f"   ⚠️  IndexPrice: Not supported by MD Gateway")

    context.log().info("✅ [Init] hf-live full market data test initialized")

def on_depth(context, depth):
    """緩存最新價格供 on_factor 使用，不做任何交易邏輯"""
    # 防御性检查
    if not depth.bid_price or not depth.ask_price:
        return
    if len(depth.bid_price) == 0 or len(depth.ask_price) == 0:
        return

    # 緩存最新價格（供 on_factor 下單使用）
    context.set_object("last_bid", depth.bid_price[0])
    context.set_object("last_ask", depth.ask_price[0])

def on_order(context, order):
    """訂單狀態回調 - 驗證發射成功"""
    config = context.get_config()
    context.log().info(f"📬 [on_order] order_id={order.order_id} status={order.status} ex_order_id='{order.ex_order_id}'")
    
    # 檢查訂單是否失敗
    if order.status == OrderStatus.Error:
        context.log().error(f"❌ [Order Error] Order {order.order_id} failed at exchange!")
        # 重置狀態，允許重試
        context.set_object("order_placed", False)
        return
    
    # 檢查訂單是否已確認（簡化邏輯，不依賴 stored_order_id）
    if order.status == OrderStatus.Submitted:
        # ✅ 新增：檢查 ex_order_id 有效性
        if not order.ex_order_id or order.ex_order_id in ["", "0"]:
            context.log().error(f"❌ [Invalid ex_order_id] Got '{order.ex_order_id}' for order {order.order_id}")
            # 這可能表示 API key 問題或交易所拒絕
            return
        
        # ex_order_id 有效，檢查是否已經顯示過（使用 ex_order_id 作為標識，防止重複處理）
        confirmed_ex_order_id = context.get_object("confirmed_ex_order_id")
        if confirmed_ex_order_id == order.ex_order_id:
            return  # 已經處理過此訂單，避免重複顯示
        
        # 首次確認此訂單
        context.set_object("confirmed_ex_order_id", order.ex_order_id)
        context.set_object("order_id", order.order_id)
        context.set_object("submit_time", context.now())  # 記錄提交時間
        
        # 顯示清晰的成功信息
        context.log().info(f"")
        context.log().info(f"=" * 80)
        context.log().info(f"🎉🎉🎉 訂單已成功提交到 Binance Futures Testnet! 🎉🎉🎉")
        context.log().info(f"")
        context.log().info(f"   📋 本地 Order ID: {order.order_id}")
        context.log().info(f"   🌐 Binance Order ID: {order.ex_order_id}")
        context.log().info(f"   💱 交易對: BTCUSDT (Futures)")
        context.log().info(f"   📊 方向: BUY (做多)")
        context.log().info(f"   📦 數量: 0.002 BTC")
        context.log().info(f"")
        context.log().info(f"   ⏰ 訂單將保持 30 秒，請立即前往 Binance 網站確認！")
        context.log().info(f"   🌐 https://testnet.binancefuture.com")
        context.log().info(f"   👉 在 Open Orders 中查找 Order ID: {order.ex_order_id}")
        context.log().info(f"")
        context.log().info(f"=" * 80)
        context.log().info(f"")
    
    elif order.status == OrderStatus.Cancelled:
        context.log().info(f"🎉 [Test Complete] Order cancelled successfully!")

def post_stop(context):
    """策略停止"""
    context.log().info("🏁 [Phase 6] Stopped")

# ========================================
# Phase 6: on_factor 回調 - 接收線性模型輸出並進行交易決策
# ========================================
def on_factor(context, symbol, timestamp, values):
    """
    根據因子信號進行交易決策

    數據流: Binance → hf-live → 15 factors → LinearModel → on_factor → 下單

    交易邏輯:
    - BULLISH (signal > 0.1): 買入 0.002 BTC @ 98% ask
    - 30 秒後自動取消未成交訂單
    """
    config = context.get_config()

    # ✅ Phase 4G 修復: 立即複製數據到 Python list, 避免懸空指針
    values = list(values)

    # 解析元數據 (如果有)
    actual_values = values
    if len(values) > 8 and values[0] == -999.0:
        actual_values = values[8:]

    if len(actual_values) < 2:
        context.log().warning(f"⚠️ Unexpected values count: {len(actual_values)}")
        return

    pred_signal = actual_values[0]
    pred_confidence = actual_values[1]

    # 信號解讀
    if pred_signal > 0.1:
        signal_text = "BULLISH"
    elif pred_signal < -0.1:
        signal_text = "BEARISH"
    else:
        signal_text = "NEUTRAL"

    context.log().info(f"🤖 [LinearModel] {symbol} Signal={pred_signal:+.4f} ({signal_text}) Conf={pred_confidence:.2%}")

    # ========== 訂單取消邏輯 (30秒後) ==========
    submit_time = context.get_object("submit_time")
    confirmed_ex_order_id = context.get_object("confirmed_ex_order_id")
    cancelled = context.get_object("cancelled")

    if submit_time and confirmed_ex_order_id and not cancelled:
        elapsed = (context.now() - submit_time) / 1_000_000_000
        if elapsed >= 30:
            order_id = context.get_object("order_id")
            context.log().info(f"⏰ 30 秒已到，取消訂單 order_id={order_id}")
            try:
                context.cancel_order(
                    config["account"], order_id, config["symbol"],
                    confirmed_ex_order_id, InstrumentType.FFuture
                )
                context.set_object("cancelled", True)
            except Exception as e:
                context.log().error(f"❌ [Cancel Failed] {e}")

    # ========== 下單邏輯 (基於 signal) ==========
    order_placed = context.get_object("order_placed") or False

    if not order_placed and pred_signal > 0.1:  # BULLISH 信號時下單
        last_ask = context.get_object("last_ask")
        if not last_ask:
            context.log().warning("⚠️ 無價格數據，跳過下單")
            return

        # 計算價格 (98% of ask, 精確到 0.1)
        test_price = int(last_ask * 0.98 * 10) / 10.0
        test_volume = 0.002

        context.log().info(f"💸 [Placing Order] Buy {test_volume} BTC @ {test_price:.1f}")

        try:
            order_id = context.insert_order(
                config["symbol"], InstrumentType.FFuture, Exchange.BINANCE,
                config["account"], test_price, test_volume,
                OrderType.Limit, Side.Buy
            )
            context.log().info(f"✅ [Order Placed] order_id={order_id}")
            context.set_object("order_placed", True)
            context.set_object("order_id", order_id)
            context.set_object("submit_time", context.now())
        except Exception as e:
            context.log().error(f"❌ [Order Failed] {e}")
            context.set_object("order_placed", True)  # 避免無限重試

def on_trade(context, trade):
    """Trade 事件由 hf-live 處理，策略層不需要處理"""
    pass

def on_ticker(context, ticker):
    """Ticker 事件由 hf-live 處理，策略層不需要處理"""
    pass


