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

    # 訂閱所有公開市場數據
    # 注意：hf-live 會自動接收並處理這些數據，計算 15 個市場因子
    context.subscribe(md_source, [symbol], InstrumentType.FFuture, Exchange.BINANCE)
    context.log().info(f"📡 Subscribed: {symbol} (Futures) - All Market Data")
    context.log().info(f"   ├─ Depth: Order book snapshots → 5 factors")
    context.log().info(f"   ├─ Trade: Market trades → 5 factors")
    context.log().info(f"   ├─ Ticker: 24h statistics → 3 factors")
    context.log().info(f"   └─ IndexPrice: Futures index → 2 factors")

    context.log().info("✅ [Init] hf-live full market data test initialized")

def on_depth(context, depth):
    """接收盤口數據 + 發送測試訂單"""
    config = context.get_config()
    
    # ✅ 防御性检查：验证深度数据有效性
    if not depth.bid_price or len(depth.bid_price) == 0:
        context.log().warning("⚠️  Depth data incomplete: no bid prices")
        return
    
    if not depth.ask_price or len(depth.ask_price) == 0:
        context.log().warning("⚠️  Depth data incomplete: no ask prices")
        return
    
    bid = depth.bid_price[0]
    ask = depth.ask_price[0]
    spread = ask - bid
    
    # 打印盤口
    context.log().info(f"📊 [on_depth] {depth.symbol} bid={bid:.2f} ask={ask:.2f} spread={spread:.2f}")
    
    # 檢查是否需要取消訂單（30秒後）
    submit_time = context.get_object("submit_time")
    confirmed_ex_order_id = context.get_object("confirmed_ex_order_id")
    cancelled = context.get_object("cancelled")
    
    if submit_time and confirmed_ex_order_id and not cancelled:
        elapsed = (context.now() - submit_time) / 1_000_000_000  # 轉換為秒
        if elapsed >= 30:
            ex_order_id = confirmed_ex_order_id
            order_id = context.get_object("order_id")
            
            context.log().info(f"")
            context.log().info(f"⏰ 30 秒已到，開始取消訂單...")
            context.log().info(f"🗑️  [Cancelling Order] order_id={order_id} ex_order_id='{ex_order_id}'")
            
            try:
                context.cancel_order(
                    config["account"], 
                    order_id, 
                    config["symbol"], 
                    ex_order_id, 
                    InstrumentType.FFuture
                )
                context.set_object("cancelled", True)
            except Exception as e:
                context.log().error(f"❌ [Cancel Failed] {str(e)}")
    
    # 安全地檢查標誌（處理 None 情況）
    order_placed = context.get_object("order_placed")
    if order_placed is None:
        order_placed = False
        context.set_object("order_placed", False)
    
    # 只發送一次測試訂單
    if not order_placed:
        # 使用合理的價格（略低於市價，不太可能成交但不會被拒絕）
        # Binance Futures BTCUSDT 限制：
        #   - tick size = 0.1（價格精度）
        #   - notional >= 100 USDT（名義價值最小值）
        raw_price = ask * 0.98  # 當前賣價的 98%（2% 折扣）
        # 使用 Decimal 確保價格精確到 0.1，完全避免浮點數表示問題
        test_price = float(Decimal(str(raw_price)).quantize(Decimal('0.1'), rounding=ROUND_DOWN))
        test_volume = 0.002  # 增加到 0.002 BTC，確保 notional >= 100 USDT
        
        notional = test_price * test_volume
        context.log().info(f"💸 [Placing Order] Buy {test_volume} BTC @ {test_price:.1f} (notional={notional:.2f} USDT)")
        
        try:
            order_id = context.insert_order(
                config["symbol"], 
                InstrumentType.FFuture, 
                Exchange.BINANCE, 
                config["account"],
                test_price, 
                test_volume, 
                OrderType.Limit, 
                Side.Buy
            )
            
            context.log().info(f"✅ [Order Placed] order_id={order_id}")
            
            # 立即設置標誌，避免重複下單
            context.set_object("order_placed", True)
            context.set_object("order_id", order_id)
            
        except Exception as e:
            context.log().error(f"❌ [Order Failed] {str(e)}")
            # 即使失敗也設置標誌，避免無限重試
            context.set_object("order_placed", True)

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
# Phase 6: on_factor 回調 - 接收線性模型輸出
# ========================================
def on_factor(context, symbol, timestamp, values):
    """
    🎊 [Phase 6] 因子回调 - 接收 LinearModel 计算的预测信号

    数据流:
    Binance → hf-live → 15 market factors → LinearModel → on_factor

    Market Factors (15):
    - Depth: spread, mid_price, bid_ask_ratio, depth_imbalance, weighted_mid
    - Trade: trade_volume_ma, trade_direction, trade_intensity, vwap, trade_volatility
    - Ticker: ticker_spread, ticker_volume_ratio, ticker_momentum
    - IndexPrice: basis, basis_pct

    LinearModel Outputs (2):
    - pred_signal: 加权因子信号 (-∞, +∞)，正值看涨，负值看跌
    - pred_confidence: 信号置信度 [0.5, 1.0]，基于信号强度的 sigmoid

    当 HF_TIMING_METADATA=ON 编译时，values 前 8 列为延迟元数据:
    [0] marker = -999.0 (识别标记)
    [1] tick_wait_us (行情等待延迟)
    [2] factor_calc_us (因子计算耗时)
    [3] factor_elapsed_us (从行情到计算完成)
    [4] scan_elapsed_us (扫描延迟)
    [5] total_elapsed_us (总端到端延迟)
    [6] output_count (输出数量)
    [7] reserved (保留)

    Args:
        symbol: 交易对 (如 'BTCUSDT')
        timestamp: 时间戳 (纳秒)
        values: 模型输出列表 [pred_signal, pred_confidence] 或带元数据
    """
    # ✅ Phase 4G 修復: 立即複製數據到 Python list,避免懸空指針
    values = list(values)

    # 检测延迟元数据 (HF_TIMING_METADATA=ON 时注入)
    latency_info = None
    actual_values = values
    if len(values) > 8 and values[0] == -999.0:
        # 解析元数据
        latency_info = {
            'tick_wait_us': values[1],
            'factor_calc_us': values[2],
            'factor_elapsed_us': values[3],
            'scan_elapsed_us': values[4],
            'total_elapsed_us': values[5],
            'output_count': int(values[6]),
        }
        # 去除元数据头，获取实际值
        actual_values = values[8:]

        context.log().info(f"")
        context.log().info(f"📊 [Latency] tick_wait={latency_info['tick_wait_us']:.1f}us "
                          f"calc={latency_info['factor_calc_us']:.1f}us "
                          f"total={latency_info['total_elapsed_us']:.1f}us")

    # 当前版本：期望 2 个线性模型输出
    if len(actual_values) >= 2:
        pred_signal = actual_values[0]
        pred_confidence = actual_values[1]

        # 生成交易信号解读
        if pred_signal > 0.1:
            signal_emoji = "📈"
            signal_text = "BULLISH"
        elif pred_signal < -0.1:
            signal_emoji = "📉"
            signal_text = "BEARISH"
        else:
            signal_emoji = "➡️"
            signal_text = "NEUTRAL"

        # 格式化输出
        context.log().info(f"")
        context.log().info(f"🤖 [LinearModel] {symbol} @ {timestamp}")
        context.log().info(f"   {signal_emoji} Signal: {pred_signal:+.4f} ({signal_text})")
        context.log().info(f"   🎯 Confidence: {pred_confidence:.2%}")
        context.log().info(f"")
    else:
        context.log().warning(f"⚠️  Unexpected values count: {len(actual_values)} (expected >= 2)")
        context.log().warning(f"   Raw values: {actual_values}")

