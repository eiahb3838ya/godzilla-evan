"""
test_hf_live - 端到端測試策略（漸進式驗證）
Phase 4B: 基礎訂單流測試（無 hf-live）
測試 Binance → Python 數據流 + 訂單發射驗證
"""
from kungfu.wingchun.constants import *
from pywingchun.constants import InstrumentType, OrderType, Side, OrderStatus
import math
from decimal import Decimal, ROUND_DOWN

def pre_start(context):
    """策略初始化"""
    context.log().info("🏁 [Phase 4E] Pre-Start - Testing hf-live Data Flow")

    # 訂閱市場數據 - 只測試數據流，不添加交易帳號
    config = context.get_config()
    context.subscribe(config["md_source"], [config["symbol"]], InstrumentType.FFuture, Exchange.BINANCE)
    context.log().info(f"📡 Subscribed: {config['symbol']} (Futures) - Market Data Only")

    context.log().info("✅ [Init] hf-live data flow test initialized")

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
    context.log().info("🏁 [Phase 4B] Stopped")

# ========================================
# Phase 4F: on_factor 回調（暫時註釋）
# 等待 Phase 4C-4E 完成後再啟用
# ========================================
def on_factor(context, symbol, timestamp, values):
    """
    🎊 [Phase 4I] 因子回调 - 接收 libsignal.so 计算的模型输出

    注意：当前版本只接收模型输出（2个值）
    因子值（spread, mid_price, bid_volume）在 FactorEngine 计算后
    直接发送到 ModelEngine，不经过 Python 层。

    未来版本可扩展为接收完整的因子+模型数据（5个值）。

    当 HF_TIMING_METADATA=ON 编译时，values 前 8 列为延迟元数据:
    [0] marker = -999.0 (识别标记)
    [1] tick_wait_us (行情等待延迟)
    [2] factor_calc_us (因子计算耗时)
    [3] factor_elapsed_us (从行情到计算完成)
    [4] scan_elapsed_us (扫描延迟)
    [5] total_elapsed_us (总端到端延迟)
    [6] factor_count (因子数量)
    [7] reserved (保留)

    Args:
        symbol: 交易对 (如 'btcusdt')
        timestamp: 时间戳 (纳秒)
        values: 模型输出列表 [pred_signal, pred_confidence] 或带元数据
    """
    # ✅ Phase 4G 修復: 立即複製數據到 Python list,避免懸空指針
    # C++ 側的 factor_values 可能在回調返回後析構,導致 pybind11 綁定的 values 指向已釋放記憶體
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
            'factor_count': int(values[6]),
        }
        # 去除元数据头，获取实际值
        actual_values = values[8:]

        context.log().info(f"")
        context.log().info(f"📊 [Latency] tick_wait={latency_info['tick_wait_us']:.1f}μs "
                          f"calc={latency_info['factor_calc_us']:.1f}μs "
                          f"total={latency_info['total_elapsed_us']:.1f}μs")

    context.log().info(f"")
    context.log().info(f"🎊🎊🎊 [on_factor] Factor data received! 🎊🎊🎊")
    context.log().info(f"  Symbol: {symbol}")
    context.log().info(f"  Timestamp: {timestamp}")
    context.log().info(f"  Values count: {len(actual_values)} (raw: {len(values)})")
    context.log().info(f"  Values: {actual_values[:5]}..." if len(actual_values) > 5 else f"  Values: {actual_values}")
    context.log().info(f"")

    # 当前版本：只期望 2 个模型输出
    if len(actual_values) >= 2:
        pred_signal = actual_values[0]
        pred_confidence = actual_values[1]

        context.log().info(f"  🤖 Model Predictions:")
        context.log().info(f"     pred_signal={pred_signal:.4f}")
        context.log().info(f"     pred_confidence={pred_confidence:.4f}")
        context.log().info(f"")
        context.log().info(f"  ✅ 🎊 E2E TEST PASSED! 🎊 ✅")
        context.log().info(f"")
    else:
        context.log().error(f"  ❌ Unexpected values count: {len(actual_values)}")
        context.log().error(f"  Expected: >= 2 (model outputs)")
