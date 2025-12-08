"""
test_hf_live - 端到端測試策略（漸進式驗證）
Phase 4B: 基礎訂單流測試（無 hf-live）
測試 Binance → Python 數據流 + 訂單發射驗證
"""
from kungfu.wingchun.constants import *
from pywingchun.constants import InstrumentType, OrderType, Side, OrderStatus

def pre_start(context):
    """策略初始化"""
    context.log().info("🏁 [Phase 4B] Pre-Start - Testing Order Placement")
    context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)
    context.log().info("📡 Subscribed: btcusdt (Spot)")
    
    # 存儲狀態
    context.set_object("order_placed", False)
    context.set_object("order_confirmed", False)
    context.set_object("order_id", None)
    context.set_object("ex_order_id", None)

def on_depth(context, depth):
    """接收盤口數據 + 發送測試訂單"""
    bid = depth.bid_price[0]
    ask = depth.ask_price[0]
    spread = ask - bid
    
    # 打印盤口
    context.log().info(f"📊 [on_depth] {depth.symbol} bid={bid:.2f} ask={ask:.2f} spread={spread:.2f}")
    
    # 只發送一次測試訂單
    if not context.get_object("order_placed"):
        # 極低價格（不會成交）
        test_price = ask - 10000.0
        test_volume = 0.001
        
        context.log().info(f"💸 [Placing Order] Buy {test_volume} BTC @ {test_price:.2f} (ask - 10000)")
        
        try:
            order_id = context.insert_order(
                "btcusdt", 
                InstrumentType.Spot, 
                Exchange.BINANCE, 
                "gz_user1",  # 使用 binance_test 的帳號
                test_price, 
                test_volume, 
                OrderType.Limit, 
                Side.Buy
            )
            
            context.log().info(f"✅ [Order Placed] order_id={order_id}")
            context.set_object("order_placed", True)
            context.set_object("order_id", order_id)
            
        except Exception as e:
            context.log().error(f"❌ [Order Failed] {str(e)}")

def on_order(context, order):
    """訂單狀態回調 - 驗證發射成功"""
    context.log().info(f"📬 [on_order] order_id={order.order_id} status={order.status} ex_order_id='{order.ex_order_id}'")
    
    # 檢查是否是我們的測試訂單
    our_order_id = context.get_object("order_id")
    if order.order_id == our_order_id:
        
        # 檢查是否成功提交到 Binance
        if order.status == OrderStatus.Submitted and order.ex_order_id != "":
            if not context.get_object("order_confirmed"):
                context.log().info("🎉 [Order Fired!] Successfully submitted to Binance")
                context.log().info(f"   ├─ Local ID: {order.order_id}")
                context.log().info(f"   ├─ Exchange ID: {order.ex_order_id}")
                context.log().info(f"   └─ Status: Submitted")
                
                context.set_object("order_confirmed", True)
                context.set_object("ex_order_id", order.ex_order_id)
                
                # 取消測試訂單（清理）
                context.log().info(f"🗑️ [Cancelling Order] order_id={order.order_id} ex_order_id='{order.ex_order_id}'")
                context.cancel_order("gz_user1", order.order_id, "btcusdt", order.ex_order_id, InstrumentType.Spot)
        
        # 檢查是否被拒絕
        elif order.status == OrderStatus.Error:
            context.log().error(f"❌ [Order Rejected] error_code={order.error_code}")
        
        # 確認取消成功
        elif order.status == OrderStatus.Cancelled:
            context.log().info("✅ [Order Cancelled] Successfully cleaned up")

def post_stop(context):
    """策略停止"""
    context.log().info("🏁 [Phase 4B] Stopped")

# ========================================
# Phase 4F: on_factor 回調（暫時註釋）
# 等待 Phase 4C-4E 完成後再啟用
# ========================================
# def on_factor(context, symbol, timestamp, values):
#     """
#     Phase 4F: 驗證完整數據流 (Depth → Factor → Model → Python)
#     
#     Args:
#         symbol: 標的代碼 (如 'BTCUSDT')
#         timestamp: 時間戳 (納秒)
#         values: 模型輸出值 [pred_signal, pred_confidence]
#     """
#     context.log().info(f"🎉 [on_factor] {symbol} @ {timestamp}")
#     context.log().info(f"   Model Output: {values}")
#     
#     if len(values) >= 2:
#         pred_signal = values[0]
#         pred_confidence = values[1]
#         context.log().info(f"   ✅ pred_signal={pred_signal:.4f}, pred_confidence={pred_confidence:.4f}")
#         context.log().info("   🎊 E2E TEST PASSED!")
#     else:
#         context.log().error(f"   ❌ Unexpected output size: {len(values)}")
