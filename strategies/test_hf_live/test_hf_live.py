"""
test_hf_live - 端到端測試策略（漸進式驗證）
Phase 4B: 測試基礎 on_depth 回調（無 signal library）
"""
from kungfu.wingchun.constants import *
from pywingchun.constants import InstrumentType

def pre_start(context):
    """Phase 4B: 策略啟動前初始化"""
    context.log().info("🏁 [test_hf_live] Pre-Start (Phase 4B)")
    context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)

def on_depth(context, depth):
    """Phase 4B: 驗證 Binance 數據接收"""
    bid = depth.bid_price[0]
    ask = depth.ask_price[0]
    context.log().info(f"✅ [on_depth] {depth.symbol} bid={bid:.2f} ask={ask:.2f}")

def post_stop(context):
    """策略停止"""
    context.log().info("🏁 [test_hf_live] Stopped")

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
