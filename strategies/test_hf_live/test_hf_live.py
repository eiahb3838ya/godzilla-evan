"""
test_hf_live - 端到端測試策略
驗證數據流: Binance → Factor → Model → on_factor callback
"""

def pre_start(ctx):
    """策略啟動前初始化"""
    ctx.logger.info("🏁 [test_hf_live] Pre-Start - Waiting for callbacks...")

def on_depth(ctx, depth):
    """
    盤口回調 - 驗證 Binance 數據接收
    """
    ctx.logger.info(f"✅ [on_depth] {depth.symbol} "
                   f"bid={depth.bid_price[0]:.2f} "
                   f"ask={depth.ask_price[0]:.2f}")

def on_factor(ctx, symbol, timestamp, values):
    """
    因子回調 - 驗證完整數據流 (Depth → Factor → Model → Python)
    
    Args:
        symbol: 標的代碼 (如 'BTCUSDT')
        timestamp: 時間戳 (納秒)
        values: 模型輸出值 [pred_signal, pred_confidence]
    """
    ctx.logger.info(f"🎉 [on_factor] {symbol} @ {timestamp}")
    ctx.logger.info(f"   Model Output ({len(values)}): {values}")
    
    if len(values) >= 2:
        pred_signal = values[0]
        pred_confidence = values[1]
        ctx.logger.info(f"   ✅ pred_signal={pred_signal:.4f}, pred_confidence={pred_confidence:.4f}")
        ctx.logger.info("   🎊 E2E TEST PASSED: Data flow verified!")
    else:
        ctx.logger.error(f"   ❌ Unexpected output size: {len(values)}")

def post_stop(ctx):
    """策略停止後清理"""
    ctx.logger.info("🏁 [test_hf_live] Stopped")
