"""
HF-Live Factor Strategy Demo
演示如何使用 hf-live 因子框架
"""

def pre_start(ctx):
    """策略啟動前初始化"""
    ctx.logger.info("=== Factor Strategy Pre-Start ===")
    ctx.logger.info("Waiting for factor callbacks from hf-live...")

    # 訂閱市場數據 (如果需要)
    # ctx.subscribe(source='binance', exchange='binance', symbol='btcusdt', is_level2=True)

def on_factor(ctx, symbol, timestamp, values):
    """
    因子回調 - 接收 hf-live 計算的因子值

    Args:
        ctx: 策略上下文
        symbol: 標的代碼 (如 'btcusdt')
        timestamp: 時間戳 (納秒)
        values: 因子值列表 (list of float)
    """
    ctx.logger.info(f"[Factor] {symbol} @ {timestamp}")
    ctx.logger.info(f"  Values ({len(values)}): {values[:5]}...")  # 僅顯示前5個

    # 示例: 簡單的因子策略邏輯
    if len(values) > 0:
        factor_0 = values[0]

        # 示例決策邏輯
        if factor_0 > 0.5:
            ctx.logger.info(f"  🔥 Signal: BUY (factor_0={factor_0:.4f})")
            # ctx.insert_order(...)  # 實際下單邏輯
        elif factor_0 < -0.5:
            ctx.logger.info(f"  🔥 Signal: SELL (factor_0={factor_0:.4f})")
            # ctx.insert_order(...)
        else:
            ctx.logger.info(f"  ⏸️  Signal: HOLD (factor_0={factor_0:.4f})")

def on_depth(ctx, depth):
    """
    盤口回調 (可選)
    如果需要同時處理原始盤口數據
    """
    # ctx.logger.debug(f"Depth: {depth.symbol} bid={depth.bid_price[0]}")
    pass

def post_stop(ctx):
    """策略停止後清理"""
    ctx.logger.info("=== Factor Strategy Stopped ===")
