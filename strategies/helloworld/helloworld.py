'''
This is source code under the Apache License 2.0.
Original Author: kx@godzilla.dev
Original date: March 3, 2025
'''
from kungfu.wingchun.constants import *
from pywingchun.constants import Side, InstrumentType, OrderType

exchange = Exchange.BINANCE
instrument_type = InstrumentType.FFuture  # Futures (USDT-Margined)

def pre_start(context):
    config = context.get_config()
    context.subscribe(config["md_source"], [config["symbol"]], instrument_type, exchange)

def on_depth(context, depth):
    # 打印最优买卖价
    bid_price = depth.bid_price[0]
    ask_price = depth.ask_price[0]
    bid_volume = depth.bid_volume[0]
    ask_volume = depth.ask_volume[0]
    
    context.log().info(f"[{depth.symbol}] Bid: {bid_price:.2f} (Vol: {bid_volume:.4f}) | Ask: {ask_price:.2f} (Vol: {ask_volume:.4f}) | Spread: {(ask_price - bid_price):.2f}")
