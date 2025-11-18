'''
This is source code under the Apache License 2.0.
Original Author: kx@godzilla.dev
Original date: March 3, 2025
'''
import kungfu.yijinjing.time as kft
from kungfu.wingchun.constants import *
from pywingchun.constants import Side, InstrumentType, OrderType

exchange = Exchange.BINANCE
instrument_type = InstrumentType.Spot

def pre_start(context):
    breakpoint()
    context.log().info(f"pre run strategy, books: {len(context.books)}")
    config = context.get_config()
    # context.add_account(config["td_source"], config["account"])
    symbol_info = context.get_market_info(config["symbol"], exchange, instrument_type)
    print(symbol_info)

    # context.set_account_cash_limit(config["td_source"], exchange, config["account"], config["base_coin"], config["base_limit"])
    # context.set_account_cash_limit(config["td_source"], exchange, config["account"], config["quote_coin"], config["quote_limit"])
    context.subscribe(config["md_source"], [config["symbol"]], instrument_type, exchange)
    # context.subscribe_trade(config["md_source"], [config["symbol"]], instrument_type, exchange)
    context.subscribe_ticker(config["md_source"], [config["symbol"]], instrument_type, exchange)
    context.set_object("total_orders", 0)
    # api = context.get_account_api(config["td_source"], config["account"])
    # context.set_object("api", api)
    # context.log().info(api.balance('usdt'))

def on_depth(context, depth):
    breakpoint()
    config = context.get_config()
    if depth.symbol != config['symbol']:
        context.log().info(f"not subscribed symbol: {depth.symbol}, {config}")
        return
    # if depth.exchange_id == exchange:
    #     context.set_object("xt_depth", depth)

    context.log().info(depth)
    book = context.get_account_book(config["td_source"], config["account"])
    context.log().info(f"active orders: {len(book.active_orders)}")
    if config["action"] == "cancel":
        for o in book.active_orders:
            if o['ex_order_id'] != "":
                context.cancel_order(config["account"], o['order_id'], config["symbol"], o['ex_order_id'], instrument_type)
            else:
                context.log().info(o)
    elif config["action"] == "single":
        if len(book.active_orders) == 0:
            order_id = context.insert_order(config["symbol"], instrument_type, exchange, config["account"], 1, 2, OrderType.Limit, Side.Buy)
    elif config["action"] == "multiple":
        orders = context.get_object("total_orders")
        context.log().info(orders)
        # if len(book.active_orders) <= 30:
        if orders < 400:
            order_id = context.insert_order(config["symbol"], instrument_type, exchange, config["account"], 1, 2, OrderType.Limit, Side.Buy)
            context.log().info(f"order id: {order_id}, {len(book.active_orders)}")
            orders += 1
            context.set_object("total_orders", orders)
        # else:
            # for o in book.active_orders:
            #     if o["ex_order_id"] == "":
            #         context.query_order(config["account"], o["order_id"], o["ex_order_id"], instrument_type)
                # else:
                #     context.cancel_order(config["account"], o['order_id'], config["symbol"], o['ex_order_id'], instrument_type)


def on_ticker(context, ticker):
    context.log().info(ticker)
    context.reload_config()
    config = context.get_config()
    context.log().info(f'{config["name"]}')

def on_transac       tion(context, transaction):
    context.log().info("{} {}".format(transaction.symbol, transaction.exchange_id))


def on_order(context, order):
    breakpoint()
    config = context.get_config()
    context.log().info(f'order received: {order}')
    if order.status == OrderStatus.Cancelled:
        context.log().info(f"order cancelled: {order.ex_order_id}")
        # context.adjust_leverage(config["account"], "btc_usdt", Direction.Long, 3)
    elif order.status == OrderStatus.Submitted:
        context.log().info(f"cancel order: {order.order_id} {order.ex_order_id}")
        context.cancel_order(config["account"], order.order_id, order.symbol, order.ex_order_id, instrument_type)


def on_trade(context, trade):
    context.log().info('trade received: {} [trade_id]{} [volume]{} [price]{}'.format(kft.strftime(trade.trade_time), trade.trade_id, trade.volume, trade.price))


def on_union_response(context, sub_msg):
    context.log().info(f"sub_msg: {sub_msg}")