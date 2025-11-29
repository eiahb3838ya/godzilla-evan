'''
This is source code under the Apache License 2.0.
Original Author: kx@godzilla.dev
Original date: March 3, 2025
'''
import kungfu.yijinjing.time as kft
from kungfu.wingchun.constants import *
from pywingchun.constants import Side, InstrumentType, OrderType

exchange = Exchange.BINANCE
instrument_type = InstrumentType.FFuture


def pre_start(context):
    context.log().info(f"pre run strategy, books: {len(context.books)}")
    config = context.get_config()
    context.add_account(config["td_source"], config["account"])
    # symbol_info = context.get_market_info(config["symbol"], exchange, instrument_type)
    # context.log().info(symbol_info)

    context.set_account_cash_limit(config["td_source"], exchange, config["account"], config["base_coin"],
                                   config["base_limit"])
    context.set_account_cash_limit(config["td_source"], exchange, config["account"], config["quote_coin"],
                                   config["quote_limit"])

    # 訂閱期貨市場深度數據（Depth）
    # index_price 目前 Binance 擴展未實作，改用 subscribe
    context.subscribe(config["md_source"], [config["symbol"]], instrument_type, exchange)


def on_depth(context, depth):
    context.log().info(depth)
    config = context.get_config()
    if depth.symbol != config['symbol']:
         context.log().info(f"not subscribed symbol: {depth.symbol}, {config}")
         return
    book = context.get_account_book(config["td_source"], config["account"])
    context.log().info(f"active orders: {len(book.active_orders)}")
    if len(book.active_orders) <= 0:
        order_id = context.insert_order(config["symbol"], instrument_type, exchange, config["account"], 1700, 0.005,
                                        OrderType.Limit, Side.Buy)
        context.log().info(order_id)
    else:
        context.log().info(book.active_orders)
        for o in book.active_orders:
            if o["ex_order_id"] == "":
                context.query_order(config["account"], o["order_id"], o["ex_order_id"], instrument_type, o["symbol"])
            # else:
            #     context.cancel_order(config["account"], o['order_id'], o["symbol"], o['ex_order_id'], instrument_type)

    # else:
    # book = context.get_account_book(td_source, account)
    # if not context.oid:
    #     for o in book.active_orders:
    #         if o['ex_order_id']:
    #             context.cancel_order(account, o['order_id'], market, o['ex_order_id'])
    # else:
    # print(depth.ask_price[0]-10)
    # for i in range(2):
    #     context.insert_order(market, InstrumentType.Spot, exchange, account, depth.ask_price[0]+1000, 0.0001*(i+1), OrderType.Limit, Side.Buy)
    # context.oid = 1

def on_index_price(context, price):
    context.log().info(f'index price received: [{price.symbol}] [{price.price}]')
    config = context.get_config()
    # context.merge_positions(config["account"], config["symbol"], 1)
    context.query_positions(config["account"], config["symbol"])

def on_ticker(context, ticker):
    context.log().info(ticker)


def on_transaction(context, transaction):
    context.log().info("{} {}".format(transaction.symbol, transaction.exchange_id))

def on_position(context, position):
    context.log().info(f"on position {position}")

def on_order(context, order):
    # config = context.get_config()
    context.log().info(f'order received: {order}')
    # if order.status == OrderStatus.Cancelled:
    #     context.log().info("Cancelld")
        # context.adjust_leverage(config["account"], config["symbol"], Direction.Long, 3)
    # elif order.status == OrderStatus.Submitted:
    #     context.cancel_order(config["account"], order.order_id, order.symbol, order.ex_order_id, instrument_type)


def on_trade(context, trade):
    context.log().info(
        'trade received: {} {} [trade_id]{} [volume]{} [price]{}'.format(kft.strftime(trade.trade_time), trade.symbol, trade.trade_id,
                                                                      trade.volume, trade.price))


def on_union_response(context, sub_msg):
    context.log().info(f"union response sub_msg: {sub_msg}")
