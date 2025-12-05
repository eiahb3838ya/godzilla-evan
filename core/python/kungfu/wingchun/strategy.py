'''
This is source code modified under the Apache License 2.0.
Original Author: Keren Dong
Modifier: kx@godzilla.dev
Modification date: March 3, 2025
'''
import os
import sys
import json
import pandas as pd
import time
import importlib
import pyyjj
import pywingchun
from kungfu.wingchun.book.book import AccountBook
from kungfu.wingchun.constants import *
import kungfu.msg.utils as msg_utils
import kungfu.yijinjing.time as kft
from kungfu.yijinjing.log import create_logger
from kungfu.data.sqlite.data_proxy import MarketInfoDB, LedgerDB, AccountsDB
from .backtest.engine import BacktestEngine


class AlgoOrderContext:
    def __init__(self, wc_ctx):
        self._wc_ctx = wc_ctx
        self.orders = {}

    def insert_algo_order(self, order):
        order_id = self._wc_ctx.add_order(order)
        if order_id > 0:
            self.orders[order_id] = order


class Strategy(pywingchun.Strategy):
    def __init__(self, ctx):
        pywingchun.Strategy.__init__(self)
        ctx.strftime = kft.strftime
        ctx.strptime = kft.strptime
        ctx.inst_infos = {}
        self.ctx = ctx
        self.ctx.account_source = {}
        self.ctx.db = LedgerDB(self.ctx.location, self.ctx.name)
        self.ctx.book = None
        self.ctx.books = {}
        self.ctx.objects = {}
        self.config_path = ctx.config_path
        self.config = ctx.config
        self.ctx.strategy_loggers = {}
        self.__init_strategy(ctx.path)

    def __add_account(self, source: str, account: str):
        self.wc_context.add_account(source, account)
        book = self.__get_account_book(source, account)
        book.set_strategy_id(self.get_uid())
        if self.ctx.backtest:
            book.set_backtest()
        book.subject.subscribe(self.on_book_event)
        location = pyyjj.location(
            pyyjj.mode.LIVE, pyyjj.category.TD, source, account, self.ctx.locator)
        self.book_context.add_book(location, book)
        self.ctx.logger.info(
            "added book {}:{}@{}".format(source, account, self.get_uid()))

    def __get_account_book(self, source: str, account: str):
        location = pyyjj.location(
            pyyjj.mode.LIVE, pyyjj.category.TD, source, account, self.ctx.locator)
        if self.ctx.get_current_strategy_index() not in self.ctx.books or location.uid not in self.ctx.books[
                self.ctx.get_current_strategy_index()]:
            book = self.ctx.db.strategy_load_book(
                self.ctx, location, self.ctx.get_current_strategy_index())
            if not book:
                self.ctx.logger.info(
                    "failed to load book from sqlite for {} [{:08x}]".format(location.uname, location.uid))
                book = AccountBook(self.ctx, location=location)
            if self.ctx.get_current_strategy_index() not in self.ctx.books:
                self.ctx.books[self.ctx.get_current_strategy_index()] = {}
            self.ctx.books[self.ctx.get_current_strategy_index()
                           ][location.uid] = book
            self.ctx.logger.info("success to init book for {} [{:08x}]".format(
                location.uname, location.uid))
        self.ctx.account_source[account] = source
        return self.ctx.books[self.ctx.get_current_strategy_index()][location.uid]

    def __get_inst_info(self, symbol, exchange_id):
        return msg_utils.object_as_dict(self.book_context.get_inst_info(symbol, exchange_id))

    def __init_strategy(self, path):
        strategy_dir = os.path.dirname(path)
        name_no_ext = os.path.split(os.path.basename(path))
        sys.path.append(os.path.relpath(strategy_dir))
        impl = importlib.import_module(os.path.splitext(name_no_ext[1])[0])
        self._pre_start = getattr(impl, 'pre_start', lambda ctx: None)
        self._post_start = getattr(impl, 'post_start', lambda ctx: None)
        self._pre_stop = getattr(impl, 'pre_stop', lambda ctx: None)
        self._post_stop = getattr(impl, 'post_stop', lambda ctx: None)
        self._on_ticker = getattr(impl, "on_ticker", lambda ctx, bar: None)
        self._on_index_price = getattr(
            impl, "on_index_price", lambda ctx, bar: None)
        self._on_depth = getattr(impl, 'on_depth', lambda ctx, depth: None)
        self._on_transaction = getattr(
            impl, "on_transaction", lambda ctx, transaction: None)
        self._on_order = getattr(impl, 'on_order', lambda ctx, order: None)
        self._on_trade = getattr(impl, 'on_trade', lambda ctx, trade: None)
        self._on_position = getattr(
            impl, 'on_position', lambda ctx, order: None)
        self._on_order_action_error = getattr(
            impl, 'on_order_action_error', lambda ctx, error: None)
        self._on_union_response = getattr(
            impl, 'on_union_response', lambda ctx, error: None)
        self._on_factor = getattr(
            impl, 'on_factor', lambda ctx, symbol, timestamp, values: None)

    def __init_book(self):
        location = pyyjj.location(pyyjj.mode.LIVE, pyyjj.category.STRATEGY, self.ctx.group, self.ctx.name,
                                  self.ctx.locator)
        self.ctx.book = AccountBook(self.ctx, location)
        self.book_context.add_book(location, self.ctx.book)

    def __init_algo(self):
        class AlgoOrderContext:
            def __init__(self, wc_ctx):
                self._wc_ctx = wc_ctx
                self.orders = {}

            def insert_algo_order(self, order):
                order_id = self._wc_ctx.add_order(order)
                if order_id > 0:
                    self.orders[order_id] = order

        self.algo_context = AlgoOrderContext(self.algo_context)
        self.ctx.insert_algo_order = self.algo_context.insert_algo_order

    def __add_timer(self, nanotime, callback):
        def wrap_callback(event):
            callback(self.ctx, event)

        self.wc_context.add_timer(nanotime, wrap_callback)

    def __add_time_interval(self, duration, callback):
        def wrap_callback(event):
            callback(self.ctx, event)

        self.wc_context.add_time_interval(duration, wrap_callback)

    def pre_start(self, wc_context):
        self.ctx.logger.info("pre start")
        self.persistent = True
        self.bt = BacktestEngine()
        self.wc_context = wc_context
        self.book_context = wc_context.book_context
        self.algo_context = wc_context.algo_context
        self.ctx.now = wc_context.now
        self.ctx.reload_config = self.__reload_config
        self.ctx.add_timer = self.__add_timer
        self.ctx.add_time_interval = self.__add_time_interval
        self.ctx.subscribe = wc_context.subscribe
        self.ctx.unsubscribe = wc_context.unsubscribe
        self.ctx.subscribe_trade = wc_context.subscribe_trade
        self.ctx.subscribe_ticker = wc_context.subscribe_ticker
        self.ctx.subscribe_index_price = wc_context.subscribe_index_price
        self.ctx.subscribe_all = wc_context.subscribe_all
        self.ctx.add_account = self.__add_account
        self.ctx.adjust_leverage = wc_context.adjust_leverage
        self.ctx.merge_positions = wc_context.merge_positions
        self.ctx.query_positions = wc_context.query_positions
        self.ctx.get_current_strategy_index = wc_context.get_current_strategy_index
        self.ctx.list_accounts = wc_context.list_accounts
        self.ctx.get_account_cash_limit = wc_context.get_account_cash_limit
        self.ctx.set_account_cash_limit = self.__set_account_cash_limit
        self.ctx.insert_order = self.__insert_order
        self.ctx.query_order = wc_context.query_order
        self.ctx.cancel_order = self.__cancel_order
        self.ctx.get_account_book = self.__get_account_book
        self.ctx.get_inst_info = self.__get_inst_info
        self.ctx.get_market_info = self.__get_market_info
        self.ctx.set_object = self.__set_object
        self.ctx.get_object = self.__get_object
        self.ctx.get_config = self.__get_config
        self.ctx.set_persistent = self.__set_persistent
        self.ctx.log = self.log
        self.ctx.get_account_api = self.__get_account_api

        # self.__init_book()
        # self.__init_algo()
        self._pre_start(self.ctx)
        self.ctx.log().info('strategy prepare to run')

    def log(self):
        strategy_id = self.ctx.get_current_strategy_index()
        if strategy_id not in self.ctx.strategy_loggers:
            self.ctx.strategy_loggers[strategy_id] = create_logger(hex(abs(strategy_id)), self.ctx.log_level,
                                                                   self.ctx.location)
        return self.ctx.strategy_loggers[strategy_id]

    def __set_persistent(self, p: bool) -> None:
        self.ctx.strategies[
            self.ctx.get_current_strategy_index()].persistent = p

    def __get_account_api(self, source_id, account_id, api_type='SPOT'):
        acct_db = AccountsDB(pyyjj.location(pyyjj.mode.LIVE, pyyjj.category.SYSTEM, 'etc', 'kungfu', self.ctx.locator),
                             'accounts')
        config = acct_db.get_td_account_config(
            source_id, source_id + '_' + account_id)
        '''not implemented'''
        return None

    def __get_market_info(self, symbol, exchange_id, instrument_type):
        market_db = MarketInfoDB(
            pyyjj.location(pyyjj.mode.LIVE, pyyjj.category.SYSTEM,
                           'etc', 'kungfu', self.ctx.locator),
            pywingchun.utils.get_shm_db())
        return market_db.get_market_info(symbol, exchange_id, instrument_type)

    def _set_init_asset(self, coin, active_orders, limit):
        '''set frozen and avail assets'''
        if not active_orders:
            return limit, 0
        df = pd.DataFrame(active_orders)
        symbol = df.symbol.iloc[0]
        base, _ = symbol.split("_")
        if coin == base:
            frozen = df[df.side == Side.Sell].volume_left.sum()
            avail = limit - frozen
            return avail, frozen
        frozen = (df[df.side == Side.Buy].volume_left *
                  df[df.side == Side.Buy].price).sum()
        avail = limit - frozen
        return avail, frozen

    def __set_account_cash_limit(self, source: str, exchange_id: str, account: str, coin: str, limit: int):
        book = self.__get_account_book(source, account)
        if not book.get_asset(coin):
            avail, frozen = self._set_init_asset(
                coin, book.active_orders, limit)
            book.set_asset(self.ctx.get_current_strategy_index(),
                           exchange_id, account, coin, avail, frozen)
        self.wc_context.set_account_cash_limit(account, coin, limit)

    def __set_object(self, name: str, obj: object):
        object_key = name + str(self.ctx.get_current_strategy_index())
        self.ctx.objects[object_key] = obj

    def __get_object(self, name: str):
        object_key = name + str(self.ctx.get_current_strategy_index())
        if object_key in self.ctx.objects:
            return self.ctx.objects[object_key]
        else:
            return None

    def __get_config(self):
        return self.ctx.strategies[self.ctx.get_current_strategy_index()].config

    def on_book_event(self, event):
        e = event.as_dict()
        self.ctx.logger.debug(
            f"book event received: {e['msg_type']}, {e['data']['account_id']}")
        if self.ctx.backtest:
            orders = self.bt.report_order()
            for o in orders:
                book = self.__get_account_book(
                    self.ctx.account_source[o.account_id], o.account_id)
                book.on_order(event, o)
                self.on_order(None, o)
        self.ctx.db.on_book_event(e)

    def post_start(self, wc_context):
        self._post_start(self.ctx)
        self.ctx.log().info('strategy ready to run')

    def pre_stop(self, wc_context):
        if self.ctx.backtest:
            self.bt.report_pnl()
        self._pre_stop(self.ctx)

    def post_stop(self, wc_context):
        self._post_stop(self.ctx)

    def on_depth(self, wc_context, depth):
        if self.ctx.backtest:
            self.bt.update_depth(depth)
        self._on_depth(self.ctx, depth)

    def on_ticker(self, wc_context, ticker):
        if self.ctx.backtest:
            self.bt.update_ticker(ticker)
        self._on_ticker(self.ctx, ticker)

    def on_index_price(self, wc_context, ip):
        self._on_index_price(self.ctx, ip)

    def on_transaction(self, wc_context, transaction):
        self._on_transaction(self.ctx, transaction)

    def on_order(self, wc_context, order):
        if order.status != OrderStatus.Error and self.persistent:
            order_dict = msg_utils.object_as_dict(order)
            order_dict["strategy_id"] = self.ctx.get_current_strategy_index()
            self.ctx.db.add_order(**order_dict)
        self._on_order(self.ctx, order)

    def on_position(self, wc_context, position):
        self._on_position(self.ctx, position)

    def on_order_action_error(self, wc_context, error):
        self._on_order_action_error(self.ctx, error)

    def on_union_response(self, wc_context, sub_msg):
        self._on_union_response(self.ctx, sub_msg)

    def on_trade(self, wc_context, trade):
        self._on_trade(self.ctx, trade)

    def on_factor(self, wc_context, symbol, timestamp, values):
        """因子回調 - 接收 hf-live 計算的因子值"""
        self._on_factor(self.ctx, symbol, timestamp, values)

    def __insert_order(self, symbol, inst_type, exchange, account, limit_price, volume, order_type, side,
                       time=TimeCondition.GTC, position_side=Direction.Long, reduce_only=False):
        if self.ctx.backtest:
            order_id = self.bt.insert_order(
                symbol, inst_type, exchange, account, limit_price,
                volume, order_type, side, time, position_side, reduce_only)
        else:
            order_id = self.wc_context.insert_order(
                symbol, inst_type, exchange, account, limit_price,
                volume, order_type, side, time, position_side, reduce_only)
        if order_id > 0:
            order_input = pywingchun.OrderInput()
            order_input.strategy_id = self.ctx.get_current_strategy_index()
            order_input.order_id = order_id
            order_input.symbol = symbol
            order_input.instrument_type = inst_type
            order_input.exchange_id = exchange
            order_input.account_id = account
            order_input.source_id = self.ctx.account_source[account]
            order_input.price = limit_price
            order_input.stop_price = limit_price
            order_input.volume = volume
            order_input.order_type = order_type
            order_input.side = side
            order_input.time_condition = time
            order_input.position_side = position_side
            order_input.reduce_only = reduce_only
            book = self.__get_account_book(
                self.ctx.account_source[account], account)
            book.on_order_input(order_input)
        return order_id

    def __cancel_order(self, account, order_id, symbol, ex_order_id, instrument_type) -> int:
        book = self.__get_account_book(self.ctx.account_source[account], account)
        if order_id not in book._active_orders.keys():
            return 0
        if self.ctx.backtest:
            self.bt.cancel_order(
                account, order_id, symbol, ex_order_id, instrument_type)
        else:
            self.wc_context.cancel_order(
                account, order_id, symbol, ex_order_id, instrument_type)
        order = book._active_orders[order_id]
        order['status'] = OrderStatus.Pending
        order['update_time'] = time.time()
        return int(order_id)

    def __reload_config(self):
        s = self.ctx.strategies[self.ctx.get_current_strategy_index()]
        try:
            with open(s.config_path) as f:
                str_conf = json.load(f)
                s.config = str_conf
        except Exception as e:
            print(f"invalid config file {str(e)}: {s.config_path}")
