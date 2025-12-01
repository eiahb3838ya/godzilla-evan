'''
This is source code modified under the Apache License 2.0.
Original Author: Keren Dong
Modifier: kx@godzilla.dev
Modification date: March 3, 2025
'''
import os
import time
import json
from .models import *
from . import object_as_dict
from . import make_url
from itertools import groupby
import pyyjj
from kungfu.yijinjing.journal import make_location_from_dict
import kungfu.wingchun.utils as wc_utils
import kungfu.wingchun.msg as wc_msg
import kungfu.wingchun.constants as wc_constants
import kungfu.wingchun.book as kwb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import not_


class SessionFactoryHolder:
    def __init__(self, location, filename):
        if location is None:
            self.engine = create_engine(f"sqlite:///{filename}")
        else:
            self.engine = create_engine(make_url(location, filename))
        # SQLite defaults to autocommit mode, no need to set isolation_level
        # self.engine = self.engine.execution_options(isolation_level="AUTOCOMMIT")
        self.session_factory = sessionmaker(bind=self.engine)


class AccountsDB(SessionFactoryHolder):
    def __init__(self, location, filename):
        super(AccountsDB, self).__init__(location, filename)
        Account.metadata.create_all(self.engine)

    def get_accounts(self):
        with session_scope(self.session_factory) as session:
            return [object_as_dict(obj) for obj in session.query(Account).all()]

    def find_account(self, account_id):
        with session_scope(self.session_factory) as session:
            account = session.query(Account).filter(Account.account_id == account_id).first()
            return object_as_dict(account) if account is not None else {}

    def list_source_accounts(self, source_name):
        with session_scope(self.session_factory) as session:
            accounts = session.query(Account).filter(Account.source_name == source_name)
            return [object_as_dict(obj) for obj in accounts]

    def get_td_account_config(self, source_name, account_id):
        with session_scope(self.session_factory) as session:
            account = session.query(Account). \
                filter(Account.source_name == source_name). \
                filter(Account.account_id == account_id).first()
            if account:
                config_value = account.config
                result = json.dumps(config_value)
                return result
            else:
                return ""

    def get_md_account_config(self, source_name):
        with session_scope(self.session_factory) as session:
            account = session.query(Account). \
                filter(Account.source_name == source_name). \
                filter(Account.receive_md).first()
            return json.dumps(object_as_dict(account)['config'])

    def reset_receive_md(self, source_name):
        with session_scope(self.session_factory) as session:
            for obj in session.query(Account).filter(Account.source_name == source_name):
                obj.receive_md = False
                session.merge(obj)

    def add_account(self, **kwargs):
        with session_scope(self.session_factory) as session:
            session.merge(Account(**kwargs))

    def delete_account(self, account_id):
        with session_scope(self.session_factory) as session:
            session.query(Account).filter(Account.account_id == account_id).delete()


class CalendarDB(SessionFactoryHolder):
    def __init__(self, location, filename):
        super(CalendarDB, self).__init__(location, filename)
        Holiday.metadata.create_all(self.engine)

    def get_holidays(self, region=wc_constants.Region.CN):
        with session_scope(self.session_factory) as session:
            return [obj.holiday for obj in session.query(Holiday).filter(Holiday.region == region).all()]


# this table is stored in /dev/shm
class MarketInfoDB(SessionFactoryHolder):
    def __init__(self, location, filename):
        super(MarketInfoDB, self).__init__(None, filename)
        MarketInfo.metadata.create_all(self.engine)

    def get_market_info(self, symbol, exchange_id, instrument_type=wc_constants.InstrumentType.Spot):
        with session_scope(self.session_factory) as session:
            info = session.query(MarketInfo).filter(
                MarketInfo.symbol == symbol,
                MarketInfo.exchange_id == exchange_id,
                MarketInfo.instrument_type == int(instrument_type)).first()
            return info.config

    def update_market_info(self, symbol, exchange_id, config):
        with session_scope(self.session_factory) as session:
            info = MarketInfo(symbol=symbol, exchange_id=exchange_id, config=config, update_time=time.time())
            session.merge(info)


class AlgoDB(SessionFactoryHolder):
    def __init__(self, location, filename):
        super(AlgoDB, self).__init__(location, filename)
        AlgoOrder.metadata.create_all(self.engine)

    def add_order(self, **kwargs):
        with session_scope(self.session_factory) as session:
            order = AlgoOrder(**kwargs)
            session.merge(order)

    def all_active_orders(self):
        with session_scope(self.session_factory) as session:
            orders = session.query(AlgoOrder).filter(AlgoOrder.active == True).all()
            return [object_as_dict(order) for order in orders]


class LedgerDB(SessionFactoryHolder):
    def __init__(self, location, filename):
        super(LedgerDB, self).__init__(location, filename)
        Base.metadata.create_all(self.engine)
        self.location = location
        self.filename = filename

    def get_location(self, ctx, uid):
        with session_scope(self.session_factory) as session:
            info = session.query(Location).get(uid).info
            location = make_location_from_dict(ctx, info)
            return location

    def add_location(self, location):
        with session_scope(self.session_factory) as session:
            if not session.query(Location).get(location.uid):
                info = {"uid": location.uid, "uname": location.uname, "mode": pyyjj.get_mode_name(location.mode),
                        "category": pyyjj.get_category_name(location.category), "group": location.group,
                        "name": location.name}
                loc_obj = Location(uid=location.uid, info=info)
                session.add(loc_obj)

    def on_book_event(self, event):
        with session_scope(self.session_factory) as session:
            cls = self.get_model_cls(event["msg_type"])
            obj = cls(**event["data"])
            session.merge(obj)

    def add_trade(self, **kwargs):
        with session_scope(self.session_factory) as session:
            session.add(Trade(**kwargs))

    def add_order(self, **kwargs):
        with session_scope(self.session_factory) as session:
            session.merge(Order(**kwargs))
            #TODO, need backup order?
            if not os.getenv("KEEP_INACTIVE_ORDER"):
                session.query(Order).filter(Order.status!=1, Order.status!=7).delete()

    def get_order(self, order_id):
        with session_scope(self.session_factory) as session:
            order = session.query(Order).get(str(order_id))
            return {} if order is None else object_as_dict(order)

    def get_strategy_orders(self, strategy_id, is_active=True):
        with session_scope(self.session_factory) as session:
            records = session.query(Order).filter(Order.strategy_id==strategy_id).all()
            orders = []
            for o in records:
                if is_active and o.status != wc_constants.OrderStatus.Submitted \
                    and o.status != wc_constants.OrderStatus.PartialFilledActive \
                    and o.status != wc_constants.OrderStatus.PreSend:
                    continue
                orders.append(o)
            return [object_as_dict(order) for order in orders]

    def cancel_strategy_orders(self, strategy_id):
        with session_scope(self.session_factory) as session:
            records = session.query(Order).filter(Order.strategy_id==strategy_id).all()
            for o in records:
                if o.status == wc_constants.OrderStatus.Submitted or o.status == wc_constants.OrderStatus.PartialFilledActive:
                    o.status = wc_constants.OrderStatus.Cancelled
                    session.add(o)
            # session.commit()

    def mark_orders_status_unknown(self, source_id, account_id):
        with session_scope(self.session_factory) as session:
            orders = session.query(Order).filter(Order.account_id == account_id).all()
            pending_orders = []
            for order in orders:
                if order.status not in wc_constants.AllFinalOrderStatus:
                    order.status = wc_constants.OrderStatus.Unknown
                    pending_orders.append(order)
            return [object_as_dict(order) for order in pending_orders]

    def get_commission(self, account_id, symbol, exchange_id):
        pass

    def set_instruments(self, instruments):
        with session_scope(self.session_factory) as session:
            session.query(FutureInstrument).delete()
            objects = [FutureInstrument(**inst) for inst in instruments]
            session.bulk_save_objects(objects)

    def get_instrument_info(self, symbol):
        with session_scope(self.session_factory) as session:
            obj = session.query(FutureInstrument).filter(FutureInstrument.symbol == symbol).first()
            return {} if obj is None else object_as_dict(obj)

    def all_instrument_infos(self):
        with session_scope(self.session_factory) as session:
            objs = session.query(FutureInstrument).all()
            return [object_as_dict(obj) for obj in objs]

    def _drop_book(self, session, uid):
        for cls in [Asset, Position]:
            session.query(cls).filter(cls.holder_uid == uid).delete()

    def load_book(self, ctx, location):
        with session_scope(self.session_factory) as session:
            args = {}
            assets = session.query(Asset).filter(Asset.holder_uid == location.uid).all()
            args["assets"] = [object_as_dict(a) for a in assets]
            orders = session.query(Order).filter(
                Order.source_id == location.group, Order.account_id == location.name).all()
            args["orders"] = list(filter(lambda o: o["status"] in [wc_constants.OrderStatus.Submitted,
                                                                   wc_constants.OrderStatus.PreSend,
                                                                   wc_constants.OrderStatus.PartialFilledActive],
                                         [object_as_dict(o) for o in orders]))
            pos_objs = session.query(Position).filter(Position.holder_uid == location.uid).all()
            args.update({"positions": [object_as_dict(pos) for pos in pos_objs]})
            book = kwb.book.AccountBook(ctx=ctx, location=location, **args)
            return book

    def strategy_load_book(self, ctx, location, strategy_id):
        with session_scope(self.session_factory) as session:
            args = {}
            assets = session.query(Asset).filter(Asset.strategy_id == strategy_id).all()
            args["assets"] = [object_as_dict(a) for a in assets]
            orders = session.query(Order).filter(Order.strategy_id == strategy_id).all()
            args["orders"] = list(filter(lambda o: o["status"] in [wc_constants.OrderStatus.Submitted,
                                                                   wc_constants.OrderStatus.PreSend,
                                                                   wc_constants.OrderStatus.PartialFilledActive],
                                         [object_as_dict(o) for o in orders]))
            pos_objs = session.query(Position).filter(Position.strategy_id == strategy_id).all()
            args.update({"positions": [object_as_dict(pos) for pos in pos_objs]})
            book = kwb.book.AccountBook(ctx=ctx, location=location, **args)
            return book

    def dump_book(self, book):
        with session_scope(self.session_factory) as session:
            self._drop_book(session, book.location.uid)
            events = [book.make_asset_event(asset) for asset in book.assets] + [position.event for position in
                                                                                book.positions]
            objects = []
            for e in events:
                e_dict = e.as_dict()
                cls = self.get_model_cls(e_dict["msg_type"])
                obj = cls(**e_dict["data"])
                objects.append(obj)
            session.bulk_save_objects(objects)

    def remove_book(self, uid):
        with session_scope(self.session_factory) as session:
            self._drop_book(session, uid)

    def get_model_cls(self, msg_type):
        if msg_type == wc_msg.Asset:
            return Asset
        elif msg_type == wc_msg.Position:
            return Position
        elif msg_type == wc_msg.AssetSnapshot:
            return AssetSnapshot
        else:
            raise ValueError('could not find class for msg_type {}'.format(msg_type))
