/**
 * This is source code modified under the Apache License 2.0.
 * Original Author: Keren Dong
 * Modifier: kx@godzilla.dev
 * Modification date: March 3, 2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <kungfu/yijinjing/common.h>
#include <kungfu/wingchun/msg.h>
#include <kungfu/wingchun/common.h>
#include <kungfu/wingchun/commander.h>
#include <kungfu/wingchun/broker/marketdata.h>
#include <kungfu/wingchun/broker/trader.h>
#include <kungfu/wingchun/service/ledger.h>
#include <kungfu/wingchun/service/algo.h>
#include <kungfu/wingchun/service/bar.h>
#include <kungfu/wingchun/strategy/context.h>
#include <kungfu/wingchun/strategy/runner.h>
#include <kungfu/wingchun/book/book.h>
#include <kungfu/wingchun/algo/algo.h>
#include <kungfu/wingchun/strategy/metric_controller.h>

namespace py = pybind11;
namespace kwb = kungfu::wingchun::book;
using namespace kungfu::yijinjing;
using namespace kungfu::wingchun;
using namespace kungfu::wingchun::broker;
using namespace kungfu::wingchun::service;
using namespace kungfu::wingchun::msg::data;

class PyMarketData: public MarketData
{
public:
    using MarketData::MarketData;
    bool subscribe(const std::vector<Instrument> &instruments) override
    { PYBIND11_OVERLOAD_PURE(bool, MarketData, subscribe, instruments); }
    bool subscribe_trade(const std::vector<Instrument> &instruments) override
    { PYBIND11_OVERLOAD_PURE(bool, MarketData, subscribe_trade, instruments); }
    bool subscribe_ticker(const std::vector<Instrument> &instruments) override
    { PYBIND11_OVERLOAD_PURE(bool, MarketData, subscribe_ticker, instruments); }
    bool subscribe_index_price(const std::vector<Instrument> &instruments) override
    { PYBIND11_OVERLOAD_PURE(bool, MarketData, subscribe_index_price, instruments); }
    bool subscribe_all() override
    { PYBIND11_OVERLOAD_PURE(bool, MarketData, subscribe_all); }
    bool unsubscribe(const std::vector<Instrument> &instruments) override
    { PYBIND11_OVERLOAD_PURE(bool, MarketData, unsubscribe,instruments); }
    void on_start() override
    {PYBIND11_OVERLOAD(void, MarketData, on_start, );}
};

class PyTrader: public Trader
{
public:
    using Trader::Trader;
    AccountType get_account_type() const override
    { PYBIND11_OVERLOAD_PURE(const AccountType, Trader, get_account_type,); }
    bool insert_order(const kungfu::yijinjing::event_ptr &event) override
    { PYBIND11_OVERLOAD_PURE(bool, Trader, insert_order, event); }
    bool query_order(const kungfu::yijinjing::event_ptr &event) override
    { PYBIND11_OVERLOAD_PURE(bool, Trader, query_order, event); }
    bool cancel_order(const kungfu::yijinjing::event_ptr &event) override
    { PYBIND11_OVERLOAD_PURE(bool, Trader, cancel_order, event); }
    bool adjust_leverage(const kungfu::yijinjing::event_ptr &event) override
    { PYBIND11_OVERLOAD_PURE(bool, Trader, adjust_leverage, event); }
    bool req_position() override
    { PYBIND11_OVERLOAD_PURE(bool, Trader, req_position,); }
    bool req_account() override
    { PYBIND11_OVERLOAD_PURE(bool, Trader, req_account,); }
    void on_start() override
    {PYBIND11_OVERLOAD(void, Trader, on_start, );}
};

class PyBook: public kwb::Book
{
public:
    using kwb::Book::Book;
    void on_depth(event_ptr event, const Depth &depth) override
    {PYBIND11_OVERLOAD_PURE(void, kwb::Book, on_depth, event, depth); }
    void on_order_input(event_ptr event, const OrderInput &input) override
    {PYBIND11_OVERLOAD_PURE(void, kwb::Book, on_order_input, event, input); }
    void on_order(event_ptr event, const Order &order) override
    {PYBIND11_OVERLOAD_PURE(void, kwb::Book, on_order, event, order); }
    void on_trade(event_ptr event, const MyTrade &trade) override
    {PYBIND11_OVERLOAD_PURE(void, kwb::Book, on_trade, event, trade); }
    void on_position(event_ptr event, const Position& position) override
    {PYBIND11_OVERLOAD_PURE(void, kwb::Book, on_position, position); }
    virtual void on_asset(event_ptr event, const Asset& asset) override
    {PYBIND11_OVERLOAD_PURE(void, kwb::Book, on_asset, event, asset); }
};

class PyAlgoOrder: public algo::AlgoOrder
{
    using algo::AlgoOrder::AlgoOrder;

    const std::string dumps() const override
    {PYBIND11_OVERLOAD_PURE(const std::string, algo::AlgoOrder, dumps); }

    void on_start(algo::AlgoContext_ptr context) override
    {PYBIND11_OVERLOAD(void, algo::AlgoOrder, on_start, context); }

    void on_stop(algo::AlgoContext_ptr context) override
    {PYBIND11_OVERLOAD(void, algo::AlgoOrder, on_stop, context); }

    void on_child_order(algo::AlgoContext_ptr context, const Order& order) override
    {PYBIND11_OVERLOAD(void, algo::AlgoOrder, on_child_order, context, order); }

    void on_child_trade(algo::AlgoContext_ptr context, const Trade& trade) override
    {PYBIND11_OVERLOAD(void, algo::AlgoOrder, on_child_trade, context, trade); }

    void on_order_report(algo::AlgoContext_ptr context, const std::string& report_msg) override
    {PYBIND11_OVERLOAD(void, algo::AlgoOrder, on_order_report, context, report_msg); }

};

class PyAlgoService: public service::Algo
{
    using service::Algo::Algo;
    void insert_order(const event_ptr &event, const std::string& msg) override
    {PYBIND11_OVERLOAD_PURE(void, service::Algo, insert_order, event, msg) }
    void cancel_order(const event_ptr &event, const OrderAction& action) override
    {PYBIND11_OVERLOAD_PURE(void, service::Algo, cancel_order, event, action) }
    void modify_order(const event_ptr &event, const std::string& msg) override
    {PYBIND11_OVERLOAD_PURE(void, service::Algo, modify_order, event, msg) }
};

class PyLedger : public Ledger
{
public:
    using Ledger::Ledger;

    std::string handle_request(const event_ptr &event, const std::string &msg) override
    {PYBIND11_OVERLOAD_PURE(std::string, Ledger, handle_request, event, msg) }

    void handle_instrument_request(const event_ptr &event) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, handle_instrument_request, event) }

    void handle_asset_request(const event_ptr &event, const data::location_ptr &app_location) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, handle_asset_request, event, app_location) }

    void on_app_location(int64_t trigger_time, const data::location_ptr &app_location) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, on_app_location, trigger_time, app_location) }

    void on_depth(event_ptr event, const Depth &depth) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, on_depth, event, depth) }

    void on_trade(event_ptr event, const Trade &trade) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, on_trade, event, trade) }

    void on_order(event_ptr event, const Order &order) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, on_order, event, order) }

    void on_transaction(event_ptr event, const MyTrade &trade) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, on_transaction, event, trade) }

    void on_instruments(const std::vector<Instrument> &instruments) override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, on_instruments, instruments) }

    void pre_start() override
    {PYBIND11_OVERLOAD_PURE(void, Ledger, pre_start) }
};

class PyRunner : public strategy::Runner
{
public:
    using strategy::Runner::Runner;
};

class PyStrategy : public strategy::Strategy
{
public:
    using strategy::Strategy::Strategy; // Inherit constructors

    void pre_start(strategy::Context_ptr context) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, pre_start, context); }

    void post_start(strategy::Context_ptr context) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, post_start, context); }

    void pre_stop(strategy::Context_ptr context) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, pre_stop, context); }

    void post_stop(strategy::Context_ptr context) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, post_stop, context); }

    void on_depth(strategy::Context_ptr context, const Depth &depth) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_depth, context, depth); }

    void on_ticker(strategy::Context_ptr context, const Ticker &ticker) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_ticker, context, ticker); }

    void on_index_price(strategy::Context_ptr context, const IndexPrice &ip) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_index_price, context, ip); }

    void on_transaction(strategy::Context_ptr context, const MyTrade &transaction) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_transaction, context, transaction); }

    void on_order(strategy::Context_ptr context, const Order &order) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_order, context, order); }

    void on_position(strategy::Context_ptr context, const Position &position) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_position, context, position); }

    void on_union_response(strategy::Context_ptr context, const std::string &msg) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_union_response, context, msg); }

    void on_order_action_error(strategy::Context_ptr context, const OrderActionError &error) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_order_action_error, context, error); }

    void on_trade(strategy::Context_ptr context, const Trade &trade) override
    {PYBIND11_OVERLOAD(void, strategy::Strategy, on_trade, context, trade); }

    void on_factor(strategy::Context_ptr context, const std::string &symbol, long long timestamp, const std::vector<double> &values) override
    {
        py::gil_scoped_acquire acquire;
        py::list py_values;
        for (const auto& v : values) {
            py_values.append(v);
        }
        PYBIND11_OVERLOAD(void, strategy::Strategy, on_factor, context, symbol, timestamp, py_values);
    }

};

PYBIND11_MODULE(pywingchun, m)
{
    auto m_utils = m.def_submodule("utils");
//     m_utils.def("get_symbol_id", &kungfu::wingchun::get_symbol_id);
    m_utils.def("get_symbol_id", py::overload_cast<const std::string &, const std::string &, InstrumentType, const std::string &, uint32_t>(&kungfu::wingchun::get_symbol_id));
    m_utils.def("is_valid_price", &kungfu::wingchun::is_valid_price);
    m_utils.def("is_final_status", &kungfu::wingchun::is_final_status);
    m_utils.def("get_shm_db", &kungfu::wingchun::get_shm_db);
    m_utils.def("order_from_input", [](const kungfu::wingchun::msg::data::OrderInput &input)
    {
        kungfu::wingchun::msg::data::Order order = {};
        kungfu::wingchun::msg::data::order_from_input(input, order);
        return order;
    });

    auto m_constants = m.def_submodule("constants");

    py::enum_<kungfu::wingchun::InstrumentType>(m_constants, "InstrumentType", py::arithmetic())
            .value("Unknown", kungfu::wingchun::InstrumentType::Unknown)
            .value("FFuture", kungfu::wingchun::InstrumentType::FFuture)
            .value("Future", kungfu::wingchun::InstrumentType::Future)
            .value("DFuture", kungfu::wingchun::InstrumentType::DFuture)
            .value("Swap", kungfu::wingchun::InstrumentType::Swap)
            .value("Spot", kungfu::wingchun::InstrumentType::Spot)
            .value("Index", kungfu::wingchun::InstrumentType::Index)
            .value("Etf", kungfu::wingchun::InstrumentType::Etf)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::InstrumentType &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::ExecType>(m_constants, "ExecType", py::arithmetic())
            .value("Unknown", kungfu::wingchun::ExecType::Unknown)
            .value("Cancel", kungfu::wingchun::ExecType::Cancel)
            .value("Trade", kungfu::wingchun::ExecType::Trade)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::ExecType &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::Side>(m_constants, "Side", py::arithmetic())
            .value("Buy", kungfu::wingchun::Side::Buy)
            .value("Sell", kungfu::wingchun::Side::Sell)
            .value("Lock", kungfu::wingchun::Side::Lock)
            .value("Unlock", kungfu::wingchun::Side::Unlock)
            .value("Exec", kungfu::wingchun::Side::Exec)
            .value("Drop", kungfu::wingchun::Side::Drop)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::Side &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::Offset>(m_constants, "Offset", py::arithmetic())
            .value("Open", kungfu::wingchun::Offset::Open)
            .value("Close", kungfu::wingchun::Offset::Close)
            .value("CloseToday", kungfu::wingchun::Offset::CloseToday)
            .value("CloseYesterday", kungfu::wingchun::Offset::CloseYesterday)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::Offset &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::BsFlag>(m_constants, "BsFlag", py::arithmetic())
            .value("Unknown", kungfu::wingchun::BsFlag::Unknown)
            .value("Buy", kungfu::wingchun::BsFlag::Buy)
            .value("Sell", kungfu::wingchun::BsFlag::Sell)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::BsFlag &a, int b)
                {
                    return static_cast<int>(a) == b;
                });

    py::enum_<kungfu::wingchun::OrderStatus>(m_constants, "OrderStatus", py::arithmetic())
            .value("Unknown", kungfu::wingchun::OrderStatus::Unknown)
            .value("Submitted", kungfu::wingchun::OrderStatus::Submitted)
            .value("Pending", kungfu::wingchun::OrderStatus::Pending)
            .value("Cancelled", kungfu::wingchun::OrderStatus::Cancelled)
            .value("Error", kungfu::wingchun::OrderStatus::Error)
            .value("Filled", kungfu::wingchun::OrderStatus::Filled)
            .value("PartialFilledNotActive", kungfu::wingchun::OrderStatus::PartialFilledNotActive)
            .value("PartialFilledActive", kungfu::wingchun::OrderStatus::PartialFilledActive)
            .value("PreSend", kungfu::wingchun::OrderStatus::PreSend)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::OrderStatus &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::Direction>(m_constants, "Direction", py::arithmetic())
            .value("Long", kungfu::wingchun::Direction::Long)
            .value("Short", kungfu::wingchun::Direction::Short)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::Direction &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::OrderType>(m_constants, "OrderType", py::arithmetic())
            .value("Unknown", kungfu::wingchun::OrderType::UnKnown)
            .value("Limit", kungfu::wingchun::OrderType::Limit)
            .value("Market", kungfu::wingchun::OrderType::Market)
            .value("Mock", kungfu::wingchun::OrderType::Mock)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::OrderType &a, int b) {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::VolumeCondition>(m_constants, "VolumeCondition", py::arithmetic())
            .value("Any", kungfu::wingchun::VolumeCondition::Any)
            .value("Min", kungfu::wingchun::VolumeCondition::Min)
            .value("All", kungfu::wingchun::VolumeCondition::All)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::VolumeCondition &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::TimeCondition>(m_constants, "TimeCondition", py::arithmetic())
            .value("IOC", kungfu::wingchun::TimeCondition::IOC)
            .value("FOK", kungfu::wingchun::TimeCondition::FOK)
            .value("GTC", kungfu::wingchun::TimeCondition::GTC)
            .value("GTX", kungfu::wingchun::TimeCondition::GTX)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::TimeCondition &a, int b)
                {
                    return static_cast<int>(a) == b;
                }
            );

    py::enum_<kungfu::wingchun::OrderActionFlag>(m_constants, "OrderActionFlag", py::arithmetic())
            .value("Cancel", kungfu::wingchun::OrderActionFlag::Cancel)
            .value("Query", kungfu::wingchun::OrderActionFlag::Query)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::OrderActionFlag &a, int b)
                {
                    return static_cast<int>(a) == b;
                });

    py::enum_<kungfu::wingchun::LedgerCategory>(m_constants, "LedgerCategory", py::arithmetic())
            .value("Account", kungfu::wingchun::LedgerCategory::Account)
            .value("Strategy", kungfu::wingchun::LedgerCategory::Strategy)
            .export_values()
            .def("__eq__",
                [](const kungfu::wingchun::LedgerCategory &a, int b)
                {
                    return static_cast<int>(a) == b;
                });

    py::class_<Instrument>(m, "Instrument")
            .def(py::init<>())
            .def_readwrite("instrument_type", &Instrument::instrument_type)
            .def_property("symbol", &Instrument::get_symbol, &Instrument::set_symbol)
            .def_property("exchange_id", &Instrument::get_exchange_id, &Instrument::set_exchange_id)
            .def_readwrite("contract_multiplier", &Instrument::contract_multiplier)
            .def_readwrite("price_tick", &Instrument::price_tick)
            .def_readwrite("delivery_year", &Instrument::delivery_year)
            .def_readwrite("delivery_month", &Instrument::delivery_month)
            .def_readwrite("is_trading", &Instrument::is_trading)
            .def_readwrite("long_margin_ratio", &Instrument::long_margin_ratio)
            .def_readwrite("short_margin_ratio", &Instrument::short_margin_ratio)
            .def_property("product_id", &Instrument::get_product_id, &Instrument::set_product_id)
            .def_property("open_date", &Instrument::get_open_date, &Instrument::set_open_date)
            .def_property("create_date", &Instrument::get_create_date, &Instrument::set_create_date)
            .def_property("expire_date", &Instrument::get_expire_date, &Instrument::set_expire_date)
            .def_property_readonly("raw_address", [](const Instrument &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Instrument*>(addr); })
            .def("__sizeof__", [](const Instrument &a) { return sizeof(a); })
            .def("__hash__", [](const Instrument &a) { return get_symbol_id(a.get_symbol(), a.get_exchange_id());})
            .def("__eq__",[](const Instrument &a, const Instrument &b){ return strcmp(a.symbol, b.symbol) == 0 && strcmp(a.exchange_id, b.exchange_id) == 0;})
            .def("__repr__",[](const Instrument &a){return to_string(a);});

    py::class_<Depth>(m, "Depth")
            .def(py::init<>())
            .def_property("source_id", &Depth::get_source_id, &Depth::set_source_id)
            .def_readwrite("data_time", &Depth::data_time)
            .def_property("symbol", &Depth::get_symbol, &Depth::set_symbol)
            .def_property("exchange_id", &Depth::get_exchange_id, &Depth::set_exchange_id)
            .def_readwrite("instrument_type", &Depth::instrument_type)
            .def_property("bid_price", &Depth::get_bid_price, &Depth::set_bid_price)
            .def_property("ask_price", &Depth::get_ask_price, &Depth::set_ask_price)
            .def_property("bid_volume", &Depth::get_bid_volume, &Depth::set_bid_volume)
            .def_property("ask_volume", &Depth::get_ask_volume, &Depth::set_ask_volume)
            .def_property_readonly("raw_address", [](const Depth &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Depth*>(addr); })
            .def("__sizeof__", [](const Depth &a) { return sizeof(a); })
            .def("__repr__",[](const Depth &a){return to_string(a);});

    py::class_<Ticker>(m, "Ticker")
            .def(py::init<>())
            .def_property("source_id", &Ticker::get_source_id, &Ticker::set_source_id)
            .def_property("symbol", &Ticker::get_symbol, &Ticker::set_symbol)
            .def_property("exchange_id", &Ticker::get_exchange_id, &Ticker::set_exchange_id)
            .def_readwrite("data_time", &Ticker::data_time)
            .def_readwrite("instrument_type", &Ticker::instrument_type)
            .def_readwrite("bid_price", &Ticker::bid_price)
            .def_readwrite("ask_price", &Ticker::ask_price)
            .def_readwrite("bid_volume", &Ticker::bid_volume)
            .def_readwrite("ask_volume", &Ticker::ask_volume)
            .def_property_readonly("raw_address", [](const Ticker &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Ticker*>(addr); })
            .def("__sizeof__", [](const Ticker &a) { return sizeof(a); })
            .def("__repr__",[](const Ticker &a){return to_string(a);});

    py::class_<Trade>(m, "Trade")
            .def(py::init<>())
        //     .def_property("source_id", &Trade::get_source_id, &Trade::set_source_id)
            .def_readwrite("trade_time", &Trade::trade_time)
            .def_property("client_id", &Trade::get_client_id, &Trade::set_client_id)
            .def_property("symbol", &Trade::get_symbol, &Trade::set_symbol)
            .def_property("exchange_id", &Trade::get_exchange_id, &Trade::set_exchange_id)
            .def_readwrite("instrument_type", &Trade::instrument_type)
            .def_readwrite("price", &Trade::price)
            .def_readwrite("volume", &Trade::volume)
            .def_readwrite("side", &Trade::side)
            .def_readwrite("position_side", &Trade::position_side)
            .def_readwrite("ask_id", &Trade::ask_id)
            .def_readwrite("bid_id", &Trade::bid_id)
            .def_readwrite("trade_id", &Trade::trade_id)
            .def_property_readonly("raw_address", [](const Trade &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Trade*>(addr); })
            .def("__sizeof__", [](const Trade &a) { return sizeof(a); })
            .def("__repr__",[](const Trade &a){return to_string(a);});

    py::class_<IndexPrice>(m, "IndexPrice")
            .def(py::init<>())
            .def_property("symbol", &IndexPrice::get_symbol, &IndexPrice::set_symbol)
            .def_property("exchange_id", &IndexPrice::get_exchange_id, &IndexPrice::set_exchange_id)
            .def_readwrite("instrument_type", &IndexPrice::instrument_type)
            .def_readwrite("price", &IndexPrice::price)
            .def_property_readonly("raw_address", [](const IndexPrice &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<IndexPrice*>(addr); })
            .def("__sizeof__", [](const IndexPrice &a) { return sizeof(a); })
            .def("__repr__",[](const IndexPrice &a){return to_string(a);});

    py::class_<Bar>(m, "Bar")
            .def(py::init<>())
            .def_property("symbol", &Bar::get_symbol, &Bar::set_symbol)
            .def_property("exchange_id", &Bar::get_exchange_id, &Bar::set_exchange_id)
            .def_readwrite("start_time", &Bar::start_time)
            .def_readwrite("end_time", &Bar::end_time)
            .def_readwrite("open", &Bar::open)
            .def_readwrite("close", &Bar::close)
            .def_readwrite("high", &Bar::high)
            .def_readwrite("low", &Bar::low)
            .def_readwrite("volume", &Bar::volume)
            .def_readwrite("start_volume", &Bar::start_volume)
            .def_readwrite("trade_count", &Bar::trade_count)
            .def_readwrite("interval", &Bar::interval)
            .def_property_readonly("raw_address", [](const Bar &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Bar*>(addr); })
            .def("__sizeof__", [](const Bar &a) { return sizeof(a); })
            .def("__repr__",[](const Bar &a){return to_string(a);});

    py::class_<OrderInput>(m, "OrderInput")
            .def(py::init<>())
            .def_readwrite("strategy_id", &OrderInput::strategy_id)
            .def_readwrite("order_id", &OrderInput::order_id)
            .def_readwrite("instrument_type", &OrderInput::instrument_type)
            .def_readwrite("price", &OrderInput::price)
            .def_readwrite("stop_price", &OrderInput::stop_price)
            .def_readwrite("volume", &OrderInput::volume)
            .def_readwrite("side", &OrderInput::side)
            .def_readwrite("time_condition", &OrderInput::time_condition)
            .def_readwrite("position_side", &OrderInput::position_side)
            .def_readwrite("order_type", &OrderInput::order_type)
            .def_readwrite("reduce_only", &OrderInput::reduce_only)
            .def_property("symbol", &OrderInput::get_symbol, &OrderInput::set_symbol)
            .def_property("exchange_id", &OrderInput::get_exchange_id, &OrderInput::set_exchange_id)
            .def_property("account_id", &OrderInput::get_account_id, &OrderInput::set_account_id)
            .def_property("source_id", &OrderInput::get_source_id, &OrderInput::set_source_id)
            .def_property_readonly("raw_address", [](const OrderInput &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<OrderInput*>(addr); })
            .def("__sizeof__", [](const OrderInput &a) { return sizeof(a); })
            .def("__repr__",[](const OrderInput &a){return to_string(a);});

    py::class_<Order>(m, "Order")
            .def(py::init<>())
            .def_readwrite("strategy_id", &Order::strategy_id)
            .def_readwrite("order_id", &Order::order_id)
            .def_readwrite("insert_time", &Order::insert_time)
            .def_readwrite("update_time", &Order::update_time)
            .def_readwrite("instrument_type", &Order::instrument_type)
            .def_readwrite("price", &Order::price)
            .def_readwrite("stop_price", &Order::stop_price)
            .def_readwrite("avg_price", &Order::avg_price)
            .def_readwrite("volume", &Order::volume)
            .def_readwrite("volume_traded", &Order::volume_traded)
            .def_readwrite("volume_left", &Order::volume_left)
            .def_readwrite("close_pnl", &Order::close_pnl)
            .def_readwrite("fee", &Order::fee)
            .def_readwrite("status", &Order::status)
            .def_readwrite("time_condition", &Order::time_condition)
            .def_readwrite("side", &Order::side)
            .def_readwrite("position_side", &Order::position_side)
            .def_readwrite("order_type", &Order::order_type)
            .def_property("fee_currency", &Order::get_fee_currency, &Order::set_fee_currency)
            .def_property("symbol", &Order::get_symbol, &Order::set_symbol)
            .def_property("ex_order_id", &Order::get_ex_order_id, &Order::set_ex_order_id)
            .def_property("exchange_id", &Order::get_exchange_id, &Order::set_exchange_id)
            .def_property("account_id", &Order::get_account_id, &Order::set_account_id)
            .def_property("source_id", &Order::get_source_id, &Order::set_source_id)
            .def_property("error_code", &Order::get_error_code, &Order::set_error_code)
            .def_property_readonly("active", [](const Order& o) { return not is_final_status(o.status); })
            .def_property_readonly("raw_address", [](const Order &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Order*>(addr); })
            .def("__sizeof__", [](const Order &a) { return sizeof(a); })
            .def("__repr__",[](const Order &a){return to_string(a);});

    py::class_<OrderAction>(m, "OrderAction")
            .def(py::init<>())
            .def_readwrite("strategy_id", &OrderAction::strategy_id)
            .def_readwrite("order_id", &OrderAction::order_id)
            .def_readwrite("order_action_id", &OrderAction::order_action_id)
            .def_property("ex_order_id", &OrderAction::get_ex_order_id, &OrderAction::set_ex_order_id)
            .def_readwrite("instrument_type", &OrderAction::instrument_type)
            .def_readwrite("action_flag", &OrderAction::action_flag)
            .def_property_readonly("raw_address", [](const OrderAction &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<OrderAction*>(addr); })
            .def("__sizeof__", [](const OrderAction &a) { return sizeof(a); })
            .def("__repr__",[](const OrderAction &a){return to_string(a);});

    py::class_<OrderActionError>(m, "OrderActionError")
            .def(py::init<>())
            .def_readwrite("order_id", &OrderActionError::order_id)
            .def_readwrite("order_action_id", &OrderActionError::order_action_id)
            .def_readwrite("error_id", &OrderActionError::error_id)
            .def_property("error_msg", &OrderActionError::get_error_msg, &OrderActionError::set_error_msg)
            .def_property_readonly("raw_address", [](const OrderActionError &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<OrderActionError*>(addr); })
            .def("__sizeof__", [](const OrderActionError &a) { return sizeof(a); })
            .def("__repr__",[](const OrderActionError &a){return to_string(a);});

    py::class_<MyTrade>(m, "MyTrade")
            .def(py::init<>())
            .def_readwrite("strategy_id", &MyTrade::strategy_id)
            .def_readwrite("trade_id", &MyTrade::trade_id)
            .def_readwrite("order_id", &MyTrade::order_id)
            .def_readwrite("trade_time", &MyTrade::trade_time)
            .def_readwrite("instrument_type", &MyTrade::instrument_type)
            .def_readwrite("side", &MyTrade::side)
            .def_readwrite("offset", &MyTrade::offset)
            .def_readwrite("price", &MyTrade::price)
            .def_readwrite("volume", &MyTrade::volume)
            .def_readwrite("fee", &MyTrade::fee)
            .def_property("fee_currency", &MyTrade::get_fee_currency, &MyTrade::set_fee_currency)
            .def_property("symbol", &MyTrade::get_symbol, &MyTrade::set_symbol)
            .def_property("exchange_id", &MyTrade::get_exchange_id, &MyTrade::set_exchange_id)
            .def_property("account_id", &MyTrade::get_account_id, &MyTrade::set_account_id)
            .def_property("ex_order_id", &MyTrade::get_ex_order_id, &MyTrade::set_ex_order_id)
            .def_property("source_id", &MyTrade::get_source_id, &MyTrade::set_source_id)
            .def_property_readonly("raw_address", [](const MyTrade &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<MyTrade*>(addr); })
            .def("__sizeof__", [](const MyTrade &a) { return sizeof(a); })
            .def("__repr__",[](const MyTrade &a){return to_string(a);});

    py::class_<Asset>(m, "Asset")
            .def(py::init<>())
            .def_readwrite("update_time", &Asset::update_time)
            .def_readwrite("avail", &Asset::avail)
            .def_readwrite("margin", &Asset::margin)
            .def_readwrite("frozen", &Asset::frozen)
            .def_readwrite("holder_uid", &Asset::holder_uid)
            .def_readwrite("ledger_category", &Asset::ledger_category)
            .def_property("coin", &Asset::get_coin, &Asset::set_coin)
            .def_property("account_id", &Asset::get_account_id, &Asset::set_account_id)
            .def_property("exchange_id", &Asset::get_exchange_id, &Asset::set_exchange_id)
            .def_property_readonly("raw_address", [](const Asset &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Asset*>(addr); })
            .def("__sizeof__", [](const Asset &a) { return sizeof(a); })
            .def("__repr__",[](const Asset &a){return to_string(a);});

    py::class_<Position>(m, "Position")
            .def(py::init<>())
            .def_readwrite("strategy_id", &Position::strategy_id)
            .def_readwrite("update_time", &Position::update_time)
            .def_readwrite("instrument_type", &Position::instrument_type)
            .def_readwrite("direction", &Position::direction)
            .def_readwrite("volume", &Position::volume)
            .def_readwrite("frozen_total", &Position::frozen_total)
            .def_readwrite("last_price", &Position::last_price)
            .def_readwrite("avg_open_price", &Position::avg_open_price)
            .def_readwrite("settlement_price", &Position::settlement_price)
            .def_readwrite("margin", &Position::margin)
            .def_readwrite("realized_pnl", &Position::realized_pnl)
            .def_readwrite("unrealized_pnl", &Position::unrealized_pnl)
            .def_readwrite("holder_uid", &Position::holder_uid)
            .def_readwrite("ledger_category", &Position::ledger_category)
            .def_property("symbol", &Position::get_symbol, &Position::set_symbol)
            .def_property("exchange_id", &Position::get_exchange_id, &Position::set_exchange_id)
            .def_property("source_id", &Position::get_source_id, &Position::set_source_id)
            .def_property("account_id", &Position::get_account_id, &Position::set_account_id)
            .def_property_readonly("raw_address", [](const Position &a) { return reinterpret_cast<uintptr_t>(&a);})
            .def("from_raw_address",[](uintptr_t addr) { return * reinterpret_cast<Position*>(addr); })
            .def("__sizeof__", [](const Position &a) { return sizeof(a); })
            .def("__repr__",[](const Position &a){return to_string(a);});

    py::class_<kwb::Book, PyBook, kwb::Book_ptr>(m, "Book")
            .def(py::init())
            .def_property_readonly("ready", &kwb::Book::is_ready)
            .def("on_depth", &kwb::Book::on_depth)
            .def("on_order_input", &kwb::Book::on_order_input)
            .def("on_order", &kwb::Book::on_order)
            .def("on_trade", &kwb::Book::on_trade)
            .def("on_position", &kwb::Book::on_position)
            .def("on_asset", &kwb::Book::on_asset)
            ;

    py::class_<kwb::BookContext, std::shared_ptr<kwb::BookContext>>(m, "BookContext")
            .def("add_book", &kwb::BookContext::add_book)
            .def("pop_book", &kwb::BookContext::pop_book)
            .def("get_inst_info", &kwb::BookContext::get_inst_info)
            ;

    py::class_<MarketData, PyMarketData, kungfu::practice::apprentice, std::shared_ptr<MarketData>>(m, "MarketData")
            .def(py::init<bool, data::locator_ptr, const std::string&>())
            .def_property_readonly("io_device", &MarketData::get_io_device)
            .def("is_live", &MarketData::is_live)
            .def("subscribe", &MarketData::subscribe)
            .def("subscribe_trade", &MarketData::subscribe_trade)
            .def("subscribe_ticker", &MarketData::subscribe_ticker)
            .def("subscribe_index_price", &MarketData::subscribe_index_price)
            .def("subscribe_all", &MarketData::subscribe_all)
            .def("unsubscribe", &MarketData::unsubscribe)
            .def("on_start", &MarketData::on_start)
            .def("add_time_interval", &MarketData::add_time_interval)
            .def("get_writer", &MarketData::get_writer)
            .def("now", &MarketData::now)
            .def("run", &MarketData::run);

    py::class_<Trader, PyTrader, kungfu::practice::apprentice, std::shared_ptr<Trader>>(m, "Trader")
            .def(py::init<bool, data::locator_ptr, const std::string&, const std::string&>())
            .def_property_readonly("io_device", &Trader::get_io_device)
            .def("on_start", &Trader::on_start)
            .def("get_writer", &Trader::get_writer)
            .def("get_account_type", &Trader::get_account_type)
            .def("add_time_interval", &Trader::add_time_interval)
            .def("insert_order", &Trader::insert_order)
            .def("query_order", &Trader::query_order)
            .def("cancel_order", &Trader::cancel_order)
            .def("adjust_leverage", &Trader::adjust_leverage)
            .def("now", &Trader::now)
            .def("run", &Trader::run);

    py::class_<Ledger, PyLedger, kungfu::practice::apprentice, std::shared_ptr<Ledger>>(m, "Ledger")
            .def(py::init<data::locator_ptr, data::mode, bool>())
            .def_property_readonly("io_device", &Ledger::get_io_device)
            .def_property_readonly("book_context", &Ledger::get_book_context)
            .def("now", &Ledger::now)
            .def("has_location", &Ledger::has_location)
            .def("get_location", &Ledger::get_location)
            .def("get_writer", &Ledger::get_writer)
            .def("publish", &Ledger::publish)
            .def("publish_broker_states", &Ledger::publish_broker_states)
            .def("new_order_single", &Ledger::new_order_single)
            .def("cancel_order", &Ledger::cancel_order)
            .def("handle_request", &Ledger::handle_request)
            .def("handle_instrument_request", &Ledger::handle_instrument_request)
            .def("handle_asset_request", &Ledger::handle_asset_request)
            .def("on_app_location", &Ledger::on_app_location)
            .def("on_depth", &Ledger::on_depth)
            .def("on_order", &Ledger::on_order)
            .def("on_trade", &Ledger::on_trade)
            .def("on_transaction", &Ledger::on_transaction)
            .def("on_instruments", &Ledger::on_instruments)
            .def("set_begin_time", &Ledger::set_begin_time)
            .def("set_end_time", &Ledger::set_end_time)
            .def("pre_start", &Ledger::pre_start)
            .def("add_timer", &Ledger::add_timer)
            .def("add_time_interval", &Ledger::add_time_interval)
            .def("run", &Ledger::run);

    py::class_<strategy::Runner, PyRunner, kungfu::practice::apprentice, std::shared_ptr<strategy::Runner>>(m, "Runner")
            .def(py::init<kungfu::yijinjing::data::locator_ptr, const std::string &, const std::string &, data::mode, bool>())
            .def("set_begin_time", &strategy::Runner::set_begin_time)
            .def("set_end_time", &strategy::Runner::set_end_time)
            .def("run", &strategy::Runner::run)
            .def("add_strategy", &strategy::Runner::add_strategy);

    py::class_<strategy::Context, std::shared_ptr<strategy::Context>>(m, "Context")
            .def_property_readonly("book_context", &strategy::Context::get_book_context)
            .def_property_readonly("algo_context", &strategy::Context::get_algo_context)
            .def("now", &strategy::Context::now)
            .def("add_timer", &strategy::Context::add_timer)
            .def("add_time_interval", &strategy::Context::add_time_interval)
            .def("get_market_info", &strategy::Context::get_market_info)
            .def("add_account", &strategy::Context::add_account)
            .def("list_accounts", &strategy::Context::list_accounts)
            .def("get_account_cash_limit", &strategy::Context::get_account_cash_limit)
            .def("set_account_cash_limit", &strategy::Context::set_account_cash_limit)
            .def("subscribe", &strategy::Context::subscribe)
            .def("unsubscribe", &strategy::Context::unsubscribe)
            .def("subscribe_trade", &strategy::Context::subscribe_trade)
            .def("subscribe_ticker", &strategy::Context::subscribe_ticker)
            .def("subscribe_index_price", &strategy::Context::subscribe_index_price)
            .def("adjust_leverage", &strategy::Context::adjust_leverage)
            .def("subscribe_all", &strategy::Context::subscribe_all)
            .def("get_current_strategy_index", &strategy::Context::get_current_strategy_index)
            .def("insert_order", &strategy::Context::insert_order)
            .def("query_order", &strategy::Context::query_order, py::arg("account"), py::arg("order_id"), py::arg("ex_order_id"), py::arg("inst_type"), py::arg("symbol")="")
            .def("merge_positions", &strategy::Context::merge_positions, py::arg("account"), py::arg("symbol"), py::arg("amount"), py::arg("inst_type")=InstrumentType::FFuture)
            .def("query_positions", &strategy::Context::query_positions, py::arg("account"), py::arg("symbol"), py::arg("inst_type")=InstrumentType::FFuture)
            .def("cancel_order", &strategy::Context::cancel_order)
            ;

    py::class_<strategy::MetricController, std::shared_ptr<strategy::MetricController>>(m, "MetricController")
            .def_static("GetInstance", &strategy::MetricController::GetInstance, py::return_value_policy::reference_internal)
            .def("update_trade_amount", &strategy::MetricController::update_trade_amount)
            .def("check_trade_amount", &strategy::MetricController::check_trade_amount)
            .def("check_order_amount", &strategy::MetricController::check_order_amount)
            ;


    py::class_<strategy::Strategy, PyStrategy, strategy::Strategy_ptr>(m, "Strategy")
            .def(py::init())
            .def("pre_start", &strategy::Strategy::pre_start)
            .def("post_start", &strategy::Strategy::post_start)
            .def("pre_stop", &strategy::Strategy::pre_stop)
            .def("post_stop", &strategy::Strategy::post_stop)
            .def("on_depth", &strategy::Strategy::on_depth)
            .def("on_ticker", &strategy::Strategy::on_ticker)
            .def("on_index_price", &strategy::Strategy::on_index_price)
            .def("on_transaction", &strategy::Strategy::on_transaction)
            .def("on_order", &strategy::Strategy::on_order)
            .def("on_trade", &strategy::Strategy::on_trade)
            .def("on_position", &strategy::Strategy::on_position)
            .def("on_union_response", &strategy::Strategy::on_union_response)
            .def("on_factor", &strategy::Strategy::on_factor,
                 py::arg("context"), py::arg("symbol"), py::arg("timestamp"), py::arg("values"),
                 "Factor callback")
            .def("get_uid", &strategy::Strategy::get_uid)
            ;

    py::class_<algo::AlgoOrder, PyAlgoOrder, algo::AlgoOrder_ptr>(m, "AlgoOrder")
            .def(py::init<uint64_t>())
            .def_property_readonly("order_id", &algo::AlgoOrder::get_order_id)
            .def("dumps", &algo::AlgoOrder::dumps)
            .def("on_start", &algo::AlgoOrder::on_start)
            .def("on_stop", &algo::AlgoOrder::on_stop)
            .def("on_depth", &algo::AlgoOrder::on_depth)
            .def("on_child_order", &algo::AlgoOrder::on_child_order)
            .def("on_child_trade", &algo::AlgoOrder::on_child_trade)
            .def("on_order_report", &algo::AlgoOrder::on_order_report)
            ;

    py::class_<algo::AlgoContext, std::shared_ptr<algo::AlgoContext>>(m, "AlgoContext")
            .def("insert_child_order", &algo::AlgoContext::insert_order)
            .def("now", &algo::AlgoContext::now)
            .def("add_timer", &algo::AlgoContext::add_timer)
            .def("add_order", &algo::AlgoContext::add_order);

    py::class_<service::Algo, PyAlgoService, service::Algo_ptr>(m, "AlgoService")
            .def(py::init<data::locator_ptr, data::mode, bool>())
            .def_property_readonly("algo_context", &service::Algo::get_algo_context)
            .def_property_readonly("io_device", &service::Algo::get_io_device)
            .def("now", &service::Algo::now)
            .def("get_location", &service::Algo::get_location)
            .def("get_writer", &service::Algo::get_writer)
            .def("has_location", &service::Algo::has_location)
            .def("has_writer", &service::Algo::has_writer)
            .def("run", &service::Algo::run)
            .def("add_order", &service::Algo::add_order)
            .def("insert_order", &service::Algo::insert_order)
            .def("cancel_order", &service::Algo::cancel_order)
            .def("modify_order", &service::Algo::modify_order)
            ;

    py::class_<BarGenerator, kungfu::practice::apprentice, std::shared_ptr<BarGenerator>>(m, "BarGenerator")
            .def(py::init<data::locator_ptr, data::mode, bool, std::string&>())
            .def("run", &service::BarGenerator::run);
}

