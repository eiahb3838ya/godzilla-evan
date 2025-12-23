/**
 * This is source code under the Apache License 2.0.
 * Original Author: kx@godzilla.dev
 * Original date: March 3, 2025
 */

#include <cmath>
#include <utility>
#include <kungfu/yijinjing/log/setup.h>
#include "marketdata_binance.h"
#include "type_convert_binance.h"
#include "market_db.h"


using namespace kungfu::wingchun::msg::data;
using namespace kungfu::yijinjing;

namespace kungfu {
    namespace wingchun {
        namespace binance {
            MarketDataBinance::MarketDataBinance(bool low_latency, yijinjing::data::locator_ptr locator, const std::string& json_config):
                MarketData(low_latency, std::move(locator), SOURCE_BINANCE) {
                yijinjing::log::copy_log_settings(get_io_device()->get_home(), SOURCE_BINANCE);
                config_ = nlohmann::json::parse(json_config);
                ws_ptr_ = std::make_shared<binapi::ws::websockets>(
                    ioctx_
                    , config_.spot_wss_host                   //"stream.binance.com"
                    , std::to_string(config_.spot_wss_port)   //"9443"
                    );
                rest_ptr_ = std::make_shared<binapi::rest::api>(
                    ioctx_
                    , config_.spot_rest_host                  //"api.binance.com"
                    , std::to_string(config_.spot_rest_port)  //"443"
                    , config_.access_key
                    , config_.secret_key
                    , 10000
                    );
                frest_ptr_ = std::make_shared<binapi::rest::api>(
                    ioctx_
                    , config_.ubase_rest_host                  //"api.binance.com"
                    , std::to_string(config_.ubase_rest_port)  //"443"
                    , config_.access_key
                    , config_.secret_key
                    , 10000
                    );
                fws_ptr_ = std::make_shared<binapi::ws::websockets>(
                    ioctx_
                    , config_.ubase_wss_host                    //"fstream.binance.com"
                    , std::to_string(config_.ubase_wss_port)    //"443"
                    );
                dws_ptr_ = std::make_shared<binapi::ws::websockets>(
                    ioctx_
                    , config_.cbase_wss_host                    //"dstream.binance.com"
                    , std::to_string(config_.cbase_wss_port)    //"443"
                    );
            }

            MarketDataBinance::~MarketDataBinance() {
                SPDLOG_INFO("MarketDataBinance destructor: stopping ASIO event loop");

                // 1. Stop ASIO event loop - this causes ioctx_.run() to return
                ioctx_.stop();

                // 2. Wait for task thread to terminate completely
                if (task_thread_ && task_thread_->joinable()) {
                    SPDLOG_INFO("MarketDataBinance destructor: joining task thread");
                    task_thread_->join();
                }

                // 3. Explicitly reset WebSocket connections
                // (shared_ptr will auto-destroy, but explicit reset ensures correct order)
                ws_ptr_.reset();
                fws_ptr_.reset();
                dws_ptr_.reset();

                // 4. Reset REST API clients
                rest_ptr_.reset();
                frest_ptr_.reset();

                SPDLOG_INFO("MarketDataBinance destructor: cleanup complete");
            }

            std::string MarketDataBinance::get_runtime_folder() const {
                auto home = get_io_device()->get_home();
                return home->locator->layout_dir(home, yijinjing::data::layout::LOG);
            }

            void MarketDataBinance::on_start() {
                MarketData::on_start();
                std::string runtime_folder = get_runtime_folder();
                SPDLOG_INFO(
                    "Connecting binance MD with {} // {} // {} // {} with runtime folder {}",
                    config_.spot_rest_host, config_.spot_wss_host, config_.ubase_rest_host, config_.ubase_wss_host, runtime_folder);
                publish_state(msg::data::BrokerState::LoggedIn);
                {
                    task_thread_ = std::make_shared<std::thread>(
                        [this]() {
                            boost::asio::io_context::work worker(this->ioctx_);
                            this->ioctx_.run();
                            return 0;
                        });
                }
                {
                    SPDLOG_DEBUG("Update Spot info");
                    auto symbols = rest_ptr_->exchange_info();
                    if (symbols.ec == 0) {
                        MarketInfoDB db(get_shm_db());
                        for (auto& symbol : symbols.v.symbols) {
                            if (symbol.second.status == "TRADING") {
                                std::stringstream oss;
                                oss << symbol.second;
                                SPDLOG_TRACE(oss.str());
                                db.add_market_info(from_binance_symbol(symbol.first), EXCHANGE_BINANCE, InstrumentType::Spot, oss.str(), now());
                            }
                        }
                    }
                }
                {
                    SPDLOG_DEBUG("Update FFuture info");
                    auto symbols = frest_ptr_->future_exchange_info();
                    if (symbols.ec == 0) {
                        MarketInfoDB db(get_shm_db());
                        nlohmann::json j = nlohmann::json::parse(symbols.v.configs);
                        for (auto& symbol : j.at("symbols")) {
                            std::stringstream oss;
                            oss << symbol.dump();
                            db.add_market_info(from_binance_symbol(symbol["symbol"]), EXCHANGE_BINANCE, InstrumentType::FFuture, oss.str(), now());
                        }
                    }
                }
                add_time_interval(time_unit::NANOSECONDS_PER_SECOND * 5, std::bind(&MarketDataBinance::_check_status, this, std::placeholders::_1));
                {
                    wingchun::msg::data::Instrument inst;
                    strcpy(inst.symbol, "ltc_usdt");
                    strcpy(inst.exchange_id, EXCHANGE_BINANCE);
                    inst.instrument_type = InstrumentType::FFuture;
                    std::vector<wingchun::msg::data::Instrument> insts;
                    insts.push_back(inst);
                    // subscribe(insts);
                    // subscribe_trade(insts);
                    // subscribe_ticker(insts);
                }
            }

            bool MarketDataBinance::subscribe(const std::vector<wingchun::msg::data::Instrument>& instruments) {
                SPDLOG_TRACE("subscribe size: {}", instruments.size());
                for (const auto& inst : instruments) {
                    std::string symbol = to_binance_symbol(inst.symbol);
                    std::string orig_symbol(inst.symbol);
                    if (strcmp(inst.exchange_id, EXCHANGE_BINANCE) != 0) {
                        SPDLOG_INFO("Not suppported exchange in {} ext: {}", EXCHANGE_BINANCE, inst.exchange_id);
                        continue;
                    }
                    auto symbol_id = get_symbol_id(symbol, "depth", inst.instrument_type, EXCHANGE_BINANCE, 0);
                    auto it = channel_cache_.find(symbol_id);
                    if (it != channel_cache_.end() and it->second.ready) {
                        SPDLOG_TRACE("add duplicated depth channel: {}", inst.symbol);
                        continue;
                    }
                    auto cb = [this, instrument_type=inst.instrument_type, orig_symbol](const char* fl, int ec, std::string errmsg, binapi::ws::part_depths_t msg) {
                        if (ec) {
                            SPDLOG_ERROR("fail to get depth: ec({}), errmsg({})", ec, errmsg);
                            return false;
                        }
                        std::stringstream oss;
                        oss << "depth message: " << msg;
                        SPDLOG_TRACE(oss.str());
                        msg::data::Depth& depth = this->get_writer(0)->open_data<msg::data::Depth>(0, msg::type::Depth);
                        strcpy(depth.source_id, SOURCE_BINANCE);
                        depth.data_time = now();
                        strcpy(depth.symbol, orig_symbol.c_str());
                        strcpy(depth.exchange_id, EXCHANGE_BINANCE);
                        depth.instrument_type = instrument_type;
                        for (int i = 0; i < 10; i++) {
                            //FIXME, may not be safe in direct convertion
                            depth.ask_price[i] = msg.a[i].price.convert_to<double>();
                            depth.ask_volume[i] = msg.a[i].amount.convert_to<double>();
                            depth.bid_price[i] = msg.b[i].price.convert_to<double>();
                            depth.bid_volume[i] = msg.b[i].amount.convert_to<double>();
                        }
                        get_writer(0)->close_data();
                        if (ec) {
                            std::cerr << "subscribe part_depth error: fl=" << fl << ", ec=" << ec << ", emsg=" << errmsg << std::endl;
                            return false;
                        }
                        return true;
                    };
                    binapi::ws::websockets::handle h;
                    if (inst.instrument_type == InstrumentType::Spot) {
                        //TODO, should make these hardcoded number configurable.
                        h = ws_ptr_->part_depth(symbol.c_str(), binapi::e_levels::_20, binapi::e_freq::_100ms, cb);
                    } else if (inst.instrument_type == InstrumentType::FFuture) {
                        h = fws_ptr_->future_part_depth(symbol.c_str(), binapi::e_levels::_20, binapi::e_freq::_100ms, cb);
                    } else if (inst.instrument_type == InstrumentType::DFuture) {
                        h = dws_ptr_->future_part_depth(symbol.c_str(), binapi::e_levels::_20, binapi::e_freq::_100ms, cb);
                    } else {
                        return false;
                    }
                    if (it == channel_cache_.end()) {
                        channel_cache_.emplace(std::make_pair(symbol_id, ChannelInfo{orig_symbol, "depth", inst.instrument_type, h, true}));
                    } else {
                        it->second.ready = true;
                        it->second.handle = h;
                    }
                }
                return true;
            }

            bool MarketDataBinance::subscribe_trade(const std::vector<wingchun::msg::data::Instrument>& instruments) {
                SPDLOG_TRACE("size: {}", instruments.size());
                for (const auto& inst : instruments) {
                    std::string symbol = to_binance_symbol(inst.symbol);
                    std::string orig_symbol(inst.symbol);
                    if (strcmp(inst.exchange_id, EXCHANGE_BINANCE) != 0) {
                        SPDLOG_INFO("Not suppported exchange in {} ext: {}", EXCHANGE_BINANCE, inst.exchange_id);
                        continue;
                    }
                    auto symbol_id = get_symbol_id(symbol, "trade", inst.instrument_type, EXCHANGE_BINANCE, 0);
                    auto it = channel_cache_.find(symbol_id);
                    if (it != channel_cache_.end() and it->second.ready) {
                        SPDLOG_INFO("add duplicated trade channel: {}", inst.symbol);
                        continue;
                    }
                    auto future_cb = [this, instrument_type=inst.instrument_type, orig_symbol](
                            const char* fl, int ec, std::string errmsg, binapi::ws::agg_trade_t msg) {
                        if (ec) {
                            SPDLOG_ERROR("future trade error: ec={}, emsg={}", ec, errmsg);
                            return true;
                        }
                        std::stringstream oss;
                        oss << "trade message: " << msg;
                        SPDLOG_TRACE(oss.str());
                        if (std::abs(int(msg.E - msg.T)) > 500 or std::abs(int(now() / 1000000 - msg.T)) > 500)
                            SPDLOG_WARN("future trade delay: event_time={}, trade_time={}, now={}", msg.E, msg.T, now());
                        msg::data::Trade& trade = this->get_writer(0)->open_data<msg::data::Trade>(0, msg::type::Trade);
                        trade.trade_time = msg.T * 1000 * 1000;
                        trade.set_symbol(orig_symbol);
                        trade.set_exchange_id(EXCHANGE_BINANCE);
                        trade.instrument_type = instrument_type;
                        trade.trade_id = msg.a;
                        trade.price = msg.p.convert_to<double>();
                        trade.volume = msg.q.convert_to<double>();
                        trade.ask_id = 0;
                        trade.bid_id = 0;
                        if (msg.m) {
                            trade.side = kungfu::wingchun::Side::Sell;
                        } else {
                            trade.side = kungfu::wingchun::Side::Buy;
                        }
                        get_writer(0)->close_data();
                        return true;
                    };
                    binapi::ws::websockets::handle h = nullptr;
                    if (inst.instrument_type == InstrumentType::Spot) {
                        h = ws_ptr_->trade(symbol.c_str(),
                            [this, orig_symbol](const char* fl, int ec, std::string errmsg, binapi::ws::trade_t msg) {
                                std::stringstream oss;
                                oss << "trade message: " << msg;
                                SPDLOG_TRACE(oss.str());
                                if (std::abs(int(msg.E - msg.T)) > 500 or std::abs(int(now() / 1000000 - msg.T)) > 500)
                                    SPDLOG_WARN("spot trade delay: event_time={}, trade_time={}, now={}", msg.E, msg.T, now());
                                msg::data::Trade& trade = this->get_writer(0)->open_data<msg::data::Trade>(0, msg::type::Trade);
                                trade.trade_time = msg.T * 1000 * 1000;
                                trade.set_symbol(orig_symbol);
                                trade.set_exchange_id(EXCHANGE_BINANCE);
                                trade.instrument_type = InstrumentType::Spot;
                                trade.trade_id = msg.t;
                                trade.price = msg.p.convert_to<double>();
                                trade.volume = msg.q.convert_to<double>();
                                trade.ask_id = msg.a;
                                trade.bid_id = msg.b;
                                if (msg.m) {
                                    trade.side = kungfu::wingchun::Side::Sell;
                                } else {
                                    trade.side = kungfu::wingchun::Side::Buy;
                                }
                                get_writer(0)->close_data();
                                if (ec) {
                                    SPDLOG_ERROR("subscribe [trade] error: ec={}, emsg={}", ec, errmsg);
                                    return false;
                                }
                                return true;
                            });
                    } else if (inst.instrument_type == InstrumentType::FFuture) {
                        h = fws_ptr_->agg_trade(symbol.c_str(), future_cb);
                    } else if (inst.instrument_type == InstrumentType::DFuture) {
                        h = dws_ptr_->agg_trade(symbol.c_str(), future_cb);
                    } else {
                        continue;
                    }
                    if (it == channel_cache_.end()) {
                        channel_cache_.emplace(std::make_pair(symbol_id, ChannelInfo{orig_symbol, "trade", inst.instrument_type, h, true}));
                    } else {
                        it->second.ready = true;
                        it->second.handle = h;
                    }
                }
                return true;
            }

            bool MarketDataBinance::subscribe_ticker(const std::vector<wingchun::msg::data::Instrument>& instruments) {
                SPDLOG_TRACE("size: {}", instruments.size());
                for (const auto& inst : instruments) {
                    std::string symbol = to_binance_symbol(inst.symbol);
                    std::string orig_symbol(inst.symbol);
                    if (strcmp(inst.exchange_id, EXCHANGE_BINANCE) != 0) {
                        SPDLOG_INFO("Not suppported exchange in {} ext: {}", EXCHANGE_BINANCE, inst.exchange_id);
                        continue;
                    }
                    auto symbol_id = get_symbol_id(symbol, "ticker", inst.instrument_type, EXCHANGE_BINANCE, 0);
                    auto it = channel_cache_.find(symbol_id);
                    if (it != channel_cache_.end() and it->second.ready) {
                        SPDLOG_INFO("add duplicated ticker channel: {}/ticker", inst.symbol);
                        continue;
                    }
                    auto cb = [&, orig_symbol, inst_type=inst.instrument_type](const char* fl, int ec, std::string errmsg, binapi::ws::book_ticker_t msg) {
                        if (ec) {
                            SPDLOG_ERROR("subscribe [ticker] error: ec={}, emsg={}", ec, errmsg);
                            return false;
                        }
                        std::stringstream oss;
                        oss << "ticker message: " << msg;
                        SPDLOG_TRACE(oss.str());
                        if (msg.T > 0 and (std::abs(int(msg.E - msg.T)) > 500 or std::abs(int(now() / 1000000 - msg.T)) > 500))
                            SPDLOG_WARN("ticker delay: event_time={}, ticker_time={}, now={}", msg.E, msg.T, now());
                        msg::data::Ticker& ticker = this->get_writer(0)->open_data<msg::data::Ticker>(0, msg::type::Ticker);
                        ticker.set_source_id(SOURCE_BINANCE);
                        if (msg.T > 0)
                            ticker.data_time = msg.T * 1000 * 1000;
                        else
                            ticker.data_time = now();
                        ticker.set_symbol(orig_symbol);
                        ticker.set_exchange_id(EXCHANGE_BINANCE);
                        ticker.instrument_type = inst_type;
                        //FIXME, may not be safe in direct convertion
                        ticker.ask_price = msg.a.convert_to<double>();
                        ticker.ask_volume = msg.A.convert_to<double>();
                        ticker.bid_price = msg.b.convert_to<double>();
                        ticker.bid_volume = msg.B.convert_to<double>();
                        get_writer(0)->close_data();
                        return true;
                    };
                    binapi::ws::websockets::handle h;
                    if (inst.instrument_type == InstrumentType::Spot) {
                        h = ws_ptr_->book(symbol.c_str(), cb);
                    } else if (inst.instrument_type == InstrumentType::FFuture) {
                        h = fws_ptr_->book(symbol.c_str(), cb);
                    } else if (inst.instrument_type == InstrumentType::DFuture) {
                        h = dws_ptr_->book(symbol.c_str(), cb);
                    } else {
                        return false;
                    }
                    if (it == channel_cache_.end()) {
                        channel_cache_.emplace(std::make_pair(symbol_id, ChannelInfo{orig_symbol, "ticker", inst.instrument_type, h, true}));
                    } else {
                        it->second.ready = true;
                        it->second.handle = h;
                    }
                }
                return true;
            }

            bool MarketDataBinance::subscribe_index_price(const std::vector<wingchun::msg::data::Instrument>& instruments) {
                SPDLOG_INFO("Not suppported subscribe_index_price in [{}] ext", SOURCE_BINANCE);
                return false;
            }

            bool MarketDataBinance::subscribe_all() {
                SPDLOG_INFO("subscribe all [not support]");
                return true;
            }

            bool MarketDataBinance::unsubscribe(const std::vector<wingchun::msg::data::Instrument>& instruments) {
                SPDLOG_INFO("unsubscribe not supported.");
                // for (auto& inst : instruments) {
                //     std::string symbol = inst.symbol;
                //     if (strcmp(inst.exchange_id, EXCHANGE_BINANCE) != 0) {
                //         SPDLOG_INFO("Not suppported exchange in binance ext: {}", inst.exchange_id);
                //         continue;
                //     }
                //     auto symbol_id = get_symbol_id(symbol, "ticker", inst.instrument_type, EXCHANGE_BINANCE, 0);
                //     auto iter = channel_cache_.find(symbol_id);
                //     if (iter == channel_cache_.end()) {
                //         SPDLOG_INFO("remove not existed channel: {}", inst.symbol);
                //         continue;
                //     }
                //     if (inst.instrument_type == InstrumentType::Spot)
                //         ws_ptr_->unsubscribe(iter->second.handle);
                //     else
                //         fws_ptr_->unsubscribe(iter->second.handle);
                //     channel_cache_.erase(iter);
                // }
                return true;
            }

            void MarketDataBinance::_check_status(kungfu::yijinjing::event_ptr) {
                SPDLOG_DEBUG("MarketDataBinance::_check_status, connections: {}", channel_cache_.size());
                std::vector<wingchun::msg::data::Instrument> depth_insts;
                std::vector<wingchun::msg::data::Instrument> trade_insts;
                std::vector<wingchun::msg::data::Instrument> ticker_insts;
                for (auto &it: channel_cache_)
                {
                    SPDLOG_DEBUG("handle in MD: {}", fmt::ptr(it.second.handle));
                    bool is_ready = true;
                    if (it.second.inst_type == InstrumentType::FFuture)
                        is_ready = fws_ptr_->is_ready(it.second.handle);
                    else
                        is_ready = ws_ptr_->is_ready(it.second.handle);
                    if (not is_ready) {
                        it.second.ready = false;
                        wingchun::msg::data::Instrument inst;
                        strcpy(inst.symbol, it.second.symbol.c_str());
                        strcpy(inst.exchange_id, EXCHANGE_BINANCE);
                        inst.instrument_type = it.second.inst_type;
                        if (it.second.sub_type == "depth")
                            depth_insts.push_back(inst);
                        else if (it.second.sub_type == "trade")
                            trade_insts.push_back(inst);
                        else if (it.second.sub_type == "ticker")
                            ticker_insts.push_back(inst);
                        SPDLOG_WARN("MarketDataBinance::_check_status re-subscribe:{} {}@{} {}", it.first, it.second.symbol, it.second.sub_type, it.second.ready);
                    }
                }
                if (not depth_insts.empty()) {
                    subscribe(depth_insts);
                }
                if (not trade_insts.empty()) {
                    subscribe_trade(trade_insts);
                }
                if (not ticker_insts.empty()) {
                    subscribe_ticker(ticker_insts);
                }
            }
        }
    }
}
