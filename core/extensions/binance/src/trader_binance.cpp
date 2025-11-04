/**
 * This is source code under the Apache License 2.0.
 * Original Author: kx@godzilla.dev
 * Original date: March 3, 2025
 */

#include <utility>
#include <algorithm>
#include <string>
#include "kungfu/wingchun/utils.h"
#include "trader_binance.h"
#include "type_convert_binance.h"

using namespace kungfu::wingchun::msg::data;
using namespace kungfu::yijinjing;

namespace kungfu {
    namespace wingchun {
        namespace binance {
            static inline void rtrim_zero(std::string& s) {
                s.erase(s.find_last_not_of('0') + 1, std::string::npos);
            }

            static inline std::string to_string(double v, int precision = 11) {
                std::ostringstream s;
                s << std::fixed << std::setprecision(precision);
                s << v;
                std::string str = s.str();
                rtrim_zero(str);
                if (str.back() == '.') {
                    str.pop_back();
                }
                return str;
            }

            TraderBinance::TraderBinance(bool low_latency, yijinjing::data::locator_ptr locator, const std::string& account_id, const std::string& json_config):
                Trader(low_latency, std::move(locator), SOURCE_BINANCE, account_id) {
                yijinjing::log::copy_log_settings(get_io_device()->get_home(), SOURCE_BINANCE);
                
                auto first_parse = nlohmann::json::parse(json_config);
                nlohmann::json config_json;
                if (first_parse.is_string()) {
                    config_json = nlohmann::json::parse(first_parse.get<std::string>());
                } else {
                    config_json = first_parse;
                }
                
                config_ = config_json.get<Configuration>();
                
                // Initialize Spot market components (Guard Clause Pattern - ADR-004)
                if (config_.enable_spot) {
                    rest_ptr_ = std::make_shared<binapi::rest::api>(
                        ioctx_
                        , config_.spot_rest_host                      //"api.binance.com"
                        , std::to_string(config_.spot_rest_port)      //"443"
                        , config_.access_key
                        , config_.secret_key
                        , 10000
                        );
                    ws_ptr_ = std::make_shared<binapi::ws::websockets>(
                        ioctx_, config_.spot_wss_host, std::to_string(config_.spot_wss_port));
                } else {
                    SPDLOG_INFO("Spot market disabled by configuration");
                }
                
                // Initialize Futures market components (Guard Clause Pattern - ADR-004)
                if (config_.enable_futures) {
                    frest_ptr_ = std::make_shared<binapi::rest::api>(
                        ioctx_
                        , config_.ubase_rest_host                      //"fapi.binance.com"
                        , std::to_string(config_.ubase_rest_port)      //"443"
                        , config_.access_key
                        , config_.secret_key
                        , 10000
                        );
                    fws_ptr_ = std::make_shared<binapi::ws::websockets>(
                        ioctx_, config_.ubase_wss_host, std::to_string(config_.ubase_wss_port));
                } else {
                    SPDLOG_INFO("Futures market disabled by configuration");
                }
            }

            TraderBinance::~TraderBinance() {}

            std::string TraderBinance::get_runtime_folder() const {
                auto home = get_io_device()->get_home();
                return home->locator->layout_dir(home, yijinjing::data::layout::LOG);
            }

            void TraderBinance::on_start() {
                Trader::on_start();
                task_thread_ = std::make_shared<std::thread>([this]() {
                    boost::asio::io_context::work worker(this->ioctx_);
                    this->ioctx_.run();
                    return 0;
                });
                std::string runtime_folder = get_runtime_folder();
                SPDLOG_INFO(
                    "Connecting BINANCE TD for {} (Spot: {}, Futures: {}) with runtime folder {}",
                    config_.user_id, 
                    config_.enable_spot ? "enabled" : "disabled",
                    config_.enable_futures ? "enabled" : "disabled",
                    runtime_folder);
                
                // Guard Clause Pattern: Only initialize enabled markets (ADR-004)
                if (config_.enable_futures && frest_ptr_) {
                    _start_userdata(InstrumentType::FFuture);
                } else {
                    SPDLOG_INFO("Skipping Futures initialization (disabled or client unavailable)");
                }
                
                if (config_.enable_spot && rest_ptr_) {
                    _start_userdata(InstrumentType::Spot);
                } else {
                    SPDLOG_INFO("Skipping Spot initialization (disabled or client unavailable)");
                }
                
                add_time_interval(time_unit::NANOSECONDS_PER_SECOND * 5, std::bind(&TraderBinance::_check_status, this, std::placeholders::_1));
                publish_state(BrokerState::Ready);
                SPDLOG_INFO("login success");
            }

            bool TraderBinance::insert_order(const yijinjing::event_ptr& event) {
                const OrderInput& input = event->data<OrderInput>();
                int64_t nano = kungfu::yijinjing::time::now_in_nano();
                auto source = event->source();
                msg::data::Order order{};
                order_from_input(input, order);
                order.insert_time = nano;
                order.update_time = nano;
                {
                    std::lock_guard<std::mutex> lock(order_mtx_);
                    order_data_.emplace(input.order_id, order_map_record(input.order_id, "", source, nano, order));
                }
                if (input.instrument_type == InstrumentType::Spot) {
                    auto order_ptr = std::make_shared<binapi::rest::api>(
                        ioctx_
                        , config_.spot_rest_host                      //"api.binance.com"
                        , std::to_string(config_.spot_rest_port)      //"443"
                        , config_.access_key
                        , config_.secret_key
                        , 10000
                        );
                    order_ptr->new_order(
                        to_binance_symbol(input.symbol),
                        to_binance(input.side),
                        to_binance(input.order_type),
                        binapi::e_time::GTC,
                        binapi::e_trade_resp_type::FULL,
                        to_string(input.volume),
                        to_string(input.price),
                        std::to_string(input.order_id),
                        std::string(""),
                        std::string(""),
                        [&, source, nano, ptr=std::move(order_ptr)](const char* fl, int ec, std::string errmsg, binapi::rest::new_order_resp_type res) {
                            if (ec) {
                                SPDLOG_ERROR("{} (input){} (ErrorId){}, (ErrorMsg){}", fl, nlohmann::json(input).dump(), ec, errmsg);
                            }
                            auto writer = get_writer(source);
                            msg::data::Order& order = writer->open_data<msg::data::Order>(nano, msg::type::Order);
                            order_from_input(input, order);
                            order.insert_time = nano;
                            order.update_time = nano;
                            if (!res.is_valid_responce_type()) {
                                order.status = OrderStatus::Error;
                                writer->close_data();
                                SPDLOG_ERROR("(input){} (ErrorId){}, (ErrorMsg){}", nlohmann::json(input).dump(), ec, errmsg);
                                return false;
                            } else {
                                auto ex_order_id = res.get_order_id();
                                strcpy(order.ex_order_id, std::to_string(ex_order_id).c_str());
                                order.status = OrderStatus::Submitted;
                                writer->close_data();
                                SPDLOG_TRACE("success to insert order, (order_id){} (xtp_order_id) {}", input.order_id, ex_order_id);
                                return true;
                            }
                            return true;
                        });
                } else if (input.instrument_type == InstrumentType::FFuture) {
                    auto order_ptr = std::make_shared<binapi::rest::api>(
                        ioctx_
                        , config_.ubase_rest_host                      //"fapi.binance.com"
                        , std::to_string(config_.ubase_rest_port)      //"443"
                        , config_.access_key
                        , config_.secret_key
                        , 10000
                        );
                    order_ptr->future_new_order(
                        to_binance_symbol(input.symbol),
                        to_binance(input.side),
                        to_binance(input.position_side),
                        to_binance(input.order_type),
                        std::string(""),
                        to_string(input.price),
                        to_string(input.volume),
                        std::to_string(input.order_id),
                        std::string(""),
                        std::string(""),
                        std::string(""),
                        std::string(""),
                        binapi::e_time::GTC,
                        std::string(""),
                        std::string(""),
                        binapi::e_trade_resp_type::FULL,
                        0,
                        [&, source, nano, ptr=std::move(order_ptr)](const char* fl, int ec, std::string errmsg, binapi::rest::future_new_order_resp_t res) {
                            if (ec) {
                                auto writer = get_writer(source);
                                msg::data::Order& order = writer->open_data<msg::data::Order>(nano, msg::type::Order);
                                order_from_input(input, order);
                                order.insert_time = nano;
                                order.update_time = nano;
                                order.status = OrderStatus::Error;
                                writer->close_data();
                                SPDLOG_ERROR("{} (input){} (ErrorId){}, (ErrorMsg){}", fl, nlohmann::json(input).dump(), ec, errmsg);
                            } else {
                                std::stringstream oss;
                                oss << res;
                                SPDLOG_INFO("(res){}", oss.str());
			    }
                            return true;
                        });
                }
                return true;
            }

            bool TraderBinance::query_order(const yijinjing::event_ptr& event) {
                const OrderAction& action = event->data<OrderAction>();
                std::stringstream sstream(action.ex_order_id);
                std::size_t ex_order_id_int;
                sstream >> ex_order_id_int;
                auto cb = [&](const char* fl, int ec, std::string errmsg, binapi::rest::order_info_t res) {
                        if (ec) {
                            SPDLOG_ERROR("{} (action){} (ErrorId){}, (ErrorMsg){}", fl, nlohmann::json(action).dump(), ec, errmsg);
                            return true;
                        }
                        auto writer = get_writer(event->source());
                        msg::data::Order& order = writer->open_data<msg::data::Order>(event->gen_time(), msg::type::Order);
                        order.order_id = action.order_id;
                        order.insert_time = res.time;
                        order.update_time = res.updateTime;
                        strcpy(order.symbol, action.symbol);
                        strcpy(order.exchange_id, EXCHANGE_BINANCE);
                        strcpy(order.account_id, get_account_id().c_str());
                        strcpy(order.source_id, get_source().c_str());
                        order.price = res.price.convert_to<double>();
                        order.volume = res.origQty.convert_to<double>();
                        order.status = from_binance(binapi::e_status_from_string(res.status.c_str()));
                        order.side = from_binance(binapi::e_side_from_string(res.side.c_str()));
                        order.order_type = from_binance(binapi::e_type_from_string(res.type.c_str()));
                        strcpy(order.ex_order_id, std::to_string(res.orderId).c_str());
                        writer->close_data();
                        SPDLOG_TRACE("success to insert order, (order_id){} (binance_order_id) {}", action.order_id, action.ex_order_id);
                        return true;
                    };
                if (action.instrument_type == InstrumentType::Spot) {
                    rest_ptr_->order_info(
                        to_binance_symbol(action.symbol)
                        , ex_order_id_int
                        , std::to_string(action.order_id)
                        , cb
                    );
                } else if (action.instrument_type == InstrumentType::FFuture) {
                    std::cout << "query: " << action.symbol << std::endl;
                    frest_ptr_->future_order_info(
                        to_binance_symbol(action.symbol)
                        , ex_order_id_int
                        , std::to_string(action.order_id)
                        , cb
                    );
                } else {
                    SPDLOG_ERROR("fail to query order {}, unknown instrument type", static_cast<int>(action.instrument_type));
                    return false;
                }
                return true;
            }

            bool TraderBinance::cancel_order(const yijinjing::event_ptr& event) {
                const OrderAction& action = event->data<OrderAction>();
                std::stringstream sstream(action.ex_order_id);
                std::size_t ex_order_id_int;
                sstream >> ex_order_id_int;
                auto cb = [&](const char* fl, int ec, std::string errmsg, binapi::rest::cancel_order_info_t res) {
                        if (ec == 0 and res.orderId != 0 and res.status == "CANCELED") {
                            SPDLOG_TRACE("{} success to request cancel order {}, ex_order_id: {}, symbol: {}", fl, action.order_id, action.ex_order_id, action.symbol);
                            return true;
                        } else {
                            SPDLOG_ERROR("failed to cancel order {}, ex_order_id: {} symbol: {} error_id: {} error_msg: {}",
                            action.order_id, action.ex_order_id, action.symbol, ec, errmsg);
                            return false;
                        }
                    };
                if (action.ex_order_id != 0 or action.order_id > 0) {
                    if (action.instrument_type == InstrumentType::Spot)
                        rest_ptr_->cancel_order(
                            to_binance_symbol(action.symbol)
                            , ex_order_id_int
                            , std::string()
                            , std::to_string(action.order_id)
                            , cb
                        );
                    else if (action.instrument_type == InstrumentType::FFuture)
                        frest_ptr_->future_cancel_order(
                            to_binance_symbol(action.symbol)
                            , ex_order_id_int
                            , std::to_string(action.order_id)
                            , cb
                        );
                    else {
                        SPDLOG_ERROR("fail to cancel order {}, unknown instrument type", static_cast<int>(action.instrument_type));
                        return true;
                    }
                    return true;
                } else {
                    SPDLOG_ERROR("fail to cancel order {}, can't find related binance order id", action.order_id);
                    return false;
                }
            }

            bool TraderBinance::adjust_leverage(const yijinjing::event_ptr &event) {
                auto json_str = event->data_as_string();
                auto source = event->source();
                SPDLOG_TRACE("adjust_leverage request: {}", json_str);
                nlohmann::json sub_msg = nlohmann::json::parse(json_str);
                Direction position_side = static_cast<Direction>(sub_msg["position_side"].get<int>());
                auto strategy_id = sub_msg["strategy_uid"].get<std::size_t>();
                SPDLOG_TRACE("strategy uid: {}", strategy_id);
                auto symbol = sub_msg["symbol"].get<std::string>();
                frest_ptr_->ffuture_adjust_leverage(
                    symbol,
                    sub_msg["leverage"].get<int>(),
                    [this, source, symbol, strategy_id](const char *fl, int ec, std::string errmsg, binapi::rest::leverage_info_t res){
                        if (ec) {
                            SPDLOG_WARN("fail to set leverage: (fl){}, (ec){}, (errmsg){}", fl, ec, errmsg);
                            return false;
                        }
                        auto writer = get_writer(source);
                        nlohmann::json data{};
                        data["symbol"] = symbol;
                        data["msg_type"] = static_cast<int>(msg::type::AdjustLeverage);
                        data["ret"] = true;
                        data["strategy_id"] = strategy_id;
                        write_union_response(writer, kungfu::yijinjing::time::now_in_nano(), data.dump());
                        return true;
                    });
                return true;
            }

            bool TraderBinance::req_position() {
                return true;
            }

            bool TraderBinance::req_account() {
                return true;
            }

            void TraderBinance::_check_status(kungfu::yijinjing::event_ptr) {
                SPDLOG_DEBUG("TraderBinance::_check_status");
                static int check_time = 0;
                check_time += 5;
                if (check_time > 3000) {
                    check_time = 0;
                    for (auto &k: listenKeys) {
                        frest_ptr_->future_ping_user_data_stream(k,
                            [this, k](const char *fl, int ec, std::string errmsg, binapi::rest::ping_user_data_stream_t res){
                                SPDLOG_TRACE("ping listenkey: {}", k);
                                return true;
                            });
                    }
                }
                // Guard Clause Pattern: Only reconnect enabled markets (ADR-004)
                if (config_.enable_spot && rest_ptr_ && ws_ptr_ && ws_ptr_->fetch_reconnect_flag()) {
                    _start_userdata(InstrumentType::Spot);
                    SPDLOG_TRACE("TraderXTC::_update_order spot: {}", order_data_.size());
                }
                if (config_.enable_futures && frest_ptr_ && fws_ptr_ && fws_ptr_->fetch_reconnect_flag()) {
                    _start_userdata(InstrumentType::FFuture);
                    SPDLOG_TRACE("TraderXTC::_update_order ffuture: {}", order_data_.size());
                }
            }

            void TraderBinance::_start_userdata(const InstrumentType type){
                if (type == InstrumentType::FFuture) {
                    auto start_uds = frest_ptr_->future_start_user_data_stream();
                    if (!start_uds) {
                        publish_state(msg::data::BrokerState::LoggedInFailed);
                        SPDLOG_ERROR("future login failed, error_id: {}, error_msg: {}", start_uds.ec, start_uds.errmsg);
                        // throw wingchun_error("cannot login future");
                        return;
                    }
                    listenKeys.push_back(start_uds.v.listenKey);
                    fws_ptr_->future_order_data(start_uds.v.listenKey.c_str(),
                        [this](const char* fl, int ec, std::string errmsg, binapi::userdata::future_order_update_t msg) {
                            if (ec) {
                                SPDLOG_ERROR("{} (future_order_update_t) (ErrorId){}, (ErrorMsg){}", fl, ec, errmsg);
                                std::ostringstream oss;
                                oss << msg;
                                SPDLOG_ERROR("msg when error occurred: {}", oss.str());
                                return true;
                            }
                            std::size_t i_coid;
                            std::stringstream sstream(msg.o.c);
                            sstream >> i_coid;
                            auto itr = order_data_.find(i_coid);
                            if (itr != order_data_.end()) {
                                std::ostringstream oss;
                                oss << msg;
                                SPDLOG_INFO("order update: {} | {}", itr->second.source, oss.str());
                                auto writer = get_writer(itr->second.source);
                                msg::data::Order& order = writer->open_data<msg::data::Order>(0, msg::type::Order);
                                order.strategy_id = itr->second.order.strategy_id;
                                order.order_id = i_coid;
                                order.set_ex_order_id(std::to_string(msg.o.i));
                                order.set_symbol(from_binance_symbol(msg.o.s));
                                order.instrument_type = itr->second.order.instrument_type;
                                order.set_exchange_id(EXCHANGE_BINANCE);
                                order.set_account_id(get_account_id());
                                order.set_source_id(get_source());
                                order.set_fee_currency(msg.o.N);
                                order.price = itr->second.order.price;
                                order.volume = itr->second.order.volume;
                                order.volume_traded = msg.o.z.convert_to<double>();
                                order.volume_left = order.volume - order.volume_traded;
                                order.stop_price = itr->second.order.stop_price;
                                order.avg_price = msg.o.ap.convert_to<double>();
                                order.fee = msg.o.n.convert_to<double>();
                                order.status = from_binance(binapi::e_status_from_string(msg.o.X.c_str()));
                                order.side = itr->second.order.side;
                                order.order_type = itr->second.order.order_type;
                                order.position_side = itr->second.order.position_side;
                                order.insert_time = itr->second.order.insert_time;
                                order.update_time = msg.T;
                                order.close_pnl = msg.o.rp.convert_to<double>();
                                writer->close_data();
                                {
                                    std::lock_guard<std::mutex> lock(order_mtx_);
                                    if (order.status != OrderStatus::PartialFilledActive and order.status != OrderStatus::Submitted) {
                                        order_data_.erase(i_coid);
                                    }
                                }
                            }
                            return true;
                        }
                    );
                } else {
                    auto start_uds = rest_ptr_->start_user_data_stream();
                    if (!start_uds) {
                        publish_state(msg::data::BrokerState::LoggedInFailed);
                        SPDLOG_ERROR("spot login failed, error_id: {}, error_msg: {}", start_uds.ec, start_uds.errmsg);
                        // throw wingchun_error("cannot login spot");
                        return;
                    }
                    listenKeys.push_back(start_uds.v.listenKey);
                    ws_ptr_->userdata(start_uds.v.listenKey.c_str(),
                        [](const char *fl, int ec, std::string errmsg, binapi::userdata::account_update_t msg) {
                            return true;
                        },
                        [](const char *fl, int ec, std::string errmsg, binapi::userdata::balance_update_t msg) {
                            return true;
                        },
                        [this](const char* fl, int ec, std::string errmsg, binapi::userdata::order_update_t msg) {
                            if (ec) {
                                SPDLOG_ERROR("{} (future_order_update_t) (ErrorId){}, (ErrorMsg){}", fl, ec, errmsg);
                                return true;
                            }
                            std::size_t i_coid;
                            std::stringstream sstream(msg.c);
                            sstream >> i_coid;
                            auto itr = order_data_.find(i_coid);
                            if (itr != order_data_.end()) {
                                std::ostringstream oss;
                                oss << msg;
                                SPDLOG_INFO("order update: {} | {}", itr->second.source, oss.str());
                                auto writer = get_writer(itr->second.source);
                                msg::data::Order& order = writer->open_data<msg::data::Order>(0, msg::type::Order);
                                order.strategy_id = itr->second.order.strategy_id;
                                order.order_id = i_coid;
                                order.set_ex_order_id(std::to_string(msg.i));
                                order.set_symbol(from_binance_symbol(msg.s));
                                order.instrument_type = itr->second.order.instrument_type;
                                order.set_exchange_id(EXCHANGE_BINANCE);
                                order.set_account_id(get_account_id());
                                order.set_source_id(get_source());
                                order.set_fee_currency("");
                                order.price = itr->second.order.price;
                                order.volume = itr->second.order.volume;
                                order.volume_traded = msg.z.convert_to<double>();
                                order.volume_left = order.volume - order.volume_traded;
                                order.stop_price = itr->second.order.stop_price;
                                order.avg_price = msg.Z.convert_to<double>() / order.volume_traded;
                                order.fee = msg.n.convert_to<double>();
                                order.status = from_binance(binapi::e_status_from_string(msg.X.c_str()));
                                order.side = itr->second.order.side;
                                order.order_type = itr->second.order.order_type;
                                order.position_side = itr->second.order.position_side;
                                order.insert_time = itr->second.order.insert_time;
                                order.update_time = msg.T;
                                // order.close_pnl = msg.o.rp.convert_to<double>();
                                writer->close_data();
                                {
                                    std::lock_guard<std::mutex> lock(order_mtx_);
                                    if (order.status != OrderStatus::PartialFilledActive and order.status != OrderStatus::Submitted) {
                                        order_data_.erase(i_coid);
                                    }
                                }
                            }
                            return true;
                        }
                    );
                }
            }

        }
    }
}
