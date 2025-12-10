/**
 * This is source code modified under the Apache License 2.0.
 * Original Author: Keren Dong
 * Modifier: kx@godzilla.dev
 * Modification date: March 3, 2025
 */

#include <utility>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <fmt/format.h>
#include <kungfu/yijinjing/log/setup.h>
#include <kungfu/yijinjing/time.h>
#include <kungfu/wingchun/strategy/runner.h>
#include <kungfu/wingchun/strategy/metric_controller.h>

using namespace kungfu::practice;
using namespace kungfu::rx;
using namespace kungfu::yijinjing;
using namespace kungfu::yijinjing::data;
using namespace kungfu::wingchun::msg::data;

namespace kungfu
{
    namespace wingchun
    {
        namespace strategy
        {
            Runner::Runner(yijinjing::data::locator_ptr locator, const std::string &group, const std::string &name, yijinjing::data::mode m,
                           bool low_latency)
                    : apprentice(location::make(m, category::STRATEGY, group, name, std::move(locator)), low_latency)
            {}

            Runner::~Runner()
            {
                std::cerr << "\n[DESTRUCTOR] ============================================" << std::endl;
                std::cerr << "[DESTRUCTOR] ~Runner() called" << std::endl;
                std::cerr << "[DESTRUCTOR] signal_engine_handle_: " << signal_engine_handle_ << std::endl;
                std::cerr << "[DESTRUCTOR] signal_lib_handle_: " << signal_lib_handle_ << std::endl;
                std::cerr << "[DESTRUCTOR] ============================================\n" << std::endl;
                
                if (signal_destroy_ && signal_engine_handle_)
                {
                    std::cerr << "[DESTRUCTOR] Calling signal_destroy()..." << std::endl;
                    signal_destroy_(signal_engine_handle_);
                    signal_engine_handle_ = nullptr;
                    std::cerr << "[DESTRUCTOR] signal_destroy() completed" << std::endl;
                }
                
                if (signal_lib_handle_)
                {
                    std::cerr << "[DESTRUCTOR] Calling dlclose()..." << std::endl;
                    // 确保所有函数指针失效
                    signal_create_ = nullptr;
                    signal_register_callback_ = nullptr;
                    signal_on_data_ = nullptr;
                    signal_destroy_ = nullptr;
                    
                    dlclose(signal_lib_handle_);
                    signal_lib_handle_ = nullptr;
                    std::cerr << "[DESTRUCTOR] dlclose() completed" << std::endl;
                }
                
                std::cerr << "[DESTRUCTOR] ~Runner() completed successfully\n" << std::endl;
            }

            Context_ptr Runner::make_context()
            {
                return std::make_shared<Context>(*this, events_);
            }

            void Runner::register_location(int64_t trigger_time, const yijinjing::data::location_ptr &app_location)
            {
                if (context_ and context_->used_account_location(app_location->uid))
                {
                    // std::cout << app_location->uname << ", " << app_location->uid << ", " << static_cast<int>(app_location->category) << std::endl;
                    // std::cout << context_->used_account_location(app_location->uid) << std::endl;
                    throw std::runtime_error("restart due to account[TD] restart");
                }
                apprentice::register_location(trigger_time, app_location);
            }

            void Runner::add_strategy(const Strategy_ptr &strategy, const std::string &path)
            {
                auto uid = yijinjing::util::hash_str_32(path);
                strategy->set_uid(uid);
                strategies_.insert(std::make_pair(uid, strategy));
            }

            void Runner::load_signal_library()
            {
                // 從配置讀取路徑 (目前使用硬編碼,未來可從環境變數或配置文件讀取)
                const char* lib_path_env = std::getenv("SIGNAL_LIB_PATH");
                std::string lib_path = lib_path_env ? lib_path_env : "/app/hf-live/build/libsignal.so";

                // ===== 調試輸出 1: 函數入口 =====
                std::cerr << "\n============================================" << std::endl;
                std::cerr << "[DEBUG] load_signal_library() called" << std::endl;
                std::cerr << "[DEBUG] Signal library path: " << lib_path << std::endl;
                std::cerr << "============================================\n" << std::endl;
                
                SPDLOG_INFO("Attempting to load signal library from: {}", lib_path);

                // dlopen 加載動態庫 (使用 RTLD_NODELETE 防止析構時卸載，避免全局對象雙重釋放)
                signal_lib_handle_ = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_NODELETE);
                
                // ===== 調試輸出 2: dlopen 結果 =====
                if (!signal_lib_handle_)
                {
                    const char* error_msg = dlerror();
                    std::cerr << "[ERROR] ❌ Failed to load signal library!" << std::endl;
                    std::cerr << "[ERROR] dlopen error: " << (error_msg ? error_msg : "unknown") << std::endl;
                    std::cerr << "[ERROR] Attempted path: " << lib_path << std::endl;
                    std::cerr << "============================================\n" << std::endl;
                    
                    SPDLOG_WARN("Failed to load signal library: {}", error_msg);
                    return;
                }

                std::cerr << "[SUCCESS] ✅ dlopen succeeded! Library loaded." << std::endl;

                // 加載函數符號
                std::cerr << "[DEBUG] Loading function symbols..." << std::endl;
                
                signal_create_ = (signal_create_fn)dlsym(signal_lib_handle_, "signal_create");
                signal_register_callback_ = (signal_register_callback_fn)dlsym(signal_lib_handle_, "signal_register_callback");
                signal_on_data_ = (signal_on_data_fn)dlsym(signal_lib_handle_, "signal_on_data");
                signal_destroy_ = (signal_destroy_fn)dlsym(signal_lib_handle_, "signal_destroy");

                // ===== 調試輸出 3: 符號加載結果 =====
                std::cerr << "[DEBUG] signal_create: " << (signal_create_ ? "✅ OK" : "❌ FAILED") << std::endl;
                std::cerr << "[DEBUG] signal_register_callback: " << (signal_register_callback_ ? "✅ OK" : "❌ FAILED") << std::endl;
                std::cerr << "[DEBUG] signal_on_data: " << (signal_on_data_ ? "✅ OK" : "❌ FAILED") << std::endl;
                std::cerr << "[DEBUG] signal_destroy: " << (signal_destroy_ ? "✅ OK" : "❌ FAILED") << std::endl;

                // 檢查必要函數是否加載成功
                if (!signal_create_ || !signal_on_data_)
                {
                    std::cerr << "[ERROR] ❌ Required functions not loaded!" << std::endl;
                    std::cerr << "[ERROR] signal_create: " << (signal_create_ ? "OK" : "NULL") << std::endl;
                    std::cerr << "[ERROR] signal_on_data: " << (signal_on_data_ ? "OK" : "NULL") << std::endl;
                    std::cerr << "============================================\n" << std::endl;
                    
                    SPDLOG_ERROR("Failed to load required signal functions (signal_create: {}, signal_on_data: {})",
                                 signal_create_ != nullptr, signal_on_data_ != nullptr);
                    dlclose(signal_lib_handle_);
                    signal_lib_handle_ = nullptr;
                    signal_create_ = nullptr;
                    signal_register_callback_ = nullptr;
                    signal_on_data_ = nullptr;
                    signal_destroy_ = nullptr;
                    return;
                }

                std::cerr << "[SUCCESS] ✅ All required functions loaded." << std::endl;

                // ===== 調試輸出 4: 創建 engine =====
                std::cerr << "[DEBUG] Calling signal_create(\"{}\")..." << std::endl;
                
                // 創建 signal engine (空配置)
                signal_engine_handle_ = signal_create_("{}");
                
                if (!signal_engine_handle_)
                {
                    std::cerr << "[ERROR] ❌ signal_create returned NULL!" << std::endl;
                    std::cerr << "[ERROR] Engine creation failed." << std::endl;
                    std::cerr << "============================================\n" << std::endl;
                    
                    SPDLOG_ERROR("signal_create returned null, engine creation failed");
                    dlclose(signal_lib_handle_);
                    signal_lib_handle_ = nullptr;
                    signal_create_ = nullptr;
                    signal_register_callback_ = nullptr;
                    signal_on_data_ = nullptr;
                    signal_destroy_ = nullptr;
                    return;
                }

                std::cerr << "[SUCCESS] ✅ Signal engine created successfully!" << std::endl;
                std::cerr << "[DEBUG] Engine handle: " << signal_engine_handle_ << std::endl;

                // ===== 調試輸出 5: 註冊回調 =====
                if (signal_register_callback_)
                {
                    std::cerr << "[DEBUG] Registering factor callback..." << std::endl;
                    
                    signal_register_callback_(signal_engine_handle_,
                        [](const char* symbol, long long ts, const double* values, int count, void* ud) {
                            Runner* self = static_cast<Runner*>(ud);
                            self->on_factor_callback(symbol, ts, values, count);
                        },
                        this);
                    
                    std::cerr << "[SUCCESS] ✅ Factor callback registered!" << std::endl;
                    SPDLOG_INFO("Signal callback registered successfully");
                }
                else
                {
                    std::cerr << "[WARNING] ⚠️  signal_register_callback is NULL, skipping..." << std::endl;
                }

                // ===== 調試輸出 6: 最終成功 =====
                std::cerr << "\n============================================" << std::endl;
                std::cerr << "[SUCCESS] 🎉 Signal library fully initialized!" << std::endl;
                std::cerr << "[SUCCESS] Library path: " << lib_path << std::endl;
                std::cerr << "============================================\n" << std::endl;
                
                SPDLOG_INFO("Signal library loaded successfully: {}", lib_path);
            }

            void Runner::on_factor_callback(const char* symbol, long long timestamp, const double* values, int count)
            {
                std::cerr << "[FACTOR] 🎊 Received factor for " << symbol 
                          << " @ " << timestamp << " (count=" << count << ")" << std::endl;
                
                SPDLOG_DEBUG("Received factor for {} @ {}: count={}", symbol, timestamp, count);

                // 調用所有策略的 on_factor 回調
                std::vector<double> factor_values(values, values + count);
                for (auto& [id, strategy] : strategies_)
                {
                    std::cerr << "[FACTOR] Calling strategy on_factor for strategy_id=" << id << std::endl;
                    context_->set_current_strategy_index(id);
                    strategy->on_factor(context_, std::string(symbol), timestamp, factor_values);
                }
                
                std::cerr << "[FACTOR] ✅ on_factor completed" << std::endl;
            }

            void Runner::on_start()
            {
                context_ = make_context();
                context_->react();

                for (const auto &strategy : strategies_)
                {
                    context_->set_current_strategy_index(strategy.first);
                    strategy.second->pre_start(context_);
                }

                events_ | is(msg::type::Depth) |
                $([&](event_ptr event)
                {
                    for (const auto &strategy : strategies_)
                    {
                        context_->set_current_strategy_index(strategy.first);
                        if (context_->is_subscribed("depth", strategy.first, event->data<Depth>())) {
                            strategy.second->on_depth(context_, event->data<Depth>());
                        }
                    }

                    // 轉發到 signal library (type=101 for Depth)
                    if (signal_on_data_ && signal_engine_handle_)
                    {
                        signal_on_data_(signal_engine_handle_, 101, &(event->data<Depth>()));
                    }
                });

                events_ | is(msg::type::Ticker) |
                $([&](event_ptr event)
                {
                    for (const auto &strategy : strategies_)
                    {
                        context_->set_current_strategy_index(strategy.first);
                        if (context_->is_subscribed("ticker", strategy.first, event->data<Ticker>())) {
                            strategy.second->on_ticker(context_, event->data<Ticker>());
                        }
                    }
                });

                events_ | is(msg::type::Trade) |
                $([&](event_ptr event)
                {
                    for (const auto &strategy : strategies_)
                    {
                        context_->set_current_strategy_index(strategy.first);
                        if (context_->is_subscribed("trade", strategy.first, event->data<Trade>())) {
                            strategy.second->on_trade(context_, event->data<Trade>());
                        }
                    }

                    // 轉發到 signal library (type=103 for Trade)
                    if (signal_on_data_ && signal_engine_handle_)
                    {
                        signal_on_data_(signal_engine_handle_, 103, &(event->data<Trade>()));
                    }
                });

                events_ | is(msg::type::IndexPrice) |
                $([&](event_ptr event)
                {
                    for (const auto &strategy : strategies_)
                    {
                        context_->set_current_strategy_index(strategy.first);
                        if (context_->is_subscribed("index_price", strategy.first, event->data<IndexPrice>())) {
                            strategy.second->on_index_price(context_, event->data<IndexPrice>());
                        }
                    }
                });

                /* 3 */
                // events_ | is(msg::type::Bar) | filter([=](event_ptr event) { return context_->is_subscribed("bar", event->data<Bar>());}) |
                // $([&](event_ptr event)
                //   {
                //       for (const auto &strategy : strategies_)
                //       {
                //           strategy->on_bar(context_, event->data<Bar>());
                //       }
                //   });

                events_ | is(msg::type::Order) | to(context_->app_.get_home_uid()) |
                $([&](event_ptr event)
                {
                    auto order = event->data<Order>();
                    for (const auto &strategy : strategies_)
                    {
                        if (order.strategy_id == strategy.first)
                        {
                            // if ((order.status == OrderStatus::Filled or order.status == OrderStatus::Cancelled) and order.volume_traded * order.avg_price > 0)
                            // {
                            //     MetricController::GetInstance().update_trade_amount(order.strategy_id, order.symbol, order.volume_traded * order.avg_price);
                            // }
                            context_->set_current_strategy_index(strategy.first);
                            strategy.second->on_order(context_, order);
                            break;
                        }
                    }
                });

                // events_ | is(msg::type::OrderActionError) | to(context_->app_.get_home_uid()) |
                // $([&](event_ptr event)
                // {
                //     for (const auto &strategy : strategies_ | indexed(0))
                //     {
                //         if (context_->is_subscribed("order", strategy.first, event->data<Order>())) {
                //             strategy.value()->on_order_action_error(context_, event->data<OrderActionError>());
                //         }
                //     }
                // });

                events_ | is(msg::type::MyTrade) | to(context_->app_.get_home_uid()) |
                $([&](event_ptr event)
                {
                    auto myTrade = event->data<MyTrade>();
                    auto itr = strategies_.find(myTrade.strategy_id);
                    if (itr!= strategies_.end()) {
                        context_->set_current_strategy_index(itr->first);
                        itr->second->on_transaction(context_, myTrade);
                    }
                });

                events_ | is(msg::type::Position) |
                $([&](event_ptr event)
                {
                    auto position = event->data<Position>();
                    for (const auto &strategy : strategies_)
                    {
                        context_->set_current_strategy_index(strategy.first);
                        strategy.second->on_position(context_, position);
                    }
                });

                events_ | is(msg::type::UnionResponse) | to(context_->app_.get_home_uid()) |
                $([&](event_ptr event)
                {
                    nlohmann::json sub_msg = nlohmann::json::parse(event->data_as_string());
                    auto itr = strategies_.find(sub_msg["strategy_id"]);
                    if (itr!= strategies_.end()) {
                        context_->set_current_strategy_index(itr->first);
                        itr->second->on_union_response(context_, sub_msg.dump());
                    }
                });

                apprentice::on_start();

                // 加載 signal library (hf-live)
                load_signal_library();

                for (const auto &strategy : strategies_)
                {
                    context_->set_current_strategy_index(strategy.first);
                    strategy.second->post_start(context_);
                }
            }

            void Runner::on_exit()
            {
                for (const auto &strategy : strategies_)
                {
                    context_->set_current_strategy_index(strategy.first);
                    strategy.second->pre_stop(context_);
                }

                apprentice::on_exit();

                for (const auto &strategy : strategies_)
                {
                    context_->set_current_strategy_index(strategy.first);
                    strategy.second->post_stop(context_);
                }
            }

        }
    }
}
