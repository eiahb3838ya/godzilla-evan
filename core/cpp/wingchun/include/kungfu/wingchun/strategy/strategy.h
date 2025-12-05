/**
 * This is source code modified under the Apache License 2.0.
 * Original Author: Keren Dong
 * Modifier: kx@godzilla.dev
 * Modification date: March 3, 2025
 */

#ifndef WINGCHUN_STRATEGY_H
#define WINGCHUN_STRATEGY_H

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include <kungfu/practice/apprentice.h>
#include <kungfu/wingchun/msg.h>

namespace kungfu
{
    namespace wingchun
    {
        namespace strategy
        {
            FORWARD_DECLARE_PTR(Context)

            class Strategy
            {
            public:
                virtual ~Strategy() = default;

                //运行前
                virtual void pre_start(Context_ptr context)
                {};

                virtual void post_start(Context_ptr context)
                {};

                //退出前
                virtual void pre_stop(Context_ptr context)
                {};

                virtual void post_stop(Context_ptr context)
                {};

                //深度数据更新回调
                //@param depth             行情数据
                virtual void on_depth(Context_ptr context, const msg::data::Depth &depth)
                {};

                //深度数据更新回调
                //@param ticker             Ticker数据
                virtual void on_ticker(Context_ptr context, const msg::data::Ticker &ticker)
                {};

                virtual void on_index_price(Context_ptr context, const msg::data::IndexPrice &ip)
                {};

                //bar 数据更新回调
                //@param bar               bar 数据
                virtual void on_bar(Context_ptr context, const msg::data::Bar &bar)
                {};

                //逐笔成交更新回调
                //@param transaction       逐笔成交数据
                virtual void on_transaction(Context_ptr context, const msg::data::MyTrade &transaction)
                {};

                //订单信息更新回调
                //@param order             订单信息数据
                virtual void on_order(Context_ptr context, const msg::data::Order &order)
                {};

                //订单操作错误回调
                //@param order             订单信息数据
                virtual void on_order_action_error(Context_ptr context, const msg::data::OrderActionError &error)
                {};

                //订单成交回报回调
                //@param trade             订单成交数据
                virtual void on_trade(Context_ptr context, const msg::data::Trade &trade)
                {};

                virtual void on_position(Context_ptr context, const msg::data::Position &position)
                {};

                virtual void on_union_response(Context_ptr context, const std::string &msg)
                {};

                //因子數據回調 (hf-live signal library)
                //@param symbol            交易標的
                //@param timestamp         時間戳 (納秒)
                //@param values            因子值陣列
                virtual void on_factor(Context_ptr context, const std::string &symbol, long long timestamp, const std::vector<double> &values)
                {};

                //设置策略UID
                //@param id                策略UID
                void set_uid(uint32_t id) {
                    uid_ = id;
                }

                uint32_t get_uid() {
                    return uid_;
                }

            private:
                uint32_t uid_;

            };

            DECLARE_PTR(Strategy)
        }
    }
}
#endif //WINGCHUN_STRATEGY_H
