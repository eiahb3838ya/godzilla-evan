/**
 * This is source code modified under the Apache License 2.0.
 * Original Author: Keren Dong
 * Modifier: kx@godzilla.dev
 * Modification date: March 3, 2025
 */

#ifndef WINGCHUN_RUNNER_H
#define WINGCHUN_RUNNER_H

#include <kungfu/practice/apprentice.h>
#include <kungfu/wingchun/strategy/context.h>
#include <kungfu/wingchun/strategy/strategy.h>

namespace kungfu
{
    namespace wingchun
    {
        namespace strategy
        {
            class Runner : public practice::apprentice
            {
            public:
                Runner(yijinjing::data::locator_ptr locator, const std::string &group, const std::string &name, yijinjing::data::mode m, bool low_latency);

                virtual ~Runner();

                void add_strategy(const Strategy_ptr &strategy, const std::string &path);

            protected:
                std::map<uint32_t, Strategy_ptr> strategies_;

                void on_start() override ;

                void on_exit() override ;

                void register_location(int64_t trigger_time, const yijinjing::data::location_ptr &location) override;

                virtual Context_ptr make_context();

            private:
                Context_ptr context_;

                // Signal library integration (hf-live)
                void* signal_lib_handle_ = nullptr;
                void* signal_engine_handle_ = nullptr;

                // Function pointers
                typedef void* (*signal_create_fn)(const char*);
                typedef void (*signal_register_callback_fn)(void*, void (*)(const char*, long long, const double*, int, void*), void*);
                typedef void (*signal_on_data_fn)(void*, int, const void*);
                typedef void (*signal_destroy_fn)(void*);

                signal_create_fn signal_create_ = nullptr;
                signal_register_callback_fn signal_register_callback_ = nullptr;
                signal_on_data_fn signal_on_data_ = nullptr;
                signal_destroy_fn signal_destroy_ = nullptr;

                void load_signal_library();
                void on_factor_callback(const char* symbol, long long timestamp, const double* values, int count);
            };
        }
    }
}

#endif //WINGCHUN_RUNNER_H
