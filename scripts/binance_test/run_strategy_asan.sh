#!/bin/bash
# run_strategy_asan.sh - 使用 ASAN 運行策略
export ASAN_OPTIONS="detect_leaks=0:symbolize=1:abort_on_error=1:print_stacktrace=1"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.6

echo "============================================"
echo "🔍 Running with AddressSanitizer"
echo "ASAN_OPTIONS: $ASAN_OPTIONS"
echo "LD_PRELOAD: $LD_PRELOAD"
echo "============================================"

cd /app
exec python3 core/python/dev_run.py -l info strategy -n test_hf_live -p strategies/test_hf_live/test_hf_live.py -c scripts/binance_test/conf.json
