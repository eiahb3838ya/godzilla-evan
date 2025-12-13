#!/bin/bash
# run_with_asan.sh - 使用 AddressSanitizer 運行策略測試
# 用途: 捕獲內存損壞問題的精確堆棧跟踪

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "🔍 Running strategy with AddressSanitizer"
echo "============================================"

# 設置 ASAN 環境變量
export ASAN_OPTIONS="detect_leaks=0:symbolize=1:abort_on_error=1:print_stacktrace=1:halt_on_error=1"

# 預加載 ASAN 運行時 (關鍵: 因為主程序沒有用 ASAN 編譯)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.6

echo "ASAN_OPTIONS: $ASAN_OPTIONS"
echo "LD_PRELOAD: $LD_PRELOAD"
echo ""

# 確保使用 ASAN 版本的 libsignal.so
SIGNAL_LIB="/app/hf-live/build/libsignal.so"
if [ -f "$SIGNAL_LIB" ]; then
    echo "Signal library: $SIGNAL_LIB"
    file "$SIGNAL_LIB" | grep -q "shared object" && echo "  ✅ Valid shared library"
else
    echo "❌ ERROR: Signal library not found at $SIGNAL_LIB"
    exit 1
fi

echo ""
echo "Starting strategy..."
echo "============================================"
echo ""

# 運行策略 (使用相同的 pm2 配置)
# 注意: pm2 會 fork 子進程，環境變量會被繼承
pm2 start strategy_hello.json --no-daemon

echo ""
echo "============================================"
echo "Test completed. Check output for ASAN reports."
echo "============================================"
