#!/bin/bash
# Phase 4G 懸空指針修復測試腳本
# 使用方法: docker exec -it godzilla-dev bash /app/scripts/test_phase4g.sh

set -e

echo "🧪 Phase 4G: Dangling Pointer Fix Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "測試目標: 驗證 signal_sender.h:59 的懸空指針修復"
echo "數據流: Binance → MD → Factor → Model → SignalSender → Python on_factor"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Stage 1: 編譯
echo ""
echo "📦 Stage 1: 編譯 libsignal.so"
cd /app/hf-live/build
rm -f libsignal.so
make clean
echo "Running cmake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-O2 -g'
echo "Running make -j4..."
make -j4

# 驗證編譯
if [ -f "libsignal.so" ]; then
    echo "✅ 編譯成功!"
    ls -lh libsignal.so
else
    echo "❌ 編譯失敗! libsignal.so 不存在"
    exit 1
fi

# Stage 2: 清理
echo ""
echo "🧹 Stage 2: 深度清理系統"
cd /app/scripts/binance_test
bash graceful_shutdown.sh
sleep 3

# 驗證清理
echo ""
echo "驗證清理結果..."
pm2 list
JOURNAL_COUNT=$(find /app/runtime -name "*.journal" 2>/dev/null | wc -l)
SOCKET_COUNT=$(find /app/runtime -name "*.nn" -o -name "*.sock" 2>/dev/null | wc -l)
echo "Journal 文件數: $JOURNAL_COUNT (預期: 0)"
echo "Socket 文件數: $SOCKET_COUNT (預期: 0)"

if [ "$JOURNAL_COUNT" -ne "0" ] || [ "$SOCKET_COUNT" -ne "0" ]; then
    echo "⚠️  警告: 清理未完全完成"
fi

# Stage 3: 重啟
echo ""
echo "🚀 Stage 3: 重啟服務"
cd /app/scripts/binance_test
./run.sh start

echo "等待服務啟動 (5秒)..."
sleep 5

echo "啟動策略..."
pm2 start /app/scripts/test_hf_live/strategy.json

echo "等待策略穩定 (3秒)..."
sleep 3

echo ""
echo "當前服務狀態:"
pm2 list

# 檢查所有服務是否 online
ONLINE_COUNT=$(pm2 jlist | jq '[.[] | select(.pm2_env.status=="online")] | length')
echo ""
echo "Online 服務數: $ONLINE_COUNT (預期: 5)"

if [ "$ONLINE_COUNT" -lt "5" ]; then
    echo "❌ 部分服務未成功啟動!"
    exit 1
fi

# Stage 4: P0 測試
echo ""
echo "✅ Stage 4: P0 測試 (60秒)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "等待系統穩定 (10秒)..."
sleep 10

echo ""
echo "=== 測試開始時間 ==="
date
START_TIME=$(date +%s)

echo ""
echo "等待 60 秒讓數據流完整運行..."
sleep 60

echo ""
echo "=== 測試結束時間 ==="
date
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "實際運行時間: ${DURATION} 秒"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 P0 測試結果分析"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 檢查 1: 記憶體錯誤 (最關鍵!)
echo ""
echo "=== 檢查 1: 記憶體錯誤 (最關鍵!) ==="
MEMORY_ERRORS=$(tail -200 /root/.pm2/logs/strategy-test-hf-live-error.log 2>/dev/null | grep -i "free\|corruption\|segmentation" | wc -l)
if [ "$MEMORY_ERRORS" -eq "0" ]; then
    echo "✅ PASS: 無記憶體錯誤"
else
    echo "❌ FAIL: 發現 $MEMORY_ERRORS 個記憶體錯誤!"
    echo ""
    echo "錯誤日誌:"
    tail -50 /root/.pm2/logs/strategy-test-hf-live-error.log | grep -i "free\|corruption\|segmentation"
    exit 1
fi

# 檢查 2: Restart count
echo ""
echo "=== 檢查 2: Restart Count ==="
RESTART=$(pm2 jlist | jq '.[] | select(.name=="strategy_test_hf_live") | .pm2_env.restart_time')
if [ "$RESTART" -eq "0" ]; then
    echo "✅ PASS: Restart count = 0"
else
    echo "⚠️  WARNING: Restart count = $RESTART"
fi

# 檢查 3: 修復生效驗證 (關鍵!)
echo ""
echo "=== 檢查 3: 修復生效驗證 (關鍵!) ==="
FIX_COUNT=$(tail -200 /root/.pm2/logs/strategy-test-hf-live-error.log 2>/dev/null | grep -c "with safe data copy" || echo "0")
if [ "$FIX_COUNT" -gt "0" ]; then
    echo "✅ PASS: 修復已生效 (出現 $FIX_COUNT 次)"
    echo ""
    echo "修復日誌示例:"
    tail -200 /root/.pm2/logs/strategy-test-hf-live-error.log | grep "with safe data copy" | head -3
else
    echo "⚠️  WARNING: 未看到 'with safe data copy' 日誌"
    echo "這可能表示:"
    echo "  1. 數據流尚未觸發 (需要更長時間)"
    echo "  2. 修復未正確編譯"
fi

# 檢查 4: 完整數據流
echo ""
echo "=== 檢查 4: 完整數據流驗證 ==="
EMOJI_COUNT=$(tail -200 /root/.pm2/logs/strategy-test-hf-live-error.log 2>/dev/null | grep -E '🏁|📊|🔢|📤|🚀|📥|🤖|🔮|📨|🎊' | wc -l)
echo "Emoji 日誌數: $EMOJI_COUNT"

if [ "$EMOJI_COUNT" -gt "0" ]; then
    echo "✅ PASS: 發現數據流日誌"
    echo ""
    echo "最近的數據流日誌 (最多顯示 10 行):"
    tail -200 /root/.pm2/logs/strategy-test-hf-live-error.log | grep -E '🏁|📊|🔢|📤|🚀|📥|🤖|🔮|📨|🎊' | tail -10
else
    echo "⚠️  WARNING: 未發現數據流日誌"
fi

# 檢查 5: Python on_factor 回調
echo ""
echo "=== 檢查 5: Python on_factor 回調 ==="
CALLBACK_COUNT=$(grep -c "🎊 Received factor" /root/.pm2/logs/strategy-test-hf-live-error.log 2>/dev/null || echo "0")
if [ "$CALLBACK_COUNT" -gt "0" ]; then
    echo "✅ PASS: on_factor 回調成功 ($CALLBACK_COUNT 次)"
else
    echo "⚠️  WARNING: on_factor 未觸發"
fi

# 最終總結
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 測試總結"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "P0 成功標準檢查:"
echo "  1. 無記憶體錯誤: $([ "$MEMORY_ERRORS" -eq "0" ] && echo "✅ PASS" || echo "❌ FAIL")"
echo "  2. Restart = 0: $([ "$RESTART" -eq "0" ] && echo "✅ PASS" || echo "⚠️  $RESTART")"
echo "  3. 修復生效: $([ "$FIX_COUNT" -gt "0" ] && echo "✅ PASS ($FIX_COUNT)" || echo "⚠️  WARNING")"
echo "  4. 數據流完整: $([ "$EMOJI_COUNT" -gt "0" ] && echo "✅ PASS" || echo "⚠️  WARNING")"
echo "  5. on_factor回調: $([ "$CALLBACK_COUNT" -gt "0" ] && echo "✅ PASS ($CALLBACK_COUNT)" || echo "⚠️  WARNING")"
echo ""

if [ "$MEMORY_ERRORS" -eq "0" ] && [ "$RESTART" -eq "0" ] && [ "$FIX_COUNT" -gt "0" ]; then
    echo "🎉 恭喜! P0 測試通過!"
    echo ""
    echo "下一步建議:"
    echo "  - 運行 P1 測試 (2小時壓力測試):"
    echo "    sleep 7200 && pm2 list"
    echo ""
    echo "  - 檢查長時間穩定性:"
    echo "    pm2 logs strategy_test_hf_live --lines 100"
else
    echo "⚠️  P0 測試部分通過,建議檢查警告項"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "測試完成時間: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
