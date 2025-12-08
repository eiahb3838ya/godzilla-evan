#!/bin/bash
# Phase 4B 測試前清理腳本

echo "=== Phase 4B: Cleaning Environment ==="

# 1. 停止所有進程
echo "1. Stopping all PM2 processes..."
pm2 stop all 2>/dev/null
pm2 delete all 2>/dev/null
pm2 list

# 2. 清理 Runtime Journal 文件
echo "2. Cleaning runtime journals..."
rm -rf /app/runtime/strategy/default/test_hf_live/journal/live/*.journal 2>/dev/null
rm -rf /app/runtime/system/service/ledger/journal/live/*.journal 2>/dev/null
rm -rf /app/runtime/system/master/*/journal/live/*.journal 2>/dev/null

# 統計清理結果
journal_count=$(find /app/runtime -name '*.journal' 2>/dev/null | wc -l)
echo "   Remaining journals: $journal_count"

# 3. 驗證清理
echo "3. Verification..."
pm2 list
echo ""
echo "=== Environment Ready for Phase 4B ==="
