#!/bin/bash
# Phase 4B 測試前深度清理腳本

echo "=== Phase 4B: Deep Cleaning Environment ==="

# 1. Stop all PM2 processes
echo "1. Stopping all PM2 processes..."
pm2 stop all 2>/dev/null
pm2 delete all 2>/dev/null

# 2. 完全刪除所有 journal 文件（不保留任何歷史）
echo "2. Removing all journal files..."
find /app/runtime -name "*.journal" -type f -delete

# 3. 刪除所有 journal 目錄（清理 metadata）
echo "3. Removing all journal directories..."
find /app/runtime -type d -name "journal" -exec rm -rf {} + 2>/dev/null || true

# 4. 重建 journal 目錄結構
echo "4. Recreating journal directories..."
mkdir -p /app/runtime/system/service/ledger/journal/live
mkdir -p /app/runtime/md/binance/binance/journal/live
mkdir -p /app/runtime/td/binance/gz_user1/journal/live
mkdir -p /app/runtime/strategy/default/test_hf_live/journal/live

# 5. 清理 PM2 日誌
pm2 flush

# 6. 驗證清理結果
REMAINING=$(find /app/runtime -name "*.journal" 2>/dev/null | wc -l)
echo "   Remaining journals: $REMAINING (should be 0)"

# 7. PM2 狀態
pm2 list
echo ""
echo "=== Environment Fully Cleaned ==="
