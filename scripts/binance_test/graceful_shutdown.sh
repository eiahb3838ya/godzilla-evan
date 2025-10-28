#!/bin/bash

echo "=========================================="
echo "  Graceful Shutdown & Cleanup Script"
echo "=========================================="

# Step 1: Stop all PM2 processes gracefully
echo -e "\n[1/5] Stopping PM2 processes..."
pm2 stop all
sleep 2
pm2 delete all
echo "✅ PM2 processes stopped"

# Step 2: Kill any remaining Python processes
echo -e "\n[2/5] Killing remaining Python processes..."
pkill -15 python  # SIGTERM (graceful)
sleep 2
pkill -9 python   # SIGKILL (force) for any stubborn ones
echo "✅ Python processes killed"

# Step 3: Clean journal files
echo -e "\n[3/5] Cleaning journal files..."
find /app/runtime -name "*.journal" -type f -delete 2>/dev/null
find ~/.config/kungfu/app/ -name "*.journal" -type f -delete 2>/dev/null
echo "✅ Journal files cleaned"

# Step 4: Clean socket files
echo -e "\n[4/5] Cleaning socket files..."
find /app/runtime -name "*.nn" -type s -delete 2>/dev/null
find /app/runtime -name "*.sock" -type s -delete 2>/dev/null
echo "✅ Socket files cleaned"

# Step 5: Clean log files (optional)
echo -e "\n[5/5] Cleaning old log files..."
find /app/runtime/log -name "*.log" -type f -mtime +7 -delete 2>/dev/null
echo "✅ Old log files cleaned (kept recent 7 days)"

echo -e "\n=========================================="
echo "  Cleanup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
pm2 list
echo ""
echo "Remaining Python processes:"
ps aux | grep python | grep -v grep || echo "  (none)"
echo ""
echo "You can now safely restart with:"
echo "  cd /app/scripts/binance_test && bash run.sh start"

