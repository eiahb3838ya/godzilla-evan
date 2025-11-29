#!/bin/bash
# Demo Future 策略啟動腳本
# 用法: ./scripts/start_demo_future.sh

set -e  # 遇到錯誤立即退出

PROJECT_ROOT="/home/huyifan/projects/godzilla-evan"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  啟動 Demo Future 策略"
echo "=========================================="

# 檢查賬戶配置
echo ""
echo "1. 檢查 Binance 賬戶配置..."
if python3 core/python/dev_run.py account -s binance list | grep -q "gz_user1"; then
    echo "   ✓ 賬戶 gz_user1 已配置"
else
    echo "   ✗ 賬戶 gz_user1 未配置，請先運行:"
    echo "     python3 core/python/dev_run.py account -s binance add"
    exit 1
fi

# 啟動 Master
echo ""
echo "2. 啟動 Master 服務..."
python3 core/python/dev_run.py -l info master &
MASTER_PID=$!
echo "   Master PID: $MASTER_PID"
sleep 5

# 啟動 Ledger
echo ""
echo "3. 啟動 Ledger 服務..."
python3 core/python/dev_run.py -l info ledger &
LEDGER_PID=$!
echo "   Ledger PID: $LEDGER_PID"
sleep 5

# 啟動 MD (Market Data)
echo ""
echo "4. 啟動 MD 閘道器 (Binance Futures)..."
python3 core/python/dev_run.py -l trace md -s binance &
MD_PID=$!
echo "   MD PID: $MD_PID"
sleep 5

# 啟動 TD (Trading)
echo ""
echo "5. 啟動 TD 閘道器 (Binance Futures, 賬戶: gz_user1)..."
python3 core/python/dev_run.py -l info td -s binance -a gz_user1 &
TD_PID=$!
echo "   TD PID: $TD_PID"
sleep 5

# 啟動策略
echo ""
echo "6. 啟動 demo_future 策略..."
python3 core/python/dev_run.py -l info strategy -n demo_future \
    -p strategies/demo_future/demo_future.py \
    -c strategies/demo_future/config.json &
STRATEGY_PID=$!
echo "   Strategy PID: $STRATEGY_PID"

# 保存 PID 到文件
cat > /tmp/kungfu_pids.txt <<EOF
MASTER_PID=$MASTER_PID
LEDGER_PID=$LEDGER_PID
MD_PID=$MD_PID
TD_PID=$TD_PID
STRATEGY_PID=$STRATEGY_PID
EOF

echo ""
echo "=========================================="
echo "  所有服務已啟動！"
echo "=========================================="
echo ""
echo "進程 PID："
echo "  Master:   $MASTER_PID"
echo "  Ledger:   $LEDGER_PID"
echo "  MD:       $MD_PID"
echo "  TD:       $TD_PID"
echo "  Strategy: $STRATEGY_PID"
echo ""
echo "查看日誌命令："
echo "  tail -f ~/.config/kungfu/app/runtime/log/strategy/demo_future/strategy.log"
echo ""
echo "停止所有服務命令："
echo "  ./scripts/stop_all.sh"
echo ""
