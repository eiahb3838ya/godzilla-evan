#!/bin/bash

start() {
    echo "清空 journal..."
    find ~/.config/kungfu/app/ -name "*.journal" 2>/dev/null | xargs rm -f

    # 1. Start Master
    pm2 start /app/scripts/binance_test/master.json
    echo "啟動 master..."
    sleep 5

    # 2. Start Ledger
    pm2 start /app/scripts/binance_test/ledger.json
    echo "啟動 ledger..."
    sleep 5

    # 3. Start MD (Market Data)
    pm2 start /app/scripts/binance_test/md_binance.json
    echo "啟動 md_binance..."
    sleep 5

    # 4. Start TD (Trading)
    pm2 start /app/scripts/binance_test/td_binance.json
    echo "啟動 td_binance..."
    sleep 5

    # 5. Start test_hf_live Strategy
    pm2 start /app/scripts/test_hf_live/strategy.json
    echo "啟動 strategy_test_hf_live..."
    sleep 2

    echo ""
    echo "=========================================="
    echo "  🎉 test_hf_live 端到端測試已啟動！"
    echo "=========================================="
    pm2 list
}

stop() {
    pm2 stop all && pm2 delete all
}

if [ $# -lt 1 ]; then
    echo "用法: ./run.sh [start/stop]"
    exit 1
fi

if [ "$1" = "start" ]; then
    start
elif [ "$1" = "stop" ]; then
    stop
else
    echo "無效操作: $1"
fi
