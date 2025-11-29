#!/bin/bash

WORK_HOME=`dirname $0`

start() {
    echo "清空 journal..."
    find ~/.config/kungfu/app/ -name "*.journal" 2>/dev/null | xargs rm -f

    # Start Master
    pm2 start /app/scripts/binance_test/master.json
    echo "啟動 master..."
    sleep 5

    # Start Ledger
    pm2 start /app/scripts/binance_test/ledger.json
    echo "啟動 ledger..."
    sleep 5

    # Start MD (Market Data)
    pm2 start /app/scripts/binance_test/md_binance.json
    echo "啟動 md_binance..."
    sleep 5

    # Start TD (Trading)
    pm2 start /app/scripts/binance_test/td_binance.json
    echo "啟動 td_binance..."
    sleep 5

    # Start demo_future Strategy
    pm2 start $WORK_HOME/strategy_demo_future.json
    echo "啟動 strategy_demo_future..."
    sleep 2

    echo ""
    echo "=========================================="
    echo "  所有服務已啟動！"
    echo "=========================================="
    pm2 list
}

stop() {
    master_pid=`ps -ef | grep python | grep master | awk '{ print $2 }'`
    if [ "$master_pid" != "" ]; then
        kill -2 $master_pid
    fi
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
