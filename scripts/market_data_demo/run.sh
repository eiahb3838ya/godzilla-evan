#!/bin/bash

WORK_HOME=`dirname $0`

start() {
    echo "clearing journal..."
    find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f
    # start master
    pm2 start master.json
    echo "starting master..."
    sleep 5
    # start ledger
    pm2 start ledger.json
    echo "starting ledger..."
    sleep 5
    # start binance md
    pm2 start md_binance.json
    echo "starting md binance..."
    sleep 5
    # start market_data_demo strategy
    pm2 start strategy.json
    echo "starting market_data_demo strategy..."
    sleep 2
}

stop() {
    master_pid=`ps -ef | grep python |  grep master | awk '{ print $2 }'`
    if [ "$master_pid" != "" ]; then
        kill -2 $master_pid
    fi
}


if [ $# -lt 1 ]; then
    echo "please indicate action [start/stop]"
    exit 1
fi
if [ "$1" = "start" ]; then
    start
elif [ "$1" = "stop" ]; then
    stop
else
    echo "invalid action: $1"
fi
