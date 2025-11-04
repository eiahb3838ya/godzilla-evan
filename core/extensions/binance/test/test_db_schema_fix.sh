#!/bin/bash
# 测试数据库 Schema 是否正确以及 TD 是否能正确读取配置
# 用于验证数据库表结构与 SQLAlchemy Model 的匹配性

set -e

echo "========================================"
echo "测试 1: 验证表结构"
echo "========================================"

# 检查表中是否有 account_id 列（正确的列名）
SCHEMA=$(docker exec godzilla-dev python3 -c "
import sqlite3
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(account_config)')
columns = [row[1] for row in cursor.fetchall()]
conn.close()
print(','.join(columns))
")

echo "当前表列: $SCHEMA"

if [[ "$SCHEMA" == *"account_id"* ]]; then
    echo "✅ 测试通过: 表包含 account_id 列"
    SCHEMA_TEST_PASS=true
else
    echo "❌ 测试失败: 表不包含 account_id 列（发现问题）"
    echo "   实际列名: $SCHEMA"
    SCHEMA_TEST_PASS=false
fi

echo ""
echo "========================================"
echo "测试 2: 验证配置读取"
echo "========================================"

# 测试 get_td_account_config 是否能成功返回配置
CONFIG_RESULT=$(docker exec godzilla-dev python3 << 'EOF'
import sys
sys.path.insert(0, "/app/core/python")

try:
    from kungfu.data.sqlite.models import Account
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:////root/.config/kungfu/app/kungfu.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # 尝试查询（模拟 get_td_account_config 的行为）
    result = session.query(Account).filter(
        Account.source_name == "binance"
    ).filter(
        Account.account_id == "binance_gz_user1"
    ).first()
    
    session.close()
    
    if result:
        print("SUCCESS")
    else:
        print("NO_RESULT")
except Exception as e:
    print(f"ERROR:{type(e).__name__}")
EOF
)

echo "配置读取结果: $CONFIG_RESULT"

if [[ "$CONFIG_RESULT" == "SUCCESS" ]]; then
    echo "✅ 测试通过: 可以成功读取配置"
    CONFIG_TEST_PASS=true
else
    echo "❌ 测试失败: 无法读取配置（$CONFIG_RESULT）"
    CONFIG_TEST_PASS=false
fi

echo ""
echo "========================================"
echo "测试 3: 验证 TD 启动"
echo "========================================"

# 停止现有进程
docker exec godzilla-dev pm2 delete all 2>/dev/null || true
sleep 2

# 清理旧日志
docker exec godzilla-dev bash -c "find /root/.config/kungfu/app/ -name '*.journal' | xargs rm -f"

# 尝试启动 TD
echo "启动系统..."
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && bash run.sh start"
sleep 10

# 检查是否有 JSON parse error
ERROR_COUNT=$(docker exec godzilla-dev bash -c "grep -c 'parse error' /root/.pm2/logs/td-binance-gz-user1-error.log 2>/dev/null || echo 0")

echo "JSON parse error 数量: $ERROR_COUNT"

if [[ "$ERROR_COUNT" == "0" ]]; then
    echo "✅ 测试通过: TD 启动无 JSON parse error"
    TD_TEST_PASS=true
else
    echo "❌ 测试失败: TD 启动有 JSON parse error"
    echo "   错误日志片段:"
    docker exec godzilla-dev bash -c "tail -20 /root/.pm2/logs/td-binance-gz-user1-error.log" | head -10
    TD_TEST_PASS=false
fi

echo ""
echo "========================================"
echo "测试 4: 验证 Market Toggle 配置"
echo "========================================"

MARKET_CONFIG=$(docker exec godzilla-dev python3 -c "
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()
# 同时查询 account_id 和 user_id，兼容两种表结构
cursor.execute('SELECT config FROM account_config WHERE account_id=\"binance_gz_user1\" OR user_id=\"binance_gz_user1\"')
row = cursor.fetchone()
if row:
    config = json.loads(row[0])
    enable_spot = config.get('enable_spot', 'MISSING')
    enable_futures = config.get('enable_futures', 'MISSING')
    print(f\"enable_spot:{enable_spot},enable_futures:{enable_futures}\")
else:
    print('NO_CONFIG')
conn.close()
" 2>/dev/null || echo "ERROR")

echo "Market toggle 配置: $MARKET_CONFIG"

if [[ "$MARKET_CONFIG" == "enable_spot:False,enable_futures:True" ]]; then
    echo "✅ 测试通过: Market toggle 配置正确"
    MARKET_TEST_PASS=true
else
    echo "❌ 测试失败: Market toggle 配置不正确或缺失"
    echo "   预期: enable_spot:False,enable_futures:True"
    echo "   实际: $MARKET_CONFIG"
    MARKET_TEST_PASS=false
fi

echo ""
echo "========================================"
echo "测试总结"
echo "========================================"
echo "Schema 测试: $SCHEMA_TEST_PASS"
echo "配置读取测试: $CONFIG_TEST_PASS"
echo "TD 启动测试: $TD_TEST_PASS"
echo "Market Toggle 测试: $MARKET_TEST_PASS"

if [[ "$SCHEMA_TEST_PASS" == "true" ]] && [[ "$CONFIG_TEST_PASS" == "true" ]] && \
   [[ "$TD_TEST_PASS" == "true" ]] && [[ "$MARKET_TEST_PASS" == "true" ]]; then
    echo ""
    echo "✅ 所有测试通过"
    exit 0
else
    echo ""
    echo "❌ 部分测试失败（这是预期的，需要修复）"
    exit 1
fi

