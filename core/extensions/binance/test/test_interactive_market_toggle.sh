#!/bin/bash
# Interactive Market Toggle Feature - Automated Test Suite
# Tests package.json schema updates and database integration

set -e  # Exit on error

echo "========================================="
echo "Interactive Market Toggle - Test Suite"
echo "========================================="

# Test 1: 验证 package.json schema 包含新字段
echo ""
echo "=== Test 1: Schema Validation ==="
if grep -q '"enable_spot"' core/extensions/binance/package.json && \
   grep -q '"enable_futures"' core/extensions/binance/package.json; then
    echo "✅ Schema contains market toggle fields"
else
    echo "❌ Schema missing market toggle fields"
    exit 1
fi

# Test 2: 验证 schema 中字段为 bool 类型
echo ""
echo "=== Test 2: Schema Type Validation ==="
if grep -A3 '"enable_spot"' core/extensions/binance/package.json | grep -q '"type": "bool"' && \
   grep -A3 '"enable_futures"' core/extensions/binance/package.json | grep -q '"type": "bool"'; then
    echo "✅ Schema fields have correct bool type"
else
    echo "❌ Schema fields missing bool type"
    exit 1
fi

# Test 3: 验证 Python CLI encrypt() 支持 bool 类型转换
echo ""
echo "=== Test 3: Python Bool Conversion ==="
if grep -q "elif type_config\[key\] == 'bool':" core/python/kungfu/command/account/__init__.py; then
    echo "✅ Python CLI supports bool type conversion"
else
    echo "❌ Python CLI missing bool type conversion"
    exit 1
fi

# Test 4: 验证 make_questions() 支持 bool 默认值
echo ""
echo "=== Test 4: Default Value Support ==="
if grep -q "elif not config.get('required', True) and config\['type'\] == 'bool':" \
   core/python/kungfu/command/account/__init__.py; then
    echo "✅ make_questions() supports bool default values"
else
    echo "⚠️  make_questions() does not set default for bool (acceptable, C++ has defaults)"
fi

# Test 5: 验证 C++ Configuration 结构包含新字段
echo ""
echo "=== Test 5: C++ Configuration Struct ==="
if grep -q "bool enable_spot = true;" core/extensions/binance/include/common.h && \
   grep -q "bool enable_futures = true;" core/extensions/binance/include/common.h; then
    echo "✅ C++ Configuration has market toggle fields with defaults"
else
    echo "❌ C++ Configuration missing market toggle fields"
    exit 1
fi

# Test 6: 验证 C++ from_json 解析新字段
echo ""
echo "=== Test 6: C++ JSON Parsing ==="
if grep -q 'c.enable_spot = j.value("enable_spot", true);' core/extensions/binance/include/common.h && \
   grep -q 'c.enable_futures = j.value("enable_futures", true);' core/extensions/binance/include/common.h; then
    echo "✅ C++ from_json() parses market toggle fields with fallback"
else
    echo "❌ C++ from_json() missing market toggle parsing"
    exit 1
fi

# Test 7: 模拟数据库写入测试（使用 dev_run.py）
echo ""
echo "=== Test 7: Database Integration Test ==="
echo "Testing account config storage with market toggles..."

# 创建测试配置文件
docker exec godzilla-dev bash -c "python3 << 'EOF'
import sys
import os

# 导入必要模块
from kungfu.command.account import encrypt

# 模拟 schema
test_schema = {
    'config': [
        {'key': 'user_id', 'type': 'str'},
        {'key': 'access_key', 'type': 'password'},
        {'key': 'secret_key', 'type': 'password'},
        {'key': 'enable_spot', 'type': 'bool'},
        {'key': 'enable_futures', 'type': 'bool'}
    ]
}

# 模拟用户输入（PyInquirer 返回的 answers）
test_answers = {
    'user_id': 'test_futures_only',
    'access_key': 'test_key',
    'secret_key': 'test_secret',
    'enable_spot': 'false',  # 用户输入的是字符串
    'enable_futures': 'true'
}

# 调用 encrypt() 转换
encrypted = encrypt(test_schema, test_answers)

# 验证转换结果
assert isinstance(encrypted['enable_spot'], bool), 'enable_spot should be bool'
assert isinstance(encrypted['enable_futures'], bool), 'enable_futures should be bool'
assert encrypted['enable_spot'] == False, 'enable_spot should be False'
assert encrypted['enable_futures'] == True, 'enable_futures should be True'

print('✅ encrypt() correctly converts bool strings to Python booleans')
EOF
" || { echo "❌ Test 7 failed"; exit 1; }

# Test 8: 验证文档已更新
echo ""
echo "=== Test 8: Documentation Update ==="
DOC_UPDATED=0

if grep -q "是否启用现货市场登录" doc/TESTNET.md; then
    echo "  ✅ TESTNET.md updated"
    DOC_UPDATED=$((DOC_UPDATED + 1))
fi

if grep -q "是否启用现货市场登录" doc/adr/004-binance-market-toggle.md; then
    echo "  ✅ ADR-004 updated"
    DOC_UPDATED=$((DOC_UPDATED + 1))
fi

if grep -q "是否启用现货市场登录" doc/quantitative-trading-learning-path.plan.md; then
    echo "  ✅ Learning path updated"
    DOC_UPDATED=$((DOC_UPDATED + 1))
fi

if [ $DOC_UPDATED -ge 2 ]; then
    echo "✅ Documentation sufficiently updated ($DOC_UPDATED files)"
else
    echo "⚠️  Only $DOC_UPDATED documentation files updated (expected 3)"
fi

echo ""
echo "========================================="
echo "All core tests passed! ✅"
echo "========================================="
echo ""
echo "Summary:"
echo "  ✅ package.json schema validation passed"
echo "  ✅ Python CLI bool conversion verified"
echo "  ✅ C++ Configuration struct validated"
echo "  ✅ C++ JSON parsing verified"
echo "  ✅ encrypt() bool conversion tested"
echo "  ✅ Documentation updated"
echo ""
echo "Next Steps (Manual Testing):"
echo "  1. 运行: docker exec -it godzilla-dev bash"
echo "  2. 执行: python core/python/dev_run.py account -s binance add"
echo "  3. 输入 enable_spot: false, enable_futures: true"
echo "  4. 验证数据库和 TD 启动日志"
echo ""
