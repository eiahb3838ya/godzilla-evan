# Interactive Market Toggle - Manual Integration Test Guide

## 目的

验证交互式账户创建时可以正确选择 Spot/Futures 市场开关，并确保配置正确传递到 C++ TD Gateway。

## 前提条件

- ✅ Docker 容器 `godzilla-dev` 正在运行
- ✅ 自动化测试已全部通过
- ✅ 有可用的 Binance Testnet API Key（Futures-only 或 Full access）

## Test Case 1: Futures-Only Account（期货专用账户）

### Step 1: 清理现有账户（如果存在）

```bash
# 停止所有服务
docker exec godzilla-dev pm2 stop all

# 删除现有账户数据
docker exec godzilla-dev rm -rf /app/runtime/td/binance/gz_user1

# 从数据库删除账户
docker exec godzilla-dev sqlite3 /app/runtime/system/etc/kungfu/db/live/accounts.db \
    "DELETE FROM account_config WHERE account_id='binance_gz_user1'"
```

### Step 2: 交互式创建账户

```bash
# 进入容器
docker exec -it godzilla-dev bash

# 运行交互式命令
python core/python/dev_run.py account -s binance add
```

**交互式输入**:
```
? 请填写账户 user_id: gz_user1
? 请填写access_key: [YOUR_FUTURES_API_KEY]
? 请填写 secret_key: [YOUR_FUTURES_SECRET_KEY]
? 是否启用现货市场登录？(true/false) [true]: false    ← 输入 false
? 是否启用期货市场登录？(true/false) [true]: true     ← 按回车或输入 true
```

### Step 3: 验证数据库配置

```bash
# 查询数据库
docker exec godzilla-dev sqlite3 /app/runtime/system/etc/kungfu/db/live/accounts.db \
    'SELECT config FROM account_config WHERE account_id="binance_gz_user1"' | python3 -m json.tool
```

**预期输出**:
```json
{
    "user_id": "gz_user1",
    "access_key": "YOUR_KEY",
    "secret_key": "YOUR_SECRET",
    "enable_spot": false,      ← 应该是布尔值 false
    "enable_futures": true     ← 应该是布尔值 true
}
```

**验证点**:
- ✅ `enable_spot` 为布尔值 `false`（不是字符串 `"false"`）
- ✅ `enable_futures` 为布尔值 `true`

### Step 4: 启动 TD Gateway 并检查日志

```bash
# 启动服务（在容器内）
pm2 start ecosystem.config.js --only td_binance

# 等待 2-3 秒让服务启动

# 检查日志
pm2 logs td_binance --lines 30 | grep -E '(Spot|Futures|disabled|enabled|configuration)'
```

**预期日志输出**:
```
[trader_binance.cpp:63] Spot market disabled by configuration
[trader_binance.cpp:98] Connecting BINANCE TD for gz_user1 (Spot: disabled, Futures: enabled)
```

**验证点**:
- ✅ 看到 "Spot market disabled by configuration"
- ✅ **没有** 看到 `-2015` 错误（Invalid API-key for Spot）
- ✅ Futures 市场正常初始化

### Step 5: 验证服务状态

```bash
pm2 list
```

**预期输出**:
```
┌─────┬───────────────────┬─────────┬─────────┐
│ id  │ name              │ status  │ restart │
├─────┼───────────────────┼─────────┼─────────┤
│ 0   │ td_binance        │ online  │ 0       │  ← 应该是 online
└─────┴───────────────────┴─────────┴─────────┘
```

---

## Test Case 2: 默认行为（向后兼容测试）

### Step 1: 删除账户并重新创建

```bash
# 停止服务
docker exec godzilla-dev pm2 stop all

# 删除账户
docker exec godzilla-dev rm -rf /app/runtime/td/binance/gz_user1
docker exec godzilla-dev sqlite3 /app/runtime/system/etc/kungfu/db/live/accounts.db \
    "DELETE FROM account_config WHERE account_id='binance_gz_user1'"
```

### Step 2: 创建账户时使用默认值

```bash
docker exec -it godzilla-dev bash
python core/python/dev_run.py account -s binance add
```

**交互式输入**（对市场开关问题直接按回车）:
```
? 请填写账户 user_id: gz_user1
? 请填写access_key: [YOUR_API_KEY]
? 请填写 secret_key: [YOUR_SECRET_KEY]
? 是否启用现货市场登录？(true/false) [true]: ← 直接按回车
? 是否启用期货市场登录？(true/false) [true]: ← 直接按回车
```

### Step 3: 验证默认值

```bash
docker exec godzilla-dev sqlite3 /app/runtime/system/etc/kungfu/db/live/accounts.db \
    'SELECT config FROM account_config WHERE account_id="binance_gz_user1"' | python3 -m json.tool
```

**预期输出**:
```json
{
    "user_id": "gz_user1",
    "access_key": "YOUR_KEY",
    "secret_key": "YOUR_SECRET",
    "enable_spot": true,      ← 默认值 true
    "enable_futures": true    ← 默认值 true
}
```

### Step 4: 验证两个市场都尝试初始化

```bash
pm2 start ecosystem.config.js --only td_binance
pm2 logs td_binance --lines 30
```

**预期**:
- ✅ 看到 Spot 和 Futures 都尝试初始化
- ⚠️ 如果 API Key 是 Futures-only，会看到 Spot 的 `-2015` 错误（正常行为）

---

## Test Case 3: 旧账户兼容性（缺失字段）

### Purpose
验证没有 `enable_spot`/`enable_futures` 字段的旧账户仍能正常运行。

### Step 1: 手动创建旧格式账户

```bash
docker exec godzilla-dev bash -c "python3 << 'EOF'
from kungfu.data.sqlite.data_proxy import AccountsDB
import pyyjj
import kungfu.yijinjing.journal as kfj
import os

home = os.getenv('KF_HOME', '/app/runtime')
locator = kfj.Locator(home)
location = pyyjj.location(pyyjj.mode.LIVE, pyyjj.category.SYSTEM, 'etc', 'kungfu', locator)
db = AccountsDB(location, 'accounts')

# 旧格式配置（无市场开关字段）
old_config = {
    'user_id': 'gz_user1',
    'access_key': 'YOUR_KEY',
    'secret_key': 'YOUR_SECRET'
}

db.add_account('binance_gz_user1', 'binance', False, old_config)
print('Old format account created')
EOF
"
```

### Step 2: 启动 TD 并验证

```bash
docker exec godzilla-dev pm2 start ecosystem.config.js --only td_binance
docker exec godzilla-dev pm2 logs td_binance --lines 20
```

**预期**:
- ✅ TD 正常启动（C++ 使用默认值 `enable_spot=true, enable_futures=true`）
- ✅ 两个市场都尝试初始化
- ✅ 没有 crash 或 schema 错误

---

## 成功标准总结

- ✅ Test Case 1: Futures-only 账户创建成功，Spot 被禁用，无 `-2015` 错误
- ✅ Test Case 2: 默认值正确应用（两个市场都启用）
- ✅ Test Case 3: 旧账户向后兼容，无需修改仍能运行

## 清理

测试完成后清理测试数据：

```bash
docker exec godzilla-dev pm2 stop all
docker exec godzilla-dev rm -rf /app/runtime/td/binance/gz_user1
docker exec godzilla-dev sqlite3 /app/runtime/system/etc/kungfu/db/live/accounts.db \
    "DELETE FROM account_config WHERE account_id='binance_gz_user1'"
```

---

## 故障排查

### 问题 1: 交互式命令没有显示市场开关问题

**可能原因**: package.json 未正确加载

**解决方案**:
```bash
# 验证 package.json 包含新字段
grep -A3 "enable_spot" core/extensions/binance/package.json

# 重启容器
docker restart godzilla-dev
```

### 问题 2: 数据库中字段是字符串 "false" 而不是布尔值

**可能原因**: `encrypt()` 函数未正确转换

**解决方案**:
```bash
# 验证 encrypt() 支持 bool 类型
grep -A5 "elif type_config\[key\] == 'bool':" core/python/kungfu/command/account/__init__.py
```

### 问题 3: TD 启动失败或 crash

**可能原因**: C++ 未正确解析配置

**解决方案**:
```bash
# 检查 TD 日志中的 JSON 解析错误
pm2 logs td_binance --err --lines 50
```

---

## 报告结果

完成测试后，请在 ADR-004 或相关 issue 中报告：

- [ ] Test Case 1 结果：✅ Pass / ❌ Fail
- [ ] Test Case 2 结果：✅ Pass / ❌ Fail
- [ ] Test Case 3 结果：✅ Pass / ❌ Fail
- [ ] 任何意外行为或改进建议

---

**测试完成日期**: ___________
**测试人**: ___________
**环境**: Docker / Native

