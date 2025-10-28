<!-- 8e0fa114-ce37-4830-94cb-d9d2f8408752 5ef87e49-a2b8-47a2-8f78-d7bc7f44bc40 -->
# 量化交易系统学习路径

## 阶段1：理解系统基础架构（可选，建议先跳过）

> **实战优先**：如果你想快速看到效果，**直接跳到阶段2**。等系统跑起来后再回来看理论。

### 1.1 了解Journal系统（事件溯源的核心）

阅读以下代码理解事件如何被记录和读取：

- `core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h`（40-160行）
- `core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h`（80-125行）

**核心概念**：所有事件（行情、订单、成交）都会被写入journal文件，可以回放和分析。

---

## 阶段2：配置并启动完整系统 ⭐

> **参考文档**：`.doc/TESTNET.md`（包含详细故障排除）

### 系统架构
```
Master → Ledger ← Strategy
           ↓         ↓
        MD ←→ TD → Binance Testnet
```

### 目标
✅ 获取 API 凭证 → ✅ 配置数据库 → ✅ 启动服务 → ✅ 运行策略看数据

---

### 2.1 环境准备 + 获取 API

**进入容器并安装依赖**：
```bash
docker-compose up -d
docker-compose exec app bash

# 安装 PM2 和 Python 软链接
apt-get update && apt-get install -y nodejs npm
npm install -g pm2
ln -sf /usr/bin/python3 /usr/bin/python
```

**获取 Binance Futures API**：
1. 访问：https://testnet.binancefuture.com/
2. 登录后点击 "Generate API"
3. **立即保存** API Key 和 Secret Key（只显示一次）

---

### 2.2 配置账户数据库

**手动创建数据库**（在容器中执行）：
```bash
mkdir -p /root/.config/kungfu/app

python3 << 'EOF'
import sqlite3, json, os
db = '/root/.config/kungfu/app/kungfu.db'
conn = sqlite3.connect(db)
conn.execute('''CREATE TABLE IF NOT EXISTS account_config (
    user_id TEXT, source_name TEXT, receive_td INTEGER DEFAULT 1,
    config TEXT, PRIMARY KEY (user_id, source_name))''')
conn.execute('INSERT OR REPLACE INTO account_config VALUES (?,?,?,?)',
    ('gz_user1', 'binance', 1, json.dumps({
        'access_key': 'YOUR_API_KEY',      # ← 替换
        'secret_key': 'YOUR_SECRET_KEY'    # ← 替换
    })))
conn.commit()
print("✅ gz_user1 添加成功")
EOF
```

**验证**：
```bash
python core/python/dev_run.py account -s binance show
# 应显示 gz_user1 及密钥前缀
```

---

### 2.3 一键启动系统

```bash
cd /app/scripts/binance_test
bash run.sh start
sleep 30  # 等待服务稳定
pm2 list  # 应显示 4 个 online 服务
```

**如果有服务 errored**：
```bash
pm2 logs <服务名> --lines 50
# 常见问题见 .doc/TESTNET.md
```

---

### 2.4 配置并运行策略

**检查配置**：
```bash
cat /app/strategies/conf.json
# 确认：symbol: "btcusdt", account: "gz_user1"
```

**启动策略**：
```bash
cd /app/scripts/binance_test

cat > strategy_hello.json << 'EOF'
{"apps": [{
  "name": "strategy:hello",
  "cwd": "../../",
  "script": "core/python/dev_run.py",
  "exec_interpreter": "python3",
  "args": "-l info strategy -n hello -p strategies/helloworld/helloworld.py -c strategies/conf.json",
  "watch": false
}]}
EOF

pm2 start strategy_hello.json
pm2 logs strategy:hello
```

**成功标志**（持续输出）：
```
[btcusdt] Bid: 114110.80 | Ask: 114120.10 | Spread: 9.30
```

---

### 2.5 停止系统

**优雅关闭**（推荐）：
```bash
cd /app/scripts/binance_test
bash graceful_shutdown.sh
```

这个脚本会自动：
- 停止所有 PM2 进程
- 清理 journal 文件
- 清理 socket 文件
- 删除旧日志（保留7天内）

**快速关闭**（备选）：
```bash
cd /app/scripts/binance_test
bash run.sh stop
# 或：pm2 delete all
```

---

### 常见问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `bash: pm2: command not found` | PM2 未安装 | `npm install -g pm2` |
| `bash: python: command not found` | 容器无 python | `ln -sf /usr/bin/python3 /usr/bin/python` |
| TD 不断重启 | 数据库不存在 | 重做 2.2 |
| 策略崩溃 | journal 冲突 | 删除 `*.journal` 和 `*.nn` |

详细故障排除见 `.doc/TESTNET.md`

---

### 阶段2完成标志

- [x] 4 个服务全部 online（`pm2 list`）
- [x] 策略持续输出实时行情数据
- [x] 知道如何启动/停止/查日志

---

## 阶段3-6：理解代码结构（深入学习）

### 核心概念速览

**策略生命周期**：
```
pre_start() → on_depth/on_order/on_trade → pre_stop()
```

**Context API**（`core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h`）：
- `subscribe()` - 订阅行情
- `insert_order()` - 下单
- `cancel_order()` - 撤单
- `get_account_book()` - 查询账户

**数据结构**（`core/cpp/wingchun/include/kungfu/wingchun/msg.h`）：
- `Depth` - 深度数据（bid/ask price/volume）
- `Order` - 订单状态
- `Trade` - 成交回报

**运行模式**：
- `LIVE` - 实时交易
- `BACKTEST` - 回测
- `REPLAY` - 回放

**数据流**：
```
交易所 WebSocket → MD → Journal → Strategy → Ledger → TD → 交易所 REST
```

---

## 进阶实战

### 修改策略

编辑 `strategies/helloworld/helloworld.py` 添加自己的逻辑：
```python
def on_depth(context, depth):
    bid, ask = depth.bid_price[0], depth.ask_price[0]
    spread = ask - bid
    if spread < 10:  # 价差小于10时的逻辑
        context.log().info(f"Low spread detected: {spread}")
```

### 运行 demo_spot 策略

```bash
pm2 start --name demo --interpreter python3 \
  core/python/dev_run.py -- strategy -n demo \
  -p strategies/demo_spot.py -c strategies/conf.json
```

观察 `on_order()` 和 `on_trade()` 回调。

---

## 完整系统理解检查清单

完成阶段 1-6 后，你应该能够：

- [x] 启动完整系统（Master/Ledger/MD/TD/Strategy）
- [x] 接收实时行情数据
- [ ] 理解 Journal 事件溯源机制
- [ ] 读懂 Context API 文档
- [ ] 修改策略代码并测试
- [ ] 查看日志和 journal 文件
- [ ] 理解订单生命周期

---

## 参考文档

- `.doc/TESTNET.md` - Binance 配置和故障排除
- `.doc/DEBUGGING.md` - 完整调试案例
- `.doc/ARCHITECTURE.md` - 系统架构详解
- 官方文档：https://godzilla.dev/documentation/