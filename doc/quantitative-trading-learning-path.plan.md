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

# 使用官方命令添加账户（自动创建正确表结构）
python core/python/dev_run.py account -s binance add
# 交互输入：
#   user_id: gz_user1
#   access_key: YOUR_API_KEY
#   secret_key: YOUR_SECRET_KEY
```

**验证**：
```bash
python core/python/dev_run.py account -s binance show
# 应显示 gz_user1 及密钥前缀
```

**说明**：该命令会自动创建正确的数据库表结构（使用 `account_id` 列），与代码中的 SQLAlchemy Model 定义一致

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

## 阶段3：理解策略回调机制

> **目标**：掌握策略生命周期和核心回调函数，学会读懂和修改现有策略

### 3.1 策略生命周期理解

**关键文件**：`strategies/helloworld/helloworld.py`（[官方範例](https://godzilla.dev/documentation/strategies/helloworld/)）

**生命周期流程**：
```python
pre_start(context)       # 策略启动前：订阅数据、初始化状态
  ↓
on_depth(context, depth) # 收到深度数据时触发
on_ticker(context, ticker) # 收到ticker数据时触发
on_order(context, order) # 收到订单更新时触发
on_trade(context, trade) # 收到成交回报时触发
  ↓
pre_stop(context)        # 策略停止前：清理资源
```

**实操练习 3.1**：
```bash
# 1. 阅读 helloworld 源码
cat /app/strategies/helloworld/helloworld.py

# 2. 修改策略，添加价差计算
nano /app/strategies/helloworld/helloworld.py
```

修改 `on_depth` 函数：
```python
def on_depth(context, depth):
    bid_price = depth.bid_price[0]
    ask_price = depth.ask_price[0]
    bid_volume = depth.bid_volume[0]
    ask_volume = depth.ask_volume[0]
    spread = ask_price - bid_price
    spread_pct = (spread / bid_price) * 100
    
    # 只在价差大于0.01%时打印
    if spread_pct > 0.01:
        context.log().info(
            f"[{depth.symbol}] Bid: {bid_price:.2f} | Ask: {ask_price:.2f} | "
            f"Spread: {spread:.2f} ({spread_pct:.3f}%)"
        )
```

**3. 重启策略查看效果**：
```bash
pm2 restart strategy:hello
pm2 logs strategy:hello --lines 20
```

---

### 3.2 Context API 深入理解

**核心 API** ([參考官方架構](https://godzilla.dev/documentation/architecture/))：

| API | 用途 | 範例 |
|-----|------|------|
| `context.subscribe()` | 订阅深度数据 | `context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)` |
| `context.subscribe_ticker()` | 订阅ticker数据 | `context.subscribe_ticker("binance", ["btcusdt"], ...)` |
| `context.get_config()` | 获取配置 | `config = context.get_config()` |
| `context.log().info()` | 记录日志 | `context.log().info("message")` |
| `context.set_object()` | 存储状态 | `context.set_object("key", value)` |
| `context.get_object()` | 读取状态 | `value = context.get_object("key")` |

**实操练习 3.2**：添加状态管理
```python
def pre_start(context):
    config = context.get_config()
    context.subscribe(config["md_source"], [config["symbol"]], instrument_type, exchange)
    
    # 初始化状态
    context.set_object("tick_count", 0)
    context.set_object("min_spread", float('inf'))
    context.set_object("max_spread", 0)

def on_depth(context, depth):
    # 读取状态
    tick_count = context.get_object("tick_count")
    min_spread = context.get_object("min_spread")
    max_spread = context.get_object("max_spread")
    
    spread = depth.ask_price[0] - depth.bid_price[0]
    
    # 更新状态
    tick_count += 1
    min_spread = min(min_spread, spread)
    max_spread = max(max_spread, spread)
    
    context.set_object("tick_count", tick_count)
    context.set_object("min_spread", min_spread)
    context.set_object("max_spread", max_spread)
    
    # 每100次tick打印统计
    if tick_count % 100 == 0:
        context.log().info(
            f"Stats - Ticks: {tick_count} | "
            f"Min Spread: {min_spread:.2f} | Max Spread: {max_spread:.2f}"
        )
```

---

### 3.3 数据结构理解

**基於 [Binance Derivatives WebSocket Streams](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams) 官方文檔**

系統支持 5 種市場數據類型，所有結構定義在 `core/cpp/wingchun/include/kungfu/wingchun/msg.h`：

---

#### 1. **Depth（訂單簿深度）** - Line 242-302

最常用的市場數據，提供多檔買賣盤口。

```python
# 訂閱
context.subscribe(md_source, [symbol], instrument_type, exchange)
# 回調: on_depth(context, depth)

# 數據訪問
depth.symbol          # 交易對名稱，如 "btcusdt"
depth.bid_price[0]    # 最優買價（數組，最多10檔）
depth.bid_volume[0]   # 最優買量
depth.ask_price[0]    # 最優賣價
depth.ask_volume[0]   # 最優賣量
depth.bid_price[4]    # 第5檔買價
depth.data_time       # 數據時間戳（納秒）
```

**實操練習**：深度數據分析
```python
def on_depth(context, depth):
    # 計算買賣盤口總量（前5檔）
    total_bid_volume = sum(depth.bid_volume[:5])
    total_ask_volume = sum(depth.ask_volume[:5])
    
    # 計算盤口不平衡度
    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    
    context.log().info(
        f"[{depth.symbol}] Bid Vol: {total_bid_volume:.4f} | "
        f"Ask Vol: {total_ask_volume:.4f} | Imbalance: {imbalance:.3f}"
    )
```

---

#### 2. **Trade（逐筆成交）** - Line 331-400

公開的市場成交數據，可用於分析成交方向和活躍度。

```python
# 訂閱
context.subscribe_trade(md_source, [symbol], instrument_type, exchange)
# 回調: on_transaction(context, transaction)

# 數據訪問
transaction.symbol        # "btcusdt"
transaction.price         # 成交價格
transaction.volume        # 成交數量
transaction.side          # Side.Buy 或 Side.Sell（主動方向）
transaction.trade_time    # 成交時間戳（納秒）
transaction.trade_id      # 交易ID
```

**實操練習**：成交數據統計
```python
def pre_start(context):
    context.subscribe_trade(config["md_source"], [config["symbol"]], instrument_type, exchange)
    context.set_object("trade_count", 0)
    context.set_object("buy_volume", 0)
    context.set_object("sell_volume", 0)

def on_transaction(context, transaction):
    trade_count = context.get_object("trade_count") + 1
    context.set_object("trade_count", trade_count)
    
    # 統計買賣方向
    if transaction.side == Side.Buy:
        buy_vol = context.get_object("buy_volume") + transaction.volume
        context.set_object("buy_volume", buy_vol)
    else:
        sell_vol = context.get_object("sell_volume") + transaction.volume
        context.set_object("sell_volume", sell_vol)
    
    # 每100筆打印統計
    if trade_count % 100 == 0:
        buy_vol = context.get_object("buy_volume")
        sell_vol = context.get_object("sell_volume")
        context.log().info(
            f"Trades: {trade_count} | Buy Vol: {buy_vol:.4f} | "
            f"Sell Vol: {sell_vol:.4f} | Ratio: {buy_vol/sell_vol:.2f}"
        )
```

---

#### 3. **IndexPrice（指數價格）** ⭐ Futures 專用 - Line 405-444

用於監控現貨指數價格，可計算期現價差。

```python
# 訂閱（僅 Futures）
context.subscribe_index_price(md_source, [symbol], InstrumentType.FFuture, exchange)
# 回調: on_index_price(context, index_price)

# 數據訪問
index_price.symbol    # "btcusdt"
index_price.price     # 指數價格
```

**參考範例**：`strategies/demo_future.py` Line 30, 66-70

---

#### 4. **Ticker（行情快照）** - Line 176-238

輕量級的盤口數據，僅包含最優買賣價。

```python
# 訂閱
context.subscribe_ticker(md_source, [symbol], instrument_type, exchange)
# 回調: on_ticker(context, ticker)

# 數據訪問
ticker.symbol         # "btcusdt"
ticker.bid_price      # 最優買價（僅1檔）
ticker.bid_volume     # 最優買量
ticker.ask_price      # 最優賣價
ticker.ask_volume     # 最優賣量
ticker.data_time      # 時間戳
```

**參考範例**：`strategies/demo_spot.py` Line 25, 69-73

---

#### 5. **Bar（K線數據）** - Line 446-493

聚合的K線數據，用於趨勢分析。

```python
# 回調: on_bar(context, bar)

# 數據訪問
bar.symbol        # "btcusdt"
bar.open          # 開盤價
bar.high          # 最高價
bar.low           # 最低價
bar.close         # 收盤價
bar.volume        # 成交量
bar.interval      # 週期（秒）
bar.start_time    # 開始時間
bar.end_time      # 結束時間
```

---

#### 📋 數據類型選擇指南

| 策略類型 | 推薦數據 | 原因 |
|---------|---------|------|
| 高頻做市 | `Depth` | 需要完整盤口（10檔） |
| 套利策略 | `Depth` 或 `Ticker` | 快速獲取價格 |
| 趨勢跟蹤 | `Bar` | K線分析 |
| 成交分析 | `Trade` | 監控市場活躍度 |
| 期現套利 | `Depth` + `IndexPrice` | 計算價差 |

---

#### 🎯 完整訂閱範例

```python
def pre_start(context):
    config = context.get_config()
    symbol = config["symbol"]
    
    # 同時訂閱多種數據
    context.subscribe(config["md_source"], [symbol], instrument_type, exchange)
    context.subscribe_ticker(config["md_source"], [symbol], instrument_type, exchange)
    context.subscribe_trade(config["md_source"], [symbol], instrument_type, exchange)
    
    # Futures 專用
    if instrument_type == InstrumentType.FFuture:
        context.subscribe_index_price(config["md_source"], [symbol], instrument_type, exchange)

# 對應的回調函數會自動觸發
def on_depth(context, depth):
    # 處理深度數據
    pass

def on_ticker(context, ticker):
    # 處理ticker數據
    pass

def on_transaction(context, transaction):
    # 處理成交數據
    pass

def on_index_price(context, index_price):
    # 處理指數價格（Futures）
    pass
```

---

### 阶段3完成标志

- [x] 理解策略生命周期（pre_start → callbacks → pre_stop）
- [x] 能够修改 helloworld 策略添加自定义逻辑
- [x] 掌握 Context API 基本用法（订阅、日志、状态管理）
- [x] 理解 Depth 数据结构并能计算衍生指标
- [x] 策略能成功运行并输出自定义日志

---

## 阶段4：实现简单交易策略

> **目标**：学会下单、撤单、查询订单，实现完整的交易闭环

### 4.1 理解订单生命周期

**订单状态流转**：
```
PreSend → Submitted → Pending → PartialFilledActive → Filled
                            ↓
                        Cancelled / Error
```

**关键文件**：`strategies/demo_spot.py`

**核心 API**：
```python
# 下限价单
order_id = context.insert_order(
    symbol="btcusdt",           # 交易对
    instrument_type=InstrumentType.Spot,
    exchange=Exchange.BINANCE,
    account="gz_user1",
    price=50000,                # 限价
    volume=0.001,               # 数量
    order_type=OrderType.Limit,
    side=Side.Buy               # 买/卖
)

# 撤单
context.cancel_order(account, order_id, symbol, ex_order_id, instrument_type)

# 查询订单
context.query_order(account, order_id, ex_order_id, instrument_type, symbol)
```

---

### 4.2 实操：编写网格交易策略（简化版）

**策略逻辑**：在固定价格网格上挂买卖单

**创建新策略文件**：
```bash
nano /app/strategies/grid_simple.py
```

**代码**：
```python
from kungfu.wingchun.constants import *
from pywingchun.constants import Side, InstrumentType, OrderType

exchange = Exchange.BINANCE
instrument_type = InstrumentType.Spot

def pre_start(context):
    config = context.get_config()
    context.add_account(config["td_source"], config["account"])
    context.subscribe(config["md_source"], [config["symbol"]], instrument_type, exchange)
    
    # 网格参数
    context.set_object("grid_center", 50000)  # 中心价格
    context.set_object("grid_step", 100)      # 网格间距
    context.set_object("grid_size", 0.001)    # 每格下单量
    context.set_object("order_placed", False)
    
    context.log().info("Grid strategy initialized")

def on_depth(context, depth):
    config = context.get_config()
    book = context.get_account_book(config["td_source"], config["account"])
    
    # 只在无活跃订单时下单（简化版）
    if len(book.active_orders) == 0 and not context.get_object("order_placed"):
        mid_price = (depth.bid_price[0] + depth.ask_price[0]) / 2
        
        # 在当前价格下方挂买单
        buy_price = mid_price - context.get_object("grid_step")
        order_id = context.insert_order(
            config["symbol"], instrument_type, exchange, 
            config["account"], buy_price, context.get_object("grid_size"),
            OrderType.Limit, Side.Buy
        )
        context.log().info(f"Buy order placed at {buy_price:.2f}, order_id: {order_id}")
        context.set_object("order_placed", True)

def on_order(context, order):
    context.log().info(f"Order update: {order.order_id} - {order.status}")
    
    # 订单成交后重置标志
    if order.status == OrderStatus.Filled:
        context.log().info(f"Order filled: {order.symbol} at {order.price}")
        context.set_object("order_placed", False)
    
    # 订单取消后重置标志
    if order.status == OrderStatus.Cancelled:
        context.set_object("order_placed", False)

def on_trade(context, trade):
    context.log().info(f"Trade: {trade.symbol} - Vol: {trade.volume} - Price: {trade.price}")
```

**配置文件**：
```bash
cat > /app/strategies/grid_conf.json << 'EOF'
{
    "name": "grid strategy",
    "md_source": "binance",
    "td_source": "binance",
    "symbol": "btcusdt",
    "account": "gz_user1"
}
EOF
```

**运行策略**（⚠️ 测试网环境）：
```bash
pm2 start --name grid --interpreter python3 \
  core/python/dev_run.py -- strategy -n grid \
  -p /app/strategies/grid_simple.py -c /app/strategies/grid_conf.json

pm2 logs grid
```

---

### 4.3 账户查询与风控

**关键 API**：
```python
# 查询账本
book = context.get_account_book(td_source, account)

# 活跃订单
for order in book.active_orders:
    print(order['order_id'], order['status'], order['price'])

# 查询余额（需要账户API）
api = context.get_account_api(td_source, account)
balance = api.balance('usdt')
```

**实操练习 4.3**：添加风控检查
```python
def pre_start(context):
    config = context.get_config()
    # 设置资金限额
    context.set_account_cash_limit(
        config["td_source"], exchange, config["account"],
        "usdt", 100  # 最多使用100 USDT
    )

def on_depth(context, depth):
    # 检查可用额度
    available = context.get_account_cash_limit(config["account"], "usdt")
    if available < 10:
        context.log().warn("Insufficient balance, skipping trade")
        return
    
    # 继续下单逻辑...
```

---

### 阶段4完成标志

- [ ] 理解订单状态流转（Submitted → Filled/Cancelled）
- [ ] 能够使用 `insert_order()` 下单
- [ ] 能够使用 `cancel_order()` 撤单
- [ ] 理解 `on_order()` 回调处理订单更新
- [ ] 实现了简单的网格交易策略并成功下单

---

## 阶段5：策略状态管理与多数据源

> **目标**：学习复杂状态管理、多交易对订阅、条件触发

### 5.1 订阅多个交易对

**参考**：`strategies/triangular_arbitrage/triangular_arbitrage.py`

```python
def pre_start(context):
    # 订阅多个交易对
    symbols = ["btcusdt", "ethusdt", "ethbtc"]
    context.subscribe("binance", symbols, InstrumentType.Spot, Exchange.BINANCE)
    
    # 初始化每个交易对的状态
    for symbol in symbols:
        context.set_object(f"{symbol}_depth", None)

def on_depth(context, depth):
    # 根据交易对更新对应状态
    context.set_object(f"{depth.symbol}_depth", depth)
    
    # 检查是否所有数据都已接收
    btc_depth = context.get_object("btcusdt_depth")
    eth_depth = context.get_object("ethusdt_depth")
    ethbtc_depth = context.get_object("ethbtc_depth")
    
    if btc_depth and eth_depth and ethbtc_depth:
        # 三个交易对数据齐全，执行策略逻辑
        analyze_arbitrage(context, btc_depth, eth_depth, ethbtc_depth)
```

---

### 5.2 实操：实现价差监控策略

**策略逻辑**：监控 BTC/USDT 和 ETH/USDT 的相对价格变化

```python
from kungfu.wingchun.constants import *
from pywingchun.constants import Side, InstrumentType, OrderType

exchange = Exchange.BINANCE
instrument_type = InstrumentType.Spot

def pre_start(context):
    config = context.get_config()
    symbols = ["btcusdt", "ethusdt"]
    context.subscribe("binance", symbols, instrument_type, exchange)
    
    context.set_object("btc_price", None)
    context.set_object("eth_price", None)
    context.set_object("ratio_history", [])
    
    context.log().info("Spread monitor initialized")

def on_depth(context, depth):
    # 更新价格
    mid_price = (depth.bid_price[0] + depth.ask_price[0]) / 2
    
    if depth.symbol == "btcusdt":
        context.set_object("btc_price", mid_price)
    elif depth.symbol == "ethusdt":
        context.set_object("eth_price", mid_price)
    
    # 计算比率
    btc_price = context.get_object("btc_price")
    eth_price = context.get_object("eth_price")
    
    if btc_price and eth_price:
        ratio = btc_price / eth_price
        ratio_history = context.get_object("ratio_history")
        ratio_history.append(ratio)
        
        # 只保留最近100个数据点
        if len(ratio_history) > 100:
            ratio_history.pop(0)
        
        context.set_object("ratio_history", ratio_history)
        
        # 计算统计指标
        if len(ratio_history) >= 20:
            avg_ratio = sum(ratio_history) / len(ratio_history)
            deviation = (ratio - avg_ratio) / avg_ratio
            
            if abs(deviation) > 0.01:  # 偏离均值1%以上
                context.log().info(
                    f"Ratio Alert! BTC/ETH: {ratio:.2f} | "
                    f"Avg: {avg_ratio:.2f} | Deviation: {deviation*100:.2f}%"
                )
```

---

### 5.3 时间管理与定时任务

**使用系统时间**：
```python
import kungfu.yijinjing.time as kft

def on_depth(context, depth):
    now = context.now()  # 纳秒时间戳
    time_str = kft.strftime(now, "%Y-%m-%d %H:%M:%S")
    context.log().info(f"Current time: {time_str}")
```

---

### 阶段5完成标志

- [ ] 能够订阅多个交易对
- [ ] 实现跨交易对的数据关联分析
- [ ] 掌握状态管理（使用列表、字典等复杂数据结构）
- [ ] 实现条件触发逻辑（价差、比率等）

---

## 阶段6：学习复杂策略 - 三角套利

> **目标**：阅读和理解真实的量化策略代码

### 6.1 三角套利原理

**策略逻辑**（`strategies/triangular_arbitrage/triangular_arbitrage.py`）：

监控三个交易对的价格关系，寻找套利机会：
- 交易对1：AAVE/ETH
- 交易对2：ETH/USDT  
- 交易对3：AAVE/USDT

**套利条件**：
```
如果：AAVE/ETH的买价 × ETH/USDT的买价 > AAVE/USDT的卖价
则：买入AAVE（用ETH）→ 卖出AAVE（换USDT）→ 买入ETH（用USDT）
```

---

### 6.2 代码阅读任务

**阅读顺序**：
1. `pre_start()` - 初始化和订阅
2. `on_depth()` - 数据更新入口
3. `inspect()` - 套利机会检测
4. `optimized_volume()` - 计算最优下单量
5. `execute()` - 执行交易

**关键学习点**：
- 如何管理多个深度数据
- 如何计算套利机会
- 如何处理精度问题（`tick_size_rounddown`）
- 如何管理订单状态（`order_ids`, `order_status`）
- 如何实现风控（资金限额检查）

---

### 6.3 实操：修改三角套利策略

**任务**：将策略改为只监控不下单（学习模式）

```python
def on_depth(context, depth):
    arbitrager = context.get_object('arbitrager')
    triangular_arbitrage = arbitrager.inspect(depth)
    
    if triangular_arbitrage:
        # 原本会下单：arbitrager.execute(triangular_arbitrage)
        # 改为只记录
        context.log().info(f"Arbitrage opportunity detected: {triangular_arbitrage}")
        
        # 额外记录详细信息
        context.log().info(f"Base/Currency: {arbitrager.base_currency_depth}")
        context.log().info(f"Currency/Quote: {arbitrager.currency_quote_depth}")
        context.log().info(f"Base/Quote: {arbitrager.base_quote_depth}")
```

**运行监控模式**：
```bash
# 复制策略文件
cp /app/strategies/triangular_arbitrage/triangular_arbitrage.py \
   /app/strategies/triangular_arbitrage/monitor_only.py

# 修改第389行（注释掉execute调用）
nano /app/strategies/triangular_arbitrage/monitor_only.py

# 运行
pm2 start --name arb_monitor --interpreter python3 \
  core/python/dev_run.py -- strategy -n arb_monitor \
  -p /app/strategies/triangular_arbitrage/monitor_only.py \
  -c /app/strategies/triangular_arbitrage/str_para.json.sample
```

---

### 阶段6完成标志

- [ ] 理解三角套利的基本原理
- [ ] 能够阅读和理解 400 行的策略代码
- [ ] 理解精度处理和风控检查
- [ ] 理解订单状态管理机制
- [ ] 能够修改现有策略实现自定义需求

---

## 阶段7：编写自己的策略

> **最终目标**：独立设计和实现一个完整的交易策略

### 7.1 策略设计清单

在编写代码前，先回答这些问题：

- [ ] **策略类型**：套利/做市/趋势跟踪/其他？
- [ ] **数据需求**：需要哪些交易对的什么数据（深度/ticker/trade）？
- [ ] **交易逻辑**：什么条件下开仓？什么条件下平仓？
- [ ] **风控规则**：最大持仓？单笔下单量？止损条件？
- [ ] **状态管理**：需要记录哪些状态？如何初始化和更新？
- [ ] **性能考虑**：计算复杂度？数据存储量？

---

### 7.2 策略模板

```python
'''
策略名称：[你的策略名称]
策略逻辑：[简要描述]
作者：[你的名字]
日期：2025-xx-xx
'''
from kungfu.wingchun.constants import *
from pywingchun.constants import Side, InstrumentType, OrderType
import kungfu.yijinjing.time as kft

exchange = Exchange.BINANCE
instrument_type = InstrumentType.Spot  # 或 FFuture

def pre_start(context):
    """策略初始化"""
    config = context.get_config()
    
    # 1. 添加账户（如果需要交易）
    # context.add_account(config["td_source"], config["account"])
    
    # 2. 订阅数据
    context.subscribe(
        config["md_source"], 
        [config["symbol"]], 
        instrument_type, 
        exchange
    )
    
    # 3. 初始化状态
    context.set_object("state", "init")
    
    # 4. 设置风控参数
    # context.set_account_cash_limit(...)
    
    context.log().info("Strategy initialized")

def on_depth(context, depth):
    """深度数据回调"""
    # 1. 数据验证
    config = context.get_config()
    if depth.symbol != config['symbol']:
        return
    
    # 2. 提取数据
    bid_price = depth.bid_price[0]
    ask_price = depth.ask_price[0]
    
    # 3. 计算指标
    # ...
    
    # 4. 判断条件
    # ...
    
    # 5. 执行交易
    # order_id = context.insert_order(...)

def on_order(context, order):
    """订单更新回调"""
    context.log().info(f"Order: {order.order_id} - {order.status}")
    
    # 根据订单状态更新策略状态
    if order.status == OrderStatus.Filled:
        # 订单成交处理
        pass

def on_trade(context, trade):
    """成交回报回调"""
    context.log().info(f"Trade: {trade.symbol} @ {trade.price}")

def pre_stop(context):
    """策略停止前清理"""
    context.log().info("Strategy stopping")
```

---

### 7.3 建议的第一个策略

**策略名称**：动态价差监控与警报

**策略逻辑**：
1. 订阅一个交易对的深度数据
2. 计算实时价差（ask - bid）
3. 维护价差的移动平均和标准差
4. 当价差偏离均值超过2倍标准差时发出警报
5. 不下单，只监控和记录

**为什么选这个**：
- 不涉及真实下单（安全）
- 涵盖数据订阅、状态管理、统计计算
- 可以扩展为做市策略的基础

---

### 7.4 调试与优化

**日志技巧**：
```python
# 不同级别的日志
context.log().trace("详细调试信息")  # 最详细
context.log().info("正常信息")
context.log().warn("警告")
context.log().error("错误")
```

**查看日志**：
```bash
pm2 logs <策略名> --lines 100
```

**性能监控**：
```python
def on_depth(context, depth):
    start_time = context.now()
    
    # 策略逻辑...
    
    end_time = context.now()
    latency_ns = end_time - start_time
    latency_us = latency_ns / 1000
    
    if latency_us > 1000:  # 超过1ms
        context.log().warn(f"High latency: {latency_us:.2f} μs")
```

---

### 阶段7完成标志

- [ ] 完成策略设计（明确逻辑和风控）
- [ ] 编写完整策略代码
- [ ] 在测试网环境成功运行
- [ ] 记录日志并验证行为符合预期
- [ ] （可选）实现回测功能

---

## 进阶主题（可选）

### A. Futures 交易

**关键差异**：
- `InstrumentType.FFuture` vs `InstrumentType.Spot`
- 需要管理持仓（`on_position` 回调）
- 支持槓桿（`context.adjust_leverage()`）

**参考**：`strategies/demo_future.py`

### B. 回测模式

**切换到回测**：
```python
# 使用历史 journal 数据
# mode = BACKTEST
```

### C. 多账户管理

```python
# 添加多个账户
context.add_account("binance", "account1")
context.add_account("binance", "account2")

# 针对不同账户下单
context.insert_order(..., account="account1", ...)
```

---

## 学习完成检查清单

完成所有阶段后，你应该能够：

- [x] 启动完整系统（阶段2）
- [x] 理解策略回调机制（阶段3）
- [ ] 实现简单下单策略（阶段4）
- [ ] 管理多交易对状态（阶段5）
- [ ] 阅读复杂策略代码（阶段6）
- [ ] 独立编写完整策略（阶段7）

---

## 参考资料

### 官方文档
- [官方安装指南](https://godzilla.dev/documentation/installation/)
- [系统架构](https://godzilla.dev/documentation/architecture/)
- [HelloWorld 範例](https://godzilla.dev/documentation/strategies/helloworld/)

### 项目文档
- `.doc/TESTNET.md` - Binance 测试网配置
- `.doc/DEBUGGING.md` - 调试案例
- `.doc/ARCHITECTURE.md` - 架构详解
- `.doc/HACKING.md` - 开发流程

### 代码参考
- `strategies/helloworld/` - 最简单範例
- `strategies/demo_spot.py` - Spot 交易範例
- `strategies/demo_future.py` - Futures 交易範例
- `strategies/triangular_arbitrage/` - 复杂策略範例

---

**最后提醒**：
- ⚠️ 始终在测试网环境练习
- ⚠️ 理解每一行代码的作用
- ⚠️ 从简单到复杂，逐步进阶
- ⚠️ 充分测试后再考虑实盘