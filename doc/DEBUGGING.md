# 深度调试案例

本文档记录在开发和测试过程中遇到的复杂问题的调试过程，以及从中获得的经验。

**重要提示**：本文档包含两个主要案例：
- **案例 1**：展示手动启动 TD Gateway 遇到的问题（❌ **不推荐的方式**）
- **案例 2**：展示使用官方脚本的完整系统启动（✅ **正确的方式**）

通过对比这两个案例，读者可以：
1. 理解为什么必须使用官方启动脚本
2. 学习系统架构和事件流机制
3. 掌握调试分布式系统的方法论

**快速跳转**：
- 如果你想直接了解正确的启动方式 → 跳转到 [案例 2](#案例-2pm2--数据库配置完整系统启动)
- 如果你想深入理解系统架构 → 从 [案例 1](#案例-1td-gateway-启动但核心逻辑未执行) 开始阅读

---

## 案例 1：TD Gateway 启动但核心逻辑未执行

> **⚠️ 警告**：本案例展示的是 **错误的启动方式**（手动使用 `kfc td` 命令）。
> 
> 这个问题的根本原因是：**没有使用官方启动脚本**。如果按照 [案例 2](#案例-2pm2--数据库配置完整系统启动) 的方式使用官方脚本 `scripts/binance_test/run.sh start`，这个问题**根本不会发生**。
> 
> **阅读本案例的价值**：
> - 理解 kungfu 框架的事件驱动架构
> - 学习如何调试分布式系统中的消息传递问题
> - 了解 Master-Apprentice 注册机制
> - 掌握持久化状态对系统行为的影响
> 
> 如果你只想快速启动系统，请直接跳转到 [案例 2](#案例-2pm2--数据库配置完整系统启动)。

---

**日期**: 2025-10-24  
**问题严重程度**: 🔴 Critical  
**解决状态**: ✅ 已通过使用官方启动脚本完全解决（见案例 2）

### 1. 问题情境

#### 背景
- **目标**: 配置 Binance Futures Testnet，测试 TD Gateway 连接
- **配置**: 已正确配置 API Key，已修改 `common.h` 使用正确的 Testnet URLs
- **错误启动方式**: 手动运行 `kfc td -s binance -a futures_testnet`（❌ 不正确）
- **预期**: TD Gateway 启动后应该显示 "Connecting BINANCE TD" 和 "login success" 日志

#### 表面现象
```bash
# 进程状态
ps aux | grep 'kfc td'
# ✅ 进程在运行
root   1073  0.2  0.7 1038456 113872 ?  Sl  16:40  0:01 /usr/bin/python3 /usr/bin/kfc td -s binance -a futures_testnet

# 日志内容
cat /tmp/td.log
# ❌ 只有这些错误，每 5 秒重复一次
[10/24 16:40:00.123456789] [ error ] spot login failed, error_id: -2015, error_msg: Invalid API-key, IP, or permissions for action.
```

#### 异常特征
1. **进程不崩溃** - 一直在运行，没有 segfault 或 abort
2. **只有 error 日志** - 没有任何 `info`、`warning`、`trace` 级别日志
3. **缺少初始化日志** - 完全没有 "Connecting BINANCE TD" 的日志
4. **不断尝试 Spot 连接** - 使用 Futures API Key 却一直尝试 Spot
5. **日志很规律** - 每 5 秒精确重复相同的错误

### 2. 调试路径

#### 第一层：检查业务逻辑（trader_binance.cpp）

**假设**: 可能是 API Key 或 URL 配置问题

**行动**: 检查 `trader_binance.cpp` 的 `on_start()` 函数

```cpp
// core/extensions/binance/src/trader_binance.cpp:69-92
void TraderBinance::on_start() {
    Trader::on_start();
    task_thread_ = std::make_shared<std::thread>([this]() {
        boost::asio::io_context::work worker(this->ioctx_);
        this->ioctx_.run();
        return 0;
    });
    std::string runtime_folder = get_runtime_folder();
    SPDLOG_INFO(
        "Connecting BINANCE TD for {} at {}:{} with runtime folder {}",
        config_.user_id, config_.spot_rest_host, config_.spot_rest_port, runtime_folder);
    
    _start_userdata(InstrumentType::FFuture);  // ← Line 88
    add_time_interval(time_unit::NANOSECONDS_PER_SECOND * 5, std::bind(&TraderBinance::_check_status, this, std::placeholders::_1));
    publish_state(BrokerState::Ready);
    SPDLOG_INFO("login success");  // ← Line 91
}
```

**发现**: 
- Line 78: 应该输出 "Connecting BINANCE TD" 但日志中没有
- Line 91: 应该输出 "login success" 但日志中也没有
- **结论**: `on_start()` 函数根本没有被调用！

---

#### 第二层：检查框架启动逻辑（apprentice.cpp）

**假设**: `on_start()` 的调用者有问题

**行动**: 查找谁调用 `on_start()`

```bash
grep -rn "on_start()" core/cpp/yijinjing/src/practice/apprentice.cpp
# 找到 Line 191
```

**关键代码**:
```cpp
// core/cpp/yijinjing/src/practice/apprentice.cpp:185-209
if (get_io_device()->get_home()->mode != mode::BACKTEST)
{
    reader_->join(master_home_location_, 0, begin_time_);  // ← Line 187
    events_ | is(msg::type::RequestStart) | first() |      // ← Line 188
    $([&](event_ptr e)
      {
          on_start();  // ← Line 191: 这里才调用 on_start()
      },
      [&](std::exception_ptr e)
      {
          try
          { std::rethrow_exception(e); }
          catch (const rx::empty_error &ex)
          {
              SPDLOG_WARN("{}", ex.what());  // ← Line 199: 捕获空流错误
          }
          catch (const std::exception &ex)
          {
              SPDLOG_WARN("Unexpected exception before start {}", ex.what());
          }
      });
} else
{
    on_start();  // ← BACKTEST 模式直接调用
}
```

**发现**:
- Line 187: TD Gateway 尝试连接到 Master 的 journal
- Line 188: 等待接收 `msg::type::RequestStart` 消息
- Line 188: 使用 RxCPP 的 `first()` 操作符，期望至少收到一条消息
- Line 199: 如果流为空，会抛出 `rx::empty_error`，但只打印 WARNING
- **结论**: TD Gateway 在等待 Master 发送启动消息，但从未收到！

**实际日志验证**:
```bash
# 前台启动 TD Gateway
timeout 10 kfc td -s binance -a futures_testnet 2>&1

# 输出：
[warning] interrupted when receiving from ipc:///app/runtime/system/master/master/nn/live/pub.nn
[warning] first() requires a stream with at least one value
```

✅ **确认**: TD Gateway 确实在等待消息，但超时后只是警告，进程继续运行

---

#### 第三层：检查 Master 注册逻辑（master.cpp）

**假设**: Master 为什么不发送 `RequestStart` 消息？

**行动**: 检查 `master.cpp` 的 `register_app()` 函数

```cpp
// core/cpp/yijinjing/src/practice/master.cpp:45-116
void master::register_app(const event_ptr &e)
{
    auto request_loc = e->data<nlohmann::json>();
    auto app_location = std::make_shared<location>(
            static_cast<mode>(request_loc["mode"]),
            static_cast<category>(request_loc["category"]),
            request_loc["group"], request_loc["name"],
            get_io_device()->get_home()->locator
    );

    if (has_location(app_location->uid))  // ← Line 55
    {
        SPDLOG_ERROR("location {} has already been registered", app_location->uname);
        return;  // ← Line 58: 直接返回，不发送 RequestStart！
    }

    // ... 注册逻辑 ...
    
    writer->mark(e->gen_time(), msg::type::RequestStart);  // ← Line 115: 只有未注册时才发送
}
```

**发现**:
- Line 55-58: 如果 location 已经注册，Master 会拒绝并直接返回
- Line 115: `RequestStart` 消息只在首次注册时发送
- **可能原因**: 之前的 TD Gateway 崩溃/重启留下了持久化状态

**验证持久化状态**:
```bash
find /app/runtime -name '*.journal' -exec ls -lh {} \;

# 输出：
-rw------- 1 root root 4.0M Oct 24 16:52 /app/runtime/td/binance/futures_testnet/journal/live/69be3cbc.1.journal
-rw------- 1 root root 1.0M Oct 24 16:52 /app/runtime/td/binance/futures_testnet/journal/live/00000000.1.journal
-rw------- 1 root root 1.0M Oct 24 16:53 /app/runtime/system/master/487fd619/journal/live/487fd619.1.journal
-rw------- 1 root root 1.0M Oct 24 16:57 /app/runtime/system/master/master/journal/live/00000000.1.journal
```

✅ **确认**: 存在旧的 journal 文件，Master 可能记住了之前的注册

---

#### 第四层：检查启动顺序（run.sh）

**行动**: 检查官方启动脚本

```bash
# scripts/helloworld/run.sh
start() {
    echo "clearing journal..."
    find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f
    # start master
    pm2 start master.json
    sleep 5
    # start ledger
    pm2 start ledger.json
    sleep 5
    # start binance md
    pm2 start md_binance.json
    sleep 5
    # start binance td
    pm2 start td_binance.json
    sleep 5
}
```

**发现**:
1. **清理 journal 文件** - 每次启动前清理
2. **启动顺序**: Master → Ledger → MD → TD
3. **等待时间**: 每个服务启动后等待 5 秒
4. **缺少 Ledger** - 我们的测试中没有启动 Ledger！

---

### 3. 根本原因

经过四层深入分析，找到了问题的根本原因：

```
┌─────────────────────────────────────────────────────────┐
│              TD Gateway 启动失败链条                      │
└─────────────────────────────────────────────────────────┘

1. [启动层] TD Gateway 进程启动
   ↓
2. [框架层] apprentice 初始化，连接到 Master
   ↓
3. [通信层] 等待 Master 发送 msg::type::RequestStart
   ↓
4. [Master层] 检查 TD location 是否已注册
   ├─ 未注册 → 注册 + 发送 RequestStart ✅
   └─ 已注册 → 拒绝 + 不发送消息 ❌
   ↓
5. [RxCPP层] events_ | is(RequestStart) | first()
   ├─ 收到消息 → 调用 on_start() ✅
   └─ 超时/空流 → 抛出 rx::empty_error → 捕获 → 只打印 WARNING ❌
   ↓
6. [结果] on_start() 永远不会被调用
   ├─ 没有初始化日志
   ├─ 没有 Futures 连接尝试
   └─ 只有定时器触发的重连检查（每 5 秒尝试 Spot 重连）
```

**为什么一直尝试 Spot 连接？**

```cpp
// trader_binance.cpp:342-349
// _check_status() 每 5 秒被定时器调用
if (ws_ptr_->fetch_reconnect_flag()) {
    _start_userdata(InstrumentType::Spot);  // ← 重连检查会尝试 Spot
}
if (fws_ptr_->fetch_reconnect_flag()) {
    _start_userdata(InstrumentType::FFuture);
}
```

因为 `on_start()` 从未调用，WebSocket 连接从未建立，重连标志一直为 true，所以定时器不断尝试重连。

---

### 4. 获得的经验

#### 4.1 事件驱动架构的调试方法

**教训**: 在事件驱动系统中，如果某个事件没有触发，整个调用链都会卡住。

**调试技巧**:
1. **反向追踪**: 从预期的结果（日志、状态变化）反向找触发点
2. **检查事件流**: 使用 `grep` 查找事件类型定义和发送位置
3. **验证消息传递**: 检查 journal、socket 文件、IPC 通信

**代码示例**:
```bash
# 查找事件类型定义
grep -rn "RequestStart" core/cpp/yijinjing/include/kungfu/yijinjing/msg.h
# 结果: RequestStart = 10025

# 查找谁发送这个事件
grep -rn "msg::type::RequestStart" core/cpp/yijinjing/src/practice/
# 结果: master.cpp:115 (发送), apprentice.cpp:188 (接收)
```

---

#### 4.2 持久化状态的影响

**教训**: 系统的持久化状态（journal、数据库）可能导致重启后行为不一致。

**关键文件位置**:
```
/app/runtime/
├── td/binance/futures_testnet/
│   ├── journal/live/*.journal    ← TD Gateway 的事件日志
│   └── nn/live/*.nn               ← Socket 文件（可能锁定）
├── system/
│   ├── master/master/
│   │   └── journal/live/*.journal ← Master 记住的注册信息
│   └── etc/kungfu/db/live/
│       └── accounts.db            ← 账户配置
```

**清理策略**:
```bash
# 完全清理（最干净）
rm -rf /app/runtime

# 部分清理（保留配置）
rm -rf /app/runtime/td
rm -rf /app/runtime/system/master/*/journal

# 清理 socket 文件（解决 "Address already in use"）
find /app/runtime -name '*.nn' -type s -delete
```

---

#### 4.3 RxCPP 异常处理

**教训**: RxCPP 的操作符会抛出异常，但异常可能被静默捕获。

**关键操作符**:
- `first()`: 期望至少一个元素，否则抛出 `rx::empty_error`
- `last()`: 期望至少一个元素，否则抛出 `rx::empty_error`
- `element_at(n)`: 期望有第 n 个元素，否则抛出 `std::out_of_range`

**代码模式**:
```cpp
// 安全模式：捕获并处理异常
events_ | is(msg_type) | first() |
$([&](event_ptr e) {
    // 正常处理
},
[&](std::exception_ptr e) {
    try { std::rethrow_exception(e); }
    catch (const rx::empty_error &ex) {
        SPDLOG_WARN("No event received: {}", ex.what());
        // ⚠️ 问题：只警告，逻辑继续，但核心功能未初始化
    }
});

// 替代方案：使用 first_or_default()
events_ | is(msg_type) | first_or_default(nullptr) |
$([&](event_ptr e) {
    if (e) {
        // 正常处理
    } else {
        SPDLOG_ERROR("Timeout waiting for event");
        // 可以选择退出或重试
    }
});
```

---

#### 4.4 启动顺序的重要性

**教训**: 分布式系统中，组件的启动顺序和等待时间至关重要。

**kungfu 系统的正确启动顺序**:
```
1. Master  (协调者)
   ↓ 等待 5 秒
2. Ledger  (账本服务)
   ↓ 等待 5 秒
3. MD      (行情网关)
   ↓ 等待 5 秒
4. TD      (交易网关)
   ↓ 等待 5 秒
5. Strategy (策略)
```

**自动化脚本**:
```bash
#!/bin/bash
start_service() {
    local name=$1
    local cmd=$2
    local wait_time=${3:-5}
    
    echo "Starting $name..."
    nohup $cmd > /tmp/${name}.log 2>&1 &
    sleep $wait_time
    
    if pgrep -f "$cmd" > /dev/null; then
        echo "✅ $name started"
    else
        echo "❌ $name failed to start"
        cat /tmp/${name}.log
        exit 1
    fi
}

# 清理旧状态
rm -rf /app/runtime
mkdir -p /app/runtime/system/etc/kungfu/db/live

# 按顺序启动
start_service "Master" "kfc master" 5
start_service "Ledger" "kfc ledger" 5
start_service "TD" "kfc td -s binance -a futures_testnet" 8
```

---

#### 4.5 日志的诊断价值

**教训**: 空日志比错误日志更可怕，说明核心逻辑根本没执行。

**日志级别的含义**:
```
TRACE   → 详细的执行流程（循环、条件判断）
DEBUG   → 调试信息（变量值、状态）
INFO    → 正常的业务流程（"Connecting", "login success"）← 最重要
WARNING → 非致命问题（重连、降级）
ERROR   → 错误但可恢复（API 失败、超时）
CRITICAL→ 严重错误（段错误、资源耗尽）
```

**诊断技巧**:
```bash
# 1. 检查日志级别分布
grep -o '\[ [a-z]* *\]' /tmp/td.log | sort | uniq -c
# 96 [ error  ]  ← 只有 error，异常！

# 2. 搜索关键成功标志
grep -i "connecting\|success\|ready\|login" /tmp/td.log
# ❌ 无输出 → 初始化逻辑未执行

# 3. 检查是否有任何 info 日志
grep '\[ info' /tmp/td.log
# ❌ 无输出 → 确认核心逻辑未执行

# 4. 前台启动捕获所有输出
timeout 10 kfc td -s binance -a futures_testnet 2>&1
# [warning] first() requires a stream with at least one value
# ✅ 找到根本问题
```

---

#### 4.6 僵尸进程的处理

**教训**: 进程崩溃后可能留下僵尸进程，占用资源但无法清理。

**识别僵尸进程**:
```bash
ps aux | grep kfc
# root   1073  0.1  0.0      0     0 ?   Z   16:40  0:01 [kfc] <defunct>
#                   ^^^                 ^^^                     ^^^^^^^^
#                    |                   |                         |
#                    CPU%              状态Z                    defunct
```

**清理方法**:
```bash
# 方法 1: 杀死父进程（让 init 回收）
pkill -9 -f kfc

# 方法 2: 重启容器（最彻底）
docker-compose restart app

# 方法 3: 检查是否真的需要清理
# 僵尸进程不占用内存，只占用 PID，如果数量不多可以忽略
```

---

### 5. 真正的解决方案：使用官方启动脚本

> **💡 重要结论**：经过深入调试，我们发现这个问题的根本原因是 **没有使用官方启动脚本**。
> 
> 上面的调试过程虽然帮助我们理解了 kungfu 框架的内部机制，但实际上这些问题在使用官方方式启动时**根本不会发生**。

---

#### ✅ 正确的启动方式

**官方脚本会自动处理所有问题**：

```bash
# 进入脚本目录
cd /app/scripts/binance_test

# 使用官方脚本启动
bash run.sh start
```

**官方脚本做了什么？**

```bash
# scripts/binance_test/run.sh
start() {
    echo "clearing journal..."
    find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f  # ← 自动清理 journal
    
    # start master
    pm2 start master.json
    sleep 5  # ← 等待 Master 初始化
    
    # start ledger
    pm2 start ledger.json
    sleep 5  # ← 等待 Ledger 就绪
    
    # start binance md
    pm2 start md_binance.json
    sleep 5  # ← 等待 MD 连接
    
    # start binance td
    pm2 start td_binance.json
    sleep 5  # ← 等待 TD 注册
}
```

---

#### 🔍 为什么官方方式能解决所有问题？

对照上面的调试过程，官方脚本解决了以下问题：

| 问题 | 手动启动的错误 | 官方脚本如何解决 |
|------|---------------|-----------------|
| **持久化状态冲突** | Master 记住旧注册，拒绝新注册 | ✅ 每次启动前清理 `*.journal` |
| **启动顺序错误** | TD 比 Master 先启动 | ✅ 严格按 Master → Ledger → MD → TD 顺序 |
| **时序竞争** | TD 注册时 Master 还未就绪 | ✅ 每个服务启动后等待 5 秒 |
| **缺少 Ledger** | 只启动了 Master 和 TD | ✅ 完整启动所有必需服务 |
| **进程管理混乱** | 手动 `nohup` 难以管理 | ✅ 使用 PM2 统一管理 |

---

#### 📋 完整启动流程（官方方式）

**步骤 1**: 安装 PM2（首次执行）

```bash
# 在容器中
apt-get update && apt-get install -y nodejs npm
npm install -g pm2
```

**步骤 2**: 配置数据库（添加账户）

```bash
# 方法 1: 交互式
python core/python/dev_run.py account -s binance add

# 方法 2: 手动创建（非 TTY 环境）
# 使用官方命令添加账户（自动创建正确表结构）
python3 /app/core/python/dev_run.py account -s binance add
# 交互输入: user_id=gz_user1, access_key=你的Key, secret_key=你的Secret
```

**步骤 3**: 启动系统

```bash
cd /app/scripts/binance_test
bash run.sh start
```

**步骤 4**: 验证

```bash
pm2 list
pm2 logs td_binance:gz_user1 --lines 20
```

**成功标志**：
- `pm2 list` 显示所有服务 `online`
- TD 日志显示 `"Connecting BINANCE TD"` 和 `"login success"`
- 没有 `-2015` 错误

---

#### 🚫 案例 1 的教训总结

1. **永远使用官方脚本**
   - 官方脚本包含必要的清理和时序逻辑
   - 手动启动会遇到各种时序和状态问题

2. **理解 vs 使用**
   - 案例 1 的调试过程帮助理解框架内部机制
   - 但实际使用时应遵循官方最佳实践

3. **分布式系统的复杂性**
   - 启动顺序、时序、持久化状态都很关键
   - 不要低估这些因素的影响

4. **官方文档的价值**
   - 官方文档和脚本是经过验证的最佳实践
   - 遇到问题时应首先查看官方方式

---

#### 📚 下一步

**案例 1 到此结束**。我们通过深入调试理解了：
- Event-driven 架构的消息传递机制
- Master-Apprentice 注册流程
- 持久化状态对系统行为的影响

**继续阅读** [案例 2](#案例-2pm2--数据库配置完整系统启动) 查看完整的官方启动流程和遇到的实际问题。

---

### 8. 参考资料

**代码文件**（帮助理解内部机制）:
- `core/cpp/yijinjing/src/practice/apprentice.cpp` - 启动流程和 RequestStart 等待逻辑
- `core/cpp/yijinjing/src/practice/master.cpp` - 注册逻辑和 location 管理
- `core/extensions/binance/src/trader_binance.cpp` - Binance TD Gateway 实现
- `scripts/binance_test/run.sh` - 官方启动脚本（✅ 实际使用这个）

**相关文档**:
- [官方安装文档](https://godzilla.dev/documentation/installation/) - 权威启动指南
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 系统架构
- [TESTNET.md](./TESTNET.md) - 测试网配置（包含官方启动流程）
- [LOG_LOCATIONS.md](./LOG_LOCATIONS.md) - 日志位置

**外部资源**:
- [RxCPP Error Handling](https://github.com/ReactiveX/RxCpp/blob/master/Rx/v2/examples/doxygen/error_handling.cpp)
- [Binance Futures Testnet](https://testnet.binancefuture.com/)

---

## 案例 2：PM2 + 数据库配置完整系统启动

> **✅ 正确示范**：本案例展示的是 **官方推荐的启动方式**。
> 
> 与案例 1 不同，这里我们使用官方脚本 `scripts/binance_test/run.sh`，这是系统设计者预期的启动方式。
> 
> **本案例的价值**：
> - 展示官方启动流程的完整步骤
> - 记录首次配置时可能遇到的实际问题（PM2 安装、数据库配置等）
> - 提供可复制的成功范例
> 
> **与案例 1 的对比**：
> - 案例 1：手动启动 → 遇到 Master-Apprentice 通信问题 → 深入调试
> - 案例 2：官方脚本 → 遇到配置问题 → 正确配置后成功运行
> 
> 如果你是第一次配置系统，**从本案例开始**是最佳选择。

---

**日期**: 2025-10-28  
**问题严重程度**: 🟡 中等（配置问题，非系统bug）  
**解决状态**: ✅ 已完全解决

### 1. 问题情境

#### 背景
- **目标**: 使用官方脚本 `scripts/binance_test/run.sh` 启动完整交易系统（✅ 正确方式）
- **环境**: Docker 容器，Binance Futures Testnet
- **预期**: Master, Ledger, MD, TD, Strategy 全部运行，策略接收实时市场数据

#### 遇到的问题链（5 个连环错误）

```
1. PM2 未安装
   ↓
2. 数据库不存在 (JSON 解析错误)
   ↓
3. PM2 配置文件账户名不匹配
   ↓
4. Journal 状态冲突 (segmentation fault)
   ↓
5. 策略启动方式错误 (无输出)
```

---

### 2. 详细调试过程

#### 问题 1: PM2 未安装

**错误现象**:
```bash
$ cd /app/scripts/binance_test
$ bash run.sh start
run.sh: line 9: pm2: command not found
```

**诊断**:
```bash
$ which pm2
# (无输出)

$ cat run.sh
#!/bin/bash
start() {
    pm2 start master.json  # ← 依赖 PM2
    pm2 start ledger.json
    pm2 start md_binance.json
    pm2 start td_binance.json
}
```

**根本原因**: 
- 官方脚本依赖 PM2 (Node.js 进程管理器)
- Docker 镜像中未预装 PM2

**解决方案**:
```bash
# 安装 Node.js 和 npm
apt-get update
apt-get install -y nodejs npm

# 安装 PM2
npm install -g pm2

# 验证
pm2 --version
# 输出: 6.0.13
```

**经验教训**:
- 官方脚本的依赖应该在文档中明确说明
- 生产环境应该在 Dockerfile 中预装 PM2

---

#### 问题 2: 数据库不存在

**错误现象**:
```bash
$ pm2 start td_binance.json
$ pm2 logs td_binance:gz_user1

RuntimeError: [json.exception.parse_error.101] parse error at line 1, column 1: 
syntax error while parsing value - unexpected end of input; expected '[', '{', or a literal
```

**诊断过程**:

**第一步: 检查数据库文件**
```bash
$ ls /root/.config/kungfu/app/kungfu.db
ls: cannot access '/root/.config/kungfu/app/kungfu.db': No such file or directory
```

**第二步: 尝试交互式添加账户**
```bash
$ python core/python/dev_run.py account -s binance add
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/prompt_toolkit/terminal/vt100_output.py", line 424, in from_pty
    assert stdout.isatty()
AssertionError
```

**第三步: 检查 TD 如何读取配置**
```cpp
// core/python/kungfu/command/td.py:23
ext = EXTENSION_REGISTRY_TD.get_extension(source)(
    low_latency, ctx.locator, account, account_config
)
// account_config 从数据库读取，如果为空 → JSON 解析失败
```

**根本原因**:
1. 用户尝试用交互式命令添加账户，但 Docker 非 TTY 环境失败
2. 数据库文件从未被创建
3. TD 启动时读取账户配置 → 空字符串 → JSON 解析失败

**解决方案**:
```bash
# 手动创建数据库
mkdir -p /root/.config/kungfu/app

# 使用官方命令添加账户（自动创建正确表结构）
python3 /app/core/python/dev_run.py account -s binance add
# 交互输入:
#   user_id: gz_user1
#   access_key: MpFV92IITflE1iFCyzjq1nWvHlWNlhNxwQcMdJCuTQJ0UKDPqEZbv9E47kSEUxbX
#   secret_key: UX9M52UeBxuQM91aJiOTiYjdcWMuoHStL7BZzZPAJKp7oZoGYI9DdX25jOj4bXDD

# 验证
$ python core/python/dev_run.py account -s binance show
receive_md    user_id    access_key                secret_key
------------  ---------  ------------------------  ------------------------
True          gz_user1   MpFV92IITflE1iFCyzjq1n... UX9M52UeBxuQM91aJiOTi...
```

**经验教训**:
- 非交互式环境需要提供手动创建数据库的方法
- TD 的错误信息应该更友好（"config not found" vs "JSON parse error"）

---

#### 问题 3: PM2 配置文件账户名不匹配

**错误现象**:
```bash
$ pm2 logs td_binance:gz_user1
# TD 不断重启，显示相同的 JSON 解析错误
```

**诊断过程**:

**检查 PM2 配置**:
```bash
$ cat scripts/binance_test/td_binance.json
{
  "apps": [{
    "name": "td_binance:gz_user1",
    "args": "-l trace td -s binance -a eiahb3838ya@ntu.im"  # ← 旧账户名
  }]
}
```

**检查数据库**:
```bash
$ python core/python/dev_run.py account -s binance show
user_id: gz_user1  # ← 数据库中的账户名
```

**根本原因**:
- PM2 配置使用旧账户名 `eiahb3838ya@ntu.im`
- 数据库中是 `gz_user1`
- TD 启动时找不到账户配置 → 返回空 → JSON 解析失败

**解决方案**:
```bash
# 修改 PM2 配置
nano scripts/binance_test/td_binance.json
# 改为: "args": "-l trace td -s binance -a gz_user1"

# 重启 TD
pm2 delete td_binance:gz_user1
pm2 start scripts/binance_test/td_binance.json
```

**经验教训**:
- 官方脚本应该使用统一的账户名（`gz_user1`）
- 配置文件和数据库应该保持同步

---

#### 问题 4: Journal 状态冲突

**错误现象**:
```bash
$ pm2 start strategy_hello.json
$ pm2 logs strategy:hello

[error] app register timeout
[critical] segmentation violation
Bus error (core dumped)
```

**诊断过程**:

**第一层: 检查策略日志**
```bash
$ cat /tmp/strategy.log
[10/28 16:18:18.344] [critical] segmentation violation
# 没有任何 INFO 日志 → 说明 pre_start() 从未执行
```

**第二层: 检查 Master 日志**
```bash
$ pm2 logs master --lines 50
[error] location strategy/default/hello/live has already been registered
```

**第三层: 查找旧的 journal 文件**
```bash
$ find /app/runtime -name '*.journal' -type f
/app/runtime/strategy/default/hello/journal/live/00000000.1.journal  # ← 旧文件
/app/runtime/system/master/master/journal/live/00000000.1.journal    # ← 记住了旧注册
```

**根本原因**:
1. 之前策略崩溃，journal 文件没有清理
2. Master 的 journal 记住了之前的注册
3. 新策略启动 → Master 拒绝注册（"已经注册"）
4. 策略等待 `RequestStart` 消息 → 超时 → 崩溃

**调用链分析**:
```cpp
// apprentice.cpp:188
events_ | is(msg::type::RequestStart) | first() |
$([&](event_ptr e) {
    on_start();  // ← 只有收到消息才执行
},
[&](std::exception_ptr e) {
    // 超时 → 抛出 rx::empty_error → 捕获但继续运行
    SPDLOG_WARN("first() requires a stream with at least one value");
});

// master.cpp:55
if (has_location(app_location->uid)) {
    SPDLOG_ERROR("location {} has already been registered", app_location->uname);
    return;  // ← 不发送 RequestStart
}
```

**解决方案**:
```bash
# 完全清理并重启
pm2 delete all
pkill -9 python
find /app/runtime -name '*.journal' -delete
find /app/runtime -name '*.nn' -type s -delete

cd /app/scripts/binance_test
bash run.sh start
```

**经验教训**:
- 官方脚本 `run.sh` 每次启动前清理 journal（`find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f`）
- 手动启动时必须记得清理
- Master 应该检测旧注册是否仍然存活

---

#### 问题 5: 策略启动方式错误

**错误现象**:
```bash
$ cd strategies/helloworld
$ python helloworld.py
# 完全无输出，程序直接退出
```

**诊断**:
```python
# strategies/helloworld/helloworld.py
def pre_start(context):
    config = context.get_config()
    context.subscribe(config["md_source"], [config["symbol"]], instrument_type, exchange)

# 直接运行 Python 文件:
# - 没有 kungfu 框架初始化
# - context 对象不存在
# - pre_start() 从未被调用
```

**根本原因**:
- 策略不是普通 Python 脚本
- 必须通过 kungfu 框架（`kfc strategy`）加载
- 框架负责创建 context、连接服务、调用回调

**解决方案**:
```bash
# 错误方式:
python strategies/helloworld/helloworld.py

# 正确方式:
python core/python/dev_run.py strategy -n hello \
  -p strategies/helloworld/helloworld.py \
  -c strategies/conf.json

# 或使用 PM2:
pm2 start scripts/binance_test/strategy_hello.json
```

**经验教训**:
- 策略文件应该在顶部注释说明启动方式
- 文档应该明确说明策略的运行机制

---

### 3. 最终完整解决方案

**一键启动脚本**:
```bash
#!/bin/bash
set -e  # 遇到错误立即退出

echo "=== 步骤 0: 安装 PM2 ==="
if ! command -v pm2 &> /dev/null; then
    apt-get update
    apt-get install -y nodejs npm
    npm install -g pm2
fi

echo "=== 步骤 1: 创建数据库 (如果不存在) ==="
if [ ! -f /root/.config/kungfu/app/kungfu.db ]; then
    # 使用官方命令添加账户
    python3 /app/core/python/dev_run.py account -s binance add
    # 交互输入: user_id=gz_user1, access_key=你的Key, secret_key=你的Secret
fi

echo "=== 步骤 2: 清理旧状态 ==="
pm2 delete all 2>/dev/null || true
pkill -9 python 2>/dev/null || true
find /app/runtime -name '*.journal' -delete 2>/dev/null || true
find /app/runtime -name '*.nn' -type s -delete 2>/dev/null || true

echo "=== 步骤 3: 启动所有服务 ==="
cd /app/scripts/binance_test
bash run.sh start

echo "=== 步骤 4: 等待服务稳定 ==="
sleep 30

echo "=== 步骤 5: 验证服务状态 ==="
pm2 list

echo "=== 步骤 6: 启动策略 ==="
pm2 start strategy_hello.json

echo "=== 步骤 7: 查看实时数据 ==="
echo "运行以下命令查看策略输出:"
echo "  pm2 logs strategy:hello --lines 20"
```

**成功指标**:
```bash
$ pm2 list
┌────┬────────────────────────┬─────────┬──────────┬────────┬───────────┐
│ id │ name                   │ mode    │ uptime   │ ↺      │ status    │
├────┼────────────────────────┼─────────┼──────────┼────────┼───────────┤
│ 0  │ master                 │ fork    │ 2m       │ 0      │ online    │
│ 1  │ ledger                 │ fork    │ 2m       │ 0      │ online    │
│ 2  │ md_binance             │ fork    │ 2m       │ 0      │ online    │
│ 3  │ td_binance:gz_user1    │ fork    │ 2m       │ 0      │ online    │
│ 4  │ strategy:hello         │ fork    │ 1m       │ 0      │ online    │
└────┴────────────────────────┴─────────┴──────────┴────────┴───────────┘

$ pm2 logs strategy:hello --lines 5
[btcusdt] Bid: 114110.80 (Vol: 2.0720) | Ask: 114120.10 (Vol: 0.0040) | Spread: 9.30
[btcusdt] Bid: 114120.20 (Vol: 0.8670) | Ask: 114120.90 (Vol: 17.6070) | Spread: 0.70
```

---

### 4. 核心经验总结

#### 4.1 依赖管理

**问题**: 官方脚本依赖 PM2，但文档未说明

**解决**:
- 在 TESTNET.md 添加 "Step 0: Install PM2"
- 考虑在 Dockerfile 中预装 PM2

#### 4.2 数据库初始化

**问题**: 交互式命令在 Docker 中失败，无备选方案

**解决**:
- 提供手动创建数据库的 Python 脚本
- 文档中说明两种方法：交互式 + 手动

#### 4.3 配置一致性

**问题**: PM2 配置和数据库账户名不一致

**解决**:
- 统一使用 `gz_user1`
- 文档中明确说明必须使用此账户名

#### 4.4 状态清理

**问题**: Journal 文件导致重启失败

**解决**:
- 官方脚本每次清理 journal
- 文档中提供完整清理命令
- 考虑添加自动检测和清理机制

#### 4.5 启动方式

**问题**: 用户不知道策略必须通过框架启动

**解决**:
- 策略文件添加注释说明启动方式
- 文档中明确说明
- 提供 PM2 配置示例

---

### 5. 调试技巧总结

#### 技巧 1: 逐层排查

```
表面现象 → 日志分析 → 代码追踪 → 根本原因
```

**案例**: JSON 解析错误
1. 看错误：JSON parse error
2. 查日志：无其他错误
3. 追代码：account_config 为空
4. 找原因：数据库不存在

#### 技巧 2: 检查完整调用链

**工具**:
```bash
# 查找函数调用
grep -rn "function_name" core/

# 查找消息类型
grep -rn "RequestStart" core/
```

#### 技巧 3: 验证假设

**每一步都要验证**:
```bash
# 假设：数据库存在
ls /root/.config/kungfu/app/kungfu.db  # 验证

# 假设：账户正确
python core/python/dev_run.py account -s binance show  # 验证

# 假设：服务运行
pm2 list  # 验证
```

#### 技巧 4: 对比官方脚本

**当手动启动失败时**:
1. 查看官方脚本如何启动
2. 对比差异（顺序、等待时间、清理步骤）
3. 采用官方方式

---

### 6. 相关文档

- [TESTNET.md](./TESTNET.md) - PM2 安装、数据库创建、官方脚本使用
- [LOG_LOCATIONS.md](./LOG_LOCATIONS.md) - PM2 日志位置
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 系统架构和事件流

