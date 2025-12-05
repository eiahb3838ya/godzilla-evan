# hf-live 工作流設計

## 文檔元信息
- **版本**: v2.0
- **日期**: 2025-12-04
- **目標**: 定義因子大師、模型大師、策略大師的完整工作流
- **前置**: [prd_hf-live.abstract.md](prd_hf-live.abstract.md)

---

## 一、角色定位

### 1.1 策略大師 (Strategy Master)

**關注點**: 交易邏輯
- ✅ 我有一個 `.so` 文件 (黑盒)
- ✅ 我寫 `on_factor()` 決定如何下單
- ✅ 我寫 JSON 配置
- ✅ 我用 pm2 啟動服務

**不關心**:
- ❌ `.so` 裡面是什麼
- ❌ 因子如何計算
- ❌ C++ 編譯

### 1.2 因子大師 (Factor Master)

**關注點**: 因子計算
- ✅ 我寫因子計算邏輯 (C++)
- ✅ 我只關注計算,不發送數據 (框架統一處理)
- ✅ 我交付因子代碼給模型大師

**不關心**:
- ❌ 如何發送因子數據
- ❌ 模型如何使用因子
- ❌ 如何下單

### 1.3 模型大師 (Model Master)

**關注點**: 模型訓練與部署
- ✅ 我使用因子大師提供的因子訓練 ONNX 模型
- ✅ 我放置 `.onnx` 文件到指定目錄
- ✅ 我配置模型參數 (輸入因子維度、輸出預測維度)
- ✅ 我用 `make` 打包成 `.so`

**不關心**:
- ❌ 因子計算邏輯
- ❌ 如何下單
- ❌ pm2 服務管理

### 1.4 角色關係

```
因子大師 → 編寫因子代碼 → 交付給模型大師
                              ↓
模型大師 → 訓練 ONNX 模型 → 放置 .onnx → 編譯 .so → 交付
                                                    ↓
策略大師 ← 接收 .so ← 放置文件 → 配置 JSON → 啟動服務 → 交易
         (不知道接收的是因子還是預測值)
```

**常規流程** (🔥 標準):
```
因子計算 → 模型預測 → 策略接收預測值 (via on_factor)
```

**簡化流程** (少見):
```
因子計算 → 直接發送 → 策略接收因子 (via on_factor)
```

**特殊情況**: 一人身兼三職
- 因子大師 = 模型大師 = 策略大師
- 流程簡化: 寫因子 → 訓練模型 → `make` → `.so` 留在原位 → 直接使用

---

## 二、策略大師工作流

### 2.1 前置條件

- ✅ 已有 Godzilla 環境 (Docker 容器)
- ✅ 已有 `libsignal.so` 文件
- ✅ 已有交易策略想法

### 2.2 完整流程 (5 步)

#### 步驟 1: 放置 .so 文件

```bash
# 策略大師收到 .so 後放置
mkdir -p /app/hf-live/build
cp libsignal.so /app/hf-live/build/

# 驗證
ls -lh /app/hf-live/build/libsignal.so
# -rwxr-xr-x 1 root root 2.3M Dec 03 10:00 libsignal.so
```

**位置**: `/app/hf-live/build/libsignal.so` (容器內路徑)

#### 步驟 2: 編寫策略

```python
# strategies/my_factor_strategy/run.py
from kungfu.wingchun import Strategy

class MyFactorStrategy(Strategy):
    def on_quote(self, context, quote):
        """可選: 原有的逐筆行情回調"""
        pass

    def on_depth(self, context, depth):
        """可選: 原有的深度行情回調 (大多數情況不再需要)"""
        pass

    def on_factor(self, context, symbol, timestamp, values):
        """🔥 核心: 因子數據回調 - 框架自動觸發"""
        # values[0]: 因子1
        # values[1]: 因子2
        # ...

        if values[0] > 0.5:  # 示例: 做多信號
            context.insert_order(
                symbol=symbol,
                exchange_id="binance",
                side="Buy",
                offset="Open",
                price_type="Limit",
                price=context.get_last_price(symbol),
                volume=1
            )
        elif values[0] < -0.5:  # 示例: 做空信號
            context.insert_order(
                symbol=symbol,
                exchange_id="binance",
                side="Sell",
                offset="Close",
                price_type="Limit",
                price=context.get_last_price(symbol),
                volume=1
            )
```

**關鍵**: `on_factor()` 簽名與 `on_depth()` 完全一致風格

#### 步驟 3: 配置 JSON

```json
// strategies/my_factor_strategy/config.json
{
  "strategy": {
    "name": "my_factor_strategy",
    "path": "strategies/my_factor_strategy/run.py",
    "signal_lib": "/app/hf-live/build/libsignal.so"  // 🔥 新增
  },
  "md": {
    "source_id": "binance",
    "symbols": ["btc_usdt", "eth_usdt"]
  },
  "td": {
    "source_id": "binance",
    "account_id": "my_account"
  }
}
```

**新增字段**: `signal_lib` - 指定 .so 路徑

#### 步驟 4: 啟動服務 (pm2)

```bash
# 容器內操作
docker exec -it godzilla-dev bash

# 1. 啟動 Master
pm2 start /app/scripts/pm2/master.json

# 等待 5 秒
sleep 5

# 2. 啟動 Ledger
pm2 start /app/scripts/pm2/ledger.json
sleep 5

# 3. 啟動 MD (行情)
pm2 start /app/scripts/pm2/md_binance.json
sleep 5

# 4. 啟動 TD (交易)
pm2 start /app/scripts/pm2/td_binance.json
sleep 5

# 5. 🔥 啟動 Strategy (自動加載 .so)
pm2 start /app/strategies/my_factor_strategy/pm2.json

# 查看狀態
pm2 list
pm2 logs my_factor_strategy
```

**關鍵**: Strategy 啟動時框架自動加載 `signal_lib` 指定的 `.so`

#### 步驟 5: 監控與調試

```bash
# 查看實時日誌
pm2 logs my_factor_strategy --lines 100

# 查看因子回調
# 應該看到類似:
# [INFO] Factor for btc_usdt: [0.23, -0.45, 1.02, ...]
# [INFO] Inserting order: Buy btc_usdt @ 45000.0

# 查看錯誤
pm2 logs my_factor_strategy --err

# 重啟策略
pm2 restart my_factor_strategy
```

### 2.3 更新 .so 流程 (基礎版)

**簡化流程** (當因子大師提供新版本 `.so`):

```bash
# 1. 替換 .so
docker cp libsignal_v2.so godzilla-dev:/app/hf-live/build/libsignal.so

# 2. 重啟策略 (dlopen 重新載入 .so)
docker exec godzilla-dev pm2 restart my_factor_strategy

# 3. 驗證
docker exec godzilla-dev pm2 logs my_factor_strategy --lines 20
```

**進階部署**: 見 [prd_hf-live.08-build-deploy.md §3](prd_hf-live.08-build-deploy.md)
- 灰度發佈 (運行兩個版本並行驗證)
- 一鍵回滾腳本
- 生產環境部署 checklist

---

## 三、因子大師工作流

### 3.1 前置條件

- ✅ hf-live 項目源碼 (獨立倉庫)
- ✅ C++ 開發環境
- ✅ 因子計算邏輯

**因子大師是否需要理解 Godzilla?**

| 需要知道 | 不需要知道 |
|---------|-----------|
| ✅ `Depth`, `Trade` 數據結構 (已包含在 hf-live/include/) | ❌ Godzilla 的 runner.cpp 實現 |
| ✅ 如何編譯 `.so` (`make` 指令) | ❌ Godzilla 的 Strategy 基類 |
| ✅ 因子 API (GetFactors, OnDepth) | ❌ Godzilla 的 RxCpp 事件流 |
|  | ❌ Godzilla 的 pm2 配置 |

**獨立開發**: market_data_types.h 已包含在 hf-live 倉庫,零配置即可編譯

### 3.2 完整流程 (4 步)

#### 步驟 1: 開發因子

**架構說明** (參考 ref 項目設計):

```
Engine (統一入口)
  → 調用所有因子模塊更新
  → 收集結果
  → 統一發送 (SignalSender::Send)
```

**因子大師代碼** (專注因子邏輯):

```cpp
// hf-live/factors/my_factors/factor_entry.cpp
#include "market_data_types.h"  // 已包含在 hf-live/include/

class MyFactorEntry {
private:
    double factors_[10];  // 因子值存儲

public:
    void OnDepth(const Depth* depth) {
        // 🔥 核心: 僅計算因子,不負責發送

        // 示例: 買賣價差因子
        factors_[0] = (depth->bid_price[0] - depth->ask_price[0]) / depth->ask_price[0];

        // 示例: 訂單簿失衡因子
        double bid_vol = depth->bid_volume[0];
        double ask_vol = depth->ask_volume[0];
        factors_[1] = (bid_vol - ask_vol) / (bid_vol + ask_vol);

        // 示例: 深度加權中間價
        factors_[2] = (depth->bid_price[0] * ask_vol + depth->ask_price[0] * bid_vol)
                     / (bid_vol + ask_vol);

        // ... 更多因子計算 ...
        // ❌ 不在這裡調用 Send!
    }

    void OnTrade(const Trade* trade) {
        // 可選: 基於逐筆成交計算因子
    }

    const double* GetFactors() const { return factors_; }
    int GetFactorCount() const { return 10; }
};
```

**Engine 統一發送** (框架代碼,因子大師一般不需修改):

```cpp
// hf-live/app_live/engine.cpp
#include "signal_sender.h"

class Engine {
    std::vector<MyFactorEntry*> factors_;

public:
    void OnDepth(const Depth* depth) {
        // 1. 調用所有因子模塊更新
        for (auto* factor : factors_) {
            factor->OnDepth(depth);
        }

        // 2. 收集結果
        std::vector<double> all_factors;
        for (auto* factor : factors_) {
            const double* vals = factor->GetFactors();
            int count = factor->GetFactorCount();
            all_factors.insert(all_factors.end(), vals, vals + count);
        }

        // 3. 🔥 統一發送 (一次性發送所有因子)
        SignalSender::Send(
            depth->symbol,
            depth->data_time,
            all_factors.data(),
            all_factors.size()
        );
    }
};
```

**關鍵**:
- ✅ 因子模塊: 專注計算,不負責發送
- ✅ Engine: 統一收集與發送
- ✅ 清晰分離: 計算邏輯 vs 通訊機制

#### 步驟 2: 編譯打包

```bash
# 在 hf-live 項目根目錄
cd /path/to/hf-live

# 編譯 (默認 Release 優化模式)
make

# 或清理後重新編譯
make clean-build

# 結果
ls -lh build/libsignal.so
# -rwxr-xr-x 1 user user 2.3M Dec 03 10:00 build/libsignal.so

# 注: Makefile 詳見 prd_hf-live.07-implementation.md §4.2
```

**產物**: 單一 `.so` 文件

#### 步驟 3: 交付 (或自用)

**情況 A: 交付給策略大師**

```bash
# 打包
tar -czf libsignal_v1.0_20251203.tar.gz build/libsignal.so

# 傳輸 (示例)
scp libsignal_v1.0_20251203.tar.gz strategy_master@server:/tmp/

# 通知策略大師:
# - .so 版本: v1.0
# - 因子數量: 10
# - 因子含義: [價差, 失衡, 中間價, ...]
# - 更新日期: 2025-12-03
```

**情況 B: 自用 (一人大師)**

```bash
# 無需移動,直接在原位使用
# hf-live/build/libsignal.so 已就緒

# 在 Godzilla 配置中指向此路徑即可
# config.json: "signal_lib": "/app/hf-live/build/libsignal.so"
```

#### 步驟 4: 文檔化

```markdown
# libsignal v1.0 使用說明

## 因子列表 (共 10 個)

| Index | 名稱 | 計算公式 | 範圍 | 說明 |
|-------|------|----------|------|------|
| 0 | 買賣價差 | (bid-ask)/ask | [-1, 1] | 負值表示流動性好 |
| 1 | 訂單簿失衡 | (bid_vol-ask_vol)/(total) | [-1, 1] | 正值看多 |
| 2 | 深度加權中間價 | weighted_mid | [0, ∞] | 動態中間價 |
| ... | ... | ... | ... | ... |

## 更新日誌
- v1.0 (2025-12-03): 初始版本
- v1.1 (待定): 新增波動率因子
```

**關鍵**: 讓策略大師知道如何使用 `values[i]`

### 3.3 迭代開發

```bash
# 修改因子邏輯
vim app_live/my_factor_engine.cpp

# 重新編譯
make

# 本地測試 (可選)
make test

# 更新版本
# v1.0 → v1.1
mv build/libsignal.so build/libsignal_v1.1.so

# 交付或自用
# ...
```

---

## 四、模型大師工作流 (🔥 常規流程)

### 4.1 前置條件

- ✅ hf-live 項目源碼 (包含 models/ 目錄)
- ✅ 因子大師已完成因子計算模塊
- ✅ Python 訓練環境 (PyTorch/TensorFlow)
- ✅ ONNX Runtime C++ 環境

**模型大師需要知道什麼?**

| 需要知道 | 不需要知道 |
|---------|-----------|
| ✅ 因子維度 (從因子大師獲取) | ❌ 因子計算邏輯細節 |
| ✅ 目標預測任務 (漲跌、收益率等) | ❌ Godzilla 架構 |
| ✅ ONNX 模型格式 | ❌ 策略交易邏輯 |
| ✅ 模型配置文件格式 | ❌ C++ Engine 實現 |

### 4.2 完整流程 (5 步)

#### 步驟 1: 訓練 ONNX 模型

**離線訓練** (Python):

```python
# train_model.py
import torch
import torch.nn as nn

class PredictionModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 預測 [-1, 1] 範圍
        )

    def forward(self, x):
        return self.net(x)

# 訓練 (示例,實際需要歷史因子數據)
model = PredictionModel(input_dim=10, output_dim=3)
# ... 訓練邏輯 ...

# 🔥 導出 ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "prediction_model.onnx",
    input_names=["factors"],
    output_names=["predictions"],
    dynamic_axes={"factors": {0: "batch_size"}}
)

print("✅ ONNX model saved: prediction_model.onnx")
```

**產物**: `prediction_model.onnx` (可移植模型文件)

#### 步驟 2: 配置模型參數

```json
// hf-live/models/demo/model_config.json
{
  "model": {
    "name": "demo_predictor",
    "onnx_path": "models/demo/prediction_model.onnx",
    "input_dim": 10,        // 對應因子數量
    "output_dim": 3,        // 預測值數量
    "thread_num": 4,        // ONNX Runtime 線程數
    "batch_size": 1,        // 批次大小
    "warmup": true          // 啟動時預熱
  }
}
```

**關鍵字段**:
- `input_dim`: 必須與因子大師的 `GetFactorCount()` 匹配
- `output_dim`: 預測值數量 (策略大師接收)
- `onnx_path`: 相對於 hf-live 根目錄路徑

#### 步驟 3: 放置模型文件

```bash
# 在 hf-live 項目中
cd /path/to/hf-live

# 創建模型目錄
mkdir -p models/demo

# 複製 ONNX 模型
cp /path/to/prediction_model.onnx models/demo/

# 複製配置
cp /path/to/model_config.json models/demo/

# 驗證
ls -lh models/demo/
# prediction_model.onnx  (2.3M)
# model_config.json      (256B)
```

#### 步驟 4: 編譯帶模型的 .so

**修改主配置** (啟用模型):

```json
// hf-live/config/app_config.json
{
  "factor": {
    "module": "my_factors",
    "output_dim": 10
  },
  "model": {
    "enabled": true,              // 🔥 啟用模型
    "config_path": "models/demo/model_config.json"
  }
}
```

**編譯**:

```bash
# 在 hf-live 項目根目錄
make clean
make

# 驗證依賴
ldd build/libsignal.so | grep onnx
# libonnxruntime.so.1.12.0 => /usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.12.0
```

**產物**: `build/libsignal.so` (包含 Factor + Model pipeline)

#### 步驟 5: 交付與文檔

**交付清單**:

```bash
# 打包
tar -czf libsignal_with_model_v1.0.tar.gz \
  build/libsignal.so \
  models/demo/prediction_model.onnx \
  models/demo/model_config.json

# 傳輸給策略大師
scp libsignal_with_model_v1.0.tar.gz strategy_master@server:/tmp/
```

**文檔** (關鍵):

```markdown
# libsignal v1.0 (with Model) 使用說明

## 架構
- 輸入: 10 個因子 (來自 FactorCalculationEngine)
- 模型: ONNX Runtime (prediction_model.onnx)
- 輸出: 3 個預測值 (策略大師 via on_factor 接收)

## 預測值含義

| Index | 名稱 | 範圍 | 說明 |
|-------|------|------|------|
| 0 | 漲跌預測 | [-1, 1] | 正值看多,負值看空 |
| 1 | 波動率預測 | [0, 1] | 預期波動率 |
| 2 | 置信度 | [0, 1] | 預測可信度 |

## 重要提醒
- 🔥 策略大師接收的是**預測值**,不是原始因子
- 策略大師**不知道**這是模型輸出 (對他來說就是 `values[]`)
- 預測值已經過模型推理,可以直接用於交易決策

## 性能
- 推理延遲: ~0.5ms (單次,4線程)
- 吞吐量: ~2000 predictions/sec
```

### 4.3 模型更新流程

當需要更新模型 (重新訓練):

```bash
# 1. 訓練新版本
python train_model_v2.py
# → prediction_model_v2.onnx

# 2. 替換模型文件
cp prediction_model_v2.onnx models/demo/prediction_model.onnx

# 3. 重新編譯 (如果配置變更)
make clean && make

# 4. 交付新 .so
# 策略大師僅需 pm2 restart (無需改代碼)
```

**優勢**:
- ✅ 模型與代碼分離 (ONNX 文件獨立)
- ✅ 策略大師無需關心模型細節
- ✅ 模型大師可獨立迭代優化

---

## 五、協作模式

### 5.1 完整協作 (因子大師 + 模型大師 + 策略大師)

```
因子大師 (Alice)
   ↓ 編寫因子計算代碼
   ↓ 交付: factors/ 模塊 + 文檔

模型大師 (Bob)
   ↓ 基於因子訓練 ONNX 模型
   ↓ 交付: prediction_model.onnx + model_config.json
   ↓ 編譯: make → libsignal.so (Factor + Model)

策略大師 (Charlie)
   ↓ 接收 .so + 文檔
   ↓ 編寫 on_factor() 策略邏輯
   ↓ 配置 JSON + pm2 啟動
   ↓ 運行交易
```

**分工明確**:
- Alice 不知道模型細節,只關注因子質量
- Bob 不知道策略邏輯,只關注預測準確性
- Charlie 不知道因子/模型,只關注 `values[]` 信號質量

### 5.2 分離協作 (因子大師 ≠ 策略大師,無模型)

```
因子大師                           策略大師
   ↓                                  ↓
開發因子邏輯                      設計交易策略
   ↓                                  ↓
編譯 .so                          等待 .so
   ↓                                  ↓
交付 .so + 文檔 ────────────────→ 接收
   ↓                                  ↓
迭代開發 ←───────── 反饋 ←──────── 配置 JSON
   ↓                                  ↓
提供 v2 ─────────────────────→ 更新 .so
                                      ↓
                                   啟動服務
                                      ↓
                                   監控交易
```

**溝通界面**:
- `.so` 文件
- 因子文檔 (index → 含義)
- 版本號

### 5.3 合一模式 (因子大師 = 模型大師 = 策略大師)

```
你 (一人全職)
   ↓
寫因子邏輯 (hf-live/factors/)
   ↓
訓練 ONNX 模型 (Python)
   ↓
放置 .onnx + 配置 (hf-live/models/)
   ↓
make (編譯 Factor + Model)
   ↓
.so 留在原位
   ↓
寫策略邏輯 (strategies/my_strategy/run.py)
   ↓
配置 JSON (指向 .so 路徑)
   ↓
pm2 啟動
   ↓
同時監控因子/模型/交易
   ↓
迭代:
  - 改因子 → make → pm2 restart
  - 改模型 → 重新訓練 .onnx → make → pm2 restart
  - 改策略 → pm2 restart (無需 make)
```

**優勢**:
- ✅ 完全控制整個 pipeline
- ✅ 快速迭代,無溝通成本
- ✅ 可以針對策略表現反向調整因子/模型

---

## 六、配置文件詳解

### 6.1 策略配置 (新增字段)

```json
{
  "strategy": {
    "name": "my_factor_strategy",
    "path": "strategies/my_factor_strategy/run.py",

    // 🔥 新增: hf-live 配置
    "signal_lib": "/app/hf-live/build/libsignal.so",  // .so 路徑
    "signal_config": {                                  // 傳遞給 signal_create()
      "factors_enabled": [0, 1, 2],  // 可選: 啟用哪些因子
      "update_interval_ms": 100      // 可選: 因子更新頻率
    }
  },

  // 原有配置
  "md": { ... },
  "td": { ... }
}
```

### 6.2 pm2 配置

```json
// strategies/my_factor_strategy/pm2.json
{
  "apps": [{
    "name": "my_factor_strategy",
    "script": "python3",
    "args": "-m kungfu.command strategy --config /app/strategies/my_factor_strategy/config.json",
    "cwd": "/app",
    "env": {
      "PYTHONPATH": "/app",
      "LD_LIBRARY_PATH": "/app/hf-live/build"  // 🔥 確保找到 .so
    },
    "log_date_format": "YYYY-MM-DD HH:mm:ss.SSS"
  }]
}
```

---

## 七、常見場景

### 7.1 場景: 模型大師發佈新模型 (🔥 常規)

```bash
# 模型大師側
cd /path/to/hf-live

# 1. 訓練新模型
python train_model_v2.py
# → prediction_model_v2.onnx (精度提升 2%)

# 2. 替換模型文件
cp prediction_model_v2.onnx models/demo/prediction_model.onnx

# 3. 重新編譯 (包含新模型)
make clean && make

# 4. 打包交付
tar -czf libsignal_v2.0_model_improved.tar.gz \
  build/libsignal.so \
  models/demo/prediction_model.onnx

scp libsignal_v2.0_model_improved.tar.gz strategy@server:/tmp/

# 策略大師側
pm2 stop my_factor_strategy
tar -xzf /tmp/libsignal_v2.0_model_improved.tar.gz -C /app/hf-live/
pm2 restart my_factor_strategy
pm2 logs my_factor_strategy  # 驗證新預測值
```

**關鍵**: 策略大師代碼無需修改,只需重啟

### 7.2 場景: 因子大師發佈新因子 (無模型)

```bash
# 因子大師側
cd /path/to/hf-live
# 修改 factors/ 代碼...
make
mv build/libsignal.so build/libsignal_v1.2_factors_only.so
scp build/libsignal_v1.2_factors_only.so strategy@server:/tmp/

# 策略大師側
pm2 stop my_factor_strategy
cp /tmp/libsignal_v1.2_factors_only.so /app/hf-live/build/libsignal.so
pm2 restart my_factor_strategy
pm2 logs my_factor_strategy  # 驗證新因子
```

### 7.3 場景: 策略大師測試多個 .so

```json
// config_test_v1.json
{
  "strategy": {
    "name": "test_v1",
    "signal_lib": "/app/hf-live/build/libsignal_v1.so"
  }
}

// config_test_v2.json
{
  "strategy": {
    "name": "test_v2",
    "signal_lib": "/app/hf-live/build/libsignal_v2.so"
  }
}

# 同時運行 A/B 測試
pm2 start pm2_test_v1.json
pm2 start pm2_test_v2.json
```

### 7.4 場景: 緊急回滾

```bash
# 策略大師側
pm2 stop my_factor_strategy

# 恢復舊版本
cp /app/hf-live/build/libsignal_v1.1.so.bak /app/hf-live/build/libsignal.so

pm2 restart my_factor_strategy
```

---

## 七、檢查清單

### 7.1 策略大師部署前

- [ ] 已收到 `.so` 文件
- [ ] 已收到因子文檔 (index → 含義)
- [ ] 已放置 `.so` 到指定路徑
- [ ] 已編寫 `on_factor()` 邏輯
- [ ] 已配置 JSON (`signal_lib` 字段)
- [ ] 已測試 pm2 啟動流程

### 7.2 因子大師交付前

- [ ] 代碼已編譯無錯誤
- [ ] `.so` 文件可執行 (`chmod +x`)
- [ ] 已測試基本功能 (可選: 單元測試)
- [ ] 已編寫因子文檔
- [ ] 已標註版本號與日期
- [ ] 已通知策略大師

---

## 八、故障排查

### 8.1 策略大師: .so 加載失敗

**症狀**:
```
[ERROR] Failed to load libsignal.so: cannot open shared object file
```

**解決**:
```bash
# 1. 檢查文件存在
ls -lh /app/hf-live/build/libsignal.so

# 2. 檢查權限
chmod +x /app/hf-live/build/libsignal.so

# 3. 檢查 LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/app/hf-live/build:$LD_LIBRARY_PATH

# 4. 檢查依賴
ldd /app/hf-live/build/libsignal.so
```

### 8.2 策略大師: on_factor 未觸發

**症狀**:
```
[INFO] Strategy started
[INFO] MD connected
# ... 但沒有 "Factor for ..." 日誌
```

**檢查**:
```bash
# 1. 確認 .so 已加載
pm2 logs my_factor_strategy | grep "signal_lib loaded"

# 2. 確認行情數據流入
pm2 logs my_factor_strategy | grep "on_depth"

# 3. 確認 runner.cpp 已轉發數據
# (需要 Godzilla C++ 代碼已集成 signal_on_data)
```

### 8.3 因子大師: 編譯失敗

**症狀**:
```
error: 'market_data_types.h' file not found
```

**解決**:
```bash
# 檢查 header 文件是否存在
ls -l hf-live/include/market_data_types.h
# 應該存在,因為是 bundled header (直接複製到倉庫)

# 如果不存在,重新複製 (見 prd_hf-live.02-data-structure-sharing.md)
cp /path/to/godzilla-evan/core/cpp/wingchun/include/kungfu/wingchun/msg.h \
   hf-live/include/market_data_types.h

# 檢查 CMakeLists.txt 包含路徑
grep "include_directories" hf-live/CMakeLists.txt
# 應包含: ${CMAKE_CURRENT_SOURCE_DIR}/include
```

---

## 九、性能考慮

### 9.1 策略大師

- ✅ `.so` 熱更新無需重啟整個系統 (僅重啟 Strategy)
- ✅ `on_factor()` 與 `on_depth()` 性能相當
- ⚠️ 避免在 `on_factor()` 中執行長時間運算 (應該 <1ms)

### 9.2 因子大師

- ✅ 因子計算應盡量優化 (目標: <100μs per depth update)
- ✅ 使用 `SignalSender::Send()` 無額外開銷
- ⚠️ 避免在 `OnDepth()` 中執行 I/O 操作

---

## 十、總結

### 策略大師視角

```
1. 拿到 .so
2. 放置文件
3. 寫 on_factor()
4. 配置 JSON
5. pm2 啟動
6. 監控交易
```

**核心**: 把 `.so` 當作黑盒,專注交易邏輯

### 因子大師視角

```
1. 寫因子邏輯
2. make 編譯
3. 交付 .so + 文檔
4. 迭代
```

**核心**: 把策略邏輯當作未知,專注因子計算

---

**下一步**:
- [prd_hf-live.04-implementation.md](prd_hf-live.04-implementation.md) - 框架內部實現細節
- [prd_hf-live.05-build-deploy.md](prd_hf-live.05-build-deploy.md) - CMake 與 CI/CD
