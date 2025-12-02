---
title: Development Guide
updated_at: 2025-12-01
owner: core-dev
lang: en
tokens_estimate: 1000
layer: archive
tags: [development, workflow, contribution]
purpose: "Quick development workflow guide"
---

# Development Guide

**快速開始**: 閱讀 [NAVIGATION.md](../NAVIGATION.md) 找到對應的開發任務

---

## 開發環境

### 必要工具
- **Docker** + **Docker Compose** (所有開發在容器內)
- **Git** (版本控制)
- **VSCode** (推薦 IDE)

### 容器設定
```bash
# 啟動開發容器
docker-compose up -d

# 進入容器
docker exec -it godzilla-dev bash
```

---

## 程式碼結構

```
godzilla-evan/
├── core/
│   ├── cpp/                    # C++ 核心
│   │   ├── yijinjing/         # 事件系統
│   │   └── wingchun/          # 交易引擎
│   ├── python/kungfu/         # Python 綁定 + CLI
│   ├── extensions/binance/    # Binance 交易所
│   └── build/                 # 編譯輸出
├── strategies/                # 交易策略
├── scripts/                   # PM2 配置
└── .doc/                      # 文檔系統
```

**詳細索引**: [`CODE_INDEX.md`](../CODE_INDEX.md)

---

## 開發工作流

### 1. 開發新策略

```bash
# 1. 創建策略檔案
mkdir -p strategies/my_strategy
cat > strategies/my_strategy/my_strategy.py << 'PYTHON'
from kungfu.wingchun import Strategy

class MyStrategy(Strategy):
    def pre_start(self, context):
        context.add_account("binance", "test_account")
        context.subscribe("binance", ["btc_usdt"], InstrumentType.Spot, Exchange.BINANCE)
    
    def on_depth(self, context, depth):
        context.log().info(f"BTC price: {depth.ask_price[0]}")
PYTHON

# 2. 創建 PM2 配置
cat > scripts/my_strategy/strategy_my_strategy.json << 'JSON'
{
  "apps": [{
    "name": "strategy_my_strategy",
    "script": "/app/core/python/dev_run.py",
    "args": "-l info strategy -n my_strategy -p /app/strategies/my_strategy/my_strategy.py",
    "env": {"KF_HOME": "/app/runtime"}
  }]
}
JSON

# 3. 啟動策略 (在容器內)
pm2 start /app/scripts/my_strategy/strategy_my_strategy.json
pm2 logs my_strategy
```

**詳細指南**: [`modules/strategy_framework.md`](../modules/strategy_framework.md)

---

### 2. 修改 C++ 核心

```bash
# 在容器內編譯
cd /app/core/build
make -j$(nproc)

# 驗證 Python 綁定
python3 -c "from kungfu.wingchun import Strategy; print('OK')"
```

**編譯選項**:
- `Release`: 生產環境 (-O3)
- `Debug`: 開發除錯 (-O0 -g)
- `RelWithDebInfo`: 效能分析 (-O3 -g)

**詳細指南**: [`modules/python_bindings.md`](../modules/python_bindings.md)

---

### 3. 除錯

```bash
# 查看日誌
pm2 logs <service_name>

# 檢查服務狀態
pm2 list

# 查看 Journal 事件
# (暫無工具,需手動讀取 binary)
```

**除錯指南**: [`operations/debugging_guide.md`](../operations/debugging_guide.md)

---

## 貢獻指南

### Git 工作流

```bash
# 1. 創建功能分支
git checkout -b feature/my-feature

# 2. 開發並提交
git add .
git commit -m "feat: add my feature"

# 3. 推送並創建 PR
git push origin feature/my-feature
```

### 提交訊息規範

使用 Conventional Commits:
- `feat:` - 新功能
- `fix:` - Bug 修復
- `docs:` - 文檔更新
- `refactor:` - 重構
- `test:` - 測試
- `chore:` - 其他維護

**範例**:
```
feat(binance): add futures market support

- Implement /fapi/v1 REST endpoints
- Add futures WebSocket subscription
- Update config schema

Closes #123
```

---

## 測試

### 單元測試 (C++)

```bash
# 編譯測試
cd /app/core/build
cmake -DBUILD_TESTS=ON ..
make tests

# 執行測試
./tests/test_yijinjing
./tests/test_wingchun
```

### 整合測試 (Python)

```bash
# 執行策略測試
python3 -m pytest strategies/tests/

# 回測
# (暫無框架,需手動實作)
```

---

## 新增交易所

### 步驟

1. **創建 Extension 目錄**:
   ```bash
   mkdir -p core/extensions/my_exchange
   ```

2. **實作介面**:
   ```cpp
   // marketdata_my_exchange.cpp
   class MarketDataMyExchange : public MarketData {
       void subscribe(...) override { ... }
   };
   
   // trader_my_exchange.cpp
   class TraderMyExchange : public Trader {
       uint64_t insert_order(...) override { ... }
       uint64_t cancel_order(...) override { ... }
   };
   ```

3. **註冊 Extension**:
   ```cpp
   EXTENSION_REGISTRY_MD.register_extension<MarketDataMyExchange>("my_exchange");
   EXTENSION_REGISTRY_TD.register_extension<TraderMyExchange>("my_exchange");
   ```

4. **創建配置契約文檔**:
   ```bash
   cp .doc/contracts/binance_config_contract.md \
      .doc/contracts/my_exchange_config_contract.md
   # 編輯配置格式
   ```

**參考實作**: [`modules/binance_extension.md`](../modules/binance_extension.md)

---

## 文檔更新

### 修改程式碼後

**遵循 DRY 原則**: 每項資訊只在一處維護

| 修改類型 | 需更新文檔 |
|---------|-----------|
| 資料結構 (msg.h) | `contracts/*_object_contract.md` + `CODE_INDEX.md` |
| API (context.cpp) | `contracts/strategy_context_api.md` |
| 生命週期 (runner.cpp) | `modules/strategy_framework.md` |
| Python 綁定 | `modules/python_bindings.md` |
| 配置格式 | `config/CONFIG_REFERENCE.md` |
| 架構決策 | 新增 `adr/00X-decision-name.md` |

**驗證工具**:
```bash
# 驗證程式碼引用
python3 .doc/operations/scripts/verify_code_refs.py

# 驗證連結
python3 .doc/operations/scripts/check_links.py
```

**詳細指南**: [`NAVIGATION.md#文檔維護指南`](../NAVIGATION.md#六文檔維護指南)

---

## 常見問題

### Q: 如何在本機除錯 Python 策略?

**A**: 所有開發必須在 Docker 容器內,使用 PM2 + logs:
```bash
docker exec -it godzilla-dev bash
cd /app/strategies/my_strategy
# 添加 context.log().info(...) 到策略
pm2 restart my_strategy
pm2 logs my_strategy
```

### Q: 如何測試 C++ 更改?

**A**: 在容器內重新編譯並重啟服務:
```bash
docker exec -it godzilla-dev bash -c "cd /app/core/build && make -j\$(nproc)"
docker exec godzilla-dev pm2 restart all
```

### Q: 如何清除 Journal 重新開始?

**A**: 刪除 journal 檔案 (僅開發環境):
```bash
docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"
```

---

## 延伸閱讀

### 入門
- [`NAVIGATION.md`](../NAVIGATION.md) - 任務導向導航
- [`operations/QUICK_START.md`](../operations/QUICK_START.md) - 快速指令參考

### 架構
- [`modules/yijinjing.md`](../modules/yijinjing.md) - 事件溯源
- [`modules/wingchun.md`](../modules/wingchun.md) - 交易引擎
- [`ARCHITECTURE.md`](ARCHITECTURE.md) - 系統架構概覽

### API
- [`contracts/strategy_context_api.md`](../contracts/strategy_context_api.md) - Context API 參考
- [`contracts/order_object_contract.md`](../contracts/order_object_contract.md) - Order 物件

---

**最後更新**: 2025-12-01  
**預估 Token**: ~1000
