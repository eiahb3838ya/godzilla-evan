# 數據結構共享方案

## 文檔元信息
- **版本**: v4.0-final
- **日期**: 2025-12-04
- **結論**: 直接複製 header 到 hf-live 倉庫 (極簡方案)

---

## 問題陳述

hf-live (private submodule) 如何獲知 godzilla-evan 數據結構以解析 `void*`?

**核心需求**:
- ✅ 編譯時確定結構大小 (性能)
- ✅ 零拷貝 (void* 直接轉型)
- ✅ 場景 A (一人大師 - godzilla 內) 零配置
- ✅ 場景 B (獨立編譯 - 因子/模型大師) 零配置
- ✅ 極低維護成本

---

## 方案演進史

### ❌ 方案 1: Symbolic Link (v1.0-v3.0)

```bash
cd hf-live/include
ln -s ../../core/cpp/wingchun/.../market_data_types.h market_data.h
```

**致命問題**:
```
場景 A (godzilla-evan 內):
  make  # ✅ symlink 有效

場景 B (獨立編譯):
  git clone <hf-live-repo>
  make  # ❌ symlink 斷裂!找不到 ../../core/...
```

**根本矛盾**: Symlink 是**路徑依賴**,不是真正獨立。

### ❌ 方案 2: Header Package + 自動化腳本

```bash
./scripts/setup_dependencies.sh
curl https://artifacts.../godzilla_headers.tar.gz
```

**問題**: 過度工程 + 增加認知負擔 + 網絡依賴

---

## ✅ 最終方案: 直接複製 (Bundled Header)

### 核心思想

**market_data_types.h 是准靜態依賴**:
- 交易所 API 結構變動頻率: **< 1次/年**
- 變動時 = Godzilla 重大升級 = 計劃性維護窗口
- 因此: **直接包含在 hf-live 倉庫,手動同步**

類比:
```
npm 包不用 symlink 到源碼,而是複製 node_modules/ ✅
Docker 鏡像不用 mount 宿主文件,而是 COPY 到鏡像 ✅
hf-live 不用 symlink,而是直接包含 header ✅
```

---

## 實施方案

### Phase 1: 初始化 hf-live 倉庫 (一次性操作)

```bash
# 在 godzilla-evan
cd core/cpp/wingchun/include/kungfu/wingchun
cp msg.h /tmp/market_data_types.h

# 在 hf-live 倉庫
cd hf-live
mkdir -p include
cp /tmp/market_data_types.h include/

# 添加版本標記
cat > include/market_data_types.VERSION <<EOF
Version: v1.0.0
Based on: Godzilla core/cpp/wingchun/include/kungfu/wingchun/msg.h
Godzilla Version: v2.0.0
Date: 2025-12-04
Update Frequency: < 1 time per year (only when exchange API changes)
EOF

# 提交到 hf-live 倉庫
git add include/market_data_types.h include/market_data_types.VERSION
git commit -m "feat: add market_data_types.h v1.0.0 (from Godzilla v2.0.0)"
git tag v1.0.0
```

### Phase 2: 使用 (因子大師代碼)

```cpp
// hf-live/factors/my_factors/factor_entry.cpp
#include "market_data_types.h"  // 直接 include,無需任何配置

class MyFactorEntry {
public:
    void OnDepth(const Depth* depth) {
        // 直接使用 Godzilla 數據結構
        double bid = depth->bid_price[0];
        double ask = depth->ask_price[0];
        factors_[0] = (bid - ask) / ask;
    }
};
```

### Phase 3: CMakeLists.txt (極簡)

**完整 CMake 配置**: 見 [prd_hf-live.04-project-config.md §4.4](prd_hf-live.04-project-config.md)

**核心概念**:
```cmake
# hf-live/CMakeLists.txt
target_include_directories(signal_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # header 已在此,零配置 ✅
)
```

---

## 場景驗證

### 場景 A: 一人大師 (godzilla-evan 內)

```bash
cd /home/user/godzilla-evan/hf-live
make
# ✅ 直接成功,無任何配置
```

**結果**: hf-live/include/market_data_types.h 直接可用

### 場景 B: 獨立編譯 (因子大師/模型大師)

```bash
# 任意位置
cd /tmp
git clone <hf-live-private-repo>
cd hf-live
make
# ✅ 直接成功,無任何配置
```

**結果**: header 已包含在倉庫中,零配置

---

## 更新流程 (罕見事件)

### 情況: Binance API 新增字段 (例: funding_rate)

**步驟 1: Godzilla 更新** (在 main repo)

```cpp
// godzilla-evan/core/.../msg.h
struct Depth {
    // ... 原有字段 ...
    double funding_rate;  // 🔥 新增
};
```

```bash
# Godzilla 發布新版本
cd godzilla-evan
git tag v3.0.0
git push --tags
```

**步驟 2: hf-live 同步** (計劃性維護)

```bash
# 在 godzilla-evan 倉庫
cd godzilla-evan

# 複製最新 header
cp core/cpp/wingchun/include/kungfu/wingchun/msg.h \
   hf-live/include/market_data_types.h

# 更新版本標記
cat > hf-live/include/market_data_types.VERSION <<EOF
Version: v2.0.0
Based on: Godzilla msg.h v3.0.0
Date: 2026-06-15
Changes: Added funding_rate field to Depth struct
Compatibility: Requires Godzilla v3.0.0+
EOF

# 提交到 hf-live submodule
cd hf-live
git add include/market_data_types.h include/market_data_types.VERSION
git commit -m "feat: sync market_data_types to Godzilla v3.0.0 - add funding_rate"
git tag v2.0.0
git push origin v2.0.0

# 更新 godzilla-evan 的 submodule 引用
cd ..
git add hf-live
git commit -m "chore: update hf-live submodule to v2.0.0"
```

**步驟 3: 通知所有用戶**

```markdown
# Release Notes: hf-live v2.0.0

## Breaking Changes
- Requires Godzilla v3.0.0+
- Depth struct updated with new `funding_rate` field

## Migration
```bash
git pull
git checkout v2.0.0
make clean && make
```

## Optional: Use new field
```cpp
factors_[10] = depth->funding_rate;
```
```

**頻率**: 每 6-12 個月一次 (與交易所 API 變動同步)

---

## 維護成本分析

| 事件 | 頻率 | 操作時間 | 年度成本 |
|------|------|---------|---------|
| 交易所 API 變動 | 0.5-1 次/年 | 5 分鐘 | **< 10 分鐘/年** |
| 日常開發 | 每天 | 0 分鐘 | 0 |
| 初始設置 | 一次 | 5 分鐘 | 0 |

**對比**:
- Symlink 方案: 場景 B 無法工作 ❌
- 自動化腳本: 維護腳本 + 網絡依賴 = 100+ 分鐘/年 ❌
- **直接複製: < 10 分鐘/年** ✅

---

## Single Source of Truth 驗證

**問**: 複製 header 是否違反 Single Source of Truth?

**答**: ❌ 不違反

**理由**:
1. **Godzilla msg.h 仍是唯一真相來源**
   - 所有修改只在 Godzilla 進行
   - hf-live header 是**只讀快照** (read-only snapshot)

2. **類比其他生態**:
   ```
   React 源碼 (GitHub)     → Single Source of Truth ✅
   node_modules/react/     → 版本化快照 ✅
   沒有人認為 node_modules 違反 SST

   Godzilla msg.h          → Single Source of Truth ✅
   hf-live/include/*.h     → 版本化快照 ✅
   ```

3. **版本化保證一致性**:
   ```markdown
   # hf-live README.md
   Dependencies: Godzilla market_data_types v1.0.0
   Compatible with: Godzilla v2.0.0 ~ v2.9.0
   ```

**關鍵**: 不會有人修改 hf-live 的 header,所有修改在 Godzilla → 同步到 hf-live

---

## 性能保證

```cpp
// 編譯時 (兩邊完全一致,因為是同一個文件)
sizeof(Depth) = 336 bytes  // godzilla-evan 編譯
sizeof(Depth) = 336 bytes  // hf-live 編譯 (相同版本)

// 運行時 (零開銷)
void* data = ...;  // godzilla 傳遞
const Depth* d = static_cast<const Depth*>(data);  // 僅指針轉型,0ns
double price = d->bid_price[0];  // 直接內存訪問
```

**結果**: ✅ 編譯時大小確定 + 零拷貝 + 內存佈局完全一致

---

## README 文檔 (hf-live/README.md)

```markdown
# hf-live - High-Frequency Factor & Model Framework

## Dependencies

### Godzilla Market Data Types
- **Version**: v1.0.0
- **Source**: Godzilla core/cpp/wingchun/include/kungfu/wingchun/msg.h
- **Compatibility**: Godzilla v2.0.0+
- **Update Frequency**: < 1 time per year (only when exchange API changes)

### Why Bundled?
This is a snapshot from Godzilla core, included directly in the repository because:
1. **Stability**: Exchange API structures rarely change (< 1 time/year)
2. **Simplicity**: Zero configuration for independent compilation
3. **Versioning**: Clear dependency tracking

### Sync Procedure (for maintainers only)

When Godzilla msg.h changes (rare event):
```bash
# In godzilla-evan repo
cp core/cpp/wingchun/include/kungfu/wingchun/msg.h \
   hf-live/include/market_data_types.h

# Update VERSION file
vim hf-live/include/market_data_types.VERSION

# Commit & tag
cd hf-live
git commit -am "feat: sync to Godzilla vX.X.X"
git tag vX.X.X
```

## Compilation

### Scenario A: Inside godzilla-evan
```bash
cd godzilla-evan/hf-live
make  # ✅ Works out of box
```

### Scenario B: Independent clone
```bash
git clone <hf-live-private-repo>
cd hf-live
make  # ✅ Works out of box
```

No setup scripts, no network dependencies, just works.
```

---

## 總結

### 為什麼選擇直接複製?

| 方案 | 場景 A 成本 | 場景 B 成本 | 維護成本 | 認知負擔 |
|------|-----------|-----------|---------|---------|
| Symlink | 低 | ❌ 無法工作 | 低 | 中 |
| Header Package | 低 | 中 (腳本) | 中 (腳本) | 高 |
| **直接複製** | **極低** | **極低** | **極低** | **極低** |

### 核心原則

> "Choose simplicity over automation when the event frequency doesn't justify the complexity."

**事實**:
- market_data_types.h 變動頻率: **< 1次/年**
- 手動同步時間: **5 分鐘**
- 為此設計自動化 = 過度工程

**結論**: 直接複製是唯一合理選擇 ✅

---

**版本**: v4.0-final (2025-12-04)
**決策**: 直接複製 header,手動同步 (< 1次/年)
