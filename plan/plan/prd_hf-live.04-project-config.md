# hf-live 項目配置與 Git 管理

## 文檔元信息
- **版本**: v2.0
- **日期**: 2025-12-04
- **目標**: 定義 hf-live 作為 Godzilla submodule 的完整配置方案 (極簡版)
- **前置**: [prd_hf-live.abstract.md](prd_hf-live.abstract.md)
- **更新**: 採用直接複製 header 方案,移除所有 `_external/` 和 symlink 複雜性

---

## 一、核心需求

### 1.1 項目關係

```
godzilla-evan (Public Repo)
  ├── main 分支 (不涉及 hf-live)
  └── feature/hf-live-support 分支 🔥
      ├── core/cpp/wingchun/ (新增 hf-live 集成代碼)
      └── hf-live/ (Submodule,不上傳源碼)

hf-live (Private Repo) 🔥
  └── 完全獨立項目 (可單獨 clone + 編譯)
```

### 1.2 設計目標

| 目標 | 方案 | 狀態 |
|------|------|------|
| ✅ hf-live 不上傳到 godzilla | `.gitignore` + Git Submodule 配置 | § 2 |
| ✅ 獨立更新 submodule | `git submodule update --remote` | § 3.3 |
| ✅ hf-live 獨立編譯 | 直接包含 `market_data_types.h` | § 4 |
| ✅ Godzilla 新分支支持 | `feature/hf-live-support` 分支 | § 5 |

---

## 二、Git Submodule 配置 (Godzilla 端)

### 2.1 創建 feature 分支

```bash
# 在 godzilla-evan 倉庫
cd /home/huyifan/projects/godzilla-evan

# 1. 從 main 創建新分支
git checkout -b feature/hf-live-support

# 2. 添加 hf-live 為 submodule (Private Repo)
git submodule add <private-repo-url>/hf-live.git hf-live

# 3. 初始化 submodule
git submodule update --init --recursive

# 4. 查看狀態
git status
# 新增文件:
#   .gitmodules
#   hf-live (commit hash)
```

**結果**: `.gitmodules` 文件內容

```ini
[submodule "hf-live"]
	path = hf-live
	url = <private-repo-url>/hf-live.git
	branch = main
```

### 2.2 配置 .gitignore (關鍵!)

**目標**: 不上傳 hf-live 源碼,僅跟蹤 submodule commit hash

```bash
# godzilla-evan/.gitignore
cat >> .gitignore << 'EOF'

# ===== hf-live Submodule 配置 =====
# 策略: 僅跟蹤 submodule commit,不上傳源碼與編譯產物

# 不上傳 hf-live 源碼與因子
hf-live/src/
hf-live/include/
hf-live/adapter/
hf-live/_comm/
hf-live/app_live/
hf-live/factors/
hf-live/models/
hf-live/*.cpp
hf-live/*.h

# 不上傳編譯中間產物
hf-live/build/*.o
hf-live/build/*.d
hf-live/build/CMakeFiles/

# 可選: 允許上傳編譯好的 .so (如果需要分發)
!hf-live/build/libsignal.so

# 不上傳 submodule 的 .git 目錄 (已由 Git 自動管理)
hf-live/.git

EOF
```

### 2.3 驗證配置

```bash
# 檢查哪些文件會被 Git 跟蹤
git status --ignored

# 應該看到:
#   modified:   .gitignore
#   new file:   .gitmodules
#   new file:   hf-live (commit xxx)  # 僅 commit hash

# 不應該看到:
#   hf-live/src/
#   hf-live/factors/
#   hf-live/*.cpp
```

### 2.4 提交配置

```bash
# 提交 submodule 配置 (不含源碼)
git add .gitmodules hf-live .gitignore
git commit -m "feat: add hf-live as private submodule

- Add hf-live submodule (private repo)
- Configure .gitignore to exclude source code
- Only track submodule commit hash

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 推送分支
git push -u origin feature/hf-live-support
```

---

## 三、Submodule 管理工作流

### 3.1 初次克隆 (新開發者)

```bash
# 克隆 godzilla-evan (含 hf-live 分支)
git clone <godzilla-url> godzilla-evan
cd godzilla-evan

# 切換到 hf-live 支持分支
git checkout feature/hf-live-support

# 初始化 submodule (需要 Private Repo 權限)
git submodule update --init --recursive

# 查看 submodule 狀態
git submodule status
# 輸出: <commit-hash> hf-live (main)
```

**注意**: 需要對 hf-live private repo 的訪問權限

### 3.2 獨立更新 hf-live (代碼管理人)

```bash
# 進入 submodule 目錄
cd hf-live

# 查看當前版本
git log -1 --oneline

# 拉取最新代碼
git pull origin main

# 或者切換到特定版本
git checkout v1.2.0

# 返回 godzilla 根目錄
cd ..

# 查看 submodule 變更
git status
# 輸出: modified: hf-live (new commits)

# 提交 submodule 版本更新
git add hf-live
git commit -m "chore: update hf-live to v1.2.0"
git push
```

**關鍵**: 更新 hf-live 不影響 godzilla 代碼,僅更新 commit hash

### 3.3 自動追蹤 hf-live 最新版本

```bash
# 配置 submodule 自動追蹤遠端分支
git config -f .gitmodules submodule.hf-live.branch main
git config -f .gitmodules submodule.hf-live.update rebase

# 一鍵更新到最新版本
git submodule update --remote hf-live

# 查看變更
git diff --submodule

# 提交
git add hf-live
git commit -m "chore: update hf-live to latest"
```

### 3.4 移除 submodule (如果需要)

```bash
# 1. 取消註冊
git submodule deinit -f hf-live

# 2. 刪除 .git/modules 目錄
rm -rf .git/modules/hf-live

# 3. 刪除工作目錄
git rm -f hf-live

# 4. 提交變更
git commit -m "chore: remove hf-live submodule"
```

---

## 四、hf-live 獨立編譯配置

### 4.1 核心思想

**market_data_types.h 直接包含在 hf-live 倉庫** (詳細理由見 [prd_hf-live.data-structure-sharing.md](prd_hf-live.data-structure-sharing.md))

**為什麼?**
- 交易所 API 結構變動頻率: **< 1次/年**
- 直接複製 = 極簡方案,零配置

### 4.2 hf-live 項目結構 (獨立倉庫)

```
hf-live/                              # Private Repo (完全獨立)
├── .gitignore
├── CMakeLists.txt                    # 獨立編譯配置
├── Makefile                          # 簡化編譯
│
├── include/
│   ├── market_data_types.h          # 🔥 直接包含 (從 Godzilla 複製)
│   └── market_data_types.VERSION    # 版本標記
│
├── adapter/
│   ├── api.h                        # C API 聲明
│   └── adapter.cpp                  # 數據分發
│
├── _comm/
│   ├── signal_sender.h
│   ├── signal_sender.cpp
│   └── engine_base.h
│
├── app_live/
│   ├── engine.h
│   ├── engine.cpp                   # 統一調度與發送
│   └── entry.cpp                    # .so 入口
│
├── factors/
│   ├── _comm/
│   │   └── factor_base.h
│   └── my_factors/                  # 🔥 因子大師編寫
│       ├── factor_entry.h
│       └── factor_entry.cpp
│
└── build/
    └── libsignal.so                 # 編譯產物
```

### 4.3 初始化 hf-live 倉庫 (一次性設置)

```bash
# 1. 創建 hf-live 倉庫
mkdir -p hf-live
cd hf-live
git init

# 2. 創建必要目錄結構
mkdir -p include adapter _comm app_live factors/_comm build

# 3. 🔥 直接複製 Godzilla 數據結構 (一次性)
cp /path/to/godzilla-evan/core/cpp/wingchun/include/kungfu/wingchun/msg.h \
   include/market_data_types.h

# 4. 添加版本標記
cat > include/market_data_types.VERSION <<EOF
Version: v1.0.0
Based on: Godzilla core/cpp/wingchun/include/kungfu/wingchun/msg.h
Godzilla Version: v2.0.0
Date: 2025-12-04
Update Frequency: < 1 time per year (only when exchange API changes)
EOF

# 5. 創建 .gitignore
cat > .gitignore << 'EOF'
# Build artifacts
build/*.o
build/*.d
build/CMakeFiles/
build/CMakeCache.txt

# Keep .so
!build/libsignal.so

# IDE
.vscode/
.idea/

# OS
.DS_Store
EOF

# 6. 首次提交
git add .
git commit -m "feat: initial commit with bundled market_data_types.h v1.0.0"
git remote add origin <private-repo-url>/hf-live.git
git push -u origin main
```

### 4.4 CMakeLists.txt (極簡)

```cmake
# hf-live/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(hf-live VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== 獨立編譯配置 =====
# market_data_types.h 已在 include/ 目錄,無需外部依賴

# 包含路徑
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include      # market_data_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/_comm
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/app_live
    ${CMAKE_CURRENT_SOURCE_DIR}/factors/_comm
)

# 編譯選項
add_compile_options(-Wall -Wextra -O3 -fPIC)

# ===== 源文件 =====
set(SOURCES
    adapter/adapter.cpp
    _comm/signal_sender.cpp
    app_live/engine.cpp
    app_live/entry.cpp
    factors/my_factors/factor_entry.cpp  # 因子大師編寫
)

# ===== 生成 .so =====
add_library(signal SHARED ${SOURCES})

# 輸出到 build/
set_target_properties(signal PROPERTIES
    OUTPUT_NAME "signal"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)

# ===== 安裝規則 (可選) =====
install(TARGETS signal
    LIBRARY DESTINATION lib
)
```

### 4.5 Makefile (對齊 ref 項目設計)

**完整設計**: 見 [prd_hf-live.07-implementation.md §4.2](prd_hf-live.07-implementation.md)

**核心特性**:
- ✅ 根目錄 Makefile 封裝 CMake 複雜度
- ✅ 簡單命令: `make`, `make clean`, `make clean-build`
- ✅ 自動檢測 CPU 核心數並行編譯
- ✅ 帶顏色輸出與錯誤處理
- ✅ 對齊 ref 項目用戶體驗

**示例**:
```bash
cd hf-live
make              # 構建 libsignal.so
make clean-build  # 清理並重新構建
```

### 4.6 驗證獨立編譯

```bash
# 在 hf-live 倉庫 (完全獨立,無需 Godzilla)
cd hf-live

# 查看幫助
make help

# 編譯 (默認目標)
make
# 🔵 開始構建: libsignal.so
# -- Found factor module: my_factors
# [100%] Linking CXX shared library libsignal.so
# ✅ 構建完成: libsignal.so

# 驗證
ls -lh build/libsignal.so
# 輸出: -rwxr-xr-x 1 user user 2.3M Dec 03 10:00 build/libsignal.so

# 檢查依賴 (應該無 Godzilla 路徑)
ldd build/libsignal.so
# 輸出應僅包含系統庫:
#   linux-vdso.so.1
#   libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6
#   libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
```

---

## 五、Godzilla 集成配置 (feature 分支)

### 5.1 分支策略

```
godzilla-evan
├── main                           # 不涉及 hf-live
├── feature/hf-live-support        # 🔥 支持 hf-live 的分支
│   ├── core/cpp/wingchun/src/strategy/runner.cpp  # 新增轉發邏輯
│   ├── core/python/kungfu/wingchun/strategy.py    # 新增 on_factor 支持
│   └── hf-live/                   # Submodule (僅 commit hash)
└── release/v2.x-hflive            # 未來穩定版本
```

### 5.2 需要修改的 Godzilla 文件

**文件 1**: `core/cpp/wingchun/src/strategy/runner.cpp`

```cpp
// 新增 #include
#include <dlfcn.h>  // dlopen, dlsym

// 新增成員變量
class Runner {
private:
    void* signal_handle_ = nullptr;  // hf-live .so 句柄
    // ...
};

// 新增初始化邏輯
void Runner::setup() {
    // ... 現有代碼 ...

    // 🔥 加載 hf-live .so (如果配置)
    std::string signal_lib_path = config_["signal_lib"];
    if (!signal_lib_path.empty()) {
        void* lib = dlopen(signal_lib_path.c_str(), RTLD_LAZY);
        if (!lib) {
            LOG_ERROR("Failed to load signal lib: {}", dlerror());
            return;
        }

        auto signal_create = (void*(*)(const char*))dlsym(lib, "signal_create");
        signal_handle_ = signal_create("{}");

        // 註冊回調 (由 Python 端處理)
    }

    // 🔥 轉發市場數據到 hf-live
    events_ | is(msg::type::Depth) |
    $([&](event_ptr event) {
        // 原有策略回調
        for (const auto &strategy : strategies_) {
            strategy.second->on_depth(context_, event->data<Depth>());
        }

        // 🔥 轉發給 signal (零拷貝)
        if (signal_handle_) {
            auto signal_on_data = (void(*)(void*, int, const void*))
                dlsym(lib, "signal_on_data");
            signal_on_data(signal_handle_, 101, event->data_address());
        }
    });
}
```

**文件 2**: `core/python/kungfu/wingchun/strategy.py`

```python
class Strategy:
    def __init__(self, ...):
        # ... 現有代碼 ...

        # 🔥 加載 hf-live .so (如果配置)
        self._signal_lib = None
        self._signal_handle = None
        if "signal_lib" in config:
            self._load_signal_lib(config["signal_lib"])

    def _load_signal_lib(self, lib_path):
        """框架內部: 加載 .so 並註冊回調"""
        import ctypes
        self._signal_lib = ctypes.CDLL(lib_path)

        # 創建
        create_fn = self._signal_lib.signal_create
        create_fn.argtypes = [ctypes.c_char_p]
        create_fn.restype = ctypes.c_void_p
        self._signal_handle = create_fn(b'{}')

        # 註冊回調
        @ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int64,
                          ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_void_p)
        def callback(symbol, ts, vals, cnt, ud):
            self.on_factor(
                self.context_,
                symbol.decode('utf-8'),
                ts,
                [vals[i] for i in range(cnt)]
            )

        register_fn = self._signal_lib.signal_register_callback
        register_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        register_fn(self._signal_handle, callback, None)

    def on_factor(self, context, symbol, timestamp, values):
        """用戶可覆寫的回調"""
        pass
```

### 5.3 提交 Godzilla 集成代碼

```bash
# 在 feature/hf-live-support 分支
git add core/cpp/wingchun/src/strategy/runner.cpp
git add core/python/kungfu/wingchun/strategy.py
git commit -m "feat: integrate hf-live signal framework

- Add dlopen support for hf-live .so loading
- Forward market data to signal_on_data()
- Add Strategy.on_factor() callback
- Zero-copy data forwarding

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin feature/hf-live-support
```

---

## 六、數據結構同步策略 (罕見事件)

### 6.1 核心原則

**market_data_types.h 是准靜態依賴**:
- 變動頻率: **< 1次/年**
- 變動時機: 交易所 API 變更 = Godzilla 重大升級
- 同步方式: **手動同步** (簡單、可控、可驗證)

### 6.2 同步工作流 (當交易所 API 變動時)

**步驟 1: Godzilla 更新** (在 main repo)

```bash
# 修改數據結構
cd godzilla-evan
vim core/cpp/wingchun/include/kungfu/wingchun/msg.h
# 例: 新增 Depth.funding_rate 字段

git commit -m "feat: add funding_rate to Depth structure"
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
Date: 2025-06-15
Changes: Added funding_rate field to Depth struct
Compatibility: Requires Godzilla v3.0.0+
EOF

# 提交到 hf-live submodule
cd hf-live
git add include/market_data_types.h include/market_data_types.VERSION
git commit -m "feat: sync to Godzilla v3.0.0 - add funding_rate"
git tag v2.0.0
git push origin v2.0.0

# 更新 godzilla-evan 的 submodule 引用
cd ..
git add hf-live
git commit -m "chore: update hf-live submodule to v2.0.0"
```

**步驟 3: 通知所有用戶**

```markdown
# hf-live v2.0.0 Release Notes

## Breaking Changes
- Requires Godzilla v3.0.0+
- Depth struct updated with `funding_rate` field

## Migration
bash
git pull
git checkout v2.0.0
make clean && make


## Optional: Use new field
cpp
factors_[10] = depth->funding_rate;

```

### 6.3 維護成本

| 事件 | 頻率 | 操作時間 | 年度成本 |
|------|------|---------|---------|
| 交易所 API 變動 | 0.5-1 次/年 | 5 分鐘 | **< 10 分鐘/年** |
| 日常開發 | 每天 | 0 分鐘 | 0 |

**對比其他方案**:
- Symlink: 場景 B 無法工作 ❌
- 自動化腳本: 維護成本 > 100 分鐘/年 ❌
- **手動同步: < 10 分鐘/年** ✅

---

## 七、完整工作流示例

### 7.1 場景: 新開發者加入

```bash
# 1. 克隆 Godzilla (需要 Private Repo 權限)
git clone <godzilla-url> godzilla-evan
cd godzilla-evan

# 2. 切換到 hf-live 分支
git checkout feature/hf-live-support

# 3. 初始化 submodule (需要 hf-live Private Repo 權限)
git submodule update --init --recursive

# 4. 編譯 hf-live
cd hf-live
make

# 5. 返回 godzilla 並編譯
cd ..
# ... 編譯 godzilla ...
```

### 7.2 場景: 因子大師獨立開發

```bash
# 1. 僅克隆 hf-live (無需 Godzilla) ✅
git clone <private-repo-url>/hf-live.git
cd hf-live

# 2. 驗證數據結構
cat include/market_data_types.h
cat include/market_data_types.VERSION
# Version: v1.0.0, Compatible with: Godzilla v2.0.0+

# 3. 開發因子
vim factors/my_factors/factor_entry.cpp
# #include "market_data_types.h"  // ✅ 直接可用
# void OnDepth(const Depth* depth) { ... }

# 4. 編譯 (零配置)
make
# ✅ 成功! 無需任何 setup

# 5. 測試
ls -lh build/libsignal.so
# -rwxr-xr-x 1 user user 2.3M

# 6. 提交
git add factors/my_factors/
git commit -m "feat: add momentum factor"
git push
```

### 7.3 場景: 代碼管理人更新 submodule

```bash
# 在 godzilla-evan 倉庫
cd godzilla-evan
git checkout feature/hf-live-support

# 更新 hf-live 到最新版本
git submodule update --remote hf-live

# 查看變更
cd hf-live
git log -3 --oneline
cd ..

# 提交 submodule 版本更新
git add hf-live
git commit -m "chore: update hf-live to v1.3.0

New features:
- Add momentum factors
- Optimize memory usage

Source: hf-live commit <hash>"
git push
```

---

## 八、.gitignore 完整配置

### 8.1 godzilla-evan/.gitignore

```gitignore
# ===== hf-live Submodule 配置 =====
# 策略: 僅跟蹤 submodule commit hash,不上傳源碼

# 不上傳 hf-live 源碼
hf-live/include/
hf-live/adapter/
hf-live/_comm/
hf-live/app_live/
hf-live/factors/
hf-live/models/
hf-live/*.cpp
hf-live/*.h
hf-live/CMakeLists.txt
hf-live/Makefile
hf-live/README.md

# 不上傳編譯產物 (除非需要分發)
hf-live/build/*.o
hf-live/build/*.d
hf-live/build/CMakeFiles/
hf-live/build/CMakeCache.txt

# 可選: 允許上傳編譯好的 .so
!hf-live/build/libsignal.so

# Submodule 的 .git 目錄由 Git 自動管理
hf-live/.git
```

### 8.2 hf-live/.gitignore

```gitignore
# Build artifacts
build/*.o
build/*.d
build/CMakeFiles/
build/CMakeCache.txt
build/Makefile
build/cmake_install.cmake

# Keep .so binary
!build/libsignal.so

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.bak
```

---

## 九、檢查清單

### 9.1 Godzilla 端 (feature 分支)

- [ ] 創建 `feature/hf-live-support` 分支
- [ ] 添加 hf-live 為 submodule
- [ ] 配置 `.gitignore` (不上傳源碼)
- [ ] 修改 `runner.cpp` (數據轉發)
- [ ] 修改 `strategy.py` (回調支持)
- [ ] 驗證 `git status` (僅看到 commit hash)
- [ ] 提交並推送分支

### 9.2 hf-live 端 (獨立倉庫)

- [ ] 創建項目結構
- [ ] 直接複製 `market_data_types.h` 到 `include/`
- [ ] 創建 `market_data_types.VERSION` 版本標記
- [ ] 編寫 `CMakeLists.txt` (獨立編譯)
- [ ] 編寫 `Makefile` (簡化編譯)
- [ ] 測試獨立編譯 (`make clean && make`)
- [ ] 驗證 `.so` 不依賴 Godzilla 路徑 (`ldd`)
- [ ] 配置 `.gitignore`
- [ ] 首次提交到 Private Repo

### 9.3 數據同步 (罕見事件)

- [ ] 文檔化手動同步流程 (< 5 分鐘)
- [ ] 測試同步流程 (當 Godzilla 更新數據結構時)
- [ ] 記錄同步時間點 (VERSION 文件)

---

## 十、總結

### 核心設計決策

| 需求 | 方案 | 優勢 |
|------|------|------|
| ✅ hf-live 不上傳源碼 | `.gitignore` + Submodule | 僅跟蹤 commit hash |
| ✅ 獨立更新 | `git submodule update --remote` | 解耦 hf-live 與 Godzilla 開發 |
| ✅ 獨立編譯 | 直接包含 `market_data_types.h` | **零配置,極簡** |
| ✅ 數據一致性 | 手動同步 (< 1次/年) | **極低維護成本** |
| ✅ 新分支隔離 | `feature/hf-live-support` | 不影響 main 分支 |

### 優勢

1. **完全獨立**: hf-live 可單獨 clone + 編譯 (零配置)
2. **私密性**: 源碼不上傳到 godzilla,僅 commit hash
3. **解耦開發**: 因子大師與策略大師獨立迭代
4. **極簡維護**: 數據同步 < 10 分鐘/年
5. **分支隔離**: main 分支不受影響,穩定性有保障

### 關鍵決策: 直接複製 vs Symlink

**為什麼選擇直接複製?**
- market_data_types.h 變動頻率: **< 1次/年**
- Symlink 方案: 場景 B (獨立編譯) 無法工作 ❌
- 直接複製: 場景 A/B 零配置,維護成本 < 10 分鐘/年 ✅

詳見: [prd_hf-live.data-structure-sharing.md](prd_hf-live.data-structure-sharing.md)

---

**版本**: v2.0
**日期**: 2025-12-04
