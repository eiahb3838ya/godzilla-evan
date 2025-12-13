# Phase 4H: SPMC Buffer Data Race Fix Specification

**Version**: 1.0
**Date**: 2024-12-13
**Status**: PENDING APPROVAL
**Target Audience**: Junior Engineer

---

## 0. Pre-requisites Checklist

Before starting, confirm:

- [ ] You have access to the godzilla-evan repository
- [ ] Docker container `godzilla-dev` is running
- [ ] You understand basic C++ atomic operations
- [ ] You have read the root cause analysis in `debug_hf-live.02-spmc-buffer-data-race.md`

**CRITICAL WARNING**:
```
!! DO NOT MODIFY ANY FILES IN ref/hf-stock-live-demo-main/ !!
!! The ref project is READ-ONLY reference code !!
```

---

## 1. Overview

### 1.1 What We're Fixing

**File**: `hf-live/app_live/data/spmc_buffer.hpp`

**Problem**: The variable `write_num_` uses `volatile` instead of `std::atomic`, causing a data race between producer and consumer threads.

**Symptom**: Crashes with "pure virtual method called", "bus error", "corrupted size vs. prev_size" when hf-live is loaded via dlopen into Godzilla.

### 1.2 Scope of Changes

| File | Action |
|------|--------|
| `hf-live/app_live/data/spmc_buffer.hpp` | MODIFY |
| `ref/*` | DO NOT TOUCH |
| All other files | DO NOT TOUCH |

---

## 2. Step-by-Step Modification Guide

### 2.1 Backup Original File

```bash
# Step 1: Create backup
cd /home/huyifan/projects/godzilla-evan
cp hf-live/app_live/data/spmc_buffer.hpp hf-live/app_live/data/spmc_buffer.hpp.backup
```

### 2.2 Modification #1: Change Variable Declaration

**Location**: Line 187

**Find this code**:
```cpp
alignas(SPMC_BUFFER_CACHE_LINE_SIZE) volatile size_t write_num_{0}; ///< 已写入的总数据量
```

**Replace with**:
```cpp
alignas(SPMC_BUFFER_CACHE_LINE_SIZE) std::atomic<size_t> write_num_{0}; ///< 已写入的总数据量 (atomic for thread safety)
```

### 2.3 Modification #2: Update push() Function (const ref version)

**Location**: Lines 104-120

**Find this code**:
```cpp
void push(const T& item) {
    // 写入数据
    blocks_[write_block_id_][write_pos_] = item;
    // 更新**写数量**
    write_num_++;
    // 更新下一次写入位置
    if ((write_pos_ + 1) == size_per_block_) { // 进入下一个内存块
        write_block_id_++;
        write_pos_ = 0;
        if (write_block_id_ == blocks_.size()) {
            blocks_.emplace_back();
            blocks_.back().resize(size_per_block_);
        }
    } else { // 还在当前内存块
        write_pos_++;
    }
}
```

**Replace with**:
```cpp
void push(const T& item) {
    // 写入数据
    blocks_[write_block_id_][write_pos_] = item;
    // 更新下一次写入位置 (BEFORE incrementing write_num_)
    if ((write_pos_ + 1) == size_per_block_) { // 进入下一个内存块
        write_block_id_++;
        write_pos_ = 0;
        if (write_block_id_ == blocks_.size()) {
            blocks_.emplace_back();
            blocks_.back().resize(size_per_block_);
        }
    } else { // 还在当前内存块
        write_pos_++;
    }
    // 更新**写数量** - use release semantics to ensure data is visible
    write_num_.fetch_add(1, std::memory_order_release);
}
```

### 2.4 Modification #3: Update push() Function (rvalue ref version)

**Location**: Lines 127-143

**Find this code**:
```cpp
void push(T&& item) {
    // 写入数据
    blocks_[write_block_id_][write_pos_] = std::move(item);
    // 更新**写数量**
    write_num_++;
    // 更新下一次写入位置
    if ((write_pos_ + 1) == size_per_block_) { // 进入下一个内存块
        write_block_id_++;
        write_pos_ = 0;
        if (write_block_id_ == blocks_.size()) {
            blocks_.emplace_back();
            blocks_.back().resize(size_per_block_);
        }
    } else { // 还在当前内存块
        write_pos_++;
    }
}
```

**Replace with**:
```cpp
void push(T&& item) {
    // 写入数据
    blocks_[write_block_id_][write_pos_] = std::move(item);
    // 更新下一次写入位置 (BEFORE incrementing write_num_)
    if ((write_pos_ + 1) == size_per_block_) { // 进入下一个内存块
        write_block_id_++;
        write_pos_ = 0;
        if (write_block_id_ == blocks_.size()) {
            blocks_.emplace_back();
            blocks_.back().resize(size_per_block_);
        }
    } else { // 还在当前内存块
        write_pos_++;
    }
    // 更新**写数量** - use release semantics to ensure data is visible
    write_num_.fetch_add(1, std::memory_order_release);
}
```

### 2.5 Modification #4: Update try_read() Function

**Location**: Lines 152-168

**Find this code**:
```cpp
bool try_read(SPMCBufferConsumerToken& consumer_token, T& out) {
    // 如果消费者令牌中的编号不在预设范围则退出
    if (consumer_token.consumer_id >= max_consumers_) return false;
    // 如果已经读满，还没有新数据，则退出
    if (consumer_token.read_num == write_num_) return false;
    // 读取新数据
    out = blocks_[consumer_token.read_block_id][consumer_token.read_pos];
    // ... rest of function
```

**Replace with**:
```cpp
bool try_read(SPMCBufferConsumerToken& consumer_token, T& out) {
    // 如果消费者令牌中的编号不在预设范围则退出
    if (consumer_token.consumer_id >= max_consumers_) return false;
    // 如果已经读满，还没有新数据，则退出
    // Use acquire semantics to ensure we see the data written by producer
    if (consumer_token.read_num == write_num_.load(std::memory_order_acquire)) return false;
    // 读取新数据
    out = blocks_[consumer_token.read_block_id][consumer_token.read_pos];
    // ... rest of function (unchanged)
```

### 2.6 Modification #5: Update is_consumer_finished() Function

**Location**: Lines 177-179

**Find this code**:
```cpp
bool is_consumer_finished(SPMCBufferConsumerToken& consumer_token) const noexcept {
    return consumer_token.read_num == write_num_;
}
```

**Replace with**:
```cpp
bool is_consumer_finished(SPMCBufferConsumerToken& consumer_token) const noexcept {
    return consumer_token.read_num == write_num_.load(std::memory_order_acquire);
}
```

---

## 3. Verification Steps

### 3.1 Syntax Check

After modifications, verify the file compiles:

```bash
cd /home/huyifan/projects/godzilla-evan/hf-live

# Quick syntax check
g++ -std=c++17 -fsyntax-only -I. app_live/data/spmc_buffer.hpp
```

**Expected Result**: No output (no errors)

### 3.2 Full Build

```bash
cd /home/huyifan/projects/godzilla-evan/hf-live
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Expected Result**: Build completes with no errors

### 3.3 Copy to Container

```bash
docker cp /home/huyifan/projects/godzilla-evan/hf-live/build/libsignal.so godzilla-dev:/app/hf-live/build/libsignal.so
```

---

## 4. Testing Protocol

### 4.1 Pre-Test Cleanup

```bash
docker exec godzilla-dev pm2 delete all 2>/dev/null || true
docker exec godzilla-dev rm -rf /shared/kungfu/runtime/*
```

### 4.2 Start Services

```bash
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"
```

### 4.3 Monitor for 5 Minutes

```bash
# In terminal 1: Watch PM2 status
watch -n 5 'docker exec godzilla-dev pm2 list'

# In terminal 2: Watch logs for errors
docker exec godzilla-dev pm2 logs strategy:hello --lines 100
```

### 4.4 Success Criteria

| Metric | Pass Condition |
|--------|----------------|
| PM2 Restart Count | 0 (no restarts) |
| Runtime | > 5 minutes without crash |
| Error Logs | No "pure virtual method called" |
| Error Logs | No "bus error" |
| Error Logs | No "corrupted size vs. prev_size" |
| Error Logs | No "segmentation violation" |

### 4.5 Failure Actions

If tests fail:

1. **Restore backup**:
   ```bash
   cp hf-live/app_live/data/spmc_buffer.hpp.backup hf-live/app_live/data/spmc_buffer.hpp
   ```

2. **Collect crash logs**:
   ```bash
   docker exec godzilla-dev pm2 logs strategy:hello --lines 500 > crash_logs_phase4h.txt
   ```

3. **Report to senior engineer** with crash logs

---

## 5. Expected Diff

After all modifications, run this to verify your changes:

```bash
diff -u hf-live/app_live/data/spmc_buffer.hpp.backup hf-live/app_live/data/spmc_buffer.hpp
```

**Expected output** (approximately):

```diff
--- hf-live/app_live/data/spmc_buffer.hpp.backup
+++ hf-live/app_live/data/spmc_buffer.hpp
@@ -104,11 +104,11 @@
 	void push(const T& item) {
 		// 写入数据
 		blocks_[write_block_id_][write_pos_] = item;
-		// 更新**写数量**
-		write_num_++;
 		// 更新下一次写入位置
 		if ((write_pos_ + 1) == size_per_block_) { // 进入下一个内存块
 			write_block_id_++;
@@ -119,6 +119,8 @@
 		} else { // 还在当前内存块
 			write_pos_++;
 		}
+		// 更新**写数量** - use release semantics to ensure data is visible
+		write_num_.fetch_add(1, std::memory_order_release);
 	}

 	/**
@@ -127,11 +129,11 @@
 	void push(T&& item) {
 		// 写入数据
 		blocks_[write_block_id_][write_pos_] = std::move(item);
-		// 更新**写数量**
-		write_num_++;
 		// 更新下一次写入位置
 		if ((write_pos_ + 1) == size_per_block_) { // 进入下一个内存块
 			write_block_id_++;
@@ -142,6 +144,8 @@
 		} else { // 还在当前内存块
 			write_pos_++;
 		}
+		// 更新**写数量** - use release semantics to ensure data is visible
+		write_num_.fetch_add(1, std::memory_order_release);
 	}

 	/**
@@ -153,7 +157,8 @@
 		// 如果消费者令牌中的编号不在预设范围则退出
 		if (consumer_token.consumer_id >= max_consumers_) return false;
 		// 如果已经读满，还没有新数据，则退出
-		if (consumer_token.read_num == write_num_) return false;
+		// Use acquire semantics to ensure we see the data written by producer
+		if (consumer_token.read_num == write_num_.load(std::memory_order_acquire)) return false;
 		// 读取新数据
 		out = blocks_[consumer_token.read_block_id][consumer_token.read_pos];
 		// 更新**读数量**
@@ -174,7 +179,7 @@
 	 * @return 是否完成所有数据的读取
 	 */
 	bool is_consumer_finished(SPMCBufferConsumerToken& consumer_token) const noexcept {
-		return consumer_token.read_num == write_num_;
+		return consumer_token.read_num == write_num_.load(std::memory_order_acquire);
 	}

 private:
@@ -184,7 +189,7 @@
 	alignas(SPMC_BUFFER_CACHE_LINE_SIZE) const size_t size_per_block_; ///< 每个内存块的大小
 	alignas(SPMC_BUFFER_CACHE_LINE_SIZE) std::atomic<size_t> active_consumer_num_{0}; ///< 当前注册的消费者数量
 	alignas(SPMC_BUFFER_CACHE_LINE_SIZE) std::atomic<size_t> next_consumer_id_{0}; ///< 下一个消费者的唯一标识符
-	alignas(SPMC_BUFFER_CACHE_LINE_SIZE) volatile size_t write_num_{0}; ///< 已写入的总数据量
+	alignas(SPMC_BUFFER_CACHE_LINE_SIZE) std::atomic<size_t> write_num_{0}; ///< 已写入的总数据量 (atomic for thread safety)
 	alignas(SPMC_BUFFER_CACHE_LINE_SIZE) size_t write_block_id_{0}; ///< 当前写的内存块位置
 	alignas(SPMC_BUFFER_CACHE_LINE_SIZE) size_t write_pos_{0}; ///< 当前内存块内的写入位置
```

---

## 6. Rollback Procedure

If issues are found after deployment:

```bash
# 1. Stop all services
docker exec godzilla-dev pm2 delete all

# 2. Restore original file
cd /home/huyifan/projects/godzilla-evan
cp hf-live/app_live/data/spmc_buffer.hpp.backup hf-live/app_live/data/spmc_buffer.hpp

# 3. Rebuild
cd hf-live/build
make -j$(nproc)

# 4. Redeploy
docker cp build/libsignal.so godzilla-dev:/app/hf-live/build/libsignal.so
```

---

## 7. Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Spec Author | Claude | 2024-12-13 | ✓ |
| Reviewer | | | |
| Executor | | | |
| Tester | | | |

---

## Appendix A: Why This Fix Works

### A.1 Memory Ordering Explanation

```
Producer                          Consumer
────────                          ────────
1. Write data to blocks_[]
2. write_num_.fetch_add(1, release)
   └── "release" guarantees:
       All writes BEFORE this
       are visible to threads
       that "acquire" this value
                                  3. write_num_.load(acquire)
                                     └── "acquire" guarantees:
                                         All writes that happened
                                         BEFORE the "release"
                                         are now visible
                                  4. Read data from blocks_[]
                                     └── Data is guaranteed to be there!
```

### A.2 Why volatile Failed

`volatile` only prevents compiler from optimizing away reads/writes. It does NOT:
- Prevent CPU instruction reordering
- Guarantee atomicity
- Provide memory barriers

In dlopen environment with high-frequency data, CPU reordering exposed this latent bug.
