# Godzilla 公開市場數據 vs ref 數據結構對照

**目的**: 羅列 Godzilla 所有公開交易數據,對照 ref 項目的數據結構

---

## 一、Godzilla 公開市場數據類型 (msg::type)

| 類型編號 | 結構名稱 | 文件位置 |
|---------|---------|---------|
| 101 | `Ticker` | [msg.h:176-210](../core/cpp/wingchun/include/kungfu/wingchun/msg.h#L176) |
| 102 | `Depth` | [msg.h:242-302](../core/cpp/wingchun/include/kungfu/wingchun/msg.h#L242) |
| 103 | `Trade` | [msg.h:331-369](../core/cpp/wingchun/include/kungfu/wingchun/msg.h#L331) |
| 104 | `IndexPrice` | [msg.h:405-428](../core/cpp/wingchun/include/kungfu/wingchun/msg.h#L405) |
| 110 | `Bar` | [msg.h:446-474](../core/cpp/wingchun/include/kungfu/wingchun/msg.h#L446) |

---

## 二、數據結構字段對照

### 2.1 Ticker (102)

**Godzilla 結構**:
```cpp
struct Ticker {
    char source_id[SOURCE_ID_LEN];
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    int64_t data_time;
    InstrumentType instrument_type;
    double bid_price;                  // 买单最优挂单价格
    double bid_volume;                 // 买单最优挂单数量
    double ask_price;                  // 卖单最优挂单价格
    double ask_volume;                 // 卖单最优挂单数量
};
```

**ref 對應**: ❌ **無直接對應結構**

**說明**: Ticker 是 Depth 的子集(僅最優一檔),ref 項目無獨立 Ticker 結構

---

### 2.2 Depth (101)

**Godzilla 結構**:
```cpp
struct Depth {
    char source_id[SOURCE_ID_LEN];
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    int64_t data_time;
    InstrumentType instrument_type;
    double bid_price[10];              // 申买价
    double ask_price[10];              // 申卖价
    double bid_volume[10];             // 申买量
    double ask_volume[10];             // 申卖量
};
```

**ref 對應**: `Stock_Internal_Book` ([my_stock.h:52-83](../ref/hf-stock-live-demo-main/sdp_handler/quote_struct/my_stock.h#L52))

**字段對照**:

| Godzilla Depth | 類型 | ref Stock_Internal_Book | 類型 | 差異 |
|---------------|------|------------------------|------|------|
| `symbol` | char[32] | `ticker` / `wind_code` | char[32] | ✅ 相同 |
| `exchange_id` | char[16] | _(無此字段)_ | - | ❌ ref 無 |
| `data_time` | int64 | `exch_time` | int | ❌ 類型不同: ns vs HHMMSSmmm |
| `bid_price[10]` | double | `bp_array[10]` | uint32 | ❌ 類型不同: double vs uint32×10000 |
| `ask_price[10]` | double | `ap_array[10]` | uint32 | ❌ 類型不同: double vs uint32×10000 |
| `bid_volume[10]` | double | `bv_array[10]` | uint32 | ❌ 類型不同 |
| `ask_volume[10]` | double | `av_array[10]` | uint32 | ❌ 類型不同 |
| _(無此字段)_ | - | `last_px` | uint32 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `open_px` | uint32 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `high_px` | uint32 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `low_px` | uint32 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `total_vol` | int64 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `total_notional` | int64 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `num_of_trades` | uint32 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `pre_close_px` | uint32 | ❌ Godzilla 無 |

**ref 入口**: `FactorEngine::OnTick(Stock_Internal_Book*)`

---

### 2.3 Trade (103)

**Godzilla 結構**:
```cpp
struct Trade {
    char client_id[CLIENT_ID_LEN];
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    InstrumentType instrument_type;
    int64_t trade_id;                  // 交易ID
    int64_t ask_id;                    // 卖方订单ID (公開市場)
    int64_t bid_id;                    // 买方订单ID (公開市場)
    double price;                      // 成交价
    double volume;                     // 成交量
    Side side;                         // 主动成交方向
    Direction position_side;
    int64_t trade_time;                // 成交时间
};
```

**ref 對應**: `Stock_Transaction_Internal_Book_New` ([my_stock_transaction_new.h:9-23](../ref/hf-stock-live-demo-main/sdp_handler/quote_struct/my_stock_transaction_new.h#L9))

**字段對照**:

| Godzilla Trade | 類型 | ref Stock_Transaction_New | 類型 | 差異 |
|---------------|------|---------------------------|------|------|
| `symbol` | char[32] | `symbol` | char[9] | ❌ 長度不同 |
| `trade_id` | int64 | `trade_index` | int64 | ✅ 相同 |
| `ask_id` | int64 | `sell_id` | int64 | ✅ 相同 |
| `bid_id` | int64 | `buy_id` | int64 | ✅ 相同 |
| `price` | double | `trade_price` | int | ❌ 類型不同: double vs int×10000 |
| `volume` | double | `trade_volume` | int64 | ❌ 類型不同 |
| `trade_time` | int64 | `int_time` | int | ❌ 類型不同: ns vs HHMMSSmmm |
| `side` | enum Side | `bsflag` | char | ❌ 類型不同: enum vs 'B'/'S' |
| _(無此字段)_ | - | `channel` | uint16 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `market` | uint8 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `trade_type` | char | ❌ Godzilla 無 |
| _(無此字段)_ | - | `trade_amount` | int64 | ❌ Godzilla 無 |
| _(無此字段)_ | - | `biz_index` | int64 | ❌ Godzilla 無 |

**ref 入口**: `FactorEngine::OnTrans(Stock_Transaction_Internal_Book_New*)`

---

### 2.4 IndexPrice (104)

**Godzilla 結構**:
```cpp
struct IndexPrice {
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    InstrumentType instrument_type;
    double price;                      // 指数价格
};
```

**ref 對應**: ❌ **無對應結構**

**說明**: ref 項目無指數價格處理

---

### 2.5 Bar (110)

**Godzilla 結構**:
```cpp
struct Bar {
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    int64_t start_time;                // 开始时间
    int64_t end_time;                  // 结束时间
    double open;                       // 开
    double close;                      // 收
    double low;                        // 低
    double high;                       // 高
    double volume;                     // 区间交易量
    double start_volume;               // 初始总交易量
    int32_t trade_count;               // 区间交易笔数
    int interval;                      // 周期(秒数)
};
```

**ref 對應**: ❌ **無對應結構**

**說明**: ref 項目無 Bar 數據處理,`FactorEngine` 僅有 `OnTick/OnTrans/OnOrder` 三個入口

---

## 三、ref 項目數據入口

ref 項目 `FactorEngine` 提供三個數據入口 ([factor_calculation_engine.h:53-59](../ref/hf-stock-live-demo-main/app_live/engine/factor_calculation_engine.h#L53)):

```cpp
void OnTick(Stock_Internal_Book* quote, start_time_t t);
void OnTrans(Stock_Transaction_Internal_Book_New* quote, start_time_t t);
void OnOrder(Stock_Order_Internal_Book_New* quote, start_time_t t);
```

### ref OnOrder 說明

`Stock_Order_Internal_Book_New` ([my_stock_order_new.h:9-21](../ref/hf-stock-live-demo-main/sdp_handler/quote_struct/my_stock_order_new.h#L9)):

```cpp
struct Stock_Order_Internal_Book_New {
    char order_type;
    uint8_t market;
    char bsflag;
    char symbol[9];
    int int_time;             // 委托时间
    int64_t order_index;
    int64_t order_volume;     // 委托数量
    uint64_t orderorino;
    uint64_t biz_index;
    int order_price;          // 委托价格
    uint16_t channel;
};
```

**Godzilla 對應**: ❌ **無對應結構**

**說明**:
- ref 的 `OnOrder` 接收**公開市場上所有人的委託單數據**
- Godzilla `Order(203)` 是**用戶自己的私有訂單**,不是公開市場數據
- Godzilla 沒有提供公開市場委託簿數據

---

## 四、總結

### Godzilla → ref 覆蓋情況

| Godzilla | msg::type | ref 對應 | 字段匹配度 |
|---------|-----------|---------|----------|
| `Ticker` | 102 | _(無)_ | N/A |
| `Depth` | 101 | `Stock_Internal_Book` | ~40% (僅 10檔報價) |
| `Trade` | 103 | `Stock_Transaction_New` | ~70% (缺元數據字段) |
| `IndexPrice` | 104 | _(無)_ | N/A |
| `Bar` | 110 | _(無)_ | N/A |
| _(無)_ | - | `Stock_Order_New` | ❌ Godzilla 無此數據 |

### 主要差異類型

1. **字段類型不同**:
   - 價格: Godzilla `double` vs ref `uint32×10000`
   - 時間: Godzilla `int64 ns` vs ref `int HHMMSSmmm`
   - 方向: Godzilla `enum Side` vs ref `char 'B'/'S'`

2. **Godzilla 缺失字段**:
   - Depth 缺: `last_px`, `open/high/low_px`, `total_vol`, `num_of_trades`, `pre_close_px`
   - Trade 缺: `channel`, `market`, `trade_type`, `trade_amount`, `biz_index`

3. **ref 缺失數據類型**:
   - 無 Ticker 結構
   - 無 IndexPrice 結構
   - 無 Bar 結構
   - 無公開市場委託簿數據 (Godzilla 也無)

---

**版本**: v1.0
**日期**: 2025-12-02
