# Demo Factor Implementation Summary

## Created Files
All files created in `/home/huyifan/projects/godzilla-evan/hf-live/factors/demo/`

### 1. factor_entry.h
- **Class**: `factors::demo::FactorEntry` inheriting from `factors::comm::FactorEntryBase`
- **Constructor**: Takes asset string, FactorMetadata, and FactorEntryConfig
- **Methods**:
  - `DoOnAddQuote(const hf::Depth &quote)` - Updates bid_ask_ratio from order book
  - `DoOnAddTrans(const hf::Trade &trans)` - Updates trade volume moving average
  - `DoOnUpdateFactors(int64_t timestamp)` - Frame update hook (minimal for demo)
- **Members**:
  - `last_ask_price_` (double) - Stores current ask price
  - `recent_volumes_` (vector<double>) - Circular buffer of last 10 trade volumes
  - `MA_WINDOW` (const) - Moving average window size = 10
- **Registration**: Uses REGISTER_FACTOR_AUTO(demo, FactorEntry) macro

### 2. factor_entry.cpp
- **Implementation Details**:
  - Constructor initializes fvals_[0] and fvals_[1] to 0.0
  - DoOnAddQuote calculates: bid_ask_ratio = bid_volume[0] / ask_volume[0]
    - Safely checks ask_volume[0] > 0 to avoid division by zero
    - Stores in fvals_[0] as float
  - DoOnAddTrans updates moving average:
    - Pushes new trade volume to recent_volumes_
    - Keeps only last 10 volumes (MA_WINDOW)
    - Calculates arithmetic mean
    - Stores in fvals_[1] as float
  - DoOnUpdateFactors is empty (factors already updated by callbacks)

### 3. meta_config.h
- **Constants**:
  - `kFactorSetName` = "demo"
  - `kFactorSize` = 2 (number of factors)
- **Factor Names** (in order):
  - Index 0: "bid_ask_ratio" - Buy/sell pressure indicator
  - Index 1: "trade_volume_ma" - Trade activity indicator
- **Metadata**: Static `kFactorMetadata` struct with set name, size, and names
- **Function**: `GetMetadata()` - Returns reference to static metadata

### 4. CMakeLists.txt
- **Target**: `factor_demo` static library
- **Sources**: All *.cpp files in demo directory (auto-discovered)
- **Include Paths**:
  - Current source directory
  - Parent parent directory (for hf-live root includes)
  - _comm directory (for common factor headers)
- **Properties**:
  - C++ Standard: 17
  - Position Independent Code: ON (for library integration)
- **Installation**: Archive to CMAKE_INSTALL_LIBDIR

## Design Patterns Used
1. **Inheritance**: FactorEntry extends FactorEntryBase
2. **Virtual Methods**: All three Do* methods override base class virtuals
3. **Static Metadata**: Metadata defined at compile time, used by registry
4. **Auto-Registration**: REGISTER_FACTOR_AUTO macro handles registration
5. **Factor Value Vector**: Uses fvals_ inherited vector for storing results
6. **Type Safety**: Proper float casting from calculation doubles

## Data Flow
1. Quote arrives → DoOnAddQuote() → bid_ask_ratio calculated → stored in fvals_[0]
2. Trade arrives → DoOnAddTrans() → volume added to buffer → MA calculated → stored in fvals_[1]
3. UpdateFactors() called → returns fvals_ vector to framework

## Validation Points
- ✓ All virtual methods implemented (DoOnAddQuote, DoOnAddTrans, DoOnUpdateFactors)
- ✓ Proper inheritance from FactorEntryBase
- ✓ Metadata struct properly defined with 2 factors
- ✓ Factor names match the two implemented calculations
- ✓ Registration macro properly placed in header
- ✓ Safe division check in bid_ask_ratio calculation
- ✓ Proper handling of circular buffer for moving average
- ✓ Type conversions (double to float) for factor storage
- ✓ C++17 compatibility (std::vector<double>)
