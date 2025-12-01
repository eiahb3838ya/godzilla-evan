/**
 * Test suite for Binance market toggle configuration (ADR-004)
 * Verifies enable_spot and enable_futures flags work correctly
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "../include/common.h"

using namespace kungfu::wingchun::binance;
using json = nlohmann::json;

// Test 1: Backward compatibility - No flags specified
TEST(BinanceMarketToggle, DefaultBothEnabled) {
    json config_json = {
        {"user_id", "test_user"},
        {"access_key", "test_key"},
        {"secret_key", "test_secret"}
    };
    
    Configuration config = config_json.get<Configuration>();
    
    EXPECT_TRUE(config.enable_spot) << "Spot should be enabled by default";
    EXPECT_TRUE(config.enable_futures) << "Futures should be enabled by default";
    EXPECT_EQ(config.user_id, "test_user");
    EXPECT_EQ(config.access_key, "test_key");
}

// Test 2: Disable Spot only
TEST(BinanceMarketToggle, DisableSpotOnly) {
    json config_json = {
        {"user_id", "futures_user"},
        {"access_key", "futures_key"},
        {"secret_key", "futures_secret"},
        {"enable_spot", false}
    };
    
    Configuration config = config_json.get<Configuration>();
    
    EXPECT_FALSE(config.enable_spot) << "Spot should be disabled";
    EXPECT_TRUE(config.enable_futures) << "Futures should be enabled by default";
}

// Test 3: Disable Futures only
TEST(BinanceMarketToggle, DisableFuturesOnly) {
    json config_json = {
        {"user_id", "spot_user"},
        {"access_key", "spot_key"},
        {"secret_key", "spot_secret"},
        {"enable_futures", false}
    };
    
    Configuration config = config_json.get<Configuration>();
    
    EXPECT_TRUE(config.enable_spot) << "Spot should be enabled by default";
    EXPECT_FALSE(config.enable_futures) << "Futures should be disabled";
}

// Test 4: Explicitly enable both
TEST(BinanceMarketToggle, ExplicitlyEnableBoth) {
    json config_json = {
        {"user_id", "both_user"},
        {"access_key", "both_key"},
        {"secret_key", "both_secret"},
        {"enable_spot", true},
        {"enable_futures", true}
    };
    
    Configuration config = config_json.get<Configuration>();
    
    EXPECT_TRUE(config.enable_spot);
    EXPECT_TRUE(config.enable_futures);
}

// Test 5: Disable both markets
TEST(BinanceMarketToggle, DisableBoth) {
    json config_json = {
        {"user_id", "disabled_user"},
        {"access_key", "disabled_key"},
        {"secret_key", "disabled_secret"},
        {"enable_spot", false},
        {"enable_futures", false}
    };
    
    Configuration config = config_json.get<Configuration>();
    
    EXPECT_FALSE(config.enable_spot);
    EXPECT_FALSE(config.enable_futures);
}

// Test 6: URL defaults are still set correctly
TEST(BinanceMarketToggle, URLDefaultsPreserved) {
    json config_json = {
        {"user_id", "test"},
        {"access_key", "key"},
        {"secret_key", "secret"},
        {"enable_spot", false}
    };
    
    Configuration config = config_json.get<Configuration>();
    
    // Testnet defaults should still be set
    EXPECT_EQ(config.spot_rest_host, "testnet.binance.vision");
    EXPECT_EQ(config.spot_rest_port, 443);
    EXPECT_EQ(config.ubase_rest_host, "testnet.binancefuture.com");
    EXPECT_EQ(config.ubase_rest_port, 443);
}

// Test 7: JSON with extra fields doesn't break parsing
TEST(BinanceMarketToggle, ExtraFieldsIgnored) {
    json config_json = {
        {"user_id", "test"},
        {"access_key", "key"},
        {"secret_key", "secret"},
        {"enable_spot", false},
        {"unknown_field", "should_be_ignored"},
        {"another_unknown", 123}
    };
    
    // Should not throw
    EXPECT_NO_THROW({
        Configuration config = config_json.get<Configuration>();
        EXPECT_FALSE(config.enable_spot);
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


