/**
 * This is source code under the Apache License 2.0.
 * Original Author: kx@godzilla.dev
 * Original date: March 3, 2025
 */

#ifndef GODZILLA_BINANCE_EXT_COMMON_H
#define GODZILLA_BINANCE_EXT_COMMON_H

#include <nlohmann/json.hpp>

namespace kungfu
{
    namespace wingchun
    {
        namespace binance
        {
            struct Configuration
            {
                std::string user_id;
                std::string access_key;
                std::string secret_key;
                
                // Market toggle flags (ADR-004)
                // Defaults to true for backward compatibility
                bool enable_spot = true;
                bool enable_futures = true;
                
                std::string spot_rest_host;  // 现货rest域名
                int spot_rest_port;          // 现货rest端口
                std::string spot_wss_host;   // 现货websocket域名
                int spot_wss_port;           // 现货websocket端口
                std::string ubase_rest_host; // u本位合约rest域名
                int ubase_rest_port;         // u本位合约rest端口
                std::string ubase_wss_host;  // u本位合约websocket域名
                int ubase_wss_port;          // u本位合约wesocket端口
                std::string cbase_rest_host; // 币本位合约rest域名
                int cbase_rest_port;         // 币本位合约rest端口
                std::string cbase_wss_host;  // 币本位合约websocket域名
                int cbase_wss_port;          // 币本位合约websocket端口
            };

            inline void from_json(const nlohmann::json &j, Configuration &c)
            {
                j.at("user_id").get_to(c.user_id);
                j.at("access_key").get_to(c.access_key);
                j.at("secret_key").get_to(c.secret_key);
                
                // Market toggle flags (ADR-004)
                // Use value() with default for backward compatibility
                c.enable_spot = j.value("enable_spot", true);
                c.enable_futures = j.value("enable_futures", true);
                
                // Binance Testnet 配置
                // Spot Testnet (testnet.binance.vision API key)
                c.spot_rest_host = "testnet.binance.vision";
                c.spot_rest_port = 443;
                c.spot_wss_host = "stream.testnet.binance.vision";
                c.spot_wss_port = 443;
                
                // Futures Testnet (需要单独的 Futures Testnet API key)
                // 参考: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
                c.ubase_rest_host = "testnet.binancefuture.com";
                c.ubase_rest_port = 443;
                c.ubase_wss_host = "stream.binancefuture.com";
                c.ubase_wss_port = 443;  // 修正：使用标准 HTTPS 端口
                c.cbase_rest_host = "testnet.binancefuture.com";
                c.cbase_rest_port = 443;
                c.cbase_wss_host = "dstream.binancefuture.com";
                c.cbase_wss_port = 443;  // 修正：使用标准 HTTPS 端口
            }
        }
    }
}
#endif //GODZILLA_BINANCE_EXT_COMMON_H
