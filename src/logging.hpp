#pragma once
// Avoid conflicts with common Windows or project logging macros
#ifdef DEBUG
#undef DEBUG
#endif
#ifdef TRACE
#undef TRACE
#endif
#ifdef INFO
#undef INFO
#endif
#ifdef WARN
#undef WARN
#endif
#ifdef ERROR
#undef ERROR
#endif

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace ow::nn {

enum class LogLevel { ERROR = 0, WARN = 1, INFO = 2, DEBUG = 3, TRACE = 4 };

struct LogConfig {
    LogLevel level{LogLevel::INFO};
    bool tag_filter_enabled{false};
    std::unordered_set<std::string> allowed_tags; // empty => allow all
};

inline LogLevel parse_level(const char* envv) {
    if (!envv || envv[0] == '\0') return LogLevel::INFO;
    std::string s(envv);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "0" || s == "silent" || s == "off") return LogLevel::ERROR; // only errors
    if (s == "error" || s == "1") return LogLevel::ERROR;
    if (s == "warn" || s == "warning" || s == "2") return LogLevel::WARN;
    if (s == "info" || s == "3") return LogLevel::INFO;
    if (s == "debug" || s == "4") return LogLevel::DEBUG;
    if (s == "trace" || s == "5") return LogLevel::TRACE;
    return LogLevel::INFO;
}

inline const LogConfig& get_log_config() {
    static LogConfig cfg;
    static bool inited = false;
    static std::mutex mtx;
    if (!inited) {
        std::lock_guard<std::mutex> g(mtx);
        if (!inited) {
            cfg.level = parse_level(std::getenv("OWNN_LOG_LEVEL"));
            const char* tags = std::getenv("OWNN_LOG_TAGS");
            if (tags && tags[0] != '\0') {
                cfg.tag_filter_enabled = true;
                std::string s(tags);
                std::stringstream ss(s);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
                    if (!tok.empty()) cfg.allowed_tags.insert(tok);
                }
            }
            inited = true;
        }
    }
    return cfg;
}

inline bool log_enabled(const std::string& tag, LogLevel lvl) {
    const auto& cfg = get_log_config();
    if ((int)lvl > (int)cfg.level) return false;
    if (!cfg.tag_filter_enabled) return true;
    return cfg.allowed_tags.count(tag) > 0 || cfg.allowed_tags.count("ALL") > 0;
}

inline void log(LogLevel lvl, const std::string& tag, const std::string& msg) {
    if (!log_enabled(tag, lvl)) return;
    std::cout << "[" << tag << "] " << msg << std::endl;
}

inline bool verbose_tag(const std::string& tag) { return log_enabled(tag, LogLevel::DEBUG); }

// One-time logging by key to avoid spam
inline void log_once(const std::string& tag, const std::string& key, const std::string& msg, LogLevel lvl = LogLevel::INFO) {
    static std::unordered_set<std::string> seen;
    std::string k = tag + ":" + key;
    if (seen.insert(k).second) {
        log(lvl, tag, msg);
    }
}

// Convenience gates matching existing tags
inline bool ow_verbose_mha() { return verbose_tag("MHA"); }
inline bool ow_verbose_moe() { return verbose_tag("MoE"); }
inline bool ow_verbose_rope() { return verbose_tag("RoPE"); }
inline bool ow_verbose_matmul() { return verbose_tag("MatMul"); }
inline bool ow_verbose_gen() { return verbose_tag("GEN"); }

// Formatting helpers
inline std::string format_params_count(size_t params) {
    double p = static_cast<double>(params);
    std::ostringstream oss;
    if (p >= 1e9) { oss.setf(std::ios::fixed); oss.precision(2); oss << (p/1e9) << " B"; }
    else if (p >= 1e6) { oss.setf(std::ios::fixed); oss.precision(2); oss << (p/1e6) << " M"; }
    else if (p >= 1e3) { oss.setf(std::ios::fixed); oss.precision(2); oss << (p/1e3) << " K"; }
    else { oss << params; }
    return oss.str();
}

inline size_t dtype_size_by_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    if (s == "F32" || s == "FLOAT32") return 4;
    if (s == "F16" || s == "FP16" || s == "BF16") return 2;
    if (s == "F64" || s == "FLOAT64") return 8;
    if (s == "I8" || s == "U8") return 1;
    if (s == "I16") return 2;
    if (s == "I32") return 4;
    if (s == "I64") return 8;
    if (s == "FP8_E4M3" || s == "FP8_E5M2") return 1;
    return 4;
}

} // namespace ow::nn