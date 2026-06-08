#pragma once

#include <cstdarg>
#include <cstdio>

namespace quadtrix {
namespace cuda {

enum class LogLevel {
    Info,
    Warn,
    Error,
};

inline const char* log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Info:
            return "info";
        case LogLevel::Warn:
            return "warn";
        case LogLevel::Error:
            return "error";
    }
    return "unknown";
}

inline void log_message(LogLevel level, const char* format, ...) {
    std::fprintf(level == LogLevel::Error ? stderr : stdout, "[cuda:%s] ", log_level_name(level));
    va_list args;
    va_start(args, format);
    std::vfprintf(level == LogLevel::Error ? stderr : stdout, format, args);
    va_end(args);
    std::fprintf(level == LogLevel::Error ? stderr : stdout, "\n");
}

}  // namespace cuda
}  // namespace quadtrix
