#pragma once

#include <iostream>

// Platform-specific headers and configurations
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <locale>
#endif

inline void print_banner()
{
// Handle platform-specific console encoding setups
#ifdef _WIN32
      // Force Windows console output to UTF-8 so block characters render correctly
      SetConsoleOutputCP(CP_UTF8);
#elif defined(__APPLE__) || defined(__linux__)
      // Set the locale for macOS and Linux to support standard UTF-8 text streams
      std::locale::global(std::locale(""));
#endif

      // ANSI escape codes for bold, bright white text and resetting formatting
      const char *WHITE = "\033[1;37m";
      const char *RESET = "\033[0m";

      // The large, solid-block llm.cpp banner
      const char *banner = R"(
██╗     ██╗     ███╗   ███╗        ██████╗ ██████╗ ██████╗ 
██║     ██║     ████╗ ████║       ██╔════╝ ██╔══██╗██╔══██╗
██║     ██║     ██╔████╔██║ █████╗██║      ██████╔╝██████╔╝
██║     ██║     ██║╚██╔╝██║ ╚════╝██║      ██╔═══╝ ██╔═══╝ 
███████╗███████╗██║ ╚═╝ ██║ █████╗╚██████╗ ██║     ██║     
╚══════╝╚══════╝╚═╝     ╚═╝ ╚════╝ ╚═════╝ ╚═╝     ╚═╝     
)";

      // Print the banner in bright white
      std::cout << WHITE << banner << RESET << std::endl;
}