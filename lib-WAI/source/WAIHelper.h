#ifndef WAI_HELPER_H
#define WAI_HELPER_H

// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
#    define WAI_HELPER_DLL_IMPORT __declspec(dllimport)
#    define WAI_HELPER_DLL_EXPORT __declspec(dllexport)
#    define WAI_HELPER_DLL_LOCAL
#else
#    if __GNUC__ >= 4
#        define WAI_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#        define WAI_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#        define WAI_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#    else
#        define WAI_HELPER_DLL_IMPORT
#        define WAI_HELPER_DLL_EXPORT
#        define WAI_HELPER_DLL_LOCAL
#    endif
#endif

// Now we use the generic helper definitions above to define WAI_API and WAI_LOCAL.
// WAI_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// WAI_LOCAL is used for non-api symbols.

#ifdef WAI_DLL             // defined if WAI is compiled as a DLL
#    ifdef WAI_DLL_EXPORTS // defined if we are building the WAI DLL (instead of using it)
#        define WAI_API WAI_HELPER_DLL_EXPORT
#    else
#        define WAI_API WAI_HELPER_DLL_IMPORT
#    endif // WAI_DLL_EXPORTS
#    define WAI_LOCAL WAI_HELPER_DLL_LOCAL
#else // WAI_DLL is not defined: this means WAI is a static lib.
#    define WAI_API
#    define WAI_LOCAL
#endif // WAI_DLL

#ifdef __APPLE__
#    include <TargetConditionals.h>
#    if TARGET_OS_IOS
#        define WAI_OS_MACIOS
#    else
#        define WAI_OS_MACOS
#    endif
#    define WAI_LOG(...) printf(__VA_ARGS__)
#elif defined(ANDROID) || defined(ANDROID_NDK)
#    include <android/log.h>
#    define WAI_OS_ANDROID
#    define WAI_LOG(...) __android_log_print(ANDROID_LOG_INFO, "lib-WAI", __VA_ARGS__)
#elif defined(_WIN32)
#    define WAI_OS_WINDOWS
#    define STDCALL __stdcall
#    define WAI_LOG(...) printf(__VA_ARGS__)
#elif defined(linux) || defined(__linux) || defined(__linux__)
#    define WAI_OS_LINUX
#    define WAI_LOG(...) printf(__VA_ARGS__)
#else
#    error "WAI has not been ported to this OS"
#endif

#ifdef WAI_BUILD_DEBUG
#    define wai_assert(expression) \
        if (!(expression)) { *(int*)0 = 0; }
#else
#    define wai_assert(expression)
#endif

#endif
