#!/bin/bash
set -e

OS=$1
VARIANT=$2

echo "Starting build on $OS for variant: ${VARIANT:-default}..."

if [ "$OS" == "macOS" ]; then
    clang++ -std=c++17 -O3 -DNDEBUG -arch arm64 -I. -Iinclude -o llm main.cpp

elif [ "$OS" == "iOS" ]; then
    echo "Building for iOS (arm64)..."
    clang++ -std=c++17 -O3 -DNDEBUG -arch arm64 \
        -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk \
        -I. -Iinclude -c main.cpp -o llm-ios-arm64.o
    ar rcs libllm-ios-arm64.a llm-ios-arm64.o

    echo "Building for iOS Simulator (x86_64)..."
    clang++ -std=c++17 -O3 -DNDEBUG -arch x86_64 \
        -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk \
        -I. -Iinclude -c main.cpp -o llm-ios-sim-x86_64.o
    ar rcs libllm-ios-sim-x86_64.a llm-ios-sim-x86_64.o

    echo "Creating XCFramework..."
    xcodebuild -create-xcframework \
        -library libllm-ios-arm64.a \
        -library libllm-ios-sim-x86_64.a \
        -output LLM.xcframework

elif [ "$OS" == "Ubuntu" ]; then
    CXX_COMPILER=${CXX:-g++} 
    
    if [ "$VARIANT" == "cpu" ]; then
        $CXX_COMPILER -std=c++17 -O3 -DNDEBUG -I. -Iinclude -o llm main.cpp
        
    elif [ "$VARIANT" == "vulkan" ]; then
        $CXX_COMPILER -std=c++17 -O3 -DNDEBUG -DENABLE_VULKAN -I. -Iinclude -o llm main.cpp -lvulkan
        
    elif [[ "$VARIANT" == *"sycl"* ]]; then
        export PATH="/opt/intel/sycl/bin:$PATH"
        if [ "$VARIANT" == "sycl-fp32" ]; then
            clang++ -std=c++17 -O3 -DNDEBUG -DENABLE_SYCL -fsycl -fsycl-targets=spir64 -I. -Iinclude -o llm main.cpp
        elif [ "$VARIANT" == "sycl-fp16" ]; then
            clang++ -std=c++17 -O3 -DNDEBUG -DENABLE_SYCL -DENABLE_FP16 -fsycl -fsycl-targets=spir64 -I. -Iinclude -o llm main.cpp
        fi
    fi

elif [ "$OS" == "Android" ]; then
    if [ -z "$ANDROID_NDK_HOME" ]; then
        echo "Error: ANDROID_NDK_HOME environment variable is not set."
        exit 1
    fi
    $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
        --target=aarch64-linux-android21 -std=c++17 -O3 -DNDEBUG -I. -Iinclude -o llm main.cpp

elif [ "$OS" == "Windows" ]; then
    if [ "$VARIANT" == "cpu-x64" ]; then
        cmd.exe /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat\" x64 && cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /Iinclude /I. main.cpp /Fe:llm.exe"
        
    elif [ "$VARIANT" == "cpu-arm64" ]; then
        cmd.exe /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat\" amd64_arm64 && cl /nologo /std:c++17 /O2 /DNDEBUG /EHsc /Iinclude /I. main.cpp /Fe:llm.exe"
        
    elif [ "$VARIANT" == "cuda13" ]; then
        cmd.exe /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat\" x64 && cl /nologo /std:c++17 /O2 /DNDEBUG /DENABLE_CUDA /EHsc /Iinclude /I. /I\"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include\" main.cpp /Fe:llm.exe /link \"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\lib\x64\cudart.lib\""
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

echo "Build completed successfully for $OS ($VARIANT)!"
