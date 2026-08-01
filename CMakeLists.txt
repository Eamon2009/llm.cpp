cmake_minimum_required(VERSION 3.15)

project(LLM_CPU LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(llm main.cpp)
target_include_directories(llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Query Clang internal resource directory to automatically find omp.h on Windows/MinGW
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
        OUTPUT_VARIABLE CLANG_RESOURCE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CLANG_RESOURCE_DIR)
        target_include_directories(llm PRIVATE "${CLANG_RESOURCE_DIR}/include")
    endif()
    
    target_compile_options(llm PRIVATE -fopenmp)
    target_link_libraries(llm PRIVATE -fopenmp)
elseif(MSVC)
    target_compile_options(llm PRIVATE /openmp)
else()
    # GCC and standard environments
    find_package(OpenMP REQUIRED)
    target_link_libraries(llm PRIVATE OpenMP::OpenMP_CXX)
endif()

# Cross-platform compiler optimization flags
if(MSVC)
    target_compile_options(llm PRIVATE
        /O2                
        /arch:AVX2          
        /DNOMINMAX        
        /EHsc              
    )
else()
    target_compile_options(llm PRIVATE
        -O3                
        -march=native # Optimize for the local CPU architecture
    )
endif()

# --------------------------------------------------------
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/run_pipeline.cmake"
    CONTENT "
# 1. Run the Python dataset script
message(STATUS \"Running Python script: data/data_set.py...\")
execute_process(
    COMMAND \"${Python3_EXECUTABLE}\" \"${CMAKE_CURRENT_SOURCE_DIR}/data/data_set.py\"
    WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\"
    RESULT_VARIABLE py_res
)
if(NOT py_res EQUAL 0)
    message(FATAL_ERROR \"data_set.py script failed!\")
endif()

# 2. Check for the best available input file in priority order
if(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/data/input.txt\")
    set(INPUT_FILE \"${CMAKE_CURRENT_SOURCE_DIR}/data/input.txt\")
elseif(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/data/input.txt.tokenizer.bin\")
    set(INPUT_FILE \"${CMAKE_CURRENT_SOURCE_DIR}/data/input.txt.tokenizer.bin\")
elseif(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/data/shards/shard_00000.bin\")
    set(INPUT_FILE \"${CMAKE_CURRENT_SOURCE_DIR}/data/shards/shard_00000.bin\")
else()
    message(FATAL_ERROR \"No valid input file found in the data/ directory!\")
endif()

# 3. Execute the compiled binary with the discovered file
message(STATUS \"Executing LLM binary using input: \${INPUT_FILE}\")
execute_process(
    COMMAND \"$<TARGET_FILE:llm>\" \"\${INPUT_FILE}\"
    WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\"
    RESULT_VARIABLE llm_res
)
if(NOT llm_res EQUAL 0)
    message(FATAL_ERROR \"LLM execution failed!\")
endif()
"
)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/run_pipeline.cmake"
    DEPENDS llm
    COMMENT "Starting data pipeline and executing LLM..."
)