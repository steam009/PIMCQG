cmake_minimum_required(VERSION 3.16)
project(reproduce_c C CXX)

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 设置C++标准（用于 symphonyqg 绑定）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -fopenmp -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# 查找所需的包
find_package(OpenMP REQUIRED)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../symqglib)

# 源文件
set(SOURCES
    src/main.c
    src/utils.c
    src/ivf_search.c
    src/hashtable.c
    src/error.c
    src/symqg_c_binding.cpp
)

# 头文件
set(HEADERS
    include/settings.h
    include/utils.h
    include/ivf_search.h
    include/hashtable.h
    include/error.h
    include/symqg_c_binding.h
)

# 设置输出目录 (在创建可执行文件之前设置)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# 创建可执行文件
add_executable(reproduce_c ${SOURCES} ${HEADERS})

# 显式设置目标的输出目录
set_target_properties(reproduce_c PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

# 链接库
target_link_libraries(reproduce_c 
    OpenMP::OpenMP_C
    m  # 数学库
    stdc++  # C++ 标准库（用于 symphonyqg 绑定）
)

# 如果需要链接到 symphonyqg 库，取消注释以下行
# 注意: symphonyqg 是 C++ 库，需要额外的 C++ 运行时库支持
# find_library(SYMPHONYQG_LIB symphonyqg PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build)
# if(SYMPHONYQG_LIB)
#     target_link_libraries(reproduce_c ${SYMPHONYQG_LIB})
#     # 如果 symphonyqg 是 C++ 库，还需要链接 C++ 标准库
#     target_link_libraries(reproduce_c stdc++)
# else()
#     message(WARNING "symphonyqg library not found. Please build it first.")
# endif()

# 创建输出目录
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)

# 打印一些有用的信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")
message(STATUS "Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# 添加自定义目标来清理结果
add_custom_target(clean-results
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/results
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/results
    COMMENT "Cleaning results directory"
)

# 添加自定义目标来运行程序
add_custom_target(run
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/reproduce_c
    DEPENDS reproduce_c
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    COMMENT "Running reproduce_c"
)
