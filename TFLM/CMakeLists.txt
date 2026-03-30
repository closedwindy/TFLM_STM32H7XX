cmake_minimum_required(VERSION 3.22)
set(TFLM_BASE ${CMAKE_CURRENT_SOURCE_DIR})


file(GLOB_RECURSE TFLM_CORE_SOURCES
        "${TFLM_BASE}/tensorflow/lite/micro/*.cc"
        "${TFLM_BASE}/tensorflow/lite/core/api/*.cc"
        "${TFLM_BASE}/tensorflow/lite/core/c/*.cc"
        "${TFLM_BASE}/tensorflow/lite/kernels/internal/*.cc"
        "${TFLM_BASE}/tensorflow/lite/kernels/kernel_util.cc"
        "${TFLM_BASE}/tensorflow/compiler/mlir/lite/schema/*.cc"
        "${TFLM_BASE}/tensorflow/compiler/mlir/lite/core/api/*.cc"
)

# 收集所有算子文件 (包括 CMSIS-NN 和通用算子)
file(GLOB_RECURSE TFLM_KERNEL_SOURCES
        "${TFLM_BASE}/tensorflow/lite/micro/kernels/*.cc"
)

# 过滤掉不需要的文件 (测试文件、host 端工具等)
list(FILTER TFLM_CORE_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER TFLM_CORE_SOURCES EXCLUDE REGEX ".*host.*")
list(FILTER TFLM_CORE_SOURCES EXCLUDE REGEX ".*benchmark.*")
list(FILTER TFLM_KERNEL_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER TFLM_KERNEL_SOURCES EXCLUDE REGEX ".*host.*")
#  创建 TFLM 静态库
add_library(tflm STATIC ${TFLM_CORE_SOURCES} ${TFLM_KERNEL_SOURCES})

target_include_directories(tflm PUBLIC
        ${TFLM_BASE}
        ${TFLM_BASE}/tensorflow/lite/micro
        ${TFLM_BASE}/tensorflow/lite
        ${TFLM_BASE}/tensorflow/lite/core
        ${TFLM_BASE}/tensorflow/lite/core/api
        ${TFLM_BASE}/tensorflow/lite/core/c
        ${TFLM_BASE}/tensorflow/lite/kernels
        ${TFLM_BASE}/tensorflow/lite/kernels/internal
        ${TFLM_BASE}/tensorflow/compiler/mlir/lite/core/api
        ${TFLM_BASE}/tensorflow/compiler/mlir/lite/schema
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmsis_nn
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmsis/CMSIS/Core/Include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include

        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kissfft
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gemmlowp
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ruy
)

target_compile_options(tflm PRIVATE
        -mcpu=cortex-m7
        -mthumb
        -mfpu=fpv5-d16
        -mfloat-abi=hard
        -fno-rtti
        -fno-exceptions
        -fno-threadsafe-statics
        -Wall
        -fdata-sections
        -ffunction-sections
        -DTF_LITE_STATIC_MEMORY
        -DTF_LITE_DISABLE_X86_NEON
        -DTF_LITE_MCU_DEBUG_LOG
        -DNDEBUG
        -DCMSIS_NN
)

set_target_properties(tflm PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

#  编译 CMSIS-NN 核心库
set(CMSIS_NN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmsis_nn)

file(GLOB_RECURSE CMSIS_NN_SOURCES "${CMSIS_NN_ROOT}/Source/*.c")

if(NOT CMSIS_NN_SOURCES)
    message(FATAL_ERROR "CMSIS-NN source files not found in ${CMSIS_NN_ROOT}/Source")
else()
    add_library(cmsis_nn_lib STATIC ${CMSIS_NN_SOURCES})

    target_include_directories(cmsis_nn_lib PUBLIC
            ${CMSIS_NN_ROOT}/Include
            ${CMSIS_NN_ROOT}/Source
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmsis/CMSIS/Core/Include
    )

    target_compile_options(cmsis_nn_lib PRIVATE
            -mcpu=cortex-m7
            -mthumb
            -mfpu=fpv5-d16
            -mfloat-abi=hard
            -O3
            -Wall
            -fdata-sections
            -ffunction-sections
            -DNDEBUG
    )

    target_link_libraries(tflm PRIVATE cmsis_nn_lib)
endif()

target_link_libraries(tflm PUBLIC m)
target_link_libraries(tflm PUBLIC stm32cubemx)

message(STATUS "TFLM Library configured with CMSIS-NN")