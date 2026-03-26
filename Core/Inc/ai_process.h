#pragma once

#include <stdint.h>
#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

        TfLiteStatus InitTfLM(void);

        float AI_Inference(const float* input_data);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"

#endif