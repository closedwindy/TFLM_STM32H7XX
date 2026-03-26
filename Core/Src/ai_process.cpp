//
// Created by hhw on 2026/3/25.
//

#include "ai_process.h"
#include "model.h"
constexpr int kTensorArenaSize = 180 * 1024;

tflite::MicroMutableOpResolver<5> resolver;
__attribute__((section(".ram_d1")))
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
const tflite::Model* model = nullptr;

TfLiteStatus InitTfLM() {
        TfLiteStatus status = resolver.AddFullyConnected();
        if (status != kTfLiteOk) return status;


        const tflite::Model* model = tflite::GetModel(motor_ff_controller_tflite);

        if (model->version() != TFLITE_SCHEMA_VERSION) {
                return kTfLiteError;
        }


        static tflite::MicroInterpreter static_interpreter(
            model,
            resolver,
            tensor_arena,
            sizeof(tensor_arena)
        );
        interpreter = &static_interpreter;

        status = interpreter->AllocateTensors();
        if (status != kTfLiteOk) {

                return status;
        }

        return kTfLiteOk;
}
float AI_Inference(const float* input_data) {
        if (interpreter == nullptr) {
                return 0.0f; // 或者处理错误
        }

        // 1. 获取输入张量
        // 模型输入形状是 [1, 4]，TFLM 展平为一维访问
        TfLiteTensor* input_tensor = interpreter->input(0);

        // 安全性检查：确保输入尺寸匹配
        if (input_tensor->bytes != 4 * sizeof(float)) {
                // 处理错误：输入尺寸不匹配 (可能是量化模型？如果是int8模型，这里需要转换)
                // 假设你的模型是 float32 输入
                return 0.0f;
        }

        // 2. 填充输入数据
        // 直接拷贝内存，效率最高
        // 输入是 float32 类型
        float* input_ptr = input_tensor->data.f;
        for (int i = 0; i < 4; i++) {
                input_ptr[i] = input_data[i];
        }
        // 或者直接用 memcpy:
        // memcpy(input_ptr, input_data, 4 * sizeof(float));

        // 3. 执行推理 (Invoke)
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
                // 推理失败
                return 0.0f;
        }

        // 4. 获取输出数据
        TfLiteTensor* output_tensor = interpreter->output(0);
        // 模型输出形状是 [1, 1]
        float result = output_tensor->data.f[0];

        return result;
}
extern "C" void DebugLog(const char* s) {
        // 方案 A: 简单使用 printf (需要重定向 stdout 到串口，通常在 syscalls.c 中完成)
        // 如果你的 syscalls.c 已经重定向了 _write，直接用这个：
        printf("%s", s);

        // 方案 B: 如果 printf 没重定向，直接使用 HAL (取消下面注释并修改句柄)
        /*
        if (s != NULL) {
            HAL_UART_Transmit(&huart1, (uint8_t*)s, strlen(s), 100);
        }
        */
}