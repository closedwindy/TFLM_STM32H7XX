//
// Created by hhw on 2026/3/25.
//

#include "ai_process.h"
#include "model.h"
#include "usart.h"
constexpr int kTensorArenaSize = 180 * 1024;

tflite::MicroMutableOpResolver<5> resolver;
__attribute__((section(".ram_d1")))
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
const tflite::Model* model = nullptr;
TfLiteStatus InitTfLM() {

        TfLiteStatus status;

        status = resolver.AddFullyConnected();
        if (status != kTfLiteOk) return status;
        status = resolver.AddQuantize();      // 量化算子
        if (status != kTfLiteOk) return status;
        status = resolver.AddDequantize();    // 反量化算子
        if (status != kTfLiteOk) return status;
        status = resolver.AddReshape();       // 形状变换（如果有）
        if (status != kTfLiteOk) return status;

        model = tflite::GetModel(motor_ff_controller_int8_io_tflite);

        if (model == nullptr) {
                return kTfLiteError;
        }

        if (model->version() != TFLITE_SCHEMA_VERSION) {
                return kTfLiteError;
        }

        // 创建解释器
        // 注意：MicroInterpreter 内部会保存 pointer，所以 model 和 tensor_arena 必须在整个生命周期内有效
        static tflite::MicroInterpreter static_interpreter(
            model,
            resolver,
            tensor_arena,
            kTensorArenaSize
        );

        interpreter = &static_interpreter;

        // 分配张量内存
        status = interpreter->AllocateTensors();
        if (status != kTfLiteOk) {
                // 如果失败，通常是 tensor_arena 太小或者算子没注册对
                HAL_UART_Transmit(&huart3, (uint8_t*)"AllocateTensors Failed\r\n", 24, 100);
                return status;
        }

        // 验证输入输出维度是否符合预期
        // TfLiteTensor* input_tensor = interpreter->input(0);
        // if (input_tensor->dims->data[1] != 4) { // 假设形状是 [1, 4]
        //         HAL_UART_Transmit(&huart3, (uint8_t*)"Input Dim Mismatch\r\n", 20, 100);
        // }

        return kTfLiteOk;
}
// TfLiteStatus InitTfLM() {
//         TfLiteStatus status = resolver.AddFullyConnected();
//         if (status != kTfLiteOk) return status;
//
//
//         const tflite::Model* model = tflite::GetModel(motor_ff_controller_tflite);
//
//         if (model->version() != TFLITE_SCHEMA_VERSION) {
//                 return kTfLiteError;
//         }
//
//
//         static tflite::MicroInterpreter static_interpreter(
//             model,
//             resolver,
//             tensor_arena,
//             sizeof(tensor_arena)
//         );
//         interpreter = &static_interpreter;
//
//         status = interpreter->AllocateTensors();
//         if (status != kTfLiteOk) {
//
//                 return status;
//         }
//
//         return kTfLiteOk;
// }
int8_t AI_Inference(const int8_t* input_data) {
        if (interpreter == nullptr) {
                return 0.0f; // 或者处理错误
        }

        // 获取输入张量
        // 模型输入形状是 [1, 4]，TFLM 展平为一维访问
        TfLiteTensor* input_tensor = interpreter->input(0);

        // 安全性检查：确保输入尺寸匹配
        if (input_tensor->bytes != 4 * sizeof(int8_t)) {
                return 0.0f;
        }


        int8_t* input_ptr = input_tensor->data.int8;
        for (int i = 0; i < 4; i++) {
                input_ptr[i] = input_data[i];
        }

        // 3. 执行推理 (Invoke)
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
                // 推理失败
                return 0.0f;
        }

        // 获取输出数据
        TfLiteTensor* output_tensor = interpreter->output(0);
        // 模型输出形状是 [1, 1]
        float result = output_tensor->data.int8[0];

        return result;
}
extern "C" void DebugLog(const char* s) {



        if (s != NULL) {
            HAL_UART_Transmit(&huart3, (uint8_t*)s, strlen(s), 100);
        }

}