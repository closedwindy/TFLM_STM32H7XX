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
        status = resolver.AddQuantize();
        if (status != kTfLiteOk) return status;
        status = resolver.AddDequantize();
        if (status != kTfLiteOk) return status;
        status = resolver.AddReshape();
        if (status != kTfLiteOk) return status;

        model = tflite::GetModel(motor_ff_controller_int8_io_tflite);

        if (model == nullptr) {
                return kTfLiteError;
        }

        if (model->version() != TFLITE_SCHEMA_VERSION) {
                return kTfLiteError;
        }


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
                HAL_UART_Transmit(&huart3, (uint8_t*)"AllocateTensors Failed\r\n", 24, 100);
                return status;
        }


        return kTfLiteOk;
}

int8_t AI_Inference(const int8_t* input_data) {
        if (interpreter == nullptr) {
                return 0;
        }


        TfLiteTensor* input_tensor = interpreter->input(0);

        if (input_tensor->bytes != 4 * sizeof(int8_t)) {
                return 0;
        }

        int8_t* input_ptr = input_tensor->data.int8;

        memcpy(input_tensor->data.int8, input_data, 4 * sizeof(int8_t));


        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
                return 0;
        }


        TfLiteTensor* output_tensor = interpreter->output(0);

        int8_t result = output_tensor->data.int8[0];

        return result;

}


extern "C" void DebugLog(const char* s) {

        if (s != NULL) {
            HAL_UART_Transmit(&huart3, (uint8_t*)s, strlen(s), 100);
        }

}