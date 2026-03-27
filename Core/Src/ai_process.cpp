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

extern "C" void DebugLog(const char* s) {



        if (s != NULL) {
            HAL_UART_Transmit(&huart3, (uint8_t*)s, strlen(s), 100);
        }

}