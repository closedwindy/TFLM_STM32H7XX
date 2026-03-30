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
const tflite::Model* model = tflite::GetModel(motor_ff_controller_int8_io_tflite);



TfLiteStatus TFLM_Init()
{
        // tflite::MicroMutableOpResolver<5> resolver;
        // resolver.AddFullyConnected();
        // resolver.AddRelu();
        // resolver.AddSoftmax();



        return kTfLiteOk;

}

int8 Invoke_process(const int8_t* input_data)
{
        tflite::MicroMutableOpResolver<5> resolver;
        resolver.AddFullyConnected();
        resolver.AddRelu();
        resolver.AddSoftmax();

        tflite::MicroInterpreter interpreter(model, resolver,
                                     tensor_arena, kTensorArenaSize);

        if (interpreter.AllocateTensors()!= kTfLiteOk) {
                return kTfLiteError;
        }
        TfLiteTensor* input0 = interpreter.input(0);
        TfLiteTensor* input1 = interpreter.input(1);
        TfLiteTensor* input2 = interpreter.input(2);
        TfLiteTensor* input3 = interpreter.input(3);

        if (input0->type != kTfLiteInt8 ||
            input1->type != kTfLiteInt8 ||
            input2->type != kTfLiteInt8 ||
            input3->type != kTfLiteInt8) {
                return kTfLiteError;
            }

        TfLiteTensor* output = interpreter.output(0);


        interpreter.Invoke();
        int8_t Res = output->data.int8[0];

        return Res;

}





extern "C" void DebugLog(const char* s) {



        if (s != NULL) {
            HAL_UART_Transmit(&huart3, (uint8_t*)s, strlen(s), 100);
        }

}