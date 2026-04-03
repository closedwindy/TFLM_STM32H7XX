//
// Created by hhw on 2026/4/3.
//

#include "ai_process.h"
#include "model.h"

#include <cstring>
#include <cstdio>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

__attribute__((section(".ram_d1"), aligned(16)))
static uint8_t tensor_arena[180 * 1024];


static tflite::MicroMutableOpResolver<20> resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;


static void PrintTensorData(const char* name, int8_t* data, size_t bytes) {
    printf("[%s] %d bytes: ", name, (int)bytes);
    size_t print_len = (bytes < 8) ? bytes : 8;
    for (size_t i = 0; i < print_len; i++) {
        printf("0x%02X(%d) ", (uint8_t)data[i], data[i]);
    }
    if (bytes > 8) printf("...");
    printf("\r\n");
}

extern "C" int TFLM_Init(void) {
    if (interpreter != nullptr) return 0;  // Already initialized

    // 1. Load Model
    // Note: Variable name from network_tflite_data.h
    model = tflite::GetModel(motor_ff_controller_int8_io_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\r\n");
        return -1;
    }


    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddAdd();          // For BiasAdd operations
    resolver.AddReshape();      // Common for input handling

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter = &static_interpreter;

    // 4. Allocate Tensors
    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        printf("AllocateTensors failed! (Arena too small?)\r\n");
        return -2;
    }

    // Print input shape for verification
    printf("TFLM Native Init OK. Input shape: ");
    for(int i = 0; i < interpreter->input(0)->dims->size; i++){
        printf("%d ", interpreter->input(0)->dims->data[i]);
    }
    printf("\r\n");

    return 0;
}

extern "C" int8_t TFLM_Infer(const int8_t* input_data, size_t input_bytes) {
    if (!interpreter) return -1;

    TfLiteTensor* input = interpreter->input(0);

    size_t input_size = input->bytes;
    if (input_size == 0) {
        printf("[ERROR] Input size is 0\r\n");
        return -1;
    }

    size_t copy_len = (input_bytes < input_size) ? input_bytes : input_size;
    std::memcpy(input->data.int8, input_data, copy_len);

    if (copy_len < input_size) {
        std::memset(input->data.int8 + copy_len, 0, input_size - copy_len);
    }

    PrintTensorData("INPUT", input->data.int8, input_size);

    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        printf("[ERROR] Invoke failed: %d\r\n", status);
        return -1;
    }

    TfLiteTensor* output = interpreter->output(0);

    PrintTensorData("OUTPUT", output->data.int8, output->bytes);

    return output->data.int8[0];
}
