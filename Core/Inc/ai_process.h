//
// Created by hhw on 2026/4/3.
//
#ifndef __AI_PROCESS_H
#define __AI_PROCESS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


        int TFLM_Init(void);

        int8_t TFLM_Infer(const int8_t* input_data, size_t input_bytes);

#ifdef __cplusplus
}
#endif

#endif /* __AI_PROCESS_H */
