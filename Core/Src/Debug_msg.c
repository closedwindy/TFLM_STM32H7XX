//
// Created by hhw on 2026/3/26.
//

#include "../Inc/Debug_msg.h"
#include "usart.h"
#include "tensorflow/lite/micro/debug_log.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

int __io_putchar(int ch) {
        HAL_UART_Transmit(&huart3, (uint8_t*)&ch, 1, HAL_MAX_DELAY);
        return ch; // 返回写入字节数（非0表示成功）
}
int tflm_io_write(const void *buff, uint16_t count)
{
        HAL_StatusTypeDef status;

        status = HAL_UART_Transmit(&huart3, (uint8_t *)buff, count,
                HAL_MAX_DELAY);

        return (status == HAL_OK ? count : 0);
}
void DebugLog(const char* format, va_list args)
{
#ifndef TF_LITE_STRIP_ERROR_STRINGS
        const int kMaxLogLen = 256;
        char log_buffer[kMaxLogLen];

        if (!format)
                return;

        vsnprintf(log_buffer, kMaxLogLen, format, args);

#if defined(USE_PRINTF)
#include <stdio.h>
        printf("%s", log_buffer);
#else
        size_t sl = strlen(log_buffer);
        if (sl)
                tflm_io_write(log_buffer, (uint16_t)sl);
#endif
#endif
}