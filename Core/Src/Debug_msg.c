//
// Created by hhw on 2026/3/26.
//

#include "../Inc/Debug_msg.h"
#include "usart.h"
int __io_putchar(int ch) {
        HAL_UART_Transmit(&huart3, (uint8_t*)&ch, 1, HAL_MAX_DELAY);
        return ch; // 返回写入字节数（非0表示成功）
}