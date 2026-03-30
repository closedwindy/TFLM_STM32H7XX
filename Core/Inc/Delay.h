//
// Created by hhw on 2025/7/26.
//

#ifndef DELAY_H
#define DELAY_H
#include "stdint.h"
#include "main.h"
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
class Timer {
        private:
       uint32_t start_time;
       uint32_t wait_time;
        public:
        explicit Timer(uint32_t ms) : wait_time(ms), start_time(HAL_GetTick()) {}
        bool wait() {
            if (HAL_GetTick() - start_time >= wait_time) {
                start_time = HAL_GetTick();
                return true;
            }
            return false;
        }
};
#endif


#endif //ROBOT_BSP_DELAY_H
