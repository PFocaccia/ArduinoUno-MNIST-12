#pragma once

#include <stdint.h>

struct Conv1Filter {
  int8_t weights[5][5];
  int32_t bias;
  uint16_t multiplier;
};

void load_conv_1_filter(Conv1Filter* filterInRam, uint8_t filter);

int32_t conv_1(const uint8_t (&imageInRam)[32][32], uint8_t row, uint8_t col, const int8_t (*weightsInRam)[5]);

uint8_t conv_max_relu( const uint8_t (&imageInRam)[32][32], uint8_t poolRow, uint8_t poolCol, const Conv1Filter* filterInRam);

void fill_pool1_block( const uint8_t (&imageInRam)[32][32], uint8_t firstRow, uint8_t (&poolBuffer)[5][14][8]);

void fill_pool1_row( const uint8_t (&imageInRam)[32][32], uint8_t sourceRow, uint8_t (&destination)[14][8]);

const uint8_t* fill_pool2( uint8_t (&imageInRam)[32][32], uint8_t (&poolBuffer)[5][14][8]);

int8_t dense(const uint8_t* input);
