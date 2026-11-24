#include "conv.h"
#include "weights.h"

void load_conv_1_filter(Conv1Filter* filterInRam, uint8_t filter) {

  static const int32_t biases[8] PROGMEM = {-20, -97, 12, -4, -12, -23, 5, -27};
  static const uint16_t multipliers[8] PROGMEM = {2286, 1274, 2183, 1070, 1533, 1645, 1255, 1273};

  memcpy_P(filterInRam->weights, mnist_weights::conv1_weights[filter][0], sizeof(filterInRam->weights));
  filterInRam->bias = static_cast<int32_t>( pgm_read_dword(&biases[filter]) );
  filterInRam->multiplier = pgm_read_word(&multipliers[filter]);

}

int32_t conv_1(const uint8_t (&imageInRam)[32][32], uint8_t row, uint8_t col, const int8_t (*weightsInRam)[5]) {

  int32_t sum = 0;

  const auto mac = [&](uint8_t ky, uint8_t kx) __attribute__((always_inline)) {

    const uint8_t pixel = imageInRam[row + ky][col + kx];
    const int8_t weight = weightsInRam[ky][kx];
    const int16_t product = pixel * weight;
    sum += product;

  };

  for (uint8_t ky = 0; ky < 5; ky++) {
    mac(ky, 0);
    mac(ky, 1);
    mac(ky, 2);
    mac(ky, 3);
    mac(ky, 4);
  }

  return sum;
}

uint8_t conv_max_relu( const uint8_t (&imageInRam)[32][32], uint8_t poolRow, uint8_t poolCol, const Conv1Filter* filterInRam) {

  const uint8_t row = 2 * poolRow;
  const uint8_t col = 2 * poolCol;

  int32_t maximum = conv_1(imageInRam, row, col, filterInRam->weights);

  int32_t value = conv_1(imageInRam, row, col + 1, filterInRam->weights);
  if (value > maximum) maximum = value;
  value = conv_1(imageInRam, row + 1, col, filterInRam->weights);
  if (value > maximum) maximum = value;
  value = conv_1(imageInRam, row + 1, col + 1, filterInRam->weights);
  if (value > maximum)  maximum = value;

  const int32_t biased = maximum + filterInRam->bias;
  if (biased <= 0) {
    return 0;
  }

  const uint32_t scaled = (static_cast<uint32_t>(biased) * filterInRam->multiplier + (1UL << 19)) >> 20;
  const uint8_t quantized = scaled > UINT8_MAX ? UINT8_MAX : scaled;
  return quantized;
}



void fill_pool1_block( const uint8_t (&imageInRam)[32][32], uint8_t firstRow, uint8_t (&poolBuffer)[5][14][8]) {

Conv1Filter filter;

  for (uint8_t channel = 0; channel < 8; channel++) {

    load_conv_1_filter(&filter, channel);

    for (uint8_t row = 0; row < 5; row++) {
      for (uint8_t col = 0; col < 14; col++) {
        poolBuffer[row][col][channel] = conv_max_relu(imageInRam, firstRow + row, col, &filter);
      }
    }
  }
}

void fill_pool1_row( const uint8_t (&imageInRam)[32][32], uint8_t sourceRow, uint8_t (&destination)[14][8]) {

  Conv1Filter filter;

  for (uint8_t channel = 0; channel < 8; channel++) {
    load_conv_1_filter(&filter, channel);
    for (uint8_t col = 0; col < 14; col++) {
      destination[col][channel] = conv_max_relu(imageInRam, sourceRow, col, &filter);
    }
  }
}

template<bool CheckRows, bool CheckCols>
static int32_t conv_2_sum( const uint8_t (*const rows[5])[8], uint8_t col, uint8_t filter, int32_t sum) {

    for (uint8_t ky = 0; ky < 5; ky++) {
        if (CheckRows && rows[ky] == nullptr) continue;

        for (uint8_t kx = 0; kx < 5; kx++) {
            const int8_t inputCol = col + kx - 2;

            if (CheckCols && (inputCol < 0 || inputCol >= 14)) continue;

            const uint8_t* inputPtr = rows[ky][inputCol];
            const int8_t* weightPtr = mnist_weights::conv2_weights[filter][ky][kx];

            for (uint8_t channel = 0; channel < 8; channel++) {
                sum += (*inputPtr++) * static_cast<int8_t>(pgm_read_byte(weightPtr++));
      }
    }
  }

  return sum;
}

static uint8_t conv_2( const uint8_t (*const rows[5])[8], uint8_t row, uint8_t col, uint8_t filter, int32_t sum, uint16_t multiplier) {
  const bool checkRows = row < 2 || row >= 12;
  const bool checkCols = col < 2 || col >= 12;

  if (checkRows) {
    if (checkCols) {
      sum = conv_2_sum<true, true>(rows, col, filter, sum);
    } else {
      sum = conv_2_sum<true, false>(rows, col, filter, sum);
    }
  } else if (checkCols) {
    sum = conv_2_sum<false, true>(rows, col, filter, sum);
  } else {
    sum = conv_2_sum<false, false>(rows, col, filter, sum);
  }

  if (sum <= 0) return 0;

  const uint32_t scaled = (static_cast<uint32_t>(sum) * multiplier + (1UL << 17)) >> 18;
  return scaled > UINT8_MAX ? UINT8_MAX : scaled;

}

const uint8_t* fill_pool2( uint8_t (&imageInRam)[32][32], uint8_t (&poolBuffer)[5][14][8]) {

  static const uint16_t multipliers[16] PROGMEM = {347, 317, 391, 313, 319, 344, 377, 271, 409, 346, 300, 361, 221, 434, 287, 332};

  fill_pool1_block(imageInRam, 0, poolBuffer);

  uint8_t* output = reinterpret_cast<uint8_t*>(&imageInRam);

  for (uint8_t row = 0; row < 12; row++) {
    const uint8_t newestRow = row + 2;

    if (newestRow >= 5) {
      fill_pool1_row(imageInRam, newestRow, poolBuffer[newestRow % 5]);
    }

    const uint8_t (*rows[5])[8];

    for (uint8_t ky = 0; ky < 5; ky++) {
      const int8_t sourceRow = row + ky - 2;
      rows[ky] = sourceRow < 0 ? nullptr : poolBuffer[sourceRow % 5];
    }

    for (uint8_t filter = 0; filter < 16; filter++) {
      const int32_t bias = static_cast<int32_t>(pgm_read_dword(&mnist_weights::conv2_biases[filter]));
      const uint16_t multiplier = pgm_read_word(&multipliers[filter]);
      uint8_t* outputRow = output + (filter * 4 + row / 3) * 4;

      for (uint8_t col = 0; col < 12; col++) {
        const uint8_t value = conv_2(rows, row, col, filter, bias, multiplier);
        uint8_t& maximum = outputRow[col / 3];
        if ((row % 3 == 0 && col % 3 == 0) || value > maximum) maximum = value;
      }
    }
  }

  return output;
}

int8_t dense(const uint8_t* input) {

  static const uint32_t multipliers[10] PROGMEM = {4170290, 3228102, 4296960, 3259225, 4310572, 3806946, 4095939, 4162557, 6617038, 4237083};
  int8_t bestDigit = 0;
  uint8_t bestScore = 0;

  for (int8_t digit = 0; digit < 10; digit++) {
    int32_t sum = 0;
    for (uint16_t i = 0; i < 256; i++) {
      const int8_t weight = static_cast<int8_t>(pgm_read_byte(&mnist_weights::dense_weights[i][digit]));
      sum += input[i] * weight;
    }

    const uint32_t magnitude = sum < 0 ? -sum : sum;
    const uint64_t scaled = static_cast<uint64_t>(magnitude) * pgm_read_dword(&multipliers[digit]);
    uint32_t rounded = scaled >> 32;
    const uint32_t fraction = static_cast<uint32_t>(scaled);
    if (fraction > 0x80000000UL || (fraction == 0x80000000UL && (rounded & 1))) {
      rounded++;
    }

    const int32_t quantized = 112 + (sum < 0 ? -static_cast<int32_t>(rounded) : static_cast<int32_t>(rounded));

    const uint8_t score = quantized < 0 ? 0 : (quantized > 255 ? 255 : quantized);
    if (score > bestScore) {
      bestScore = score;
      bestDigit = digit;
    }
  }

  return bestDigit;
}
