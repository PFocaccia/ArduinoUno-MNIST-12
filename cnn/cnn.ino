#include <Arduino.h>
#include <avr/interrupt.h>
#include "conv.h"

static uint8_t image[32][32];
static uint8_t poolBuffer[5][14][8];
static volatile uint16_t timer1Overflows;

ISR(TIMER1_OVF_vect) {
  timer1Overflows++;
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  const int command = Serial.read();

  if (command == 'P') {
    Serial.write('A');
    return;
  }

  if (command != 'I') return;

  memset(image, 0, sizeof(image));

  for (uint8_t row = 2; row < 30; row++) if (Serial.readBytes(&image[row][2], 28) != 28) return;

  const uint8_t savedStatus = SREG;
  noInterrupts();
  TCCR1B = 0;
  TCCR1A = 0;
  TCNT1 = 0;
  timer1Overflows = 0;
  TIFR1 = _BV(TOV1);
  TIMSK1 = _BV(TOIE1);
  TCCR1B = _BV(CS10);
  SREG = savedStatus;

  const uint8_t* output = fill_pool2(image, poolBuffer);
  const int8_t digit = dense(output);
  asm volatile("" : : "r"(digit) : "memory");

  noInterrupts();
  TCCR1B = 0;
  const uint16_t ticks = TCNT1;
  uint16_t overflows = timer1Overflows;
  if (TIFR1 & _BV(TOV1)) {
    overflows++;
  }
  TIMSK1 = 0;
  TIFR1 = _BV(TOV1);
  SREG = savedStatus;

  const uint32_t cycles = (static_cast<uint32_t>(overflows) << 16) | ticks;
  const uint8_t response[6] = {
    'R', static_cast<uint8_t>(digit),
    static_cast<uint8_t>(cycles),
    static_cast<uint8_t>(cycles >> 8),
    static_cast<uint8_t>(cycles >> 16),
    static_cast<uint8_t>(cycles >> 24),
  };
  Serial.write(response, sizeof(response));
}
