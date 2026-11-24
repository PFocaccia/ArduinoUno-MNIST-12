# ArduinoUno-MNIST-12

**Handwritten digit recognition on an 8-bit Arduino Uno R3 with 2 KB of SRAM.**

This project runs the ONNX Model Zoo's **INT8 MNIST-12** convolutional network directly on the ATmega328P. A Linux host streams 28 × 28 grayscale images over USB serial. The Uno runs the full forward pass and returns a digit and a hardware cycle count.

The Uno correctly classified **986 of the first 1,000 MNIST test images (98.6%)**, with a mean inference time of **989.368 ms** at 16 MHz.

## Benchmark results

| Metric | Mean | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| CPU cycles per inference | 15,829,886 | 15,791,529 | 15,856,746 |
| Inference time | 989.368 ms | 986.971 ms | 991.047 ms |

## Memory footprint

The firmware fits within the memory constraints of the ATmega328P without requiring external memory.

| Memory | Used | Available | Utilization |
| --- | ---: | ---: | ---: |
| Program flash | 10,472 B | 32,256 B | 32.5% |
| Static SRAM | 1,770 B | 2,048 B | 86.4% |

The compiled firmware leaves **278 bytes of SRAM** available for the stack and other runtime memory requirements.

The two main static buffers account for 1,584 bytes:

- **Input image buffer:** 1,024 bytes (32 × 32).
- **Circular activation buffer:** 560 bytes (5 × 14 × 8).

Memory usage is reported by Arduino CLI at compile time. The SRAM figure represents static memory allocation and does not include peak stack usage.

## How it fits

- **Keep the weights in flash.** The first convolution copies one 5 × 5 filter into SRAM and reuses it across pooling windows. The second convolution reads its weights from flash in channel order, using incrementing pointers for both weights and activations.
- **Keep five rows of activations.** A 560-byte circular buffer holds five rows of the first max-pool output, with 14 columns and eight channels per row. As the second convolution advances, the firmware computes one new row and replaces the oldest.
- **Reuse the image buffer.** The input occupies 1,024 bytes in SRAM, including two pixels of zero padding on each side. The second max pooling writes its 256 output bytes into the part of that buffer the first convolution has finished reading.
- **Use integer arithmetic throughout.** Convolutions and the dense layer multiply unsigned 8-bit activations by signed 8-bit weights and accumulate in 32 bits. Precomputed integer multipliers and shifts handle requantization. The host compares the resulting predictions with the dataset labels.

## Host and device

The C program on the PC drives the benchmark. It configures the serial port with POSIX `termios`, sends one image, waits for the Uno's response, and repeats. Serial transfers use `poll()` with timeout handling and support partial reads and writes.

The firmware receives each row directly into its padded image buffer. Timer1 measures the forward pass, including the dense layer, through direct hardware register configuration and an overflow interrupt service routine.

The time reported by the host is calculated from the cycle count at the Uno's nominal 16 MHz clock. Serial transfers happen outside the measured interval.

The binary protocol runs at **115200 baud, 8N1**.

| Host sends | Uno returns |
| --- | --- |
| `P` | `A`, confirming the sketch is ready |
| `I` followed by 784 pixel bytes | `R`, one predicted digit byte, and a 32-bit little-endian cycle count |

## Run the benchmark

You need an **Arduino Uno R3**, a USB data cable, **Arduino CLI**, and a Linux PC with **GCC**. Run these commands from the repository root. The Arduino AVR board support package is installed locally by Arduino CLI.

```bash
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli compile --fqbn arduino:avr:uno --output-dir cnn/build cnn
arduino-cli upload --fqbn arduino:avr:uno --port /dev/ttyACM0 \
  --input-file cnn/build/cnn.ino.hex
```

The firmware and host both use 115200 baud. If the device appears at a different path, use that path for upload and change `serial_path` in `server/server.c`. Make sure your user has permission to open the port and close any serial monitor before running the benchmark.

```bash
gcc -std=c11 -O2 -Wall -Wextra server/server.c -o server/mnist-bench
./server/mnist-bench
```

The program prints the expected digit, predicted digit, cycles, and milliseconds for each sample. It then reports sample accuracy and the mean, minimum, and maximum inference time. At roughly one second per inference, a 1,000-image run takes around 18 minutes including transfers.

## Repository layout

| File | What it contains |
| --- | --- |
| `cnn/cnn.ino` | Serial request handling, padded input buffer, Timer1 measurement, and binary replies |
| `cnn/conv.cpp`, `cnn/conv.h` | Convolutions, max pooling, circular buffering, requantization, and dense classification |
| `cnn/weights.h` | Model weights and second-convolution biases stored in flash |
| `server/server.c` | Linux serial host and benchmark statistics |
| `server/mnist-images.bin`, `server/mnist-labels.bin` | The first 1,000 test images and their labels |
| `THIRD_PARTY_NOTICES.md` | Model and dataset attribution and licensing information |

## MNIST sample format

The two sample files are derived from the original compressed MNIST test IDX files. The IDX headers were removed and the first 1,000 records were copied in their original order, preserving the pixel and label values.

- `mnist-images.bin` holds **784,000 bytes**. Image `i` starts at offset `784 × i` and contains 784 unsigned grayscale bytes in row-major order.
- `mnist-labels.bin` holds **1,000 bytes**. Byte `i` is the numeric label, from 0 to 9, for image `i`.

## Credits

The pretrained model and its INT8 export come from the [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist). The model card credits the Intel contributors who prepared the quantized export. The benchmark samples come from the [MNIST database](https://github.com/cvdfoundation/mnist) by Yann LeCun and Corinna Cortes, using the CVDF mirror. See [Third-party notices](THIRD_PARTY_NOTICES.md) for the applicable notices and details of the local conversions.
