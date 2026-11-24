#define _DEFAULT_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

static const char serial_path[] = "/dev/ttyACM0";
static const char images_path[] = "server/mnist-images.bin";
static const char labels_path[] = "server/mnist-labels.bin";

static int configure_serial(int fd) {
    struct termios config;
    if (tcgetattr(fd, &config) == -1) return -1;

    cfmakeraw(&config);
    config.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS | CSIZE);
    config.c_cflag |= CS8 | CLOCAL | CREAD;
    config.c_iflag &= ~(IXON | IXOFF | IXANY);
    config.c_cc[VMIN] = 1;
    config.c_cc[VTIME] = 0;

    if (cfsetispeed(&config, B115200) == -1 || cfsetospeed(&config, B115200) == -1) return -1;

    return tcsetattr(fd, TCSANOW, &config);
}

static int transfer(int fd, uint8_t *bytes, size_t length, int sending) {
    struct pollfd port = { .fd = fd, .events = sending ? POLLOUT : POLLIN };
    while (length > 0) {

        int ready = poll(&port, 1, 5000);

        if (ready == -1 && errno == EINTR) continue;
        if (ready <= 0) {
            if (ready == 0) errno = ETIMEDOUT;
            return -1;
        }
        if (!(port.revents & port.events)) {
            errno = EIO;
            return -1;
        }

        ssize_t count = sending ? write(fd, bytes, length) : read(fd, bytes, length);

        if (count == -1 && (errno == EINTR || errno == EAGAIN)) continue;
        if (count <= 0) {
            if (count == 0) errno = EIO;
            return -1;
        }

        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int benchmark(int fd, FILE *images, FILE *labels) {

    const double cycles_per_ms = 16000.0; // Arduino Uno R3, 16 MHz
    uint8_t request[1 + 28 * 28] = {'I'};
    uint8_t response[6];
    uint64_t total_cycles = 0;
    uint32_t min_cycles = UINT32_MAX;
    uint32_t max_cycles = 0;
    size_t count = 0;
    size_t correct = 0;

    while (count < 1000) {
        int label = fgetc(labels);
        if (label == EOF || label > 9 || fread(request + 1, 1, sizeof(request) - 1, images) != sizeof(request) - 1) {
            fprintf(stderr, "Missing or invalid MNIST data at sample %zu.\n", count);
            return -1;
        }
        if (transfer(fd, request, sizeof(request), 1) == -1 || transfer(fd, response, sizeof(response), 0) == -1) {
            perror("Serial transfer");
            return -1;
        }
        if (response[0] != 'R' || response[1] > 9) {
            fprintf(stderr, "Invalid response at sample %zu.\n", count);
            return -1;
        }

        uint32_t cycles = (uint32_t)response[2] | ((uint32_t)response[3] << 8) | ((uint32_t)response[4] << 16) | ((uint32_t)response[5] << 24);

        printf("%zu  expected=%d  predicted=%u  cycles=%" PRIu32 "  time=%.3f ms\n", count, label, (unsigned)response[1], cycles, cycles / cycles_per_ms);
        count++;
        correct += response[1] == label;
        total_cycles += cycles;
        if (cycles < min_cycles) min_cycles = cycles;
        if (cycles > max_cycles) max_cycles = cycles;
    }

    printf("Sample accuracy = %zu/%zu (%.1f%%)\n", correct, count, 100.0 * correct / count);
    printf("Mean cycles = %.1f  min = %" PRIu32 "  max = %" PRIu32 "\n",
           (double)total_cycles / count, min_cycles, max_cycles);
    printf("Mean time = %.3f ms  min = %.3f ms  max = %.3f ms\n",
           (double)total_cycles / count / cycles_per_ms, min_cycles / cycles_per_ms, max_cycles / cycles_per_ms);
    return 0;
}

int main(void) {
    FILE *images = fopen(images_path, "rb");
    if (!images) {
        perror(images_path);
        return EXIT_FAILURE;
    }
    FILE *labels = fopen(labels_path, "rb");
    if (!labels) {
        perror(labels_path);
        fclose(images);
        return EXIT_FAILURE;
    }

    int fd = open(serial_path, O_RDWR | O_NOCTTY | O_NONBLOCK);
    int result = EXIT_FAILURE;
    if (fd == -1 || configure_serial(fd) == -1) {
        perror(serial_path);
    } else {

        sleep(2);
        uint8_t byte = 'P';
        if (tcflush(fd, TCIOFLUSH) == -1 || transfer(fd, &byte, 1, 1) == -1 || transfer(fd, &byte, 1, 0) == -1) {
            perror("Arduino Uno synchronization");
        } else if (byte != 'A') {
            fprintf(stderr, "Unexpected synchronization response.\n");
        } else if (benchmark(fd, images, labels) == 0) {
            result = EXIT_SUCCESS;
        }
    }

    if (fd != -1) close(fd);
    fclose(labels);
    fclose(images);
    return result;
}
