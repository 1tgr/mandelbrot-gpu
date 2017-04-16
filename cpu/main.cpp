#include <stdio.h>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "../mandel.h"
#include "../timer.h"

int main() {
    auto timer = Timer();
    const auto width = 8400;
    const auto height = 4800;
    auto image = std::vector<char>(width * height * 3);
    printf("[%f] Init\n", timer.measure());

    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            mandel(image, width, height, x, y);
        }
    }

    printf("[%f] Kernel (CPU)\n", timer.measure());
    stbi_write_png("image.png", width, height, 3, &image[0], width * 3);
    printf("[%f] .png\n", timer.measure());
    return 0;
}
