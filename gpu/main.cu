#include <stdio.h>
#include <vector>

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "../mandel.h"
#include "../timer.h"

static __global__ void mandel_kernel(char *image, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        mandel(image, width, height, x, y);
    }
}

int main() {
    auto timer = Timer();
    const auto width = 8400;
    const auto height = 4800;
    auto image = std::vector<char>(width * height * 3);
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);
    auto device_image = thrust::device_vector<char>(image.size());
    printf("[%f] Init\n", timer.measure());
    mandel_kernel<<<gridDim, blockDim, 0>>>(thrust::raw_pointer_cast(&device_image[0]), width, height);
    thrust::copy(device_image.begin(), device_image.end(), image.begin());
    printf("[%f] Kernel (GPU)\n", timer.measure());
    stbi_write_png("image.png", width, height, 3, &image[0], width * 3);
    printf("[%f] .png\n", timer.measure());
    return 0;
}
