#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#if __NVCC__
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#define DEVICE __device__
#else
#define DEVICE
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

static DEVICE void image_to_viewport(
    int image_width,
    int image_height,
    int image_x,
    int image_y,
    double viewport_left,
    double viewport_top,
    double viewport_right,
    double viewport_bottom,
    double& viewport_x,
    double& viewport_y
) {
    auto viewport_width = viewport_right - viewport_left;
    auto viewport_height = viewport_bottom - viewport_top;
    viewport_x = (static_cast<double>(image_x) * viewport_width) / static_cast<double>(image_width) + viewport_left;
    viewport_y = (static_cast<double>(image_y) * viewport_height) / static_cast<double>(image_height) + viewport_top;
}

template<typename T> static DEVICE void mandel(T& image, int image_width, int image_height, int image_x, int image_y) {
    /*
     *    -2.5    +1.0
     * -1.0 +-------+
     *      |       |
     *      |       |
     *      |       |
     * +1.0 +-------+
     */

    double x0, y0;
    image_to_viewport(image_width, image_height, image_x, image_y, -2.5, -1.0, 1.0, 1.0, x0, y0);

    auto x = 0.0, y = 0.0;
    auto iteration = 0, max_iteration = 1000;
    while (x * x + y * y < (1 << 16) && iteration < max_iteration) {
        auto xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        iteration++;
    }

    if (iteration < max_iteration) {
        auto log_zn = log(x * x + y * y) / 2;
        auto nu = log(log_zn / log(2.0)) / log(2.0);
        iteration += 1 - nu;
    }

    /*
    auto color1 = palette[floor(iteration)];
    auto color2 = palette[floor(iteration) + 1];
    auto color = linear_interpolate(color1, color2, iteration % 1);
    plot(Px, Py, color);
    */
    auto offset = (image_y * image_width + image_x) * 3;
    image[offset] = image[offset + 1] = image[offset + 2] = iteration;
}

#ifdef __NVCC__
static __global__ void mandel_kernel(char *image, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        mandel(image, width, height, x, y);
    }
}
#endif

int main() {
    const auto width = 8400;
    const auto height = 4800;
    auto image = std::vector<char>(width * height * 3);

#ifdef __NVCC__
    printf("Using GPU\n");
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);
    auto device_image = thrust::device_vector<char>(image.capacity());
    mandel_kernel<<<gridDim, blockDim, 0>>>(thrust::raw_pointer_cast(&device_image[0]), width, height);
    thrust::copy(device_image.begin(), device_image.end(), image.begin());
#else
    printf("Using CPU\n");
    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            mandel(image, width, height, x, y);
        }
    }
#endif

    stbi_write_png("image.png", width, height, 3, &image[0], width * 3);
    return 0;
}
