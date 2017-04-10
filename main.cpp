#include <algorithm>
#include <math.h>
#include <tuple>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

static std::tuple<double, double> image_to_viewport(
        int image_width, int image_height, std::tuple<int, int> image_pt,
        std::tuple<double, double> viewport_tl, std::tuple<double, double> viewport_br) {
    int image_x, image_y;
    std::tie(image_x, image_y) = image_pt;
    double viewport_left, viewport_top, viewport_right, viewport_bottom;
    std::tie(viewport_left, viewport_top) = viewport_tl;
    std::tie(viewport_right, viewport_bottom) = viewport_br;
    auto viewport_width = viewport_right - viewport_left;
    auto viewport_height = viewport_bottom - viewport_top;
    return std::make_tuple(
        (static_cast<double>(image_x) * viewport_width) / static_cast<double>(image_width) + viewport_left,
        (static_cast<double>(image_y) * viewport_height) / static_cast<double>(image_height) + viewport_top);
}

static void mandel(char *image, int image_width, int image_height, int image_x, int image_y) {
    /*
     *    -2.5    +1.0
     * -1.0 +-------+
     *      |       |
     *      |       |
     *      |       |
     * +1.0 +-------+
     */

    double x0, y0;
    std::tie(x0, y0) = image_to_viewport(image_width, image_height, std::make_tuple(image_x, image_y), std::make_tuple(-2.5, -1.0), std::make_tuple(1.0, 1.0));

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
        auto nu = log(log_zn / log(2)) / log(2);
        iteration += 1 - nu;
    }

    /*
    auto color1 = palette[floor(iteration)];
    auto color2 = palette[floor(iteration) + 1];
    auto color = linear_interpolate(color1, color2, iteration % 1);
    plot(Px, Py, color);
    */
    auto image_ptr = image + (image_y * image_width + image_x) * 3;
    image_ptr[0] = image_ptr[1] = image_ptr[2] = iteration;
}


int main() {
    const auto width = 640;
    const auto height = 480;
    auto image = std::vector<char>(width * height * 3);
    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            mandel(&image[0], width, height, x, y);
        }
    }

    stbi_write_png("image.png", width, height, 3, &image[0], width * 3);
    return 0;
}
