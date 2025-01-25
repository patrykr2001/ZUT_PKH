#include <iostream>
#include <vector>
#include <algorithm>
extern "C" {
#include "qdbmp.h"
}

void apply_median_filter_cpu(BMP* bmp) {
    UINT width = BMP_GetWidth(bmp);
    UINT height = BMP_GetHeight(bmp);
    BMP* output = BMP_Create(width, height, BMP_GetDepth(bmp));

    for (UINT x = 1; x < width - 1; ++x) {
        for (UINT y = 1; y < height - 1; ++y) {
            std::vector<UCHAR> r_values, g_values, b_values;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    UCHAR r, g, b;
                    BMP_GetPixelRGB(bmp, x + dx, y + dy, &r, &g, &b);
                    r_values.push_back(r);
                    g_values.push_back(g);
                    b_values.push_back(b);
                }
            }
            std::sort(r_values.begin(), r_values.end());
            std::sort(g_values.begin(), g_values.end());
            std::sort(b_values.begin(), b_values.end());
            BMP_SetPixelRGB(output, x, y, r_values[4], g_values[4], b_values[4]);
        }
    }

    BMP_WriteFile(output, "output_cpu.bmp");
    BMP_Free(output);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input BMP file>" << std::endl;
        return 1;
    }

    BMP* bmp = BMP_ReadFile(argv[1]);
    if (BMP_GetError() != BMP_OK) {
        std::cerr << "Error reading BMP file!" << std::endl;
        return 1;
    }

    apply_median_filter_cpu(bmp);
    BMP_Free(bmp);

    return 0;
}