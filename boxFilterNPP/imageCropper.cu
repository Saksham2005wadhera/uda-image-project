#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

// Kernel for converting RGB to Grayscale
__global__ void rgbToGrayKernel(unsigned char* rgb, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        int rgbIdx = idx * 3;
        unsigned char r = rgb[rgbIdx];
        unsigned char g = rgb[rgbIdx + 1];
        unsigned char b = rgb[rgbIdx + 2];
        gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// Kernel for cropping the image
__global__ void cropKernel(unsigned char* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight, int offsetX, int offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight) {
        int srcIdx = (y + offsetY) * srcWidth + (x + offsetX);
        int dstIdx = y * dstWidth + x;
        dst[dstIdx] = src[srcIdx];
    }
}

// Kernel for adjusting shadows and highlights
__global__ void adjustShadowsHighlightsKernel(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        unsigned char pixel = img[idx];
        if (pixel <= 128) {
            img[idx] = static_cast<unsigned char>(min(255.0f, pixel * 1.1f)); // Increase shadows by 10%
        } else {
            img[idx] = static_cast<unsigned char>(max(0.0f, pixel * 1.05f)); // increase highlights by 5%
        }
    }
}

// Function to load a PPM image (color)
void loadPPM(const std::string& filename, unsigned char*& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + filename);
    }

    std::string magic;
    file >> magic;
    if (magic != "P6") {
        throw std::runtime_error("Invalid PPM file");
    }

    file >> width >> height;

    int maxVal;
    file >> maxVal;
    file.ignore(256, '\n'); // Skip to the next line

    data = new unsigned char[width * height * 3];
    file.read(reinterpret_cast<char*>(data), width * height * 3);

    file.close();
}

// Function to save a PGM image
void savePGM(const std::string& filename, unsigned char* data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + filename);
    }

    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char*>(data), width * height);

    file.close();
}

int main(int argc, char* argv[]) {
    printf("%s Starting...\n\n", argv[0]);

    try {
        std::vector<std::string> inputFiles = {
            "img1.ppm", "img2.ppm", "img3.ppm", "img4.ppm", "img5.ppm",
            "img6.ppm", "img7.ppm", "img8.ppm", "img9.ppm", "img10.ppm"
        };

        std::string outputDir = "processed_images";
        // Create the output directory
        #if defined(_WIN32)
            _mkdir(outputDir.c_str());
        #else 
            mkdir(outputDir.c_str(), 0755);
        #endif

        for (const auto& sFilename : inputFiles) {
            unsigned char* h_rgb = nullptr;
            int width, height;
            loadPPM(sFilename, h_rgb, width, height);

            unsigned char* h_gray = new unsigned char[width * height];
            unsigned char* d_rgb;
            unsigned char* d_gray;

            checkCudaErrors(cudaMalloc(&d_rgb, width * height * 3 * sizeof(unsigned char)));
            checkCudaErrors(cudaMalloc(&d_gray, width * height * sizeof(unsigned char)));

            checkCudaErrors(cudaMemcpy(d_rgb, h_rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 blockSize(16, 16);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
            rgbToGrayKernel<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);

            checkCudaErrors(cudaMemcpy(h_gray, d_gray, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

            int shorterSide = std::min(width, height);
            int longerSide = std::max(width, height);
            int offset = (longerSide - shorterSide) / 2;

            int dstWidth = shorterSide;
            int dstHeight = shorterSide;

            unsigned char* h_dst = new unsigned char[dstWidth * dstHeight];
            unsigned char* d_dst;

            checkCudaErrors(cudaMalloc(&d_dst, dstWidth * dstHeight * sizeof(unsigned char)));
            checkCudaErrors(cudaMemcpy(d_gray, h_gray, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

            cropKernel<<<gridSize, blockSize>>>(d_gray, d_dst, width, height, dstWidth, dstHeight, offset, 0);

            // Adjust shadows and highlights
            adjustShadowsHighlightsKernel<<<gridSize, blockSize>>>(d_dst, dstWidth, dstHeight);

            checkCudaErrors(cudaMemcpy(h_dst, d_dst, dstWidth * dstHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost));

            std::string outputFilename = outputDir + "/" + sFilename.substr(0, sFilename.find_last_of('.')) + "_cropped_adjusted.pgm";
            savePGM(outputFilename, h_dst, dstWidth, dstHeight);
            std::cout << "Saved image: " << outputFilename << std::endl;

            delete[] h_rgb;
            delete[] h_gray;
            delete[] h_dst;
            cudaFree(d_rgb);
            cudaFree(d_gray);
            cudaFree(d_dst);
        }

        exit(EXIT_SUCCESS);
    } catch (std::exception& e) {
        std::cerr << "Program error! The following exception occurred: \n" << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        exit(EXIT_FAILURE);
    }

    return 0;
}
