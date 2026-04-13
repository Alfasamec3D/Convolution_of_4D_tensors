#include <chrono>
#include <iostream>

#include "convolution.hpp"

void printHelp(const char* programName) {
  std::cout << "Exploitation: " << programName << " [mode]\n"
            << "Modes:\n"
            << "  -n,  --naive            Test naive convolution\n"
            << "  -im, --im2col           Test im2col convolution\n"
            << "  -i,  --interactive      Interactive data input, output\n"
            << "  -h,  --help             Show this help\n"
            << std::endl;
}

int main(int argc, char* argv[]) {
  size_t N, C, H, W, K;
  std::cin >> N;
  std::cin >> C;
  std::cin >> H;
  std::cin >> W;
  std::array input_dim = {N, C, H, W};
  std::vector<float> input_vector(Dim::length(input_dim));
  for (size_t i; i < input_vector.size(); ++i) std::cin >> input_vector[i];

  std::cin >> K;
  std::array kernel_dim = {C, C, K, K};
  std::vector<float> kernel_vector(Dim::length(kernel_dim));
  for (size_t i; i < kernel_vector.size(); ++i) std::cin >> kernel_vector[i];

  Linalg::Tensor input{input_dim, input_vector};
  Linalg::Tensor kernel{kernel_dim, kernel_vector};

  if (argc < 2) {
    printHelp(argv[0]);
    return 1;
  }

  std::string mode = argv[1];

  if (mode == "-n" || mode == "--naive") {
    Linalg::Tensor result = Convolution::conv_naive(input, kernel);

    for (size_t i = 0; i < 4; ++i) std::cout << result.dimension(i) << ' ';
    std::cout << std::endl;
    for (float elem : result.data()) std::cout << elem << ' ';
    std::cout << std::endl;
    return 0;
  } else if (mode == "-im" || mode == "--im2col") {
    Linalg::Tensor result = Convolution::conv_im2col(input, kernel);

    for (size_t i = 0; i < 4; ++i) std::cout << result.dimension(i) << ' ';
    std::cout << std::endl;
    for (float elem : result.data()) std::cout << elem << ' ';
    std::cout << std::endl;
    return 0;
  } else if (mode == "-i" || mode == "--interactive") {
    auto start = std::chrono::high_resolution_clock::now();
    Linalg::Tensor result1 = Convolution::conv_im2col(input, kernel);
    auto end = std::chrono::high_resolution_clock::now();
    long timeNaive =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    for (size_t i = 0; i < 4; ++i) std::cout << result1.dimension(i) << ' ';
    std::cout << std::endl;
    for (float elem : result1.data()) std::cout << elem << ' ';
    std::cout << std::endl;
    start = std::chrono::high_resolution_clock::now();
    Linalg::Tensor result2 = Convolution::conv_im2col(input, kernel);
    end = std::chrono::high_resolution_clock::now();
    long timeIm2col =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    for (size_t i = 0; i < 4; ++i) std::cout << result2.dimension(i) << ' ';
    std::cout << std::endl;
    for (float elem : result2.data()) std::cout << elem << ' ';
    std::cout << std::endl;

    std::cout << "Naive time measured: " << timeNaive << " ms" << std::endl;
    std::cout << "Im2col time measured: " << timeIm2col << " ms" << std::endl;
    return 0;
  } else if (mode == "-h" || mode == "--help") {
    printHelp(argv[0]);
    return 0;
  } else {
    std::cerr << "Error: Invalid mode '" << mode << std::endl;
    printHelp(argv[0]);
    return 1;
  }
}