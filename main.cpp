#include <chrono>
#include <format>
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
  // if user did not par arguments
  if (argc < 2) {
    std::cerr << "Error: No mode " << std::endl;
    printHelp(argv[0]);
    return 1;
  }
  std::string mode = argv[1];

  // if user needs help
  if (mode == "-h" || mode == "--help") {
    printHelp(argv[0]);
    return 0;

    // if user passed invalid argument
  } else if (mode != "-n" && mode != "--naive" && mode != "-im" &&
             mode != "--im2col" && mode != "-i" && mode != "--interactive") {
    std::cerr << "Error: Invalid mode '" << mode << std::endl;
    printHelp(argv[0]);
    return 1;
  }

  size_t N, C, H, W, K;
  std::cin >> N;
  std::cin >> C;
  std::cin >> H;
  std::cin >> W;
  std::array input_dim = {N, C, H, W};
  std::vector<float> input_vector(Dim::length(input_dim));

  // read elements of input tensor
  for (size_t i = 0; i < input_vector.size(); ++i) std::cin >> input_vector[i];

  std::cin >> K;
  std::array kernel_dim = {C, C, K, K};
  std::vector<float> kernel_vector(Dim::length(kernel_dim));

  // read elements of kernel tensor
  for (size_t i = 0; i < kernel_vector.size(); ++i)
    std::cin >> kernel_vector[i];

  Linalg::Tensor input{input_dim, input_vector};
  Linalg::Tensor kernel{kernel_dim, kernel_vector};

  // Test naive convolution
  if (mode == "-n" || mode == "--naive") {
    Linalg::Tensor result = Convolution::conv_naive(input, kernel);

    // print results
    for (size_t i = 0; i < 4; ++i) std::cout << result.dimension(i) << ' ';
    std::cout << std::endl;
    std::string float_number;
    for (float elem : result.data()) {
      float_number = std::format("{:.6f}", elem);
      float_number.pop_back();
      std::cout << float_number << ' ';
    }
    std::cout << std::endl;
    return 0;

    // Test im2col convolution
  } else if (mode == "-im" || mode == "--im2col") {
    Linalg::Tensor result = Convolution::conv_im2col(input, kernel);

    // print results
    for (size_t i = 0; i < 4; ++i) std::cout << result.dimension(i) << ' ';
    std::cout << std::endl;
    std::string float_number;
    for (float elem : result.data()) {
      float_number = std::format("{:.6f}", elem);
      float_number.pop_back();
      std::cout << float_number << ' ';
    }
    std::cout << std::endl;
    return 0;

    // interactive mode (show speed)
  } else if (mode == "-i" || mode == "--interactive") {
    // Measure naive time:
    auto start = std::chrono::high_resolution_clock::now();
    Linalg::Tensor result1 = Convolution::conv_im2col(input, kernel);
    auto end = std::chrono::high_resolution_clock::now();
    long timeNaive =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    // Measue im2col time:
    start = std::chrono::high_resolution_clock::now();
    Linalg::Tensor result2 = Convolution::conv_im2col(input, kernel);
    end = std::chrono::high_resolution_clock::now();
    long timeIm2col =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    // print time:
    std::cout << "Naive time measured: " << timeNaive << " ms" << std::endl;
    std::cout << "Im2col time measured: " << timeIm2col << " ms" << std::endl;
    // print results:
    char answer;
    std::cout << "Would you like to see the results? [Y/n]";
    std::cin >> answer;
    if (answer == 'y') {
      for (size_t i = 0; i < 4; ++i) std::cout << result1.dimension(i) << ' ';
      std::cout << std::endl;
      for (float elem : result1.data()) std::cout << elem << ' ';
      std::cout << std::endl;

      for (size_t i = 0; i < 4; ++i) std::cout << result2.dimension(i) << ' ';
      std::cout << std::endl;
      for (float elem : result2.data()) std::cout << elem << ' ';
      std::cout << std::endl;
    }
    return 0;
  }
}