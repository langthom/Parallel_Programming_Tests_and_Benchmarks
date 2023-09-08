/*
 * MIT License
 * 
 * Copyright (c) 2023 Dr. Thomas Lang
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
**/
#define NOMINMAX

#include <array>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include "../Common/MultiGPUExecution.h"
#include "../Common/CommonFunctions.h"
#include "../Common/CommonKernels.h"


std::unique_ptr< float[] > applyToData(std::ostream& out, std::array< int64_t, 3 > const& dims, std::unique_ptr< float[] >& inputData, int K) {
  int nr_gpus = 0;
  std::vector< std::string > deviceNames;
  std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
  cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

  int64_t padding            = K - 1;
  int64_t size               = dims[0] * dims[1] * dims[2];
  int64_t sizeWithoutPadding = (dims[0] - padding) * (dims[1] - padding) * (dims[2] - padding);
  auto offsets               = computeMaskOffsets(K, dims[1], dims[0]);

  auto outputData = std::make_unique< float[] >(sizeWithoutPadding);

  if (error != cudaError_t::cudaSuccess) {
    out << "[CUDA] Error during GPU information retrieval, error was: " << cudaGetErrorString(error) << '\n';
    return nullptr;
  } else {

    int maxPotentialCUDABlockSize = 1;
    cudaError_t blockSizeRetErr = getMaxPotentialBlockSize(maxPotentialCUDABlockSize, 0/* multi GPU system: use the first device here only! */);
    if (blockSizeRetErr != cudaError_t::cudaSuccess) {
      out << "[CUDA]   *** Error during retrieval of maximum potential block size: " << cudaGetErrorString(blockSizeRetErr) << '\n';
      return nullptr;
    }

    float elapsedTimeInMilliseconds = -1;
    cudaError_t cuda_kernel_error = launchKernelMultiCUDAGPU(outputData.get(), inputData.get(), size, offsets.data(), K, dims[0], dims[1], dims[2], &elapsedTimeInMilliseconds, maxPotentialCUDABlockSize);
    if (cuda_kernel_error != cudaError_t::cudaSuccess) {
      out << "[CUDA]  Error during Kernel launch, error was '" << cudaGetErrorString(cuda_kernel_error) << "'\n";
      return nullptr;
    }

    out << "Averaging on GPU(s) took " << elapsedTimeInMilliseconds << " milliseconds.\n";
  }

  return outputData;
}


std::unique_ptr< float[] > parseMHD(std::array< int64_t, 3 >& dims, std::string const& mhdPath) {
  //Asserts ushort data, binary encoded, 3D, uncompressed!

  std::string dataFile;

  std::ifstream mhd{ mhdPath };
  assert(mhd.good());
  std::string line;
  while (std::getline(mhd, line)) {
    if (line[0] == 'D') { // "DimSize = "
      std::istringstream iss{ line.substr(10) };
      iss >> dims[0] >> dims[1] >> dims[2];
    } else if (line.size() > 18 && line.substr(0, 18).compare("ElementDataFile = ") == 0) { // "ElementDataFile = "
      dataFile = line.substr(18);
    }
  }

  int64_t size = dims[0] * dims[1] * dims[2];
  auto data = std::make_unique< float[] >(size);
  {
    auto tmpData = std::make_unique< unsigned short[] >(size);
    std::ifstream dataStream{ dataFile, std::ios_base::binary };
    dataStream.read(reinterpret_cast< char* >(tmpData.get()), size * sizeof(unsigned short));
    std::copy_n(tmpData.get(), size, data.get());
  }
  
  return data;
}


void writeMHD(std::string const& outputMHD, std::string const& inputMHD, std::array< int64_t, 3 > dims, int K, std::unique_ptr< float[] > const& data) {

  std::string outDataFile = outputMHD.substr(0, outputMHD.size() - 4) + ".raw";
  int64_t padding = K - 1;
  dims[0] -= padding;
  dims[1] -= padding;
  dims[2] -= padding;

  std::ifstream inMHD{ inputMHD };
  std::ofstream outMHD{ outputMHD };
  assert(inMHD.good()); assert(outMHD.good());
  std::string line;
  while (std::getline(inMHD, line)) {
    if (line.size() > 18 && line.substr(0, 18).compare("ElementDataFile = ") == 0) { // "ElementDataFile = "
      outMHD << "ElementDataFile = " << outDataFile << "\n";
    } else if (line[0] == 'D') { // "DimSize = "
      outMHD << "DimSize = " << dims[0] << " " << dims[1] << " " << dims[2] << "\n";
    } else {
      outMHD << line << "\n";
    }
  }
  outMHD.close();

  std::ofstream dataStream{ outDataFile, std::ios_base::binary };
  int64_t size = dims[0] * dims[1] * dims[2];
  auto tmpData = std::make_unique< unsigned short[] >(size);
  std::copy_n(data.get(), size, tmpData.get());
  dataStream.write(reinterpret_cast< char const* >(tmpData.get()), size * sizeof(unsigned short));
}


/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <path/to/input/mhd> <K>\n\n";
    std::cerr << "  <path/to/input/mhd>  File path to the MHD file serving as input, also determines where the output MHD file will be generated.\n";
    std::cerr << "  <K>                  Environment size, must be bigger than one and odd.\n";
    return EXIT_FAILURE;
  }

  std::string inputMHD = argv[1];

  int K = 1;
  std::stringstream(argv[2]) >> K;

  std::array< int64_t, 3 > dims;

  auto parsingStart = std::chrono::high_resolution_clock::now();
  auto data = parseMHD(dims, inputMHD);
  auto parsingEnd = std::chrono::high_resolution_clock::now();
  std::cout << "Parsing and reading data took " << std::chrono::duration_cast< std::chrono::milliseconds >(parsingEnd - parsingStart).count() << " milliseconds.\n";

  auto outputData = applyToData(std::cout, dims, data, K);

  if (outputData) {
    auto writingStart = std::chrono::high_resolution_clock::now();
    writeMHD(inputMHD.substr(0, inputMHD.size() - 4) + "_avg" + std::to_string(K) + ".mhd", inputMHD, dims, K, outputData);
    auto writingEnd = std::chrono::high_resolution_clock::now();
    std::cout << "Writing data took " << std::chrono::duration_cast< std::chrono::milliseconds >(writingEnd - writingStart).count() << " milliseconds.\n";
  }

  return EXIT_SUCCESS;
}