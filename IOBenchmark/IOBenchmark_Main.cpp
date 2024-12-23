/*
 * MIT License
 * 
 * Copyright (c) 2024 Dr. Thomas Lang
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
#include <chrono>
#include <cstdio>       // C file API
#include <filesystem>
#include <fstream>      // C++ fstream API
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "mio.hpp"      // C++ memory mapped file I/O

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

class RawIO
{
public:
  using Dimensions = std::array<long long, 3>;

  RawIO(Dimensions const& dims, std::string const& fname) noexcept
    : Dims(dims)
    , InputFilename(fname)
  {
    auto tempFPath = std::filesystem::temp_directory_path() / std::filesystem::path(fname).filename();
    this->OutputFilename = tempFPath.string();

    std::ofstream outf{this->OutputFilename.c_str(), std::ios::binary};
    outf.seekp(this->Dims[2] * this->SliceSizeBytes() - 1, std::ios::beg);
    outf.put('\0');
  }

  virtual ~RawIO()
  {
    std::filesystem::remove(this->OutputFilename);
  }

  template< bool Input >
  float FullSliceIO(int z, float* buffer) const
  {
    auto start = std::chrono::high_resolution_clock::now();
    this->FullSliceIO(z, buffer, std::conditional_t<Input, std::true_type, std::false_type>{});
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count(); // fractional milliseconds
  }

  virtual std::string GetName() const = 0;

protected:
  virtual void FullSliceIO(int z, float* target, std::true_type)  const = 0;
  virtual void FullSliceIO(int z, float* source, std::false_type) const = 0;

  inline size_t SliceSize() const
  {
    return this->Dims.at(0) * this->Dims.at(1);
  }

  inline size_t SliceSizeBytes() const
  {
    return this->SliceSize() * sizeof(unsigned short);
  }

  template< class From, class To >
  void Copy(To* target, From* source, long long bufSize) const
  {
    #pragma omp parallel for
    for (long long i = 0; i < bufSize; ++i) {
      target[i] = static_cast< To >(source[i]);
    }
  }

protected:
  Dimensions Dims;
  std::string InputFilename;
  std::string OutputFilename;
};

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

class CFileIO : public RawIO
{
public:

  using RawIO::RawIO;

protected:

  void FullSliceIO(int z, float* target, std::true_type) const override
  {
    size_t const sliceSize      = this->SliceSize();
    size_t const sliceSizeBytes = this->SliceSizeBytes();
    auto uint16Buf = std::make_unique< unsigned short[] >(sliceSize);

    std::FILE* file = std::fopen(this->InputFilename.c_str(), "rb");
    if (file) {
      std::fseek(file, z * sliceSizeBytes, SEEK_SET);
      std::fread(uint16Buf.get(), sizeof(unsigned short), sliceSize, file);
      std::fclose(file);
    }

    this->Copy(target, uint16Buf.get(), sliceSize);
  }

  void FullSliceIO(int z, float* source, std::false_type) const override
  {
    size_t const sliceSize      = this->SliceSize();
    size_t const sliceSizeBytes = this->SliceSizeBytes();
    auto uint16Buf = std::make_unique< unsigned short[] >(sliceSize);

    this->Copy(uint16Buf.get(), source, sliceSize);

    std::FILE* file = std::fopen(this->OutputFilename.c_str(), "r+b");
    if (file) {
      std::fseek(file, z * sliceSizeBytes, SEEK_SET);
      std::fwrite(uint16Buf.get(), sizeof(unsigned short), sliceSize, file);
      std::fclose(file);
    }
  }

  std::string GetName() const override
  {
    return "C file API";
  }
};

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

class FStreamIO : public RawIO
{
public:

  using RawIO::RawIO;

protected:

  void FullSliceIO(int z, float* target, std::true_type) const override
  {
    size_t const sliceSize      = this->SliceSize();
    size_t const sliceSizeBytes = this->SliceSizeBytes();
    auto uint16Buf = std::make_unique< unsigned short[] >(sliceSize);

    std::ifstream file{this->InputFilename.c_str(), std::ios::binary};
    file.seekg(z * sliceSizeBytes);
    file.read(reinterpret_cast< char* >(uint16Buf.get()), sliceSizeBytes);

    this->Copy(target, uint16Buf.get(), sliceSize);
  }

  void FullSliceIO(int z, float* source, std::false_type) const override
  {
    size_t const sliceSize      = this->SliceSize();
    size_t const sliceSizeBytes = this->SliceSizeBytes();
    auto uint16Buf = std::make_unique< unsigned short[] >(sliceSize);

    this->Copy(uint16Buf.get(), source, sliceSize);

    std::fstream file{this->OutputFilename.c_str(), std::ios::in | std::ios::out | std::ios::binary};
    file.seekp(z * sliceSizeBytes);
    file.write(reinterpret_cast< char const* >(uint16Buf.get()), sliceSizeBytes);
  }

  std::string GetName() const override
  {
    return "C++ fstream API";
  }
};

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

class MmapIO : public RawIO
{
public:

  using RawIO::RawIO;

protected:

  void FullSliceIO(int z, float* target, std::true_type) const override
  {
    size_t const sliceSize      = this->SliceSize();
    size_t const sliceSizeBytes = this->SliceSizeBytes();

    mio::mmap_source file(this->InputFilename.c_str(), z * sliceSizeBytes, sliceSizeBytes);
    this->Copy(target, reinterpret_cast< unsigned short const* >(file.data()), sliceSize);
  }

  void FullSliceIO(int z, float* source, std::false_type) const override
  {
    size_t const sliceSize      = this->SliceSize();
    size_t const sliceSizeBytes = this->SliceSizeBytes();

    mio::mmap_sink file(this->OutputFilename.c_str(), z * sliceSizeBytes, sliceSizeBytes);
    this->Copy(reinterpret_cast< unsigned short* >(file.data()), source, sliceSize);
  }

  std::string GetName() const override
  {
    return "C++ memory mapped files";
  }
};

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

void PrintBenchmarkResults(std::string const& api, float meanReadTime, float meanWriteTime)
{
#define FMT std::setw(8) << std::scientific << std::setprecision(2)
  std::cout << "  Consdering " << api << '\n';
  std::cout << "    o Average read  time: " << FMT << meanReadTime  << " [ms]\n";
  std::cout << "    o Average write time: " << FMT << meanWriteTime << " [ms]\n";
}


std::tuple< RawIO::Dimensions, std::string > ParseMhd(std::string const& mhdFile)
{
  RawIO::Dimensions volumeDims;
  std::string rawFile;

  std::string line;
  std::ifstream mhd{mhdFile};
  while (std::getline(mhd, line)) {
    if (line[0] == 'D') { // DimSize = dimX dimY dimZ
      std::string key;
      char eq;
      std::istringstream(line) >> key >> eq >> volumeDims[0] >> volumeDims[1] >> volumeDims[2];
    } else if (line[0] == 'E' && line[7] == 'D') { // ElementDataFile = path.raw
      rawFile = line.substr(18);
    }
  }

  return std::make_tuple(volumeDims, rawFile);
}

void BenchmarkFullSliceIO(RawIO::Dimensions const& volumeDims, std::string const& rawFile, int numberOfFullSlices)
{
  std::vector<int> zs(numberOfFullSlices);
  for (int i = 0; i < numberOfFullSlices; ++i) {
    zs[i] = static_cast< int >(static_cast< float >(i) * volumeDims[2] / numberOfFullSlices);
  }

  auto buffer = std::make_unique< float[] >(volumeDims.at(0) * volumeDims.at(1));

  auto benchmark = [zs, &buffer](std::unique_ptr< RawIO > const& io) {
    float meanReadTime = 0.f, meanWriteTime = 0.f;
    for (int z : zs) {
      meanReadTime  += io->FullSliceIO<true >(z, buffer.get());
      meanWriteTime += io->FullSliceIO<false>(z, buffer.get());
    }
    meanReadTime  /= zs.size();
    meanWriteTime /= zs.size();
    PrintBenchmarkResults(io->GetName(), meanReadTime, meanWriteTime);
  };

  std::vector<std::unique_ptr<RawIO>> ios;
  ios.emplace_back(new FStreamIO{volumeDims, rawFile});
  ios.emplace_back(new CFileIO{volumeDims, rawFile});
  ios.emplace_back(new MmapIO{volumeDims, rawFile});

  std::cout << "Full slice I/O:\n";
  for (auto const& io : ios) {
    benchmark(io);
  }
}

void BenchmarkCuboidRoIsIO(RawIO::Dimensions const& volumeDims, std::string const& rawFile, int numberOfCuboidRoIs)
{
}


void RunBenchmarks(std::string const& mhdFile, int numberOfFullSlices, int numberOfCuboidRoIs)
{
  RawIO::Dimensions volumeDims;
  std::string rawFile;
  std::tie(volumeDims, rawFile) = ParseMhd(mhdFile);

  BenchmarkFullSliceIO(volumeDims, rawFile, std::min<RawIO::Dimensions::value_type>(volumeDims[2], numberOfFullSlices));
  BenchmarkCuboidRoIsIO(volumeDims, rawFile, numberOfCuboidRoIs);
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <path/to/inputfile.mhd>  <number-full-slices>  <number-cubic-rois>\n\n";
    std::cerr << "  Extracts _and_ saves Regions of Interest of the following cases from the input file and to a temporary file.\n";
    std::cerr << "    (1) Extracts  <number-full-slices>  full slices from the file.\n";
    std::cerr << "    (2) Extracts  <number-cubic-rois>  of sizes KxKxK (K = 3,5,7,9) from the file.\n\n";
    std::cerr << "  <path/to/inputfile.mhd>  Path to an existing mhd file (3D, metadata, pointing to a .raw file). MUST be uint16.\n";
    std::cerr << "  <number-full-slices>     Number of full slices to extract/store. Must be > 0.\n";
    std::cerr << "  <number-cubic-rois>      Number of cubic RoIs to extract/store. Must be > 0.\n";
    return EXIT_FAILURE;
  }

  auto parseInt = [](char const* arg) -> int {
    int value;
    std::istringstream(arg) >> value;
    return value;
  };

  RunBenchmarks(argv[1], parseInt(argv[2]), parseInt(argv[3]));
  return EXIT_SUCCESS;
}