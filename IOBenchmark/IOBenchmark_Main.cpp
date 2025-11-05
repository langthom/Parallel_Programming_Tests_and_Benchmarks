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
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "mio.hpp"      // C++ memory mapped file I/O

#include "ChunkedCopier.h"


/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

class RawIO
{
public:
  using Dimensions = std::array<long long, 3>;

  enum class IOMethod {
    C_FILE, CXX_FSTREAM, CXX_MMAP
  };


  template< IOMethod Method >
  static std::pair<float, float> BenchmarkFullSliceIO(
    Dimensions const& dims,
    std::string const& fname,
    std::vector<std::pair<Dimensions::value_type,Dimensions::value_type>> const& zs,
    float* buffer
  )
  {
    RawIO io{dims, fname};
    float meanReadTime = 0.f, meanWriteTime = 0.f;

    for (auto const [zStart, zEnd] : zs) {
      meanReadTime  += io.ReadFullSlices< Method>(zStart, zEnd, buffer);
      meanWriteTime += io.WriteFullSlices<Method>(zStart, zEnd, buffer);
    }

    meanReadTime  /= zs.size();
    meanWriteTime /= zs.size();
    return std::make_pair(meanReadTime, meanWriteTime);
  }

  template< IOMethod Method >
  static std::pair<float, float> BenchmarkSmallRegionIO(
    Dimensions const& dims,
    std::string const& fname,
    std::vector<std::array<Dimensions::value_type,6>> const& rois,
    float* buffer
  )
  {
    RawIO io{dims, fname};
    float meanReadTime = 0.f, meanWriteTime = 0.f;

    for (auto const& roi : rois) {
      meanReadTime  += io.ReadEnvironment< Method>(roi, buffer);
      meanWriteTime += io.WriteEnvironment<Method>(roi, buffer);
    }

    meanReadTime  /= rois.size();
    meanWriteTime /= rois.size();
    return std::make_pair(meanReadTime, meanWriteTime);
  }

  // -------------------


  RawIO(Dimensions const& dims, std::string const& fname) noexcept
    : Dims(dims)
    , InputFilename(fname)
  {
    auto tempFPath = std::filesystem::temp_directory_path() / std::filesystem::path(fname).filename();
    this->OutputFilename = tempFPath.string();

    std::ofstream outf{this->OutputFilename.c_str(), std::ios::binary};
    outf.seekp(this->Dims[2] * this->SliceSize() * sizeof(unsigned short) - 1, std::ios::beg);
    outf.put('\0');
  }

  virtual ~RawIO()
  {
    std::filesystem::remove(this->OutputFilename);
  }


  template< IOMethod Method >
  float ReadFullSlices(Dimensions::value_type zStart, Dimensions::value_type zEnd, float* targetBuffer) const
  {
    auto func = [&]() {
      size_t const sliceSize  = this->SliceSize();
      size_t const bufferSize = (zEnd - zStart + 1) * sliceSize;
      size_t const rawOffset  = zStart * sliceSize * sizeof(unsigned short);

      if constexpr (Method == IOMethod::C_FILE) {

        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        std::FILE* file = std::fopen(this->InputFilename.c_str(), "rb");
        if (file) {
          std::fseek(file, rawOffset, SEEK_SET);
          std::fread(uint16Buf.get(), sizeof(unsigned short), bufferSize, file);
          std::fclose(file);
        }
        this->Copy(targetBuffer, uint16Buf.get(), bufferSize);

      } else if constexpr (Method == IOMethod::CXX_FSTREAM) {
        
        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        std::ifstream file{this->InputFilename.c_str(), std::ios::binary};
        file.seekg(rawOffset);
        file.read(reinterpret_cast< char* >(uint16Buf.get()), bufferSize * sizeof(unsigned short));
        this->Copy(targetBuffer, uint16Buf.get(), bufferSize);
      
      } else if constexpr (Method == IOMethod::CXX_MMAP) {
      
        mio::mmap_source file(this->InputFilename.c_str(), rawOffset, bufferSize * sizeof(unsigned short));
        this->Copy(targetBuffer, reinterpret_cast< unsigned short const* >(file.data()), bufferSize);

      }
    };
    return this->TimeOf(func);
  }


  template< IOMethod Method >
  float WriteFullSlices(Dimensions::value_type zStart, Dimensions::value_type zEnd, float* sourceBuffer) const
  {
    auto func = [&]() {
      size_t const sliceSize  = this->SliceSize();
      size_t const bufferSize = (zEnd - zStart + 1) * sliceSize;
      size_t const rawOffset  = zStart * sliceSize * sizeof(unsigned short);

      if constexpr (Method == IOMethod::C_FILE) {

        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        this->Copy(uint16Buf.get(), sourceBuffer, bufferSize);

        std::FILE* file = std::fopen(this->OutputFilename.c_str(), "r+b");
        if (file) {
          std::fseek(file, rawOffset, SEEK_SET);
          std::fwrite(uint16Buf.get(), sizeof(unsigned short), bufferSize, file);
          std::fclose(file);
        }

      } else if constexpr (Method == IOMethod::CXX_FSTREAM) {

        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        this->Copy(uint16Buf.get(), sourceBuffer, bufferSize);

        std::fstream file{this->OutputFilename.c_str(), std::ios::in | std::ios::out | std::ios::binary};
        file.seekp(rawOffset);
        file.write(reinterpret_cast< char const* >(uint16Buf.get()), bufferSize * sizeof(unsigned short));

      } else if constexpr (Method == IOMethod::CXX_MMAP) {

        mio::mmap_sink file(this->OutputFilename.c_str(), rawOffset, bufferSize * sizeof(unsigned short));
        this->Copy(reinterpret_cast< unsigned short* >(file.data()), sourceBuffer, bufferSize);

      }
    };
    return this->TimeOf(func);
  }


  template< IOMethod Method >
  float ReadEnvironment(std::array<Dimensions::value_type, 6> const& roi, float* targetBuffer) const
  {
    // roi : xmin, xmax, ymin, ymax, zmin, zmax
    auto func = [&]() {
      size_t const envSize           = roi[1] - roi[0] + 1;
      size_t const bufferSize        = envSize * envSize * envSize;
      size_t const rawOffset         = ((roi[4] * this->Dims[1] + roi[2]) * this->Dims[0] + roi[0]) * sizeof(unsigned short);
      size_t const yJump             = this->Dims[0] * sizeof(unsigned short);
      size_t const zJump             = (this->Dims[1] - envSize) * this->Dims[0] * sizeof(unsigned short);

      if constexpr (Method == IOMethod::C_FILE) {

        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        unsigned short* uint16BufPtr = uint16Buf.get();

        std::FILE* file = std::fopen(this->InputFilename.c_str(), "rb");
        std::fseek(file, rawOffset, SEEK_SET);

        for (Dimensions::value_type z = roi[4]; z <= roi[5]; ++z) {
          for (Dimensions::value_type y = roi[2]; y <= roi[3]; ++y) {
            std::fread(uint16BufPtr, sizeof(unsigned short), envSize, file);
            uint16BufPtr += envSize;
            std::fseek(file, yJump, SEEK_CUR);
          }
          std::fseek(file, zJump, SEEK_CUR);
        }
        
        std::fclose(file);
        this->Copy(targetBuffer, uint16Buf.get(), bufferSize);

      } else if constexpr (Method == IOMethod::CXX_FSTREAM) {

        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        unsigned short* uint16BufPtr = uint16Buf.get();

        std::ifstream file{this->InputFilename.c_str(), std::ios::binary};
        file.seekg(rawOffset);

        for (Dimensions::value_type z = roi[4]; z <= roi[5]; ++z) {
          for (Dimensions::value_type y = roi[2]; y <= roi[3]; ++y) {
            file.read(reinterpret_cast< char* >(uint16BufPtr), envSize * sizeof(unsigned short));
            uint16BufPtr += envSize;
            file.seekg(yJump, std::ios::cur);
          }
          file.seekg(zJump, std::ios::cur);
        }

        this->Copy(targetBuffer, uint16Buf.get(), bufferSize);

      } else if constexpr (Method == IOMethod::CXX_MMAP) {

        size_t const mapp = ((this->Dims[1] + 1) * this->Dims[0] + 1) * envSize * sizeof(unsigned short);
        mio::mmap_source file(this->InputFilename.c_str(), rawOffset, mapp);
        float* buf = targetBuffer;
        char const* source = file.data();

        for (Dimensions::value_type z = roi[4]; z <= roi[5]; ++z) {
          for (Dimensions::value_type y = roi[2]; y <= roi[3]; ++y) {
            this->Copy(buf, reinterpret_cast< unsigned short const* >(source), envSize);
            buf += envSize;
            source += yJump;
          }
          source += zJump;
        }
      }

    };
    return this->TimeOf(func);
  }


  template< IOMethod Method >
  float WriteEnvironment(std::array<Dimensions::value_type, 6> const& roi, float* sourceBuffer) const
  {
    // roi : xmin, xmax, ymin, ymax, zmin, zmax
    auto func = [&]() {
      size_t const envSize           = roi[1] - roi[0] + 1;
      size_t const bufferSize        = envSize * envSize * envSize;
      size_t const rawOffset         = ((roi[4] * this->Dims[1] + roi[2]) * this->Dims[0] + roi[0]) * sizeof(unsigned short);
      size_t const yJump             = this->Dims[0] * sizeof(unsigned short);
      size_t const zJump             = (this->Dims[1] - envSize) * this->Dims[0] * sizeof(unsigned short);

      if constexpr (Method == IOMethod::C_FILE) {

        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        unsigned short* uint16BufPtr = uint16Buf.get();
        this->Copy(uint16Buf.get(), sourceBuffer, bufferSize);

        std::FILE* file = std::fopen(this->OutputFilename.c_str(), "r+b");
        std::fseek(file, rawOffset, SEEK_SET);

        for (Dimensions::value_type z = roi[4]; z <= roi[5]; ++z) {
          for (Dimensions::value_type y = roi[2]; y <= roi[3]; ++y) {
            std::fwrite(uint16BufPtr, sizeof(unsigned short), envSize, file);
            uint16BufPtr += envSize;
            std::fseek(file, yJump, SEEK_CUR);
          }
          std::fseek(file, zJump, SEEK_CUR);
        }

        std::fclose(file);

      } else if constexpr (Method == IOMethod::CXX_FSTREAM) {

        auto uint16Buf = std::make_unique< unsigned short[] >(bufferSize);
        unsigned short* uint16BufPtr = uint16Buf.get();
        this->Copy(uint16Buf.get(), sourceBuffer, bufferSize);

        std::fstream file{this->OutputFilename.c_str(), std::ios::in | std::ios::out | std::ios::binary};
        file.seekp(rawOffset);

        for (Dimensions::value_type z = roi[4]; z <= roi[5]; ++z) {
          for (Dimensions::value_type y = roi[2]; y <= roi[3]; ++y) {
            file.write(reinterpret_cast< char const* >(uint16BufPtr), envSize * sizeof(unsigned short));
            uint16BufPtr += envSize;
            file.seekp(yJump, std::ios::cur);
          }
          file.seekp(zJump, std::ios::cur);
        }

      } else if constexpr (Method == IOMethod::CXX_MMAP) {

        size_t const mapp = ((this->Dims[1] + 1) * this->Dims[0] + 1) * envSize * sizeof(unsigned short);
        mio::mmap_sink file(this->OutputFilename.c_str(), rawOffset, mapp);
        float const* buf = sourceBuffer;
        char* target = file.data();

        for (Dimensions::value_type z = roi[4]; z <= roi[5]; ++z) {
          for (Dimensions::value_type y = roi[2]; y <= roi[3]; ++y) {
            this->Copy(reinterpret_cast< unsigned short* >(target), buf, envSize);
            buf += envSize;
            target += yJump;
          }
          target += zJump;
        }
      }

    };
    return this->TimeOf(func);
  }


protected:

  template< class Func, class... Args >
  inline float TimeOf(Func&& func, Args&&... args) const
  {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
  }

  inline size_t SliceSize() const
  {
    return this->Dims.at(0) * this->Dims.at(1);
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


void PrintBenchmarkResults(std::string const& api, size_t sizeBytes, float meanReadTime, float meanWriteTime)
{
#define FMT std::setw(8) << std::scientific << std::setprecision(2)
  float const size_GiB = static_cast< float >(sizeBytes) / static_cast< float >(1ull << 30);
  std::cout << "  Consdering " << api << '\n';
  std::cout << "    o Read  time: " << FMT << meanReadTime  << " [ms]  (-> " << FMT << (size_GiB * 1000.f / meanReadTime)  << " [GiB/s])\n";
  std::cout << "    o Write time: " << FMT << meanWriteTime << " [ms]  (-> " << FMT << (size_GiB * 1000.f / meanWriteTime) << " [GiB/s])\n";
#undef FMT
}

void BenchmarkFullSliceIO(RawIO::Dimensions const& volumeDims, std::string const& rawFile, int numberOfFullSlices, int EnvSize)
{
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<RawIO::Dimensions::value_type> dist{0, volumeDims[2]-EnvSize-1};

  std::vector<std::pair<RawIO::Dimensions::value_type, RawIO::Dimensions::value_type>> zs;
  zs.reserve(numberOfFullSlices);

  for (int i = 0; i < numberOfFullSlices; ++i) {
    auto start = dist(gen);
    zs.emplace_back(start, start + EnvSize - 1);
  }

  size_t const bufSize = EnvSize * volumeDims.at(0) * volumeDims.at(1);
  auto buffer = std::make_unique< float[] >(bufSize);
  auto bufferPtr = buffer.get();

  auto const [api1_read, api1_write] = RawIO::BenchmarkFullSliceIO<RawIO::IOMethod::C_FILE     >(volumeDims, rawFile, zs, bufferPtr);
  auto const [api2_read, api2_write] = RawIO::BenchmarkFullSliceIO<RawIO::IOMethod::CXX_FSTREAM>(volumeDims, rawFile, zs, bufferPtr);
  auto const [api3_read, api3_write] = RawIO::BenchmarkFullSliceIO<RawIO::IOMethod::CXX_MMAP   >(volumeDims, rawFile, zs, bufferPtr);
  std::cout << "Full slice I/O   (full slices; envSize = " << EnvSize << "):\n";
  PrintBenchmarkResults("C file API",              bufSize, api1_read, api1_write);
  PrintBenchmarkResults("C++ fstream API",         bufSize, api2_read, api2_write);
  PrintBenchmarkResults("C++ memory mapped files", bufSize, api3_read, api3_write);
  std::cout << '\n';
}

void BenchmarkCuboidRoIsIO(RawIO::Dimensions const& volumeDims, std::string const& rawFile, int numberOfCuboidRoIs, int EnvSize)
{
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<RawIO::Dimensions::value_type> distX{0, volumeDims[0]-EnvSize-1};
  std::uniform_int_distribution<RawIO::Dimensions::value_type> distY{0, volumeDims[1]-EnvSize-1};
  std::uniform_int_distribution<RawIO::Dimensions::value_type> distZ{0, volumeDims[2]-EnvSize-1};

  std::vector<std::array<RawIO::Dimensions::value_type, 6>> zs;
  zs.reserve(numberOfCuboidRoIs);

  for (int i = 0; i < numberOfCuboidRoIs; ++i) {
    auto const X = distX(gen);
    auto const Y = distY(gen);
    auto const Z = distZ(gen);
    std::array<RawIO::Dimensions::value_type, 6> const roi{
      X, X+EnvSize-1, Y, Y+EnvSize-1, Z, Z+EnvSize-1
    };
    zs.emplace_back(roi);
  }

  size_t const bufSize = EnvSize * EnvSize * EnvSize;
  auto buffer = std::make_unique< float[] >(bufSize);
  auto bufferPtr = buffer.get();

  auto const [api1_read, api1_write] = RawIO::BenchmarkSmallRegionIO<RawIO::IOMethod::C_FILE     >(volumeDims, rawFile, zs, bufferPtr);
  auto const [api2_read, api2_write] = RawIO::BenchmarkSmallRegionIO<RawIO::IOMethod::CXX_FSTREAM>(volumeDims, rawFile, zs, bufferPtr);
  auto const [api3_read, api3_write] = RawIO::BenchmarkSmallRegionIO<RawIO::IOMethod::CXX_MMAP   >(volumeDims, rawFile, zs, bufferPtr);
  std::cout << "Full slice I/O   (local environments; envSize = " << EnvSize << "):\n";
  PrintBenchmarkResults("C file API",              bufSize, api1_read, api1_write);
  PrintBenchmarkResults("C++ fstream API",         bufSize, api2_read, api2_write);
  PrintBenchmarkResults("C++ memory mapped files", bufSize, api3_read, api3_write);
  std::cout << '\n';
}

void RunBenchmarks(std::string const& mhdFile, int numberOfFullSlices, int numberOfCuboidRoIs)
{
  RawIO::Dimensions volumeDims;
  std::string rawFile;
  std::tie(volumeDims, rawFile) = ParseMhd(mhdFile);

  for (int envSize : {3, 5, 7, 9}) {
    BenchmarkFullSliceIO( volumeDims, rawFile, numberOfFullSlices, envSize);
    BenchmarkCuboidRoIsIO(volumeDims, rawFile, numberOfCuboidRoIs, envSize);
  }
}


void BenchmarkQueuedCopying(std::string const& mhdFile)
{
  auto const [dims, rawFile] = ParseMhd(mhdFile);
  auto tempFPath = (std::filesystem::temp_directory_path() / std::filesystem::path(rawFile).filename()).string();

  // Initialize the target file.
  {
    auto size = std::filesystem::file_size(rawFile);
    std::ofstream targetFile{tempFPath, std::ios::binary};
    targetFile.seekp(size - 1, std::ios::cur);
    targetFile.put('\0');
  }

  size_t const numSlices = 32;
  size_t const nTiles    = (dims.at(2) + numSlices - 1) / numSlices;
  size_t const sliceSize = dims.at(0) * dims.at(1);

  std::vector<std::pair<size_t, size_t>> chunks;
  chunks.reserve(nTiles);

  for (size_t tileIx = 0; tileIx < nTiles; ++tileIx) {
    size_t const offset    = tileIx * numSlices * sliceSize;
    size_t const numSlcs   = tileIx + 1 == nTiles ? (dims.at(2) - tileIx * numSlices) : numSlices;
    size_t const chunkSize = numSlcs * sliceSize;
    chunks.emplace_back(offset, chunkSize);
  }

  io::QueueChunkedCopier copier;
  copier.SetParameters(rawFile, tempFPath, nTiles);

  float const sizeGiB = dims.at(0) * dims.at(1) * dims.at(2) * sizeof(float) / (1ull << 30);

#define FMT std::setprecision(2) << std::scientific
  std::pair<float,float> const nonThreadingTime = copier.NonThreadingCopying(chunks);
  std::cout << "[Classical copying        ]  "
            << "fstream: " << FMT << nonThreadingTime.first  << " [s] (~ " << FMT << (sizeGiB / nonThreadingTime.first)  << " [GiB/s])  |  "
            << "mmap:    " << FMT << nonThreadingTime.second << " [s] (~ " << FMT << (sizeGiB / nonThreadingTime.second) << " [GiB/s])\n";

  std::pair<float, float> const manualTime = copier.ManuallyOverlappingCopying(chunks);
  std::cout << "[Manual overlap copy      ]  "
            << "fstream: " << FMT << manualTime.first  << " [s] (~ " << FMT << (sizeGiB / manualTime.first)  << " [GiB/s])  |  "
            << "mmap:    " << FMT << manualTime.second << " [s] (~ " << FMT << (sizeGiB / manualTime.second) << " [GiB/s])\n";

  std::pair<float, float> const threadingTime = copier.ThreadingCopying(chunks);
  std::cout << "[Threaded  copying        ]  "
            << "fstream: " << FMT << threadingTime.first  << " [s] (~ " << FMT << (sizeGiB / threadingTime.first)  << " [GiB/s])  |  "
            << "mmap:    " << FMT << threadingTime.second << " [s] (~ " << FMT << (sizeGiB / threadingTime.second) << " [GiB/s])\n";

  std::pair<float, float> const threadingTime2 = copier.ThreadedWriteCopying(chunks);
  std::cout << "[Threaded write copying   ]  "
            << "fstream: " << FMT << threadingTime2.first  << " [s] (~ " << FMT << (sizeGiB / threadingTime2.first)  << " [GiB/s])  |  "
            << "mmap:    " << FMT << threadingTime2.second << " [s] (~ " << FMT << (sizeGiB / threadingTime2.second) << " [GiB/s])\n";
#undef FMT
}


/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <path/to/inputfile.mhd>  <number-full-slices>  <number-cubic-rois>\n\n";
    std::cerr << "  Extracts _and_ saves Regions of Interest of the following cases from the input file and to a temporary file.\n";
    std::cerr << "    (1) Extracts  <number-full-slices>  full slices from the file.\n";
    std::cerr << "    (2) Extracts  <number-cubic-rois>  of sizes KxKxK (K = 3,5,7,9) from the file.\n\n";
    std::cerr << "  <path/to/inputfile.mhd>  Path to an existing mhd file (3D, metadata, pointing to a .raw file). MUST be uint16.\n";
    std::cerr << "  <number-full-slices>     Number of full slices to extract/store. Must be >= 0. No region benchmark if any of both is zero.\n";
    std::cerr << "  <number-cubic-rois>      Number of cubic RoIs to extract/store. Must be >= 0. No region benchmark if any of both is zero.\n";
    return EXIT_FAILURE;
  }

  auto parseInt = [](char const* arg) -> int {
    int value;
    std::istringstream(arg) >> value;
    return value;
  };

  int fullSlices = parseInt(argv[2]);
  int regions    = parseInt(argv[3]);

  if (fullSlices > 0 && regions > 0) {
    RunBenchmarks(argv[1], fullSlices, regions);
    std::cout << std::string(80, '=') << '\n';
  }
  BenchmarkQueuedCopying(argv[1]);
  return EXIT_SUCCESS;
}