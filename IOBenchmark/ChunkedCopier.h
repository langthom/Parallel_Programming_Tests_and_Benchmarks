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

#define _NOMINMAX
#include <array>
#include <functional>
#include <string>

#include <fstream>

#include "mio.hpp"

#include <fcntl.h> // open flags
#include <io.h>

namespace io
{
  enum class IOMethod {
    FSTREAM, MMAP, SYSCALL
  };

  template<IOMethod Method>
  constexpr char const* IOMethodToString() {
    constexpr std::array<char const*, 3> names{
      "C++ fstream  ", "mmap         ", "Low-level I/O"
    };
    return names[static_cast< int >(Method)];
  }

  template<IOMethod Method>
  class IOTask
  {
  public:

    IOTask(size_t offset, size_t chunkSize) noexcept
      : Offset(offset)
      , ChunkSize(chunkSize)
      , Chunk(nullptr)
    {
    }

    void Execute(std::string const& filename, bool read)
    {
      size_t const offsetBytes    = this->Offset    * sizeof(unsigned short);
      size_t const chunkSizeBytes = this->ChunkSize * sizeof(unsigned short);

      if (read) {
        this->Chunk = std::make_unique<unsigned short[]>(this->ChunkSize);
        this->Read(filename, offsetBytes, chunkSizeBytes, EnumType<Method>{});
      } else {
        this->Write(filename, offsetBytes, chunkSizeBytes, EnumType<Method>{});
      }
    }

  private:

    template<io::IOMethod M> struct EnumType{};

    void Read(std::string const& filename, size_t offsetBytes, size_t chunkBytes, EnumType<io::IOMethod::FSTREAM>) {
      std::ifstream file{filename.c_str(), std::ios::binary};
      file.seekg(offsetBytes, std::ios::beg);
      file.read(reinterpret_cast< char* >(this->Chunk.get()), chunkBytes);
    }

    void Read(std::string const& filename, size_t offsetBytes, size_t chunkBytes, EnumType<io::IOMethod::MMAP>) {
      mio::mmap_source mappedFile(filename, offsetBytes, chunkBytes);
      std::memcpy(this->Chunk.get(), mappedFile.data(), chunkBytes);
    }

    void Read(std::string const& filename, size_t offsetBytes, size_t chunkBytes, EnumType<io::IOMethod::SYSCALL>) {
      int fId = _open(filename.c_str(), O_RDONLY | O_BINARY | O_SEQUENTIAL);
      if (fId != -1) {
        _lseeki64(fId, offsetBytes, SEEK_SET);

        // The syscall only supports 32bit. Need a loop.
        constexpr long long tileSize = 1ll << 30;
        int nTiles = static_cast<int>((chunkBytes + tileSize - 1) / tileSize);
        char* buf = reinterpret_cast< char* >(this->Chunk.get());

        for (int tile = 0; tile < nTiles; ++tile) {
          _read(fId, buf, static_cast<unsigned int>(std::min<long long>(tileSize, static_cast<long long>(chunkBytes) - tileSize * tile)));
          buf += tileSize;
        }

        _close(fId);
      }
    }


    void Write(std::string const& filename, size_t offsetBytes, size_t chunkBytes, EnumType<io::IOMethod::FSTREAM>) {
      std::fstream file{filename.c_str(), std::ios::binary | std::ios::in | std::ios::out};
      file.seekp(offsetBytes, std::ios::beg);
      file.write(reinterpret_cast< char* >(this->Chunk.get()), chunkBytes);
    }

    void Write(std::string const& filename, size_t offsetBytes, size_t chunkBytes, EnumType<io::IOMethod::MMAP>) {
      mio::mmap_sink mappedFile(filename, offsetBytes, chunkBytes);
      std::memcpy(mappedFile.data(), this->Chunk.get(), chunkBytes);
    }

    void Write(std::string const& filename, size_t offsetBytes, size_t chunkBytes, EnumType<io::IOMethod::SYSCALL>) {
      int fId = open(filename.c_str(), O_RDWR | O_BINARY | O_SEQUENTIAL);
      if (fId != -1) {
        _lseeki64(fId, offsetBytes, SEEK_SET);

        // The syscall only supports 32bit. Need a loop.
        constexpr long long tileSize = 1ll << 30;
        int nTiles = static_cast<int>((chunkBytes + tileSize - 1) / tileSize);
        char* buf = reinterpret_cast< char* >(this->Chunk.get());

        for (int tile = 0; tile < nTiles; ++tile) {
          _write(fId, buf, static_cast<unsigned int>(std::min<long long>(tileSize, static_cast<long long>(chunkBytes) - tileSize * tile)));
          buf += tileSize;
        }

        _close(fId);
      }
    }


  private:
    size_t Offset, ChunkSize;
    std::shared_ptr< unsigned short[] > Chunk;
  };


  class QueueChunkedCopier
  {
  public:
    using Timings = std::array<float, 3>;

    void SetParameters(std::string const& inputFilename, std::string const& outputFilename, size_t numberOfExecutions);

    Timings NonThreadingCopying(std::vector<std::pair<size_t, size_t>> const& chunks);

    Timings ManuallyOverlappingCopying(std::vector<std::pair<size_t, size_t>> const& chunks);

    Timings ThreadingCopying(std::vector<std::pair<size_t, size_t>> const& chunks);

    Timings ThreadedWriteCopying(std::vector<std::pair<size_t, size_t>> const& chunks);

  private:
    std::string InputFilename, OutputFilename;
  };

}
