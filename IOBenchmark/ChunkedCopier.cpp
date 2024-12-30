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
#include "ChunkedCopier.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <thread>
#include <queue>
#include <vector>

#include "mio.hpp"

namespace io::impl
{
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
        this->Chunk = std::make_unique<float[]>(this->ChunkSize);
        mio::mmap_source mappedFile(filename, offsetBytes, chunkSizeBytes);
        std::copy_n(reinterpret_cast< unsigned short const* >(mappedFile.data()), this->ChunkSize, this->Chunk.get());
      } else {
        mio::mmap_sink mappedFile(filename, offsetBytes, chunkSizeBytes);
        std::copy_n(this->Chunk.get(), this->ChunkSize, reinterpret_cast< unsigned short* >(mappedFile.data()));
      }
    }

  private:
    size_t Offset, ChunkSize;
    std::shared_ptr< float[] > Chunk;
  };

  // -------------------------------------------------------------------------------- //

  template< class T >
  class ThreadSafeQueue
  {
  public:

    void Push(T&& item)
    {
      std::lock_guard< std::mutex > _lock(this->Mutex);
      this->Items.push(std::forward<T&&>(item));
      this->CondVar.notify_one();
    }

    T Pop()
    {
      std::unique_lock< std::mutex > _lock(this->Mutex);
      this->CondVar.wait(_lock, [this]() { return !this->Items.empty(); });

      T item = this->Items.front();
      this->Items.pop();

      return item;
    }

    bool HasTasks(void)
    {
      std::lock_guard< std::mutex > _lock(this->Mutex);
      return !this->Items.empty();
    }

  public:
    std::mutex Mutex;
    std::condition_variable CondVar;

  private:
    std::queue<T> Items;
  };

  static ThreadSafeQueue< IOTask > ReadQueue;
  static ThreadSafeQueue< IOTask > WriteQueue;



  enum class IOMode {
    READ, WRITE
  };

  template<IOMode Mode>
  void ChunkReaderThread(std::atomic_bool& keepRunning, std::string const& inputFilename)
  {
    static constexpr bool isRead = Mode == IOMode::READ;
    auto& Data = isRead ? ReadQueue : WriteQueue;

    while (keepRunning || Data.HasTasks()) {
      IOTask task = Data.Pop();

      task.Execute(inputFilename, isRead);

      if constexpr (isRead) {
        WriteQueue.Push(std::move(task));
      }
    }
  }

} // namespace impl



void io::QueueChunkedCopier::SetParameters(std::string const& inputFilename, std::string const& outputFilename, size_t numberOfExecutions)
{
  this->InputFilename  = inputFilename;
  this->OutputFilename = outputFilename;
}


float io::QueueChunkedCopier::NonThreadingCopying(std::vector<std::pair<size_t, size_t>> const& chunks)
{
  auto start = std::chrono::high_resolution_clock::now();

  for (auto const& [offset, chunkSize] : chunks) {
    impl::IOTask task{offset, chunkSize};
    task.Execute(this->InputFilename,  true);
    task.Execute(this->OutputFilename, false);
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration< float >(end - start).count();
}


float io::QueueChunkedCopier::ManuallyOverlappingCopying(std::vector<std::pair<size_t, size_t>> const& chunks)
{
  auto start = std::chrono::high_resolution_clock::now();

  size_t const numChunks = chunks.size();
  auto const [off0, cs0] = chunks[0];
  impl::IOTask inTask{off0, cs0}, outTask{off0, cs0};

  for (int i = 0; i <= numChunks; ++i) {
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        // Input
        if (i < numChunks) {
          auto const [off, cs] = chunks[i];
          inTask = impl::IOTask{off, cs};
          inTask.Execute(this->InputFilename, true);
        }
      }

      #pragma omp section
      {
        // Output
        if (i > 0) {
          outTask.Execute(this->OutputFilename, false);
        }
      }
    }

    std::swap(inTask, outTask);
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration< float >(end - start).count();
}


float io::QueueChunkedCopier::ThreadingCopying(std::vector<std::pair<size_t, size_t>> const& chunks)
{
  auto start = std::chrono::high_resolution_clock::now();

  std::atomic_bool keepReading = true;
  std::atomic_bool keepWriting = true;
  std::thread  readThread(impl::ChunkReaderThread<impl::IOMode::READ>,  std::ref(keepReading), this->InputFilename);
  std::thread writeThread(impl::ChunkReaderThread<impl::IOMode::WRITE>, std::ref(keepWriting), this->OutputFilename);

  for (auto const& [offset, chunkSize] : chunks) {
    impl::ReadQueue.Push({offset, chunkSize});
  }

  // First wait for the reading thread to finish, subsequently for the writing thread.
  keepReading = false;  readThread.join();
  keepWriting = false; writeThread.join();

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration< float >(end - start).count();
}
