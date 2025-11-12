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
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <numeric>
#include <thread>
#include <queue>
#include <vector>



namespace io::impl
{

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

  template<IOMethod Method>
  static ThreadSafeQueue< IOTask<Method> > ReadQueue;

  template<IOMethod Method>
  static ThreadSafeQueue< IOTask<Method> > WriteQueue;



  enum class IOMode {
    READ, WRITE
  };

  template<IOMode Mode, IOMethod Method>
  void ChunkReaderThread(std::atomic_bool& keepRunning, std::string const& inputFilename)
  {
    static constexpr bool isRead = Mode == IOMode::READ;
    auto& Data = isRead ? ReadQueue<Method> : WriteQueue<Method>;

    while (keepRunning || Data.HasTasks()) {
      IOTask<Method> task = Data.Pop();

      task.Execute(inputFilename, isRead);

      if constexpr (isRead) {
        WriteQueue<Method>.Push(std::move(task));
      }
    }
  }

} // namespace impl

void io::QueueChunkedCopier::SetParameters(std::string const& inputFilename, std::string const& outputFilename, size_t numberOfExecutions)
{
  this->InputFilename  = inputFilename;
  this->OutputFilename = outputFilename;
}


io::QueueChunkedCopier::Timings io::QueueChunkedCopier::NonThreadingCopying(std::vector<std::pair<size_t, size_t>> const& chunks)
{
  auto start1 = std::chrono::high_resolution_clock::now();

  for (auto const& [offset, chunkSize] : chunks) {
    IOTask<IOMethod::FSTREAM> task{offset, chunkSize};
    task.Execute(this->InputFilename,  true);
    task.Execute(this->OutputFilename, false);
  }

  auto end1 = std::chrono::high_resolution_clock::now();
  float dur1 = std::chrono::duration< float >(end1 - start1).count();

  auto start2 = std::chrono::high_resolution_clock::now();

  for (auto const& [offset, chunkSize] : chunks) {
    IOTask<IOMethod::MMAP> task{offset, chunkSize};
    task.Execute(this->InputFilename,  true);
    task.Execute(this->OutputFilename, false);
  }

  auto end2 = std::chrono::high_resolution_clock::now();
  float dur2 = std::chrono::duration< float >(end2 - start2).count();

  auto start3 = std::chrono::high_resolution_clock::now();

  for (auto const& [offset, chunkSize] : chunks) {
    IOTask<IOMethod::SYSCALL> task{offset, chunkSize};
    task.Execute(this->InputFilename,  true);
    task.Execute(this->OutputFilename, false);
  }

  auto end3 = std::chrono::high_resolution_clock::now();
  float dur3 = std::chrono::duration< float >(end3 - start3).count();
  return {dur1, dur2, dur3};
}


io::QueueChunkedCopier::Timings io::QueueChunkedCopier::ManuallyOverlappingCopying(std::vector<std::pair<size_t, size_t>> const& chunks)
{
  auto start1 = std::chrono::high_resolution_clock::now();

  size_t const numChunks = chunks.size();
  auto const [off0, cs0] = chunks[0];
  IOTask<IOMethod::FSTREAM> inTask{off0, cs0}, outTask{off0, cs0};

  for (int i = 0; i <= numChunks; ++i) {
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        // Input
        if (i < numChunks) {
          auto const [off, cs] = chunks[i];
          inTask = IOTask<IOMethod::FSTREAM>{off, cs};
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

  auto end1 = std::chrono::high_resolution_clock::now();
  float dur1 = std::chrono::duration< float >(end1 - start1).count();


  auto start2 = std::chrono::high_resolution_clock::now();

  IOTask<IOMethod::MMAP> inTask2{off0, cs0}, outTask2{off0, cs0};

  for (int i = 0; i <= numChunks; ++i) {
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        // Input
        if (i < numChunks) {
          auto const [off, cs] = chunks[i];
          inTask2 = IOTask<IOMethod::MMAP>{off, cs};
          inTask2.Execute(this->InputFilename, true);
        }
      }

      #pragma omp section
      {
        // Output
        if (i > 0) {
          outTask2.Execute(this->OutputFilename, false);
        }
      }
    }

    std::swap(inTask2, outTask2);
  }

  auto end2 = std::chrono::high_resolution_clock::now();
  float dur2 = std::chrono::duration< float >(end2 - start2).count();

  auto start3 = std::chrono::high_resolution_clock::now();

  IOTask<IOMethod::SYSCALL> inTask3{off0, cs0}, outTask3{off0, cs0};

  for (int i = 0; i <= numChunks; ++i) {
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        // Input
        if (i < numChunks) {
          auto const [off, cs] = chunks[i];
          inTask3 = IOTask<IOMethod::SYSCALL>{off, cs};
          inTask3.Execute(this->InputFilename, true);
        }
      }

      #pragma omp section
      {
        // Output
        if (i > 0) {
          outTask3.Execute(this->OutputFilename, false);
        }
      }
    }

    std::swap(inTask3, outTask3);
  }

  auto end3 = std::chrono::high_resolution_clock::now();
  float dur3 = std::chrono::duration< float >(end3 - start3).count();

  return {dur1, dur2, dur3};
}


io::QueueChunkedCopier::Timings io::QueueChunkedCopier::ThreadingCopying(std::vector<std::pair<size_t, size_t>> const& chunks)
{
  auto start1 = std::chrono::high_resolution_clock::now();

  std::atomic_bool keepReading = true;
  std::atomic_bool keepWriting = true;
  std::thread  readThread(impl::ChunkReaderThread<impl::IOMode::READ, IOMethod::FSTREAM>, std::ref(keepReading), this->InputFilename);
  std::thread writeThread(impl::ChunkReaderThread<impl::IOMode::WRITE,IOMethod::FSTREAM>, std::ref(keepWriting), this->OutputFilename);

  for (auto const& [offset, chunkSize] : chunks) {
    impl::ReadQueue<IOMethod::FSTREAM>.Push({offset, chunkSize});
  }

  // First wait for the reading thread to finish, subsequently for the writing thread.
  keepReading = false;  readThread.join();
  keepWriting = false; writeThread.join();

  auto end1 = std::chrono::high_resolution_clock::now();
  float dur1 = std::chrono::duration< float >(end1 - start1).count();

  // ---------------------
  auto start2 = std::chrono::high_resolution_clock::now();

  keepReading = true;
  keepWriting = true;
  std::thread  readThread2(impl::ChunkReaderThread<impl::IOMode::READ, IOMethod::MMAP>, std::ref(keepReading), this->InputFilename);
  std::thread writeThread2(impl::ChunkReaderThread<impl::IOMode::WRITE,IOMethod::MMAP>, std::ref(keepWriting), this->OutputFilename);

  for (auto const& [offset, chunkSize] : chunks) {
    impl::ReadQueue<IOMethod::MMAP>.Push({offset, chunkSize});
  }

  // First wait for the reading thread to finish, subsequently for the writing thread.
  keepReading = false;  readThread2.join();
  keepWriting = false; writeThread2.join();

  auto end2 = std::chrono::high_resolution_clock::now();
  float dur2 = std::chrono::duration< float >(end2 - start2).count();

  // ---------------------
  auto start3 = std::chrono::high_resolution_clock::now();

  keepReading = true;
  keepWriting = true;
  std::thread  readThread3(impl::ChunkReaderThread<impl::IOMode::READ, IOMethod::SYSCALL>, std::ref(keepReading), this->InputFilename);
  std::thread writeThread3(impl::ChunkReaderThread<impl::IOMode::WRITE,IOMethod::SYSCALL>, std::ref(keepWriting), this->OutputFilename);

  for (auto const& [offset, chunkSize] : chunks) {
    impl::ReadQueue<IOMethod::SYSCALL>.Push({offset, chunkSize});
  }

  // First wait for the reading thread to finish, subsequently for the writing thread.
  keepReading = false;  readThread3.join();
  keepWriting = false; writeThread3.join();

  auto end3 = std::chrono::high_resolution_clock::now();
  float dur3 = std::chrono::duration< float >(end3 - start3).count();

  return {dur1, dur2, dur3};
}

io::QueueChunkedCopier::Timings io::QueueChunkedCopier::ThreadedWriteCopying(std::vector<std::pair<size_t, size_t>> const& chunks)
{
  auto start1 = std::chrono::high_resolution_clock::now();
  std::vector<std::future<void>> writePromises;
  for (auto const& [offset, chunkSize] : chunks) {
    // Main thread reads (and processes)
    IOTask<IOMethod::FSTREAM> task(offset, chunkSize);
    task.Execute(this->InputFilename, true);
    // Async writing
    writePromises.push_back(std::async(std::launch::async, [this](IOTask<IOMethod::FSTREAM> task){ task.Execute(this->OutputFilename, false); }, task));
  }
  for (auto& promise : writePromises) {
    promise.get();
  }
  auto end1 = std::chrono::high_resolution_clock::now();
  float dur1 = std::chrono::duration< float >(end1 - start1).count();

  // --
  writePromises.clear();
  auto start2 = std::chrono::high_resolution_clock::now();
  for (auto const& [offset, chunkSize] : chunks) {
    // Main thread reads (and processes)
    IOTask<IOMethod::MMAP> task(offset, chunkSize);
    task.Execute(this->InputFilename, true);
    // Async writing
    writePromises.push_back(std::async(std::launch::async, [this](IOTask<IOMethod::MMAP> task) { task.Execute(this->OutputFilename, false); }, task));
  }
  for (auto& promise : writePromises) {
    promise.get();
  }
  auto end2 = std::chrono::high_resolution_clock::now();
  float dur2 = std::chrono::duration< float >(end2 - start2).count();

  // --
  writePromises.clear();
  auto start3 = std::chrono::high_resolution_clock::now();
  for (auto const& [offset, chunkSize] : chunks) {
    // Main thread reads (and processes)
    IOTask<IOMethod::SYSCALL> task(offset, chunkSize);
    task.Execute(this->InputFilename, true);
    // Async writing
    writePromises.push_back(std::async(std::launch::async, [this](IOTask<IOMethod::SYSCALL> task) { task.Execute(this->OutputFilename, false); }, task));
  }
  for (auto& promise : writePromises) {
    promise.get();
  }
  auto end3 = std::chrono::high_resolution_clock::now();
  float dur3 = std::chrono::duration< float >(end3 - start3).count();

  return {dur1, dur2, dur3};
}
