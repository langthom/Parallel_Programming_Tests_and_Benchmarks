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
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <tuple>

#include <immintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

#include "RawArrayConversion.h"

std::unique_ptr< unsigned short[] > GetData(double memInGiB, long long& N) {
  N = static_cast< long long >(memInGiB * (1ull << 30) / sizeof(unsigned short));
  auto data = std::make_unique< unsigned short[] >(N);
  std::iota(data.get(), data.get()+N, 0);
  return data;

  std::random_device rd;
  std::default_random_engine re(rd());
  std::uniform_int_distribution<unsigned short> dist(0, 65535);
  std::for_each(data.get(), data.get() + N, [&dist,&re](unsigned short& f){ f = dist(re); });

  return data;
}

#define METHOD1 "serial"
void Copy1(float* target, unsigned short* source, long long N) {
  for (long long i = 0; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD2 "serial                                + restrict"
void Copy2(float* __restrict target, unsigned short* __restrict source, long long N) {
  for (long long i = 0; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD3 "                  parallel"
void Copy3(float* target, unsigned short* source, long long N) {
  #pragma omp parallel
  for (long long i = 0; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD4 "                  parallel            + restrict"
void Copy4(float* __restrict target, unsigned short* __restrict source, long long N) {
  #pragma omp parallel
  for (long long i = 0; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD5 "serial          + unrolled"
void Copy5(float* target, unsigned short* source, long long N) {
  long long i = 0;
  for (; i < N; i += 8) {
    target[i+0] = static_cast< float >(source[i+0]);
    target[i+1] = static_cast< float >(source[i+1]);
    target[i+2] = static_cast< float >(source[i+2]);
    target[i+3] = static_cast< float >(source[i+3]);
    target[i+4] = static_cast< float >(source[i+4]);
    target[i+5] = static_cast< float >(source[i+5]);
    target[i+6] = static_cast< float >(source[i+6]);
    target[i+7] = static_cast< float >(source[i+7]);
  }
  i -= 8;
  for (; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD6 "serial          + unrolled            + restrict"
void Copy6(float* __restrict target, unsigned short* __restrict source, long long N) {
  long long i = 0;
  for (; i < N; i += 8) {
    target[i+0] = static_cast< float >(source[i+0]);
    target[i+1] = static_cast< float >(source[i+1]);
    target[i+2] = static_cast< float >(source[i+2]);
    target[i+3] = static_cast< float >(source[i+3]);
    target[i+4] = static_cast< float >(source[i+4]);
    target[i+5] = static_cast< float >(source[i+5]);
    target[i+6] = static_cast< float >(source[i+6]);
    target[i+7] = static_cast< float >(source[i+7]);
  }
  i -= 8;
  for (; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD7 "std::transform"
void Copy7(float* target, unsigned short* source, long long N) {
  std::transform(source, source + N, target, [](unsigned short x) {return static_cast< float >(x); });
}

#define METHOD8 "CUDA"
void Copy8(float* __restrict target, unsigned short* __restrict source, long long N) {
  cudaRawArrayConversion(target, source, N);
}

#define METHOD9 "vectorized(8)"
void Copy9(float* target, unsigned short* source, long long N) { // taken from https://stackoverflow.com/a/16035571/10696884
  long long i = 0;
  for (; i < N; i += 8) {
    //  Load 8 16-bit ushorts.
    //  vi = {a,b,c,d,e,f,g,h}
    __m128i vi = _mm_load_si128((const __m128i*)(source + i));

    //  Convert to 32-bit integers
    //  vi0 = {a,0,b,0,c,0,d,0}
    //  vi1 = {e,0,f,0,g,0,h,0}
    __m128i vi0 = _mm_cvtepu16_epi32(vi);
    __m128i vi1 = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(vi,vi));

    //  Convert to float
    __m128 vf0 = _mm_cvtepi32_ps(vi0);
    __m128 vf1 = _mm_cvtepi32_ps(vi1);

    //  Store
    _mm_store_ps(target + i + 0,vf0);
    _mm_store_ps(target + i + 4,vf1);
  }
  i -= 8;
  for (; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD10 "vectorized(8)                        + restrict"
void Copy10(float* __restrict target, unsigned short* __restrict source, long long N) { // taken from https://stackoverflow.com/a/16035571/10696884
  long long i = 0;
  for (; i < N; i += 8) {
    //  Load 8 16-bit ushorts.
    //  vi = {a,b,c,d,e,f,g,h}
    __m128i vi = _mm_load_si128((const __m128i*)(source + i));

    //  Convert to 32-bit integers
    //  vi0 = {a,0,b,0,c,0,d,0}
    //  vi1 = {e,0,f,0,g,0,h,0}
    __m128i vi0 = _mm_cvtepu16_epi32(vi);
    __m128i vi1 = _mm_cvtepu16_epi32(_mm_unpackhi_epi64(vi,vi));

    //  Convert to float
    __m128 vf0 = _mm_cvtepi32_ps(vi0);
    __m128 vf1 = _mm_cvtepi32_ps(vi1);

    //  Store
    _mm_store_ps(target + i + 0,vf0);
    _mm_store_ps(target + i + 4,vf1);
  }
  i -= 8;
  for (; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD11 "vectorized(8)  + unrolled            + restrict"
void Copy11(float* __restrict target, unsigned short* __restrict source, long long N) { // taken from https://stackoverflow.com/a/16035571/10696884
  long long i = 0;
  for (; i < N; i += 16) {
    // Load 128 bit integer data (-> 128 bit == 8x unsigned short)
    //  a0 = [ u16_1, u16_2, u16_3, u16_4, u16_5, u16_6, u16_7, u16_8 ]
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i a1 = _mm_load_si128((const __m128i*)(source + i + 8));

    //  Split into two registers (little endian!)
    //  a0 = [ u16_1, u16_2, u16_3, u16_4 | u16_5, u16_6, u16_7, u16_8 ]
    //  b0 = [ u16_5, u16_6, u16_7, u16_8 | u16_5, u16_6, u16_7, u16_8 ]
    __m128i b0 = _mm_unpackhi_epi64(a0,a0);
    __m128i b1 = _mm_unpackhi_epi64(a1,a1);

    //  Convert to 32-bit integers
    //  a0 = [ int32(u16_1), int32(u16_2), int32(u16_3), int32(u16_4) ]
    //  b0 = [ int32(u16_5), int32(u16_6), int32(u16_7), int32(u16_8) ]
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    a1 = _mm_cvtepu16_epi32(a1);
    b1 = _mm_cvtepu16_epi32(b1);

    //  Convert to float32
    //  c0 = [ float32(u16_1), float32(u16_2), float32(u16_3), float32(u16_4) ]
    //  d0 = [ float32(u16_5), float32(u16_6), float32(u16_7), float32(u16_8) ]
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    __m128 c1 = _mm_cvtepi32_ps(a1);
    __m128 d1 = _mm_cvtepi32_ps(b1);

    //  Store
    _mm_store_ps(target + i +  0,c0);
    _mm_store_ps(target + i +  4,d0);
    _mm_store_ps(target + i +  8,c1);
    _mm_store_ps(target + i + 12,d1);
  }

  i -= 16;
  for (; i < N; ++i) {
    target[i] = static_cast< float >(source[i]);
  }
}

#define METHOD12 "vectorized(8)  + parallel"
void Copy12(float* target, unsigned short* source, long long N) {
  #pragma omp parallel for
  for (long long i = 0; i < N; i += 8) {
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i b0 = _mm_unpackhi_epi64(a0, a0);
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    _mm_store_ps(target + i + 0, c0);
    _mm_store_ps(target + i + 4, d0);
  }

  long long rem = (N / 8) * 8;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}

#define METHOD13 "vectorized(8)  + parallel            + restrict"
void Copy13(float* __restrict target, unsigned short* __restrict source, long long N) {
  #pragma omp parallel for
  for (long long i = 0; i < N; i += 8) {
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i b0 = _mm_unpackhi_epi64(a0, a0);
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    _mm_store_ps(target + i + 0, c0);
    _mm_store_ps(target + i + 4, d0);
  }

  long long rem = (N / 8) * 8;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}

#define METHOD14 "vectorized(8)  + parallel + unrolled"
void Copy14(float* target, unsigned short* source, long long N) {

  #pragma omp parallel for
  for (long long i = 0; i < N; i += 16) {
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i a1 = _mm_load_si128((const __m128i*)(source + i + 8));

    //  Split into two registers
    __m128i b0 = _mm_unpackhi_epi64(a0, a0);
    __m128i b1 = _mm_unpackhi_epi64(a1, a1);

    //  Convert to 32-bit integers
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    a1 = _mm_cvtepu16_epi32(a1);
    b1 = _mm_cvtepu16_epi32(b1);

    //  Convert to float
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    __m128 c1 = _mm_cvtepi32_ps(a1);
    __m128 d1 = _mm_cvtepi32_ps(b1);

    //  Store
    _mm_store_ps(target + i + 0, c0);
    _mm_store_ps(target + i + 4, d0);
    _mm_store_ps(target + i + 8, c1);
    _mm_store_ps(target + i + 12, d1);
  }

  long long rem = (N / 16) * 16;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}

#define METHOD15 "vectorized(8)  + parallel + unrolled + restrict"
void Copy15(float* __restrict target, unsigned short* __restrict source, long long N) {

  #pragma omp parallel for
  for (long long i = 0; i < N; i += 16) {
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i a1 = _mm_load_si128((const __m128i*)(source + i + 8));

    //  Split into two registers
    __m128i b0 = _mm_unpackhi_epi64(a0, a0);
    __m128i b1 = _mm_unpackhi_epi64(a1, a1);

    //  Convert to 32-bit integers
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    a1 = _mm_cvtepu16_epi32(a1);
    b1 = _mm_cvtepu16_epi32(b1);

    //  Convert to float
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    __m128 c1 = _mm_cvtepi32_ps(a1);
    __m128 d1 = _mm_cvtepi32_ps(b1);

    //  Store
    _mm_store_ps(target + i + 0, c0);
    _mm_store_ps(target + i + 4, d0);
    _mm_store_ps(target + i + 8, c1);
    _mm_store_ps(target + i + 12, d1);
  }

  long long rem = (N / 16) * 16;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}


// like Copy12 (seems best) but with different OpenMP schedules

#define METHOD16 "vectorized(8)  + parallel + schedule(static)"
void Copy16(float* target, unsigned short* source, long long N) {
  #pragma omp parallel for schedule(static)
  for (long long i = 0; i < N; i += 8) {
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i b0 = _mm_unpackhi_epi64(a0, a0);
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    _mm_store_ps(target + i + 0, c0);
    _mm_store_ps(target + i + 4, d0);
  }

  long long rem = (N / 8) * 8;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}

#define METHOD17 "vectorized(8)  + parallel + schedule(static,1)"
void Copy17(float* target, unsigned short* source, long long N) {
  #pragma omp parallel for schedule(static,1)
  for (long long i = 0; i < N; i += 8) {
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i b0 = _mm_unpackhi_epi64(a0, a0);
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    _mm_store_ps(target + i + 0, c0);
    _mm_store_ps(target + i + 4, d0);
  }

  long long rem = (N / 8) * 8;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}

#define METHOD18 "vectorized(8)  + parallel + schedule(static,N/8)"
void Copy18(float* target, unsigned short* source, long long N) {
  #pragma omp parallel for schedule(static,N/8)
  for (long long i = 0; i < N; i += 8) {
    __m128i a0 = _mm_load_si128((const __m128i*)(source + i + 0));
    __m128i b0 = _mm_unpackhi_epi64(a0, a0);
    a0 = _mm_cvtepu16_epi32(a0);
    b0 = _mm_cvtepu16_epi32(b0);
    __m128 c0 = _mm_cvtepi32_ps(a0);
    __m128 d0 = _mm_cvtepi32_ps(b0);
    _mm_store_ps(target + i + 0, c0);
    _mm_store_ps(target + i + 4, d0);
  }

  long long rem = (N / 8) * 8;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}

#define METHOD19 "vectorized(16) + parallel + schedule(static)"
void Copy19(float* target, unsigned short* source, long long N) {
  // 16x unsigned short = 256 bits
  #pragma omp parallel for schedule(static)
  for (long long i = 0; i < N; i += 16) {
    // Load 16 consecutive uint16_t values (32 bytes) into a single __m256i register
    __m256i v_shorts = _mm256_loadu_si256((const __m256i*)(source + i));

    // --- Step 1: Separate the 16 shorts into two 128-bit chunks (low 8 and high 8) ---
    // This is necessary because the widening intrinsic only takes a 128-bit source.
    __m128i v_low_shorts  = _mm256_extracti128_si256(v_shorts, 0); // Low 8 shorts
    __m128i v_high_shorts = _mm256_extracti128_si256(v_shorts, 1); // High 8 shorts

    // --- Step 2: Widen the 16-bit unsigned integers to 32-bit signed integers (epi32) ---
    // The intrinsic `_mm256_cvtepu16_epi32` takes a __m128i (8x u16) and produces a __m256i (8x u32).
    // It correctly handles the unsigned nature of the source data during widening.

    // Convert low 8 shorts to 8 DWORDS
    __m256i v_low_int  = _mm256_cvtepu16_epi32(v_low_shorts);

    // Convert high 8 shorts to 8 DWORDS
    __m256i v_high_int = _mm256_cvtepu16_epi32(v_high_shorts);

    // --- Step 3: Convert the 32-bit integers (DWORDS) to 32-bit floats (PS) ---
    // This is the standard conversion intrinsic.

    // Convert 8 DWORDS (low set) to 8 floats
    __m256 v_low_float  = _mm256_cvtepi32_ps(v_low_int);

    // Convert 8 DWORDS (high set) to 8 floats
    __m256 v_high_float = _mm256_cvtepi32_ps(v_high_int);

    // --- Step 4: Store the results ---
    _mm256_storeu_ps(target + i + 0, v_low_float);
    _mm256_storeu_ps(target + i + 8, v_high_float);  
  }

  long long rem = (N / 16) * 16;
  for (; rem < N; ++rem) {
    target[rem] = static_cast<float>(source[rem]);
  }
}



// Preprocessor solutions to allow for different function parameter list qualifiers (restricted) without implicit decay or so.

bool isNear(float* expected, float* got, long long N, double tol) {
  for (long long i = 0; i < N; ++i) {
    if (std::fabs(expected[i] - got[i]) > tol) {
      return false;
    }
  }
  return true;
}

#define FLOAT_FMT std::setw(4) << std::setprecision(2) << std::scientific

#define METHOD(i) METHOD ## i

#define ELAPSED_S(i) double elapsed ## i; do { \
  auto start = std::chrono::high_resolution_clock::now(); \
  Copy ## i(dataOut, dataIn, N); \
  auto end = std::chrono::high_resolution_clock::now(); \
  elapsed ## i = std::chrono::duration<double>(end-start).count(); \
} while(false);

#define BENCHMARK(i, M) do { \
  std::vector<double> times(M); \
  for (int run = 0; run < M; ++run) { \
    ELAPSED_S(i); \
    times[run] = elapsed ## i; \
    if (!isNear(dataOutExp.get(), dataOut, N, 1e-9)) { \
      std::cerr << " *** Method " << i << " produce invalid copy!\n"; \
      return EXIT_FAILURE; \
    } \
    std::memset(dataOut, 0, N * sizeof(float)); \
  } \
  double avg = std::accumulate(times.cbegin(), times.cend(), 0.0); \
  double std = 0.0; \
  for (int run = 0; run < M; ++run) { \
    std += std::pow(times[run] - avg, 2); \
  } \
  if (M > 1) { \
    std /= (M-1); \
  } \
  double gbps = sizeGB / avg; \
  std::cout << " #" << std::setw(2) << std::right << i << " [" << std::setw(48) << std::left << METHOD(i) << "]  " << FLOAT_FMT << avg << " +/- " << FLOAT_FMT << std << " [s] (~ " << gbps << " [GB/s])\n"; \
  avgs.push_back(avg); \
} while(false);


/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

double ParseDouble(char const* str) {
  std::istringstream iss{str};
  double val;
  iss >> val;
  return val;
}

int main(int argc, char** argv) {

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <disableGPU:y/n>  <sizeGB:float>...\n";
    return EXIT_FAILURE;
  }

  bool disableGPU = argv[1][0] == 'y';

  std::vector<double> sizesGB(argc-2);
  std::transform(argv+2, argv+argc, sizesGB.begin(), ParseDouble);

  int nRuns = 20;

  for (double sizeGB : sizesGB) {
    long long N;
    auto dataInSP  = GetData(sizeGB, N);
    auto dataOutSP = std::make_unique<float[]>(N);

    auto dataOutExp = std::make_unique<float[]>(N);
    Copy1(dataOutExp.get(), dataInSP.get(), N);

    std::cout << "Input has " << FLOAT_FMT << static_cast<double>(N) << " elements (size: " << FLOAT_FMT << sizeGB << " [GB])\n";

    unsigned short* dataIn = dataInSP.get();
    float* dataOut = dataOutSP.get();

    std::vector<double> avgs;
    avgs.reserve(20);

    BENCHMARK( 1, nRuns);
    BENCHMARK( 2, nRuns);
    BENCHMARK( 3, nRuns);
    BENCHMARK( 4, nRuns);
    BENCHMARK( 5, nRuns);
    BENCHMARK( 6, nRuns);
    BENCHMARK( 7, nRuns);
    if (disableGPU) {
      std::cout << " # 8 [CUDA                                            ]  disabled\n";
      avgs.push_back(std::numeric_limits<double>::max());
    } else {
      BENCHMARK(8, nRuns);
    }
    BENCHMARK( 9, nRuns);
    BENCHMARK(10, nRuns);
    BENCHMARK(11, nRuns);
    BENCHMARK(12, nRuns);
    BENCHMARK(13, nRuns);
    BENCHMARK(14, nRuns);
    BENCHMARK(15, nRuns);
    BENCHMARK(16, nRuns);
    BENCHMARK(17, nRuns);
    BENCHMARK(18, nRuns);
    BENCHMARK(19, nRuns);

    int bestMethod = static_cast< int >(std::distance(avgs.begin(), std::min_element(avgs.begin(), avgs.end()))+1/*offset in indexing*/);
    std::cout << "  => Best method: " << bestMethod << "\n\n";
  }

  return EXIT_SUCCESS;
}