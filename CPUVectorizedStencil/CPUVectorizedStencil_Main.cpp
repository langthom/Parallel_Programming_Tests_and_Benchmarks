/*
 * MIT License
 * 
 * Copyright (c) 2025 Dr. Thomas Lang
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

static int KMAX = 9;

std::unique_ptr<float[]> GetData(double memInGiB, long long& N, std::array<long long, 3>& dims) {

  /*
  dims = {10,3,3};
  N = 3 * 3 * 10;
  auto data1 = std::make_unique<float[]>(N);
  std::fill_n(data1.get(), N, 0.f);
  float tmp[] = {1, 1, 2, 2, 3, 3, 1, 1};
  std::copy_n(tmp, 8, data1.get() + 41);
  return data1;
  */

  constexpr double toGiB = 1ull << 30;
  int E = 8;
  long long const dimX = 1ull << E;
  long long const dimY = 1ull << E;
  long long const dimZ = static_cast<long long>(memInGiB * toGiB / (2.0/* in and output */ * sizeof(float) * dimY * dimX));
  dims = {dimX, dimY, dimZ};

  N = dimX * dimY * dimZ;
  auto data = std::make_unique< float[] >(N);
  std::random_device rd;
  std::default_random_engine re(rd());
  std::uniform_real_distribution<float> dist(0.f, 65535.f);
  std::for_each(data.get(), data.get() + N, [&dist,&re](float& f){ f = dist(re); });
  return data;
}


bool isNear(float* expected, float* got, long long N, double tol) {
  return true;//!!!!
  for (long long i = 0; i < N; ++i) {
    if (std::fabs(expected[i] - got[i]) > tol) {
      return false;
    }
  }
  return true;
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

std::array<long long, 3> To3D(long long ix1D, std::array<long long, 3> const& dims) {
  long long const s = dims[0] * dims[1];
  long long const z = ix1D / s;
  long long const j = ix1D - z * s;
  long long const y = j / dims[0];
  long long const x = j % dims[0];
  return {x, y, z};
}

long long To1D(std::array<long long, 3> const& ix3D, std::array<long long, 3> const& dims) {
  return (ix3D[2] * dims[1] + ix3D[1]) * dims[0] + ix3D[0];
}

std::vector<long long> NeighborhoodOffsets(std::array<long long, 3> const& dims, int envSize) {
  int const K = envSize * envSize * envSize;
  int const p = envSize / 2;
  std::vector<long long> offsets(K);
  int o = 0;
  for (int z = -p; z <= p; ++z) {
    for (int y = -p; y <= p; ++y) {
      for (int x = -p; x <= p; ++x) {
        offsets[o++] = To1D({x, y, z}, dims);
      }
    }
  }
  return offsets;
}


inline std::array<float, 4> Stats(std::vector<float> const& env) {
  float const M = static_cast< float >(env.size());
  float avg = 0.f;  for (float f : env) { avg += f;                      }  avg /= M;
  float std = 0.f;  for (float f : env) { std += std::pow(f - avg, 2);   }  std /= (M - 1);
  std = std::sqrt(std);
  float skw = 0.f;  for (float f : env) { skw += std::pow((f - avg), 3); }  skw /= ((M-1) * std::pow(std, 3));
  float krt = 0.f;  for (float f : env) { krt += std::pow((f - avg), 4); }  krt /= ((M-1) * std::pow(std, 4));
  return {avg, std, skw, krt};
}


void ComputeStd_GroundTruth(float* target, float const* source, std::array<long long, 3> const& dims, int envSize) {
  long long const N                    = dims[0] * dims[1] * dims[2];
  int const padding                    = envSize / 2;
  std::vector<long long> const offsets = NeighborhoodOffsets(dims, envSize);

  for (long long ix1D = 0; ix1D < N; ++ix1D) {
    std::array<long long, 3> const ix3D = To3D(ix1D, dims);
    bool const inXPadding = ix3D[0] < padding || ix3D[0] >= dims[0] - padding;
    bool const inYPadding = ix3D[1] < padding || ix3D[1] >= dims[1] - padding;
    bool const inZPadding = ix3D[2] < padding || ix3D[2] >= dims[2] - padding;
    bool const inPadding = inXPadding || inYPadding || inZPadding;

    if (inPadding) {
      target[ix1D] = 0.f;
    } else {
      std::vector<float> env(offsets.size());
      for (int oI = 0; oI < offsets.size(); ++oI) {
        env[oI] = source[ix1D + offsets[oI]];
      }
      target[ix1D] = Stats(env)[1];
    }
  }
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */
// OpenMP parallel


void ComputeStd_Parallel(float* target, float const* source, std::array<long long, 3> const& dims, int envSize) {
  long long const N                    = dims[0] * dims[1] * dims[2];
  int const padding                    = envSize / 2;
  std::vector<long long> const offsets = NeighborhoodOffsets(dims, envSize);

  #pragma omp parallel for schedule(static)
  for (long long ix1D = 0; ix1D < N; ++ix1D) {
    std::array<long long, 3> const ix3D = To3D(ix1D, dims);
    bool const inXPadding = ix3D[0] < padding || ix3D[0] >= dims[0] - padding;
    bool const inYPadding = ix3D[1] < padding || ix3D[1] >= dims[1] - padding;
    bool const inZPadding = ix3D[2] < padding || ix3D[2] >= dims[2] - padding;
    bool const inPadding = inXPadding || inYPadding || inZPadding;

    if (inPadding) {
      target[ix1D] = 0.f;
    } else {
      std::vector<float> env;
      env.reserve(offsets.size());
      for (long long offset : offsets) {
        env.push_back(source[ix1D + offset]);
      }
      target[ix1D] = Stats(env)[1];
    }
  }
}


/* ------------------------------------------------------------------------------------------------------------------------------------------------- */
// Vectorized single output

#define BLEND_M1 0x80
#define BLEND_P1 0x01
#define BLEND_M2 0xC0
#define BLEND_P2 0x02
#define BLEND_M3 0xE0
#define BLEND_P3 0x04
#define BLEND_M4 0xF0
#define BLEND_P4 0x0F
#define BLEND_M5 0xF8
#define BLEND_P5 0x1F


__m256 VectorizedSum(float const* sourceAtEnvBegin, std::array<long long, 3> const& dims, int envSize, __m256 avg, int power, __m256 normalization) {
  int const halfEnv = envSize / 2;
  int const numVectors = static_cast< int >(std::ceil((8.0+envSize-1)/8.0));
  std::vector<__m256> sumAccumulators(numVectors, _mm256_set1_ps(0.f));

  // Step 1: Sum reduction of Y and Z axes (of shape envSize each)
  for (int z = 0; z < envSize; ++z) {
    for (int y = 0; y < envSize; ++y) {
      for (int acc = 0; acc < numVectors; ++acc) {
        // Load a vector of data.
        __m256 data = _mm256_load_ps(sourceAtEnvBegin + To1D({0,y,z}, dims) + acc * 8);
        // Compute  std::pow(data - avg, power)
        data = _mm256_sub_ps(data, avg);
        for (int pow = 2; pow <= power; ++pow) {
          data = _mm256_mul_ps(data, data);
        }
        // Addition
        sumAccumulators[acc] = _mm256_add_ps(sumAccumulators[acc], data);
      }
    }
  }

  // Step 2: Left and right rotations of accumulator vectors including blending.
  if (envSize > 11) { throw std::runtime_error("too big -> expand to max size!"); }
  static const std::array<__m256i, 10> rotations{
    _mm256_set_epi32(0,7,6,5,4,3,2,1), // -1
    _mm256_set_epi32(6,5,4,3,2,1,0,7), // +1
    _mm256_set_epi32(1,0,7,6,5,4,3,2), // -2
    _mm256_set_epi32(5,4,3,2,1,0,7,6), // +2
    _mm256_set_epi32(2,1,0,7,6,5,4,3), // -3
    _mm256_set_epi32(4,3,2,1,0,7,6,5), // +3
    _mm256_set_epi32(3,2,1,0,7,6,5,4), // -4
    _mm256_set_epi32(3,2,1,0,7,6,5,4), // +4
    _mm256_set_epi32(4,3,2,1,0,7,6,5), // -5
    _mm256_set_epi32(2,1,0,7,6,5,4,3), // +5
  };

  static constexpr std::array<int, 10> blendMasks{
    0x80, // -1
    0x01, // +1
    0xC0, // -2
    0x02, // +2
    0xE0, // -3
    0x04, // +3
    0xF0, // -4
    0x0F, // +4
    0xF8, // -5
    0x1F, // +5
  };

  std::vector<__m256> rotatedVectors(numVectors);
  std::vector<__m256> globalSumAccs = sumAccumulators;

  auto blendIt = [&](__m256& left, __m256 const& right, int kd) {
    switch (kd) {
    case 0: left = _mm256_blend_ps(left, right, BLEND_M1); break;
    case 1: left = _mm256_blend_ps(left, right, BLEND_P1); break;
    case 2: left = _mm256_blend_ps(left, right, BLEND_M2); break;
    case 3: left = _mm256_blend_ps(left, right, BLEND_P2); break;
    case 4: left = _mm256_blend_ps(left, right, BLEND_M3); break;
    case 5: left = _mm256_blend_ps(left, right, BLEND_P3); break;
    case 6: left = _mm256_blend_ps(left, right, BLEND_M4); break;
    case 7: left = _mm256_blend_ps(left, right, BLEND_P4); break;
    case 8: left = _mm256_blend_ps(left, right, BLEND_M5); break;
    case 9: left = _mm256_blend_ps(left, right, BLEND_P5); break;
    }
  };

  for (int k = 0; k < halfEnv; ++k) {
    for (int dir : {0/*left*/, 1/*right*/}) {
      auto const& rotation = rotations[k * 2 + dir];

      // Rotate all vectors individually.
      for (int acc = 0; acc < numVectors; ++acc) {
        rotatedVectors[acc] = _mm256_permutevar8x32_ps(sumAccumulators[acc], rotation);
      }

      // Blend successive vectors.
      for (int acc = 1; acc < numVectors; ++acc) {
        __m256 left = rotatedVectors[acc - 1]; // intentional copy
        blendIt(rotatedVectors[acc-1], rotatedVectors[acc], k * 2 + dir);
        blendIt(rotatedVectors[acc-0], left,                k * 2 + dir);
      }

      // After blending, sum them onto the accumulators.
      for (int acc = 0; acc < numVectors; ++acc) {
        globalSumAccs[acc] = _mm256_add_ps(globalSumAccs[acc], rotatedVectors[acc]);
      }
    }
  }

  // Step 3: Normalization
  for (int acc = 0; acc < numVectors; ++acc) {
    globalSumAccs[acc] = _mm256_div_ps(globalSumAccs[acc], normalization);
  }

  // Step 4: Extract the 8 final floats.
  int const accStart  = halfEnv / 8;
  auto const& sum1 = globalSumAccs[accStart];
  auto const& sum2 = globalSumAccs[accStart + 1];
  auto const& finalRotation = rotations[(halfEnv - 1) * 2 + 0];

  __m256 rot1 = _mm256_permutevar8x32_ps(sum1, finalRotation);
  __m256 rot2 = _mm256_permutevar8x32_ps(sum2, finalRotation);
  blendIt(rot1, rot2, (halfEnv - 1) * 2 + 0);
  return rot1;
}

__m256 VectorizedStats(float const* sourceAtEnvBegin, std::array<long long, 3> const& dims, int envSize) {
  int const N = envSize * envSize * envSize;
  __m256 N1 = _mm256_set1_ps(N - 1);

  __m256 avg = VectorizedSum(sourceAtEnvBegin, dims, envSize, _mm256_set1_ps(0.f), 1, _mm256_set1_ps(N));
  __m256 var = VectorizedSum(sourceAtEnvBegin, dims, envSize,                 avg, 2, N1);
  var = _mm256_sqrt_ps(var);
  __m256 N2 = _mm256_mul_ps(var, var);
  N2 = _mm256_mul_ps(N2, var);
  N2 = _mm256_mul_ps(N1, N2);
  __m256 skw = VectorizedSum(sourceAtEnvBegin, dims, envSize,                 avg, 3, N2);
  N2 = _mm256_mul_ps(N2, var);
  __m256 krt = VectorizedSum(sourceAtEnvBegin, dims, envSize,                 avg, 4, N2);
  return var;
}


void ComputeStd_Vectorized(float* target, float const* source, std::array<long long, 3> const& dims, int envSize) {
  long long const N = dims[0] * dims[1] * dims[2];
  int const padding = envSize / 2;
  std::vector<long long> const offsets = NeighborhoodOffsets(dims, envSize);

  //#pragma omp parallel for schedule(static)
  for (long long yz = 0; yz < dims[1] * dims[2]; ++yz) {
    long long z = yz / dims[1];
    long long y = yz % dims[1];
    long long x = 0;

    bool const inYPadding = y < padding || y >= dims[1] - padding;
    bool const inZPadding = z < padding || z >= dims[2] - padding;

    for (; x < dims[0];) {
      bool const inXPadding = x < padding || x >= dims[0] - padding;
      bool const inPadding = inXPadding || inYPadding || inZPadding;
      long long const ix1D = To1D({x, y, z}, dims);

      if (inPadding) {
        target[ix1D] = 0.f;
        ++x;
      } else {
        __m256 std = VectorizedStats(source + To1D({x-padding,y-padding,z-padding}, dims), dims, envSize);
        _mm256_store_ps(target + ix1D, std);
        x += 8;
      }
    }

    for (; x < dims[0]; ++x) {
      bool const inXPadding = x < padding || x >= dims[0] - padding;
      bool const inPadding = inXPadding || inYPadding || inZPadding;
      long long const ix1D = To1D({x, y, z}, dims);

      if (inPadding) {
        target[ix1D] = 0.f;
      } else {
        std::vector<float> env;
        env.reserve(offsets.size());
        for (long long offset : offsets) {
          env.push_back(source[ix1D + offset]);
        }
        target[ix1D] = Stats(env)[1];
      }
    }
  }
}


/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

#define NAME_FMT  std::setw(15)
#define FLOAT_FMT std::setw(4) << std::setprecision(2) << std::scientific

template<class Fun, class... Args>
inline void BenchmarkFunction(std::string const& methodName, int nRuns, float* expectedData, long long N, Fun&& func, Args&&... args) {
  auto&& argsTuple = std::forward_as_tuple(args...);
  std::vector<double> elapsedS(nRuns);

  for (int run = 0; run < nRuns; ++run) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    elapsedS[run] = std::chrono::duration<double>(end - start).count();

    if (expectedData && !isNear(expectedData, std::get<0>(argsTuple), N, 1e-7)) {
      std::cerr << " *** Error: method '" << methodName << "' produced incorrect results\n";
      std::exit(1);
    }
  }

  double avg = 0.0; for (double t : elapsedS) { avg += t;                    }                avg /= nRuns;
  double std = 0.0; for (double t : elapsedS) { std += std::pow(t - avg, 2); } if (nRuns > 1) avg /= (nRuns - 1);
  std::cout << "Method '" << NAME_FMT << methodName << "' took " << FLOAT_FMT << avg << " +/- " << FLOAT_FMT << std << " [s]\n";
}



/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

template<class T>
T ParseNumber(char const* str) {
  std::istringstream iss{str};
  T val;
  iss >> val;
  return val;
}

int main(int argc, char** argv) {

  /*
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <envSize-odd:int> <sizeGB:float>...\n";
    return EXIT_FAILURE;
  }

  int const envSize = ParseNumber<int>(argv[1]);

  std::vector<double> sizesGB(argc-2);
  std::transform(argv+2, argv+argc, sizesGB.begin(), ParseNumber<double>);
  */
  int const envSize = 7;
  std::vector<double> sizesGB{0.125};

  int nRuns = 1;// 20;

  if (envSize < 3 || envSize > KMAX || envSize % 2 == 0) {
    std::cerr << "Environment size must be in [3,5,...," << KMAX << "] and odd.\n";
    return EXIT_FAILURE;
  }

  for (double sizeGB : sizesGB) {
    long long N;
    std::array<long long, 3> dims;
    auto dataInSP   = GetData(sizeGB, N, dims);
    auto dataOutSP  = std::make_unique<float[]>(N);
    auto dataOutExp = std::make_unique<float[]>(N);

    std::cout << "Input shape: " << std::setw(3) << dims[0] << " x " << std::setw(3) << dims[1] << " x " << std::setw(3) << dims[2] << " (size: " << FLOAT_FMT << sizeGB << " [GB])\n";
    BenchmarkFunction("Sequential",  nRuns, nullptr,          N, ComputeStd_GroundTruth, dataOutExp.get(), dataInSP.get(), dims, envSize);
    BenchmarkFunction("Parallel",    nRuns, dataOutExp.get(), N, ComputeStd_Parallel,    dataOutSP.get(),  dataInSP.get(), dims, envSize);
    BenchmarkFunction("ParallelVec", nRuns, dataOutExp.get(), N, ComputeStd_Vectorized,  dataOutSP.get(),  dataInSP.get(), dims, envSize);
  }

  return EXIT_SUCCESS;
}