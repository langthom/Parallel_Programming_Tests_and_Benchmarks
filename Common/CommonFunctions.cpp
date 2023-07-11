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

#include "CommonFunctions.h"

#include <array>

double formatMemory(double N, std::string& unit) {
  std::array< const char*, 4 > fmts{
    "B", "kiB", "MiB", "GiB"
  };
  int fmtI = 0;
  for (; N >= 1024 && fmtI < fmts.size(); ++fmtI) {
    N /= 1024.f;
  }

  unit = fmts[fmtI];
  return N;
}


std::vector< int > computeMaskOffsets(int envSize, std::size_t dimY, std::size_t dimX) {
  std::size_t voxelsInEnvironment = static_cast< std::size_t >(envSize) * envSize * envSize;
  std::vector< int > offsets(voxelsInEnvironment, 0);
  int K2 = envSize / 2;
  std::size_t _o = 0;

  for (int _z = -K2; _z <= K2; ++_z) {
    for (int _y = -K2; _y <= K2; ++_y) {
      for (int _x = -K2; _x <= K2; ++_x) {
        offsets[_o++] = (_z * dimY + _y) * dimX + _x;
      }
    }
  }

  return offsets;
}

