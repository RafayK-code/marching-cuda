#pragma once

#include "common.h"

#include <vector>

std::vector<float> MarchingCubesGPU(const float* field, const MarchingCubesConfig& config);