#pragma once

#include "common.h"

#include <vector>

std::vector<float> MarchingCubesCPU(const float* field, const MarchingCubesConfig& config);