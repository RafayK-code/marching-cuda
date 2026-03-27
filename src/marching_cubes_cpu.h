#pragma once

#include <vector>

std::vector<float> MarchingCubesCPU(const float* field, int gridSize, float min, float max, float isovalue);