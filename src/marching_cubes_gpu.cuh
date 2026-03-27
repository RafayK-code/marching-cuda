#pragma once

#include <vector>

std::vector<float> MarchingCubesGPU(const float* field, int gridSize, float min, float max, float isovalue);