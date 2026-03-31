#pragma once

#include "common.h"

#include <vector>

std::vector<float> MarchingCubesGPU(const float* field, const MarchingCubesConfig& config);

struct cudaGraphicsResource;

int MarchingCubesGPU_Interop(const float* field, const MarchingCubesConfig& config, cudaGraphicsResource* vbo);