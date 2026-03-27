#pragma once

#include <string>
#include <vector>
#include <functional>

void WritePLYFile(const float* vertices, size_t bufferSize, const std::string& filename);

std::vector<float> GenerateScalarField(int gridSize, float min, float max, std::function<float(float, float, float)> implicitFunc);

std::vector<float> LoadCTHead(const std::string& directory);