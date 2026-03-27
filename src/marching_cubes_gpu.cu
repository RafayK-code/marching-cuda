#include "marching_cubes_gpu.cuh"
#include "common.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <iostream>

__constant__ int d_MARCHING_CUBES_LUT[256][16];
__constant__ float d_VERT_TABLE[12][3];

static void InitGPUSymbols()
{
    static bool initialized = false;
    if (!initialized)
    {
        cudaMemcpyToSymbol(d_MARCHING_CUBES_LUT, MARCHING_CUBES_LUT, sizeof(MARCHING_CUBES_LUT));
        cudaMemcpyToSymbol(d_VERT_TABLE, VERT_TABLE, sizeof(VERT_TABLE));
        initialized = true;
    }
}

__global__ void CountVerticesKernel(const float* field, int* vertexCounts, int gridSize, float isovalue)
{
    int cubeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numCubes = (gridSize - 1) * (gridSize - 1) * (gridSize - 1);

    if (cubeIdx >= numCubes)
        return;

    int cubesPerSlice = (gridSize - 1) * (gridSize - 1);
    int z = cubeIdx / cubesPerSlice;
    int remainder = cubeIdx % cubesPerSlice;
    int y = remainder / (gridSize - 1);
    int x = remainder % (gridSize - 1);

    float vertValues[8];
    vertValues[0] = field[x + y * gridSize + z * gridSize * gridSize];
    vertValues[1] = field[(x + 1) + y * gridSize + z * gridSize * gridSize];
    vertValues[2] = field[(x + 1) + y * gridSize + (z + 1) * gridSize * gridSize];
    vertValues[3] = field[x + y * gridSize + (z + 1) * gridSize * gridSize];
    vertValues[4] = field[x + (y + 1) * gridSize + z * gridSize * gridSize];
    vertValues[5] = field[(x + 1) + (y + 1) * gridSize + z * gridSize * gridSize];
    vertValues[6] = field[(x + 1) + (y + 1) * gridSize + (z + 1) * gridSize * gridSize];
    vertValues[7] = field[x + (y + 1) * gridSize + (z + 1) * gridSize * gridSize];

    int cubeCase = 0;
    for (int i = 0; i < 8; i++)
    {
        if (vertValues[i] < isovalue)
            cubeCase |= (1 << i);
    }

    int count = 0;
    const int* edges = d_MARCHING_CUBES_LUT[cubeCase];

    for (int i = 0; i < 16; i++)
    {
        if (edges[i] == -1)
            break;

        count++;
    }

    vertexCounts[cubeIdx] = count;
}

__global__ void MarchingCubesKernel(const float* field, const int* vertexOffsets, float* output, int gridSize, float min, float max, float isovalue)
{
    int cubeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numCubes = (gridSize - 1) * (gridSize - 1) * (gridSize - 1);

    if (cubeIdx >= numCubes)
        return;

    int cubesPerSlice = (gridSize - 1) * (gridSize - 1);
    int z = cubeIdx / cubesPerSlice;
    int remainder = cubeIdx % cubesPerSlice;
    int y = remainder / (gridSize - 1);
    int x = remainder % (gridSize - 1);

    float vertValues[8];
    vertValues[0] = field[x + y * gridSize + z * gridSize * gridSize];
    vertValues[1] = field[(x + 1) + y * gridSize + z * gridSize * gridSize];
    vertValues[2] = field[(x + 1) + y * gridSize + (z + 1) * gridSize * gridSize];
    vertValues[3] = field[x + y * gridSize + (z + 1) * gridSize * gridSize];
    vertValues[4] = field[x + (y + 1) * gridSize + z * gridSize * gridSize];
    vertValues[5] = field[(x + 1) + (y + 1) * gridSize + z * gridSize * gridSize];
    vertValues[6] = field[(x + 1) + (y + 1) * gridSize + (z + 1) * gridSize * gridSize];
    vertValues[7] = field[x + (y + 1) * gridSize + (z + 1) * gridSize * gridSize];

    int cubeCase = 0;
    for (int i = 0; i < 8; i++)
    {
        if (vertValues[i] < isovalue)
            cubeCase |= (1 << i);
    }

    int writeOffset = vertexOffsets[cubeIdx];

    const int* edges = d_MARCHING_CUBES_LUT[cubeCase];
    float stepSize = (max - min) / (gridSize - 1);

    int vertexIdx = 0;
    for (int i = 0; i < 16; i++)
    {
        if (edges[i] == -1)
            break;

        const float* edgeVert = d_VERT_TABLE[edges[i]];

        float worldX = min + (x + edgeVert[0]) * stepSize;
        float worldY = min + (y + edgeVert[1]) * stepSize;
        float worldZ = min + (z + edgeVert[2]) * stepSize;

        int outIdx = (writeOffset + vertexIdx) * 3;
        output[outIdx + 0] = worldX;
        output[outIdx + 1] = worldY;
        output[outIdx + 2] = worldZ;

        vertexIdx++;
    }
}

std::vector<float> MarchingCubesGPU(const float* field, int gridSize, float min, float max, float isovalue)
{
    InitGPUSymbols();

    int numCubes = (gridSize - 1) * (gridSize - 1) * (gridSize - 1);
    int numGridPoints = gridSize * gridSize * gridSize;

    float* d_field;
    cudaMalloc(&d_field, numGridPoints * sizeof(float));
    cudaMemcpy(d_field, field, numGridPoints * sizeof(float), cudaMemcpyHostToDevice);

    int* d_vertexCounts;
    cudaMalloc(&d_vertexCounts, numCubes * sizeof(int));
    cudaMemset(d_vertexCounts, 0, numCubes * sizeof(int));

    int threadsPerBlock = 512;
    int numBlocks = (numCubes + threadsPerBlock - 1) / threadsPerBlock;

    CountVerticesKernel<<<numBlocks, threadsPerBlock>>>(d_field, d_vertexCounts, gridSize, isovalue);

    int* d_vertexOffsets;
    cudaMalloc(&d_vertexOffsets, numCubes * sizeof(int));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // doesnt launch kernel
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
        d_vertexCounts, d_vertexOffsets, numCubes);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
        d_vertexCounts, d_vertexOffsets, numCubes);

    int lastOffset, lastCount;
    cudaMemcpy(&lastOffset, d_vertexOffsets + numCubes - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastCount, d_vertexCounts + numCubes - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int totalVertices = lastOffset + lastCount;

    float* d_output;
    cudaMalloc(&d_output, totalVertices * 3 * sizeof(float));

    MarchingCubesKernel<<<numBlocks, threadsPerBlock>>>(d_field, d_vertexOffsets, d_output, gridSize, min, max, isovalue);

    std::vector<float> h_output(totalVertices * 3);
    cudaMemcpy(h_output.data(), d_output, totalVertices * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_temp_storage);
    cudaFree(d_vertexOffsets);
    cudaFree(d_vertexCounts);
    cudaFree(d_field);
    cudaFree(d_output);

    return h_output;
}