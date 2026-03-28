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

__global__ void CountVerticesKernel(const float* field, int* vertexCounts, int gridX, int gridY, int gridZ, float isovalue)
{
    int cubeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numCubes = (gridX - 1) * (gridY - 1) * (gridZ - 1);

    if (cubeIdx >= numCubes)
        return;

    int cubesPerSlice = (gridX - 1) * (gridY - 1);
    int z = cubeIdx / cubesPerSlice;
    int remainder = cubeIdx % cubesPerSlice;
    int y = remainder / (gridX - 1);
    int x = remainder % (gridX - 1);

    float vertValues[8];
    vertValues[0] = field[x + y * gridX + z * gridX * gridY];
    vertValues[1] = field[(x + 1) + y * gridX + z * gridX * gridY];
    vertValues[2] = field[(x + 1) + y * gridX + (z + 1) * gridX * gridY];
    vertValues[3] = field[x + y * gridX + (z + 1) * gridX * gridY];
    vertValues[4] = field[x + (y + 1) * gridX + z * gridX * gridY];
    vertValues[5] = field[(x + 1) + (y + 1) * gridX + z * gridX * gridY];
    vertValues[6] = field[(x + 1) + (y + 1) * gridX + (z + 1) * gridX * gridY];
    vertValues[7] = field[x + (y + 1) * gridX + (z + 1) * gridX * gridY];

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

__global__ void MarchingCubesKernel(const float* field, const int* vertexOffsets, float* output, MarchingCubesConfig config)
{
    int cubeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numCubes = (config.gridX - 1) * (config.gridY - 1) * (config.gridZ - 1);

    if (cubeIdx >= numCubes)
        return;

    // Convert linear index to 3D using gridX, gridY, gridZ
    int cubesPerSlice = (config.gridX - 1) * (config.gridY - 1);
    int z = cubeIdx / cubesPerSlice;
    int remainder = cubeIdx % cubesPerSlice;
    int y = remainder / (config.gridX - 1);
    int x = remainder % (config.gridX - 1);

    // Field indexing using gridX and gridY
    float vertValues[8];
    vertValues[0] = field[x + y * config.gridX + z * config.gridX * config.gridY];
    vertValues[1] = field[(x + 1) + y * config.gridX + z * config.gridX * config.gridY];
    vertValues[2] = field[(x + 1) + y * config.gridX + (z + 1) * config.gridX * config.gridY];
    vertValues[3] = field[x + y * config.gridX + (z + 1) * config.gridX * config.gridY];
    vertValues[4] = field[x + (y + 1) * config.gridX + z * config.gridX * config.gridY];
    vertValues[5] = field[(x + 1) + (y + 1) * config.gridX + z * config.gridX * config.gridY];
    vertValues[6] = field[(x + 1) + (y + 1) * config.gridX + (z + 1) * config.gridX * config.gridY];
    vertValues[7] = field[x + (y + 1) * config.gridX + (z + 1) * config.gridX * config.gridY];

    int cubeCase = 0;
    for (int i = 0; i < 8; i++)
    {
        if (vertValues[i] < config.isovalue)
            cubeCase |= (1 << i);
    }

    int writeOffset = vertexOffsets[cubeIdx];

    const int* edges = d_MARCHING_CUBES_LUT[cubeCase];
    float stepX = (config.maxX - config.minX) / (config.gridX - 1);
    float stepY = (config.maxY - config.minY) / (config.gridY - 1);
    float stepZ = (config.maxZ - config.minZ) / (config.gridZ - 1);

    int vertexIdx = 0;
    for (int i = 0; i < 16; i++)
    {
        if (edges[i] == -1)
            break;

        const float* xyz = d_VERT_TABLE[edges[i]];

        float worldX = config.minX + (x + xyz[0]) * stepX;
        float worldY = config.minY + (y + xyz[1]) * stepY;
        float worldZ = config.minZ + (z + xyz[2]) * stepZ;

        int outIdx = (writeOffset + vertexIdx) * 3;
        output[outIdx + 0] = worldX;
        output[outIdx + 1] = worldY;
        output[outIdx + 2] = worldZ;

        vertexIdx++;
    }
}

std::vector<float> MarchingCubesGPU(const float* field, const MarchingCubesConfig& config)
{
    InitGPUSymbols();

    int numCubes = (config.gridX - 1) * (config.gridY - 1) * (config.gridZ - 1);
    int numGridPoints = config.gridX * config.gridY * config.gridZ;

    float* d_field;
    cudaMalloc(&d_field, numGridPoints * sizeof(float));
    cudaMemcpy(d_field, field, numGridPoints * sizeof(float), cudaMemcpyHostToDevice);

    int* d_vertexCounts;
    cudaMalloc(&d_vertexCounts, numCubes * sizeof(int));
    cudaMemset(d_vertexCounts, 0, numCubes * sizeof(int));

    int threadsPerBlock = 512;
    int numBlocks = (numCubes + threadsPerBlock - 1) / threadsPerBlock;

    CountVerticesKernel<<<numBlocks, threadsPerBlock>>>(d_field, d_vertexCounts, config.gridX, config.gridY, config.gridZ, config.isovalue);

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

    MarchingCubesKernel<<<numBlocks, threadsPerBlock>>>(d_field, d_vertexOffsets, d_output, config);

    std::vector<float> h_output(totalVertices * 3);
    cudaMemcpy(h_output.data(), d_output, totalVertices * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_temp_storage);
    cudaFree(d_vertexOffsets);
    cudaFree(d_vertexCounts);
    cudaFree(d_field);
    cudaFree(d_output);

    return h_output;
}