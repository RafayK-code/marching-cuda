#include "utils.h"
#include "marching_cubes_cpu.h"
#include "marching_cubes_gpu.cuh"
#include <iostream>
#include <cmath>
#include <chrono>

float sphere(float x, float y, float z)
{
    return x * x + y * y + z * z;
}

float torus(float x, float y, float z)
{
    float R = 1.5f;
    float r = 0.5f;

    float temp = std::sqrt(x * x + y * y) - R;
    return temp * temp + z * z - r * r;
}

int main()
{
    MarchingCubesConfig config = {
        256, 256, 256,
        -1.0f, 1.0f,
        -2.0f, 2.0f,
        -1.0f, 1.0f,
        0.4f
    };

    //std::cout << "=== Marching Cubes CPU vs GPU Benchmark ===" << std::endl;
    //std::cout << "Grid size: " << gridSize << "^3" << std::endl;
    //std::cout << "Num cubes: " << (gridSize - 1) * (gridSize - 1) * (gridSize - 1) << std::endl << std::endl;

    //std::cout << "Generating scalar field..." << std::endl;
    //std::vector<float> field = GenerateScalarField(gridSize, min, max, sphere);

    std::cout << "Reading CT Data..." << std::endl;
    std::vector<float> field = LoadCTHead("../data/CThead");

    //std::cout << "Reading MR Data..." << std::endl;
    //std::vector<float> field = LoadMRBrain("../data/MRbrain");

    // CPU
    std::cout << "\n--- CPU Version ---" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_vertices = MarchingCubesCPU(field.data(), config);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "Vertices: " << cpu_vertices.size() / 3 << std::endl;
    std::cout << "Triangles: " << cpu_vertices.size() / 9 << std::endl;
    std::cout << "Time: " << cpu_time_ms << " ms" << std::endl;

    WritePLYFile(cpu_vertices.data(), cpu_vertices.size(), "output_cpu.ply");
    std::cout << "Wrote output_cpu.ply" << std::endl;

    // GPU
    std::cout << "\n--- GPU Version ---" << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> gpu_vertices = MarchingCubesGPU(field.data(), config);
    auto gpu_end = std::chrono::high_resolution_clock::now();

    double gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "Vertices: " << gpu_vertices.size() / 3 << std::endl;
    std::cout << "Triangles: " << gpu_vertices.size() / 9 << std::endl;
    std::cout << "Time: " << gpu_time_ms << " ms" << std::endl;

    WritePLYFile(gpu_vertices.data(), gpu_vertices.size(), "output_gpu.ply");
    std::cout << "Wrote output_gpu.ply" << std::endl;

    // SUMMARY
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "CPU Time: " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time_ms / gpu_time_ms) << "x" << std::endl;

    return 0;
}