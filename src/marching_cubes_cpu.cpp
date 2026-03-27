#include "marching_cubes_cpu.h"

#include "common.h"

std::vector<float> MarchingCubesCPU(const float* field, int gridSize, float min, float max, float isovalue)
{
    std::vector<float> res;
    float stepSize = (max - min) / (gridSize - 1);

    for (int x = 0; x < gridSize - 1; x++)
    {
        for (int y = 0; y < gridSize - 1; y++)
        {
            for (int z = 0; z < gridSize - 1; z++)
            {
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

                const int* edges = MARCHING_CUBES_LUT[cubeCase];
                for (int i = 0; i < 16; i++)
                {
                    if (edges[i] == -1)
                        break;

                    const float* xyz = VERT_TABLE[edges[i]];
                    float worldX = min + (x + xyz[0]) * stepSize;
                    float worldY = min + (y + xyz[1]) * stepSize;
                    float worldZ = min + (z + xyz[2]) * stepSize;

                    res.push_back(worldX);
                    res.push_back(worldY);
                    res.push_back(worldZ);
                }
            }
        }
    }

    return res;
}