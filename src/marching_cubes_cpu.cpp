#include "marching_cubes_cpu.h"

#include "common.h"

std::vector<float> MarchingCubesCPU(const float* field, const MarchingCubesConfig& config)
{
    std::vector<float> res;

    float stepX = (config.maxX - config.minX) / (config.gridX - 1);
    float stepY = (config.maxY - config.minY) / (config.gridY - 1);
    float stepZ = (config.maxZ - config.minZ) / (config.gridZ - 1);

    for (int x = 0; x < config.gridX - 1; x++)
    {
        for (int y = 0; y < config.gridY - 1; y++)
        {
            for (int z = 0; z < config.gridZ - 1; z++)
            {
                // Field indexing now uses gridX, gridY, gridZ
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

                const int* edges = MARCHING_CUBES_LUT[cubeCase];
                for (int i = 0; i < 16; i++)
                {
                    if (edges[i] == -1)
                        break;

                    const float* xyz = VERT_TABLE[edges[i]];

                    float worldX = config.minX + (x + xyz[0]) * stepX;
                    float worldY = config.minY + (y + xyz[1]) * stepY;
                    float worldZ = config.minZ + (z + xyz[2]) * stepZ;

                    res.push_back(worldX);
                    res.push_back(worldY);
                    res.push_back(worldZ);
                }
            }
        }
    }

    return res;
}