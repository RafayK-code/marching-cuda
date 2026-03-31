#include "marching_cubes_cpu.h"

#include "common.h"

#include <glm/glm.hpp>

glm::vec3 ComputeGradient(const float* field, int x, int y, int z, int gridX, int gridY, int gridZ)
{
    // Clamp to valid range (avoid out of bounds)
    int x0 = std::max(x - 1, 0);
    int x1 = std::min(x + 1, gridX - 1);
    int y0 = std::max(y - 1, 0);
    int y1 = std::min(y + 1, gridY - 1);
    int z0 = std::max(z - 1, 0);
    int z1 = std::min(z + 1, gridZ - 1);

    // Central differences
    float dx = field[x1 + y * gridX + z * gridX * gridY] -
        field[x0 + y * gridX + z * gridX * gridY];
    float dy = field[x + y1 * gridX + z * gridX * gridY] -
        field[x + y0 * gridX + z * gridX * gridY];
    float dz = field[x + y * gridX + z1 * gridX * gridY] -
        field[x + y * gridX + z0 * gridX * gridY];

    return glm::normalize(glm::vec3(dx, dy, dz));
}

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

                    // World coordinates
                    float worldX = config.minX + (x + xyz[0]) * stepX;
                    float worldY = config.minY + (y + xyz[1]) * stepY;
                    float worldZ = config.minZ + (z + xyz[2]) * stepZ;

                    // Grid coordinates for gradient (fractional)
                    float gridPosX = x + xyz[0];
                    float gridPosY = y + xyz[1];
                    float gridPosZ = z + xyz[2];

                    // Compute gradient at this position
                    // (Use integer position for now, could interpolate for better quality)
                    int gx = (int)round(gridPosX);
                    int gy = (int)round(gridPosY);
                    int gz = (int)round(gridPosZ);

                    glm::vec3 normal = ComputeGradient(field, gx, gy, gz, config.gridX, config.gridY, config.gridZ);

                    res.push_back(worldX);
                    res.push_back(worldY);
                    res.push_back(worldZ);

                    res.push_back(normal.x);
                    res.push_back(normal.y);
                    res.push_back(normal.z);
                }
            }
        }
    }

    return res;
}