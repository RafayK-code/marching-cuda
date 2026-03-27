#include "utils.h"

#include <fstream>
#include <iostream>

void WritePLYFile(const float* vertices, size_t bufferSize, const std::string& filename)
{
    std::ofstream plyFile(filename, std::ios::binary);
    if (!plyFile.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return;
    }

    size_t numVertices = bufferSize / 3;
    size_t numFaces = numVertices / 3;

    // ply header
    plyFile << "ply\n";
    plyFile << "format binary_little_endian 1.0\n";
    plyFile << "element vertex " << numVertices << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "element face " << numFaces << "\n";
    plyFile << "property list uchar int vertex_indices\n";
    plyFile << "end_header\n";

    plyFile.write(reinterpret_cast<const char*>(vertices), bufferSize * sizeof(float));

    // faces
    for (int i = 0; i < numVertices; i += 3)
    {
        unsigned char count = 3;
        int idx0 = i;
        int idx1 = i + 1;
        int idx2 = i + 2;

        plyFile.write(reinterpret_cast<const char*>(&count), sizeof(unsigned char));
        plyFile.write(reinterpret_cast<const char*>(&idx0), sizeof(int));
        plyFile.write(reinterpret_cast<const char*>(&idx1), sizeof(int));
        plyFile.write(reinterpret_cast<const char*>(&idx2), sizeof(int));
    }

    plyFile.close();
}

std::vector<float> GenerateScalarField(int gridSize, float min, float max, std::function<float(float, float, float)> implicitFunc)
{
    std::vector<float> field(gridSize * gridSize * gridSize);

    float stepSize = (max - min) / (gridSize - 1);

    for (int x = 0; x < gridSize; x++)
    {
        for (int y = 0; y < gridSize; y++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                // convert grid indices to world coordinates
                float worldX = min + x * stepSize;
                float worldY = min + y * stepSize;
                float worldZ = min + z * stepSize;

                float value = implicitFunc(worldX, worldY, worldZ);

                int idx = x + y * gridSize + z * gridSize * gridSize;
                field[idx] = value;
            }
        }
    }

    return field;
}

std::vector<float> LoadCTHead(const std::string& directory)
{
    int width = 256;
    int height = 256;
    int depth = 256;

    std::vector<float> field(width * height * depth, 0.0f);

    for (int y = 0; y < 113; y++)
    {
        std::string filename = directory + "/CThead." + std::to_string(y + 1);

        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << filename << std::endl;
            continue;
        }

        std::vector<uint16_t> slice(width * depth);
        file.read(reinterpret_cast<char*>(slice.data()), slice.size() * sizeof(uint16_t));

        for (int z = 0; z < depth; z++)
        {
            for (int x = 0; x < width; x++)
            {
                uint16_t value = slice[x + z * width];

                // byte swap from big-endian to little-endian
                value = ((value & 0xFF) << 8) | ((value >> 8) & 0xFF);

                int flippedY = 112 - y;
                int idx = x + flippedY * width + z * width * height;
                field[idx] = value / 4095.0f;
            }
        }
    }

    std::cout << "Loaded CT head\n";

    return field;
}