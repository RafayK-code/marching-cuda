#include "utils.h"
#include "marching_cubes_cpu.h"
#include <iostream>
#include <cmath>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_gl_interop.h>
#include "marching_cubes_gpu.cuh"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

const char* vertexShaderSource = R"(
#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 450 core
in vec3 FragPos;
in vec3 Normal;
out vec4 FragColor;

void main()
{
    vec3 color = vec3(0.85, 0.85, 0.85);
    
    // Main light
    vec3 lightDir1 = normalize(vec3(1.0, 1.0, 1.0));
    vec3 norm = normalize(Normal);
    float diff1 = max(dot(norm, lightDir1), 0.0);
    
    // Fill light (softer, from opposite side)
    vec3 lightDir2 = normalize(vec3(-0.5, 0.3, -0.5));
    float diff2 = max(dot(norm, lightDir2), 0.0) * 0.3;  // Weaker fill
    
    // Higher ambient + two lights
    vec3 finalColor = color * (diff1 * 0.5 + diff2 + 0.4);  // 0.6 ambient
    
    FragColor = vec4(finalColor, 1.0);
}
)";

// Camera state
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
float cameraDistance = 3.0f;
float cameraYaw = 0.0f;
float cameraPitch = 0.0f;

// Mouse state for camera control
double lastMouseX = 400.0;
double lastMouseY = 300.0;
bool firstMouse = true;
bool mousePressed = false;

// Mouse callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

    if (ImGui::GetIO().WantCaptureMouse)
        return;

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        mousePressed = (action == GLFW_PRESS);
        if (mousePressed)
        {
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
            firstMouse = true;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

    if (ImGui::GetIO().WantCaptureMouse)
        return;

    if (!mousePressed)
        return;

    if (firstMouse)
    {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
        return;
    }

    float xoffset = xpos - lastMouseX;
    float yoffset = ypos - lastMouseY; // Reversed
    lastMouseX = xpos;
    lastMouseY = ypos;

    float sensitivity = 0.5f;
    cameraYaw += xoffset * sensitivity;
    cameraPitch += yoffset * sensitivity;

    // Clamp pitch
    if (cameraPitch > 89.0f) cameraPitch = 89.0f;
    if (cameraPitch < -89.0f) cameraPitch = -89.0f;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);

    if (ImGui::GetIO().WantCaptureMouse)
        return;

    cameraDistance -= (float)yoffset * 0.5f;
    if (cameraDistance < 0.5f) cameraDistance = 0.5f;
    if (cameraDistance > 10.0f) cameraDistance = 10.0f;
}

int main()
{
#if 0
    MarchingCubesConfig config = {
        256, 113, 256,
        -1.0f, 1.0f,
        -1.0f, 1.0f,
        -1.0f, 1.0f,
        currentIsovalue
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
#endif
    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(1280, 720, "GLFW Triangle", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    // Load OpenGL with GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    glDisable(GL_SCISSOR_TEST);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init("#version 450");

    // Register callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Link shaders into program
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    std::cout << "Reading CT Data..." << std::endl;
    std::vector<float> field = LoadCTHead("../data/CThead");

    constexpr int maxCubes = (256 - 1) * (256 - 1) * (113 - 1);
    constexpr size_t maxVBOFloats = static_cast<size_t>(maxCubes) * 15 * 6;

    // Upload to GPU
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, maxVBOFloats * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // Get uniform locations
    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
    unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    // Global isovalue that we can change
    float currentIsovalue = 0.4f;
    bool needsRegeneration = true;

    int numVertices = 0;

    cudaGraphicsResource* vboResource;
    cudaGraphicsGLRegisterBuffer(&vboResource, VBO, cudaGraphicsMapFlagsWriteDiscard);

    // Render loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);  // Position it
        ImGui::SetNextWindowSize(ImVec2(400, 160), ImGuiCond_FirstUseEver);  // Make it bigger

        ImGui::Begin("Marching Cubes Controls");
        if (ImGui::SliderFloat("Isovalue", &currentIsovalue, 0.0f, 1.0f))
        {
            needsRegeneration = true;
        }
        ImGui::Text("Vertices: %d", numVertices);
        ImGui::Text("Triangles: %d", numVertices / 3);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::End();

        // Regenerate mesh if isovalue changed
        if (needsRegeneration)
        {
            MarchingCubesConfig config = {
                256, 113, 256,
                -1.0f, 1.0f,
                -1.0f, 1.0f,
                -1.0f, 1.0f,
                currentIsovalue
            };

            numVertices = MarchingCubesGPU_Interop(field.data(), config, vboResource);
            needsRegeneration = false;
        }

        // Clear
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update camera position
        cameraPos.x = cameraDistance * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
        cameraPos.y = cameraDistance * sin(glm::radians(cameraPitch));
        cameraPos.z = cameraDistance * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));

        // Create transforms
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1280.0f / 720.0f, 0.05f, 100.0f);

        // Send to shader
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Draw mesh
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, numVertices);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    cudaGraphicsUnregisterResource(vboResource);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}