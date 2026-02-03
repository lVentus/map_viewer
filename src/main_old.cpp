// main.cpp
// Simple graph viewer for .gm map files.
// - Mouse wheel: zoom around cursor
// - Left mouse drag: pan

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <utility>

// ----------------- Data structures -----------------

struct Bounds {
    double minx = 0.0;
    double miny = 0.0;
    double maxx = 1.0;
    double maxy = 1.0;
};

struct Node {
    double x = 0.0;
    double y = 0.0;
};

struct Edge {
    uint32_t u = 0;
    uint32_t v = 0;
    int w = 1;   // width class
    int c = 0;   // color class
};

struct Graph {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    Bounds bounds;
};

// Each group has a uniform width and color.
struct EdgeGroup {
    int w = 1;
    int c = 0;
    std::vector<uint32_t> indices; // 2 * #edges
    GLuint ebo = 0;
    GLsizei count = 0;
    float color[3] = {1.0f, 1.0f, 1.0f};
};

// ----------------- Camera state -----------------

struct Camera {
    double centerX = 0.0;
    double centerY = 0.0;
    double baseViewWidth = 1.0;   // world width when zoom = 1
    double baseViewHeight = 1.0;  // world height when zoom = 1
    double zoom = 1.0;            // > 0
    int windowWidth = 1280;
    int windowHeight = 720;
};

Camera gCamera;
bool   gMouseLeftDown = false;
double gLastCursorX = 0.0;
double gLastCursorY = 0.0;

// Forward declarations
bool loadGraphGM(const std::string& path, Graph& outGraph);
GLuint createShaderProgram();
void   initCameraFromBounds(const Bounds& b, int winW, int winH);
void   getViewRect(double& left, double& right, double& bottom, double& top);

// ----------------- Small helpers -----------------

// Read an int from a line that may contain trailing comments.
static bool readIntWithComment(std::istream& in, int& outVal) {
    std::string line;
    if (!std::getline(in, line)) {
        return false;
    }
    std::istringstream iss(line);
    return static_cast<bool>(iss >> outVal);
}

// Simple color mapping based on c.
static void colorFromType(int c, float outColor[3]) {
    const int idx = (c >= 0) ? (c % 8) : ((c % 8) + 8);
    switch (idx) {
        case 0: outColor[0] = 1.0f; outColor[1] = 1.0f; outColor[2] = 1.0f; break; // white
        case 1: outColor[0] = 0.4f; outColor[1] = 0.8f; outColor[2] = 1.0f; break; // blue-ish
        case 2: outColor[0] = 1.0f; outColor[1] = 0.4f; outColor[2] = 0.4f; break; // red-ish
        case 3: outColor[0] = 1.0f; outColor[1] = 0.8f; outColor[2] = 0.2f; break; // yellow
        case 4: outColor[0] = 0.4f; outColor[1] = 1.0f; outColor[2] = 0.4f; break; // green
        case 5: outColor[0] = 1.0f; outColor[1] = 0.4f; outColor[2] = 1.0f; break; // magenta
        case 6: outColor[0] = 0.8f; outColor[1] = 0.8f; outColor[2] = 0.8f; break; // light gray
        case 7: outColor[0] = 0.6f; outColor[1] = 0.6f; outColor[2] = 0.6f; break; // darker gray
        default: outColor[0] = 1.0f; outColor[1] = 1.0f; outColor[2] = 1.0f; break;
    }
}

// ----------------- GM loader -----------------

bool loadGraphGM(const std::string& path, Graph& outGraph) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open graph file: " << path << "\n";
        return false;
    }

    int numNodes = 0;
    int numEdges = 0;
    int numPolygons = 0;
    int numTexts = 0;

    if (!readIntWithComment(in, numNodes)) {
        std::cerr << "Failed to read node count\n";
        return false;
    }
    if (!readIntWithComment(in, numEdges)) {
        std::cerr << "Failed to read edge count\n";
        return false;
    }
    if (!readIntWithComment(in, numPolygons)) {
        std::cerr << "Failed to read polygon count\n";
        return false;
    }
    if (!readIntWithComment(in, numTexts)) {
        std::cerr << "Failed to read text count\n";
        return false;
    }

    std::string tag;
    if (!(in >> tag) || tag != "BOUNDS") {
        std::cerr << "Expected 'BOUNDS' line\n";
        return false;
    }
    Bounds b;
    if (!(in >> b.minx >> b.miny >> b.maxx >> b.maxy)) {
        std::cerr << "Failed to read bounds values\n";
        return false;
    }
    // Consume rest of line after BOUNDS
    std::string dummy;
    std::getline(in, dummy);

    if (numNodes <= 0 || numEdges < 0) {
        std::cerr << "Invalid counts in gm header\n";
        return false;
    }

    outGraph.bounds = b;
    outGraph.nodes.resize(static_cast<size_t>(numNodes));

    for (int i = 0; i < numNodes; ++i) {
        double x, y;
        if (!(in >> x >> y)) {
            std::cerr << "Failed to read node " << i << "\n";
            return false;
        }
        outGraph.nodes[i].x = x;
        outGraph.nodes[i].y = y;
    }

    outGraph.edges.clear();
    outGraph.edges.reserve(static_cast<size_t>(numEdges));

    for (int i = 0; i < numEdges; ++i) {
        uint32_t u, v;
        int w, c;
        if (!(in >> u >> v >> w >> c)) {
            std::cerr << "Failed to read edge " << i << "\n";
            return false;
        }
        if (u >= static_cast<uint32_t>(numNodes) ||
            v >= static_cast<uint32_t>(numNodes)) {
            std::cerr << "Warning: edge (" << u << ", " << v
                      << ") references unknown node id. Skipped.\n";
            continue;
        }
        Edge e;
        e.u = u;
        e.v = v;
        e.w = (w > 0) ? w : 1;
        e.c = c;
        outGraph.edges.push_back(e);
    }

    std::cout << "Total nodes read: " << outGraph.nodes.size() << "\n";
    std::cout << "Total edges read: " << outGraph.edges.size() << "\n";
    return true;
}

// ----------------- Camera helpers -----------------

// Initialize camera to show the full bounds.
void initCameraFromBounds(const Bounds& b, int winW, int winH) {
    gCamera.windowWidth = winW;
    gCamera.windowHeight = winH;

    const double width = std::max(1e-9, b.maxx - b.minx);
    const double height = std::max(1e-9, b.maxy - b.miny);

    const double aspect = static_cast<double>(winW) / static_cast<double>(winH);
    const double boundsAspect = width / height;

    if (boundsAspect >= aspect) {
        gCamera.baseViewWidth = width;
        gCamera.baseViewHeight = width / aspect;
    } else {
        gCamera.baseViewHeight = height;
        gCamera.baseViewWidth = height * aspect;
    }

    gCamera.centerX = 0.5 * (b.minx + b.maxx);
    gCamera.centerY = 0.5 * (b.miny + b.maxy);
    gCamera.zoom = 1.0;
}

// Compute current view rectangle in world coordinates.
void getViewRect(double& left, double& right, double& bottom, double& top) {
    const double vw = gCamera.baseViewWidth / gCamera.zoom;
    const double vh = gCamera.baseViewHeight / gCamera.zoom;

    left   = gCamera.centerX - vw * 0.5;
    right  = gCamera.centerX + vw * 0.5;
    bottom = gCamera.centerY - vh * 0.5;
    top    = gCamera.centerY + vh * 0.5;
}

// ----------------- GLFW callbacks -----------------

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (width <= 0 || height <= 0) return;
    gCamera.windowWidth = width;
    gCamera.windowHeight = height;
    glViewport(0, 0, width, height);

    // Adjust base view height to maintain aspect, keep current world width.
    double left, right, bottom, top;
    getViewRect(left, right, bottom, top);
    const double viewWidth = right - left;
    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    gCamera.baseViewWidth = viewWidth * gCamera.zoom;
    gCamera.baseViewHeight = gCamera.baseViewWidth / aspect;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    (void)window; (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            gMouseLeftDown = true;
            glfwGetCursorPos(window, &gLastCursorX, &gLastCursorY);
        } else if (action == GLFW_RELEASE) {
            gMouseLeftDown = false;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (!gMouseLeftDown) return;

    const double dx = xpos - gLastCursorX;
    const double dy = ypos - gLastCursorY;
    gLastCursorX = xpos;
    gLastCursorY = ypos;

    double left, right, bottom, top;
    getViewRect(left, right, bottom, top);
    const double vw = right - left;
    const double vh = top - bottom;

    const double ndc_dx = 2.0 * dx / static_cast<double>(gCamera.windowWidth);
    const double ndc_dy = 2.0 * dy / static_cast<double>(gCamera.windowHeight);

    const double world_dx = ndc_dx * vw * 0.5;
    const double world_dy = -ndc_dy * vh * 0.5; // screen y goes down

    gCamera.centerX -= world_dx;
    gCamera.centerY -= world_dy;
}

static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window; (void)xoffset;
    if (yoffset == 0.0) return;

    double zoomFactor = (yoffset > 0.0) ? 1.1 : (1.0 / 1.1);

    double left, right, bottom, top;
    getViewRect(left, right, bottom, top);
    const double vw = right - left;
    const double vh = top - bottom;

    double cx, cy;
    glfwGetCursorPos(window, &cx, &cy);
    const double nx = cx / static_cast<double>(gCamera.windowWidth);   // [0,1]
    const double ny = cy / static_cast<double>(gCamera.windowHeight);  // [0,1]

    const double world_x = left + nx * vw;
    const double world_y = top - ny * vh;

    gCamera.zoom *= zoomFactor;
    if (gCamera.zoom < 0.0001) gCamera.zoom = 0.0001;

    const double newVW = gCamera.baseViewWidth / gCamera.zoom;
    const double newVH = gCamera.baseViewHeight / gCamera.zoom;

    gCamera.centerX = world_x + (0.5 - nx) * newVW;
    gCamera.centerY = world_y + (ny - 0.5) * newVH;
}

// ----------------- Shader creation -----------------

GLuint createShaderProgram() {
    const char* vsSrc = R"(#version 330 core
layout (location = 0) in vec2 aPos;

uniform vec4 uView; // left, right, bottom, top

void main() {
    float left   = uView.x;
    float right  = uView.y;
    float bottom = uView.z;
    float top    = uView.w;

    float x_ndc = ( (aPos.x - left) / (right - left) ) * 2.0 - 1.0;
    float y_ndc = ( (aPos.y - bottom) / (top - bottom) ) * 2.0 - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, 0.0, 1.0);
}
)";

    const char* fsSrc = R"(#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
)";

    auto compileShader = [](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            GLint len = 0;
            glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
            std::string log(len, '\0');
            glGetShaderInfoLog(s, len, nullptr, log.data());
            std::cerr << "Shader compile error:\n" << log << "\n";
        }
        return s;
    };

    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ----------------- main -----------------

int main(int argc, char** argv) {
    std::string filePath = "data/test.gm";
    if (argc >= 2) {
        filePath = argv[1];
    }

    Graph graph;
    if (!loadGraphGM(filePath, graph)) {
        return EXIT_FAILURE;
    }

    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int winW = 1280;
    const int winH = 720;
    GLFWwindow* window = glfwCreateWindow(winW, winH, "Map Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return EXIT_FAILURE;
    }

    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << "OpenGL version: " << (version ? reinterpret_cast<const char*>(version) : "unknown") << "\n";

    initCameraFromBounds(graph.bounds, winW, winH);

    // Upload node positions to a VBO.
    std::vector<float> vertexData;
    vertexData.reserve(graph.nodes.size() * 2);
    for (const auto& n : graph.nodes) {
        vertexData.push_back(static_cast<float>(n.x));
        vertexData.push_back(static_cast<float>(n.y));
    }

    GLuint vao = 0;
    GLuint vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 vertexData.size() * sizeof(float),
                 vertexData.data(),
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // Group edges by (w,c) and create one EBO per group.
    std::map<std::pair<int,int>, size_t> groupIndex;
    std::vector<EdgeGroup> groups;

    for (const auto& e : graph.edges) {
        std::pair<int,int> key(e.w, e.c);
        auto it = groupIndex.find(key);
        size_t idx;
        if (it == groupIndex.end()) {
            EdgeGroup g;
            g.w = e.w;
            g.c = e.c;
            colorFromType(e.c, g.color);
            groups.push_back(g);
            idx = groups.size() - 1;
            groupIndex[key] = idx;
        } else {
            idx = it->second;
        }
        groups[idx].indices.push_back(e.u);
        groups[idx].indices.push_back(e.v);
    }

    for (auto& g : groups) {
        if (g.indices.empty()) continue;
        glGenBuffers(1, &g.ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     g.indices.size() * sizeof(uint32_t),
                     g.indices.data(),
                     GL_STATIC_DRAW);
        g.count = static_cast<GLsizei>(g.indices.size());
        // keep CPU copy small if needed
        g.indices.shrink_to_fit();
    }

    glBindVertexArray(0);

    GLuint program = createShaderProgram();
    GLint locView  = glGetUniformLocation(program, "uView");
    GLint locColor = glGetUniformLocation(program, "uColor");

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::cout << "Loaded graph: " << graph.nodes.size()
              << " nodes, " << graph.edges.size() << " edges\n";

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindVertexArray(vao);

        double left, right, bottom, top;
        getViewRect(left, right, bottom, top);
        glUniform4f(locView,
                    static_cast<float>(left),
                    static_cast<float>(right),
                    static_cast<float>(bottom),
                    static_cast<float>(top));

        for (const auto& g : groups) {
            if (g.count == 0 || g.ebo == 0) continue;

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
            glUniform3fv(locColor, 1, g.color);

            float lineWidth = static_cast<float>(g.w);
            if (lineWidth < 1.0f) lineWidth = 1.0f;
            glLineWidth(lineWidth);

            glDrawElements(GL_LINES, g.count, GL_UNSIGNED_INT, nullptr);
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    for (auto& g : groups) {
        if (g.ebo) glDeleteBuffers(1, &g.ebo);
    }
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
