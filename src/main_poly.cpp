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
#include <algorithm>
#include <numeric>

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

// 2D point helper (we intentionally avoid adding heavy dependencies like glm).
struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Polygon {
    int layer = 0;            // bigger means on top
    float color[4] = {0.2f, 0.6f, 0.3f, 0.35f};
    std::vector<Vec2> vertices; // world-space vertices in order (CW/CCW)
};

struct Graph {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Polygon> polygons;
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

// Polygon GPU resources (one mesh per polygon for now; easy to optimize later).
struct PolygonMesh {
    int layer = 0;
    float color[4] = {1,1,1,1};
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
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

// Polygon triangulation helpers
static bool triangulateEarClipping(const std::vector<Vec2>& poly, std::vector<uint32_t>& outIndices);

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

// ----------------- Polygon triangulation (ear clipping) -----------------

static float signedArea(const std::vector<Vec2>& poly) {
    // Shoelace formula, returns positive for CCW.
    double a = 0.0;
    const size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
        const Vec2& p = poly[i];
        const Vec2& q = poly[(i + 1) % n];
        a += static_cast<double>(p.x) * static_cast<double>(q.y)
           - static_cast<double>(q.x) * static_cast<double>(p.y);
    }
    return static_cast<float>(0.5 * a);
}

static float cross(const Vec2& a, const Vec2& b, const Vec2& c) {
    // (b - a) x (c - a)
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

static bool pointInTri(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c) {
    // Barycentric via same-side tests; assumes triangle is CCW.
    const float c1 = cross(a, b, p);
    const float c2 = cross(b, c, p);
    const float c3 = cross(c, a, p);
    // Allow points on edges (>= 0) to avoid precision issues.
    return (c1 >= 0.0f && c2 >= 0.0f && c3 >= 0.0f);
}

static bool isConvexCCW(const Vec2& prev, const Vec2& curr, const Vec2& next) {
    return cross(prev, curr, next) > 0.0f;
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

    // Polygons (optional). Format (one polygon per line):
    // POLY layer r g b a n x0 y0 x1 y1 ... x(n-1) y(n-1)
    outGraph.polygons.clear();
    outGraph.polygons.reserve(static_cast<size_t>(std::max(0, numPolygons)));

    for (int i = 0; i < numPolygons; ++i) {
        std::string polyTag;
        if (!(in >> polyTag)) {
            std::cerr << "Failed to read polygon tag at index " << i << "\n";
            return false;
        }
        if (polyTag != "POLY") {
            std::cerr << "Expected 'POLY' tag, got '" << polyTag << "'\n";
            return false;
        }

        Polygon poly;
        int n = 0;
        if (!(in >> poly.layer
                 >> poly.color[0] >> poly.color[1] >> poly.color[2] >> poly.color[3]
                 >> n)) {
            std::cerr << "Failed to read polygon header at index " << i << "\n";
            return false;
        }
        if (n < 3) {
            std::cerr << "Polygon " << i << " has < 3 vertices; skipped\n";
            // Consume the remaining pairs if any (defensive)
            for (int k = 0; k < std::max(0, n); ++k) {
                float x, y;
                in >> x >> y;
            }
            continue;
        }

        poly.vertices.resize(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) {
            float x, y;
            if (!(in >> x >> y)) {
                std::cerr << "Failed to read polygon vertex " << j << " for polygon " << i << "\n";
                return false;
            }
            poly.vertices[static_cast<size_t>(j)] = Vec2{x, y};
        }
        // Normalize to CCW to simplify triangulation.
        if (signedArea(poly.vertices) < 0.0f) {
            std::reverse(poly.vertices.begin(), poly.vertices.end());
        }
        outGraph.polygons.push_back(std::move(poly));
    }

    std::cout << "Total nodes read: " << outGraph.nodes.size() << "\n";
    std::cout << "Total edges read: " << outGraph.edges.size() << "\n";
    std::cout << "Total polygons read: " << outGraph.polygons.size() << "\n";
    return true;
}

// ----------------- Polygon triangulation (ear clipping) -----------------

static bool triangulateEarClipping(const std::vector<Vec2>& poly, std::vector<uint32_t>& outIndices) {
    outIndices.clear();
    if (poly.size() < 3) return false;

    // Indices into poly
    std::vector<int> idx(poly.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Safety: avoid infinite loops on degenerate polygons
    int guard = 0;
    const int guardMax = static_cast<int>(poly.size()) * static_cast<int>(poly.size());

    while (idx.size() > 3 && guard++ < guardMax) {
        bool earFound = false;

        for (size_t i = 0; i < idx.size(); ++i) {
            const int i0 = idx[(i + idx.size() - 1) % idx.size()];
            const int i1 = idx[i];
            const int i2 = idx[(i + 1) % idx.size()];

            const Vec2& a = poly[static_cast<size_t>(i0)];
            const Vec2& b = poly[static_cast<size_t>(i1)];
            const Vec2& c = poly[static_cast<size_t>(i2)];

            if (!isConvexCCW(a, b, c)) continue;

            bool contains = false;
            for (size_t j = 0; j < idx.size(); ++j) {
                const int vi = idx[j];
                if (vi == i0 || vi == i1 || vi == i2) continue;
                const Vec2& p = poly[static_cast<size_t>(vi)];
                if (pointInTri(p, a, b, c)) {
                    contains = true;
                    break;
                }
            }
            if (contains) continue;

            // Emit ear
            outIndices.push_back(static_cast<uint32_t>(i0));
            outIndices.push_back(static_cast<uint32_t>(i1));
            outIndices.push_back(static_cast<uint32_t>(i2));

            idx.erase(idx.begin() + static_cast<long>(i));
            earFound = true;
            break;
        }

        if (!earFound) {
            return false;
        }
    }

    if (idx.size() == 3) {
        outIndices.push_back(static_cast<uint32_t>(idx[0]));
        outIndices.push_back(static_cast<uint32_t>(idx[1]));
        outIndices.push_back(static_cast<uint32_t>(idx[2]));
        return true;
    }
    return false;
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
uniform vec4 uColor;
void main() {
    FragColor = uColor;
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

    // ----------------- Build polygon GPU resources -----------------
    struct PolygonGPU {
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint ebo = 0;
        GLsizei indexCount = 0;
        int layer = 0;
        float color[4] = {1,1,1,1};
    };

    std::vector<PolygonGPU> polygonGpu;
    polygonGpu.reserve(graph.polygons.size());

    for (const auto& poly : graph.polygons) {
        std::vector<uint32_t> indices;
        if (!triangulateEarClipping(poly.vertices, indices)) {
            std::cerr << "Warning: triangulation failed for a polygon (layer=" << poly.layer << ")\n";
            continue;
        }

        PolygonGPU pg;
        pg.layer = poly.layer;
        pg.color[0] = poly.color[0];
        pg.color[1] = poly.color[1];
        pg.color[2] = poly.color[2];
        pg.color[3] = poly.color[3];
        pg.indexCount = static_cast<GLsizei>(indices.size());

        glGenVertexArrays(1, &pg.vao);
        glGenBuffers(1, &pg.vbo);
        glGenBuffers(1, &pg.ebo);

        glBindVertexArray(pg.vao);

        glBindBuffer(GL_ARRAY_BUFFER, pg.vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     poly.vertices.size() * sizeof(Vec2),
                     poly.vertices.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pg.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     indices.size() * sizeof(uint32_t),
                     indices.data(),
                     GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), (void*)0);

        glBindVertexArray(0);
        polygonGpu.push_back(pg);
    }

    std::sort(polygonGpu.begin(), polygonGpu.end(),
              [](const PolygonGPU& a, const PolygonGPU& b) { return a.layer < b.layer; });

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

        double left, right, bottom, top;
        getViewRect(left, right, bottom, top);
        glUniform4f(locView,
                    static_cast<float>(left),
                    static_cast<float>(right),
                    static_cast<float>(bottom),
                    static_cast<float>(top));

        // ---- Pass 1: polygons (under roads/nodes) ----
        for (const auto& p : polygonGpu) {
            if (p.vao == 0 || p.indexCount == 0) continue;
            glBindVertexArray(p.vao);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, p.ebo);
            glUniform4f(locColor, p.color[0], p.color[1], p.color[2], p.color[3]);
            glDrawElements(GL_TRIANGLES, p.indexCount, GL_UNSIGNED_INT, nullptr);
        }

        // ---- Pass 2: roads ----
        glBindVertexArray(vao);

        for (const auto& g : groups) {
            if (g.count == 0 || g.ebo == 0) continue;

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
            glUniform4f(locColor, g.color[0], g.color[1], g.color[2], 1.0f);

            float lineWidth = static_cast<float>(g.w);
            if (lineWidth < 1.0f) lineWidth = 1.0f;
            glLineWidth(lineWidth);

            glDrawElements(GL_LINES, g.count, GL_UNSIGNED_INT, nullptr);
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    for (auto& p : polygonGpu) {
        if (p.ebo) glDeleteBuffers(1, &p.ebo);
        if (p.vbo) glDeleteBuffers(1, &p.vbo);
        if (p.vao) glDeleteVertexArrays(1, &p.vao);
    }
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
