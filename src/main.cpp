// src/main.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "app.hpp"
#include "map_cache.hpp"
#include "renderer_gl.hpp"

// ---- camera state ----

static Camera gCam;
static bool gMouseLeft = false;
static double gLastX = 0.0;
static double gLastY = 0.0;
static bool gDirty = true;

static void getViewRect(double& left, double& right, double& bottom, double& top) {
    const double vw = gCam.baseViewWidth / gCam.zoom;
    const double vh = gCam.baseViewHeight / gCam.zoom;
    left   = gCam.centerX - vw * 0.5;
    right  = gCam.centerX + vw * 0.5;
    bottom = gCam.centerY - vh * 0.5;
    top    = gCam.centerY + vh * 0.5;
}

static void initCameraFromBounds(const Bounds& b, int fbW, int fbH) {
    gCam.fbW = fbW;
    gCam.fbH = fbH;

    gCam.centerX = 0.5 * (b.minx + b.maxx);
    gCam.centerY = 0.5 * (b.miny + b.maxy);

    const double w = (b.maxx - b.minx);
    const double h = (b.maxy - b.miny);
    const double aspect = (fbH > 0) ? (double)fbW / (double)fbH : 16.0 / 9.0;

    double vw = w;
    double vh = h;
    if (vw / vh < aspect) vw = vh * aspect;
    else                 vh = vw / aspect;

    gCam.baseViewWidth  = vw * 1.05;
    gCam.baseViewHeight = vh * 1.05;
    gCam.zoom = 1.0;
}

static std::string findDefaultFontPath() {
    const std::vector<std::string> candidates = {
        "data/DejaVuSans.ttf",
        "external/fonts/DejaVuSans.ttf",
        "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    };
    for (const auto& p : candidates) {
        std::ifstream f(p, std::ios::binary);
        if (f.good()) return p;
    }
    return "";
}

// ---- input callbacks ----

static void framebufferSizeCallback(GLFWwindow* window, int w, int h) {
    (void)window;
    gCam.fbW = w;
    gCam.fbH = h;
    glViewport(0, 0, w, h);
    gDirty = true;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS) {
        gMouseLeft = true;
        glfwGetCursorPos(window, &gLastX, &gLastY);
    } else if (action == GLFW_RELEASE) {
        gMouseLeft = false;
    }
}

static void cursorPosCallback(GLFWwindow* window, double x, double y) {
    if (!gMouseLeft) return;

    const double dx = x - gLastX;
    const double dy = y - gLastY;
    gLastX = x;
    gLastY = y;

    int winW = 0, winH = 0;
    glfwGetWindowSize(window, &winW, &winH);
    if (winW <= 0 || winH <= 0) return;

    double left, right, bottom, top;
    getViewRect(left, right, bottom, top);

    const double ndc_dx = 2.0 * dx / (double)winW;
    const double ndc_dy = 2.0 * dy / (double)winH;

    const double world_dx = ndc_dx * (right - left) * 0.5;
    const double world_dy = -ndc_dy * (top - bottom) * 0.5;

    gCam.centerX -= world_dx;
    gCam.centerY -= world_dy;
    gDirty = true;
}

static void scrollCallback(GLFWwindow* window, double xoff, double yoff) {
    (void)xoff;
    if (yoff == 0.0) return;

    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(window, &cx, &cy);

    int winW = 0, winH = 0;
    glfwGetWindowSize(window, &winW, &winH);
    if (winW <= 0 || winH <= 0) return;

    double left, right, bottom, top;
    getViewRect(left, right, bottom, top);

    const double nx = cx / (double)winW;
    const double ny = cy / (double)winH;

    const double worldX = left + nx * (right - left);
    const double worldY = top - ny * (top - bottom);

    const double zoomFactor = std::pow(1.1, yoff);
    double newZoom = gCam.zoom * zoomFactor;
    newZoom = std::max(0.05, std::min(200.0, newZoom));
    gCam.zoom = newZoom;

    getViewRect(left, right, bottom, top);

    const double newWorldX = left + nx * (right - left);
    const double newWorldY = top - ny * (top - bottom);

    gCam.centerX += (worldX - newWorldX);
    gCam.centerY += (worldY - newWorldY);

    gDirty = true;
}

// ---- main ----

int main(int argc, char** argv) {
    std::string gmPath;
    if (argc >= 2) gmPath = argv[1];
    else {
        std::cerr << "Usage: " << argv[0] << " <map.gm> [font.ttf]\n";
        return 1;
    }

    std::string fontPath;
    if (argc >= 3) fontPath = argv[2];
    else fontPath = findDefaultFontPath();

    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "map_viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create window\n";
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to load GL\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    framebufferSizeCallback(window, fbW, fbH);

    RendererGL renderer;
    if (!renderer.init()) {
        std::cerr << "Renderer init failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (!fontPath.empty()) {
        if (!renderer.initFont(fontPath)) {
            std::cerr << "Font init failed, text disabled\n";
        }
    } else {
        std::cerr << "No font found. Put a TTF at data/DejaVuSans.ttf (or pass argv[2]). Text disabled.\n";
    }

    MapCache cache;
    CacheInfo ci{};
    if (!cache.openOrBuild(gmPath, ci)) {
        std::cerr << "ERROR: cache open/build failed\n";
        renderer.shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    initCameraFromBounds(ci.bounds, fbW, fbH);

    cache.start();

    double lastViewArea = -1.0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double left, right, bottom, top;
        getViewRect(left, right, bottom, top);

        const AABB viewBB{(float)left, (float)bottom, (float)right, (float)top};

        cache.nextFrame();

        if (gDirty) {
            const bool visChanged = cache.updateVisible(viewBB);
            cache.enqueueRequests();

            const double viewArea = (right - left) * (top - bottom);
            if (lastViewArea < 0.0) {
                lastViewArea = viewArea;
                cache.bumpTextRev();
            } else {
                const double denom = std::max(1.0, lastViewArea);
                const double rel = std::abs(viewArea - lastViewArea) / denom;
                if (rel > 0.002) {
                    lastViewArea = viewArea;
                    cache.bumpTextRev();
                } else if (visChanged) {
                    cache.bumpTextRev();
                }
            }

            gDirty = false;
        }

        const double viewArea = (right - left) * (top - bottom);

        cache.uploadReady(renderer, viewArea);
        cache.evict();

        renderer.beginFrame(viewBB, 0.08f, 0.09f, 0.10f);

        // Polys: collect & layer-sort (same behavior)
        if (cache.anyPolyLoaded()) {
            std::vector<const PolyBatchGpu*> polys;
            polys.reserve(1024);

            for (const auto& tk : cache.visibleTiles()) {
                const auto* t = cache.findTile(tk.tx, tk.ty);
                if (!t || !t->loaded) continue;
                for (const auto& pg : t->gpuPolys) polys.push_back(&pg);
            }

            std::sort(polys.begin(), polys.end(),
                      [](const PolyBatchGpu* a, const PolyBatchGpu* b) { return a->layer < b->layer; });

            for (const PolyBatchGpu* p : polys) {
                renderer.drawPoly(*p, viewBB);
            }
        }

        // Roads
        for (const auto& tk : cache.visibleTiles()) {
            const auto* t = cache.findTile(tk.tx, tk.ty);
            if (!t || !t->loaded) continue;
            if (!aabbIntersects(t->bb, viewBB)) continue;

            for (const auto& l : t->gpuLines) {
                renderer.drawLines(l, viewBB);
            }
        }

        // Text
        if (renderer.textReady() && cache.anyTextLoaded()) {
            for (const auto& tk : cache.visibleTiles()) {
                auto* t = cache.findTile(tk.tx, tk.ty);
                if (!t || !t->loaded) continue;
                if (t->texts.empty()) continue;

                cache.ensureTileText(renderer, tk.tx, tk.ty, viewArea);

                if (t->gpuTextValid && t->gpuText.vertCount > 0) {
                    renderer.drawText(t->gpuText, viewBB);
                }
            }
        }

        renderer.endFrame();
        glfwSwapBuffers(window);
    }

    cache.shutdown();
    renderer.shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}