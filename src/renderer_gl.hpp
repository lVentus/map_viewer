// src/renderer_gl.hpp
#pragma once

#include <glad/glad.h>
#include <string>
#include <vector>
#include <array>

#include "app.hpp"

struct PolyBatchGpu {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    float color[4] = {1, 1, 1, 1};
    int layer = 0;
};

struct LineBatchGpu {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLsizei vertCount = 0;
    float color[4] = {1, 1, 1, 1};
    float width = 1.0f;
    int w = 1;
    int c = 0;
};

struct TextGpu {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLsizei vertCount = 0;
};

class RendererGL {
public:
    bool init();
    void shutdown();

    bool initFont(const std::string& ttfPath);
    bool textReady() const { return textReady_; }

    float fontPixelSize() const { return fontPixelSize_; }
    float estimateTextWidthPx(const std::string& s) const;

    // Build text triangles for labels (uses stb baked atlas internally).
    // Caller controls visibility filtering; this just turns labels into TextVertex list.
    void buildTextVerts(const std::vector<TextLabelCPU>& labels,
                        std::vector<TextVertex>& outVerts) const;

    // GL frame helpers
    void beginFrame(const AABB& view, float clearR, float clearG, float clearB);
    void endFrame();

    // Uploads
    void uploadLines(const LineBatchCPU& cpu, LineBatchGpu& out);
    void uploadPoly(const PolyMeshCPU& cpu, PolyBatchGpu& out);
    void uploadTextQuads(const std::vector<TextVertex>& verts, TextGpu& out);

    // Destroys
    void destroy(LineBatchGpu& b);
    void destroy(PolyBatchGpu& p);
    void destroy(TextGpu& t);

    // Draws
    void drawPoly(const PolyBatchGpu& p, const AABB& view);
    void drawLines(const LineBatchGpu& b, const AABB& view);
    void drawText(const TextGpu& t, const AABB& view);

    // Same heuristic as your original implementation.
    static float normalizeTextAngle(float angle);

private:
    GLuint colorProg_ = 0;
    GLuint textProg_ = 0;

    // font atlas (baked ASCII 32..126)
    int atlasW_ = 1024;
    int atlasH_ = 1024;
    float fontPixelSize_ = 32.0f;
    GLuint textAtlasTex_ = 0;
    bool textReady_ = false;

    // opaque baked table storage (defined in .cpp)
    struct BakedTable;
    BakedTable* baked_ = nullptr;

private:
    static GLuint compile_(GLenum type, const char* src);
    static GLuint link_(GLuint vs, GLuint fs);

    bool buildPrograms_();
    bool buildFontAtlas_(const std::string& ttfPath);
};