// src/renderer_gl.cpp

// stb implementation must appear in exactly one translation unit.
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include "renderer_gl.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

// ---- private baked table ----
struct RendererGL::BakedTable {
    stbtt_bakedchar baked[96]{};
};

// ---- shader sources ----
static const char* kColorVS = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
uniform vec4 uView; // left,right,bottom,top
void main(){
    float x = (aPos.x - uView.x) / (uView.y - uView.x) * 2.0 - 1.0;
    float y = (aPos.y - uView.z) / (uView.w - uView.z) * 2.0 - 1.0;
    gl_Position = vec4(x,y,0,1);
}
)GLSL";

static const char* kColorFS = R"GLSL(
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main(){ FragColor = uColor; }
)GLSL";

static const char* kTextVS = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec4 aColor;
out vec2 vUV;
out vec4 vColor;
uniform vec4 uView; // left,right,bottom,top
void main(){
    float x = (aPos.x - uView.x) / (uView.y - uView.x) * 2.0 - 1.0;
    float y = (aPos.y - uView.z) / (uView.w - uView.z) * 2.0 - 1.0;
    gl_Position = vec4(x,y,0,1);
    vUV = aUV;
    vColor = aColor;
}
)GLSL";

static const char* kTextFS = R"GLSL(
#version 330 core
in vec2 vUV;
in vec4 vColor;
out vec4 FragColor;
uniform sampler2D uTex;
void main(){
    float a = texture(uTex, vUV).r;
    FragColor = vec4(vColor.rgb, vColor.a * a);
}
)GLSL";

// ---- helpers ----
GLuint RendererGL::compile_(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log((size_t)len, '\0');
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile failed:\n" << log << "\n";
    }
    return s;
}

GLuint RendererGL::link_(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);

    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log((size_t)len, '\0');
        glGetProgramInfoLog(p, len, nullptr, log.data());
        std::cerr << "Program link failed:\n" << log << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

bool RendererGL::buildPrograms_() {
    const GLuint vs0 = compile_(GL_VERTEX_SHADER, kColorVS);
    const GLuint fs0 = compile_(GL_FRAGMENT_SHADER, kColorFS);
    colorProg_ = link_(vs0, fs0);

    const GLuint vs1 = compile_(GL_VERTEX_SHADER, kTextVS);
    const GLuint fs1 = compile_(GL_FRAGMENT_SHADER, kTextFS);
    textProg_ = link_(vs1, fs1);

    return colorProg_ != 0 && textProg_ != 0;
}

static bool readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const size_t n = (size_t)in.tellg();
    in.seekg(0, std::ios::beg);
    out.resize(n);
    in.read((char*)out.data(), (std::streamsize)n);
    return (bool)in;
}

bool RendererGL::buildFontAtlas_(const std::string& ttfPath) {
    std::vector<uint8_t> ttf;
    if (!readFileBytes(ttfPath, ttf)) {
        std::cerr << "Font read failed: " << ttfPath << "\n";
        return false;
    }

    if (!baked_) baked_ = new BakedTable();

    std::vector<uint8_t> bitmap((size_t)atlasW_ * (size_t)atlasH_, 0);
    const int res = stbtt_BakeFontBitmap(
        ttf.data(), 0, fontPixelSize_,
        bitmap.data(), atlasW_, atlasH_,
        32, 96, baked_->baked
    );

    if (res <= 0) {
        std::cerr << "stbtt_BakeFontBitmap failed (try a different TTF or bigger atlas).\n";
        return false;
    }

    if (!textAtlasTex_) glGenTextures(1, &textAtlasTex_);
    glBindTexture(GL_TEXTURE_2D, textAtlasTex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, atlasW_, atlasH_, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    textReady_ = true;
    std::cout << "Font loaded: " << ttfPath << "\n";
    return true;
}

// ---- public API ----
bool RendererGL::init() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return buildPrograms_();
}

void RendererGL::shutdown() {
    if (textAtlasTex_) {
        glDeleteTextures(1, &textAtlasTex_);
        textAtlasTex_ = 0;
    }
    textReady_ = false;

    if (colorProg_) {
        glDeleteProgram(colorProg_);
        colorProg_ = 0;
    }
    if (textProg_) {
        glDeleteProgram(textProg_);
        textProg_ = 0;
    }

    delete baked_;
    baked_ = nullptr;
}

bool RendererGL::initFont(const std::string& ttfPath) {
    if (ttfPath.empty()) return false;
    if (!buildFontAtlas_(ttfPath)) {
        textReady_ = false;
        return false;
    }
    return true;
}

void RendererGL::beginFrame(const AABB& view, float clearR, float clearG, float clearB) {
    glClearColor(clearR, clearG, clearB, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(colorProg_);
    glUniform4f(glGetUniformLocation(colorProg_, "uView"), view.minx, view.maxx, view.miny, view.maxy);
}

void RendererGL::endFrame() {
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void RendererGL::uploadLines(const LineBatchCPU& cpu, LineBatchGpu& out) {
    destroy(out);
    if (cpu.verts.empty()) return;

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);

    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(cpu.verts.size() * sizeof(Vec2)), cpu.verts.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), (void*)0);

    glBindVertexArray(0);

    out.vertCount = (GLsizei)cpu.verts.size();
    out.w = cpu.w;
    out.c = cpu.c;
}

void RendererGL::uploadPoly(const PolyMeshCPU& cpu, PolyBatchGpu& out) {
    destroy(out);
    if (cpu.verts.empty() || cpu.idx.empty()) return;

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glGenBuffers(1, &out.ebo);

    glBindVertexArray(out.vao);

    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(cpu.verts.size() * sizeof(Vec2)), cpu.verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), (void*)0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(cpu.idx.size() * sizeof(uint32_t)), cpu.idx.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);

    out.indexCount = (GLsizei)cpu.idx.size();
    out.layer = cpu.layer;
    out.color[0] = cpu.color[0];
    out.color[1] = cpu.color[1];
    out.color[2] = cpu.color[2];
    out.color[3] = cpu.color[3];
}

void RendererGL::uploadTextQuads(const std::vector<TextVertex>& verts, TextGpu& out) {
    destroy(out);
    if (verts.empty()) return;

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);

    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(TextVertex)), verts.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)(2 * sizeof(float)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)(4 * sizeof(float)));

    glBindVertexArray(0);

    out.vertCount = (GLsizei)verts.size();
}

void RendererGL::destroy(LineBatchGpu& b) {
    if (b.vbo) { glDeleteBuffers(1, &b.vbo); b.vbo = 0; }
    if (b.vao) { glDeleteVertexArrays(1, &b.vao); b.vao = 0; }
    b.vertCount = 0;
}

void RendererGL::destroy(PolyBatchGpu& p) {
    if (p.ebo) { glDeleteBuffers(1, &p.ebo); p.ebo = 0; }
    if (p.vbo) { glDeleteBuffers(1, &p.vbo); p.vbo = 0; }
    if (p.vao) { glDeleteVertexArrays(1, &p.vao); p.vao = 0; }
    p.indexCount = 0;
}

void RendererGL::destroy(TextGpu& t) {
    if (t.vbo) { glDeleteBuffers(1, &t.vbo); t.vbo = 0; }
    if (t.vao) { glDeleteVertexArrays(1, &t.vao); t.vao = 0; }
    t.vertCount = 0;
}

void RendererGL::drawPoly(const PolyBatchGpu& p, const AABB& view) {
    glUseProgram(colorProg_);
    glUniform4f(glGetUniformLocation(colorProg_, "uView"), view.minx, view.maxx, view.miny, view.maxy);
    glUniform4f(glGetUniformLocation(colorProg_, "uColor"), p.color[0], p.color[1], p.color[2], p.color[3]);
    glBindVertexArray(p.vao);
    glDrawElements(GL_TRIANGLES, p.indexCount, GL_UNSIGNED_INT, (void*)0);
}

void RendererGL::drawLines(const LineBatchGpu& b, const AABB& view) {
    glUseProgram(colorProg_);
    glUniform4f(glGetUniformLocation(colorProg_, "uView"), view.minx, view.maxx, view.miny, view.maxy);
    glUniform4f(glGetUniformLocation(colorProg_, "uColor"), b.color[0], b.color[1], b.color[2], b.color[3]);
    glLineWidth(b.width);
    glBindVertexArray(b.vao);
    glDrawArrays(GL_LINES, 0, b.vertCount);
}

void RendererGL::drawText(const TextGpu& t, const AABB& view) {
    if (!textReady_ || !textAtlasTex_ || !t.vao || t.vertCount == 0) return;

    glUseProgram(textProg_);
    glUniform4f(glGetUniformLocation(textProg_, "uView"), view.minx, view.maxx, view.miny, view.maxy);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textAtlasTex_);
    glUniform1i(glGetUniformLocation(textProg_, "uTex"), 0);

    glBindVertexArray(t.vao);
    glDrawArrays(GL_TRIANGLES, 0, t.vertCount);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

float RendererGL::estimateTextWidthPx(const std::string& s) const {
    if (!baked_) return 0.0f;
    float x = 0.0f;
    for (unsigned char ch : s) {
        if (ch < 32 || ch > 126) continue;
        x += baked_->baked[ch - 32].xadvance;
    }
    return x;
}

float RendererGL::normalizeTextAngle(float angle) {
    const float twoPi = 6.2831853071795864769f;
    if (std::fabs(angle) > twoPi) {
        return angle * 3.14159265358979323846f / 180.0f;
    }
    return angle;
}

void RendererGL::buildTextVerts(const std::vector<TextLabelCPU>& labels,
                               std::vector<TextVertex>& outVerts) const {
    outVerts.clear();
    if (!textReady_ || !baked_) return;

    // Upper bound reserve; each glyph -> 6 verts.
    size_t estGlyphs = 0;
    for (const auto& l : labels) estGlyphs += l.text.size();
    outVerts.reserve(estGlyphs * 6);

    for (const auto& lbl : labels) {
        const float ang = normalizeTextAngle(lbl.angle);
        const float cs = std::cos(ang);
        const float sn = std::sin(ang);

        const float scale = (lbl.size > 1e-3f) ? (lbl.size / fontPixelSize_) : 1.0f;

        float penX = 0.0f;
        float penY = 0.0f;

        for (unsigned char ch : lbl.text) {
            if (ch < 32 || ch > 126) continue;

            stbtt_aligned_quad q{};
            stbtt_GetBakedQuad(baked_->baked, atlasW_, atlasH_, (int)ch - 32, &penX, &penY, &q, 1);

            const float lx0 = q.x0 * scale;
            const float ly0 = -q.y0 * scale;
            const float lx1 = q.x1 * scale;
            const float ly1 = -q.y1 * scale;

            auto xform = [&](float lx, float ly) -> Vec2 {
                const float rx = lx * cs - ly * sn;
                const float ry = lx * sn + ly * cs;
                return Vec2{lbl.pos.x + rx, lbl.pos.y + ry};
            };

            const Vec2 p0 = xform(lx0, ly0);
            const Vec2 p1 = xform(lx1, ly0);
            const Vec2 p2 = xform(lx1, ly1);
            const Vec2 p3 = xform(lx0, ly1);

            TextVertex v0{p0.x, p0.y, q.s0, q.t0, lbl.color[0], lbl.color[1], lbl.color[2], lbl.color[3]};
            TextVertex v1{p1.x, p1.y, q.s1, q.t0, lbl.color[0], lbl.color[1], lbl.color[2], lbl.color[3]};
            TextVertex v2{p2.x, p2.y, q.s1, q.t1, lbl.color[0], lbl.color[1], lbl.color[2], lbl.color[3]};
            TextVertex v3{p3.x, p3.y, q.s0, q.t1, lbl.color[0], lbl.color[1], lbl.color[2], lbl.color[3]};

            outVerts.push_back(v0); outVerts.push_back(v1); outVerts.push_back(v2);
            outVerts.push_back(v0); outVerts.push_back(v2); outVerts.push_back(v3);
        }
    }
}