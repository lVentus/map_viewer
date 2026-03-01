// src/app.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>

#include <sys/stat.h>

// ---- basic types ----

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Bounds {
    double minx = 0.0;
    double miny = 0.0;
    double maxx = 1.0;
    double maxy = 1.0;
};

struct AABB {
    float minx = 0.0f;
    float miny = 0.0f;
    float maxx = 0.0f;
    float maxy = 0.0f;
};

inline Vec2 v2(float x, float y) { return Vec2{x, y}; }

inline bool aabbIntersects(const AABB& a, const AABB& b) {
    return !(a.maxx < b.minx || a.minx > b.maxx || a.maxy < b.miny || a.miny > b.maxy);
}

// ---- camera ----

struct Camera {
    double centerX = 0.0;
    double centerY = 0.0;
    double baseViewWidth = 1.0;
    double baseViewHeight = 1.0;
    double zoom = 1.0;
    int fbW = 1280;
    int fbH = 720;
};

// ---- cache params ----
// Keep identical to the current behavior.
static constexpr int kTileSize = 65535;
static constexpr int kCacheVersion = 1;

struct TileKey { int tx = 0; int ty = 0; };

inline uint64_t tileKey64(int tx, int ty) {
    return (uint64_t)((uint32_t)tx) << 32 | (uint32_t)ty;
}

inline TileKey worldToTile(float x, float y) {
    return TileKey{
        (int)std::floor((double)x / (double)kTileSize),
        (int)std::floor((double)y / (double)kTileSize),
    };
}

inline void tilesForAABB(const AABB& bb, std::vector<TileKey>& out) {
    const int tx0 = (int)std::floor((double)bb.minx / (double)kTileSize);
    const int tx1 = (int)std::floor((double)bb.maxx / (double)kTileSize);
    const int ty0 = (int)std::floor((double)bb.miny / (double)kTileSize);
    const int ty1 = (int)std::floor((double)bb.maxy / (double)kTileSize);

    out.clear();
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            out.push_back(TileKey{tx, ty});
        }
    }
}

// ---- tile records ----

enum RecType : uint8_t {
    REC_EDGE = 1,
    REC_POLY = 2,
    REC_TEXT = 3,
};

// Edge payload: float x0 y0 x1 y1, int32 w, int32 c
struct EdgeRec {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    int32_t w = 1;
    int32_t c = 0;
};

// ---- cpu runtime types ----

struct TextVertex {
    float x, y;
    float u, v;
    float r, g, b, a;
};

struct TextLabelCPU {
    Vec2 pos;
    float angle = 0.0f;
    float size = 400.0f;
    float color[4] = {1, 1, 1, 1};
    std::string text;
};

struct LineBatchCPU {
    std::vector<Vec2> verts; // GL_LINES
    int w = 1;
    int c = 0;
};

struct PolyMeshCPU {
    std::vector<Vec2> verts;
    std::vector<uint32_t> idx; // triangles
    std::array<float, 4> color{1, 1, 1, 1};
    int layer = 0;
    AABB bb{};
};

// ---- cache info ----

struct CacheInfo {
    std::string cacheDir;
    std::string tilesDir;
    std::string manifestPath;
    std::string tilesDatPath;
    std::string tilesIdxPath;

    Bounds bounds{};
    uint64_t gmSize = 0;
    uint64_t gmMtime = 0;
    bool valid = false;
};

// ---- small utils (keep behavior identical) ----

inline bool statFile(const std::string& path, uint64_t& size, uint64_t& mtime) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return false;
    size = (uint64_t)st.st_size;
    mtime = (uint64_t)st.st_mtime;
    return true;
}

inline uint64_t fnv1a64(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t fnv1a64_str(const std::string& s) {
    return fnv1a64(s.data(), s.size());
}

inline std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return std::string(buf);
}

inline std::string baseName(const std::string& path) {
    const size_t p = path.find_last_of("/\\");
    return (p == std::string::npos) ? path : path.substr(p + 1);
}

inline std::string getCacheRoot() {
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/map_viewer";
    const char* home = std::getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.cache/map_viewer";
    return "./.cache/map_viewer";
}

inline bool ensureDir(const std::string& dir) {
    // Keep current behavior (mkdir -p via system).
    std::string cmd = "mkdir -p \"" + dir + "\"";
    return std::system(cmd.c_str()) == 0;
}

// ---- triangulation (ear clipping) ----

inline float cross2(const Vec2& a, const Vec2& b, const Vec2& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

inline bool pointInTri(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c) {
    const float c1 = cross2(a, b, p);
    const float c2 = cross2(b, c, p);
    const float c3 = cross2(c, a, p);
    const bool hasNeg = (c1 < 0) || (c2 < 0) || (c3 < 0);
    const bool hasPos = (c1 > 0) || (c2 > 0) || (c3 > 0);
    return !(hasNeg && hasPos);
}

inline bool isCCW(const std::vector<Vec2>& poly) {
    double area = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const Vec2& a = poly[i];
        const Vec2& b = poly[(i + 1) % poly.size()];
        area += (double)a.x * (double)b.y - (double)b.x * (double)a.y;
    }
    return area > 0.0;
}

// Expects simple polygon
inline bool triangulateEarClipping(const std::vector<Vec2>& poly, std::vector<uint32_t>& outIdx) {
    outIdx.clear();
    const int n = (int)poly.size();
    if (n < 3) return false;

    std::vector<int> V(n);
    if (isCCW(poly)) {
        std::iota(V.begin(), V.end(), 0);
    } else {
        for (int i = 0; i < n; ++i) V[i] = n - 1 - i;
    }

    int nv = n;
    int i = 0;
    int count = 2 * nv;

    while (nv > 2) {
        if ((count--) <= 0) return false;

        const int i0 = V[(i + nv - 1) % nv];
        const int i1 = V[i % nv];
        const int i2 = V[(i + 1) % nv];

        const Vec2& a = poly[i0];
        const Vec2& b = poly[i1];
        const Vec2& c = poly[i2];

        if (cross2(a, b, c) <= 0.0f) { ++i; continue; } // reflex

        bool anyInside = false;
        for (int j = 0; j < nv; ++j) {
            const int vi = V[j];
            if (vi == i0 || vi == i1 || vi == i2) continue;
            if (pointInTri(poly[vi], a, b, c)) { anyInside = true; break; }
        }
        if (anyInside) { ++i; continue; }

        outIdx.push_back((uint32_t)i0);
        outIdx.push_back((uint32_t)i1);
        outIdx.push_back((uint32_t)i2);

        V.erase(V.begin() + (i % nv));
        --nv;
        count = 2 * nv;
        i = 0;
    }

    return true;
}