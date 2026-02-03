// main.cpp
// Map viewer for .gm files with runtime cache tiling.
//
// Controls:
// - Mouse wheel: zoom around cursor (anchored at cursor)
// - Left mouse drag: pan
//
// GM format (unchanged):
//   <numNodes>//Node
//   <numEdges>//Edge
//   <numPolygons>//Polygon
//   <numTexts>//Text
//   BOUNDS minx miny maxx maxy
//   <numNodes lines>   x y
//   <numEdges lines>   u v w c
//   <numPolygons lines> POLY layer r g b a n x0 y0 ... x(n-1) y(n-1)
//   <numTexts lines>    TEXT x y angle size r g b a <rest of line is text>
//
// Cache (generated automatically, stored in ~/.cache/map_viewer/):
// - First open: sequentially scan .gm and write tiles (tileSize=4096 world units).
// - Next opens: validate by file size+mtime and reuse cache.
//
// Cache tile payload stores resolved geometry (edge endpoints as float coordinates),
// so runtime does NOT need to load the full original .gm or keep nodes in RAM.
//
// Text rendering:
// - Bitmap atlas via stb_truetype (ASCII 32..126).
// - Visibility rule: show only if 1% <= (labelArea / viewArea) <= 15% (with hysteresis).
// - Text mesh rebuilt on camera change (dirty), per visible tile.

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <cstring>
#include <array>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

#include <filesystem>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_set>
#include <atomic>
#include <utility>

#include <sys/mman.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <unistd.h>
#ifdef __GLIBC__
#include <malloc.h>
#endif

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

// ----------------- Math / helpers -----------------

struct Vec2 { float x=0.f, y=0.f; };
struct Bounds { double minx=0, miny=0, maxx=1, maxy=1; };

static inline Vec2 v2(float x,float y){ return Vec2{x,y}; }

struct AABB { float minx=0, miny=0, maxx=0, maxy=0; };
static inline bool aabbIntersects(const AABB& a, const AABB& b) {
    return !(a.maxx < b.minx || a.minx > b.maxx || a.maxy < b.miny || a.miny > b.maxy);
}

static bool statFile(const std::string& path, uint64_t& size, uint64_t& mtime) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return false;
    size = (uint64_t)st.st_size;
    mtime = (uint64_t)st.st_mtime;
    return true;
}

static uint64_t fnv1a64(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i=0;i<len;i++){ h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static uint64_t fnv1a64_str(const std::string& s){ return fnv1a64(s.data(), s.size()); }
static std::string hex64(uint64_t v){
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return std::string(buf);
}
static std::string baseName(const std::string& path){
    size_t p = path.find_last_of("/\\");
    return (p==std::string::npos)? path : path.substr(p+1);
}
static std::string getCacheRoot(){
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/map_viewer";
    const char* home = std::getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.cache/map_viewer";
    return "./.cache/map_viewer";
}
static bool ensureDir(const std::string& dir){
    std::string cmd = "mkdir -p \"" + dir + "\"";
    return std::system(cmd.c_str()) == 0;
}

// ----------------- Camera -----------------

struct Camera {
    double centerX=0, centerY=0;
    double baseViewWidth=1, baseViewHeight=1;
    double zoom=1.0;
    int fbW=1280, fbH=720; // framebuffer size (for viewport)
};
static Camera gCam;
static bool gMouseLeft=false;
static double gLastX=0, gLastY=0;
static bool gDirty=true;

// Compute view rect in world coordinates
static void getViewRect(double& left,double& right,double& bottom,double& top){
    const double vw = gCam.baseViewWidth / gCam.zoom;
    const double vh = gCam.baseViewHeight / gCam.zoom;
    left   = gCam.centerX - vw*0.5;
    right  = gCam.centerX + vw*0.5;
    bottom = gCam.centerY - vh*0.5;
    top    = gCam.centerY + vh*0.5;
}

static void initCameraFromBounds(const Bounds& b, int fbW, int fbH){
    gCam.fbW = fbW; gCam.fbH = fbH;
    gCam.centerX = 0.5*(b.minx + b.maxx);
    gCam.centerY = 0.5*(b.miny + b.maxy);

    const double w = (b.maxx - b.minx);
    const double h = (b.maxy - b.miny);
    const double aspect = (fbH>0)? (double)fbW/(double)fbH : 16.0/9.0;

    // Fit bounds into view with a bit of margin
    double vw = w;
    double vh = h;
    if (vw / vh < aspect) vw = vh * aspect;
    else                 vh = vw / aspect;
    gCam.baseViewWidth = vw * 1.05;
    gCam.baseViewHeight = vh * 1.05;
    gCam.zoom = 1.0;
}

// ----------------- GL utils -----------------

static GLuint compileShader(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::string log(len,'\0'); glGetShaderInfoLog(s,len,nullptr,log.data());
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return s;
}
static GLuint linkProgram(GLuint vs, GLuint fs){
    GLuint p=glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs);
    glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(len,'\0'); glGetProgramInfoLog(p,len,nullptr,log.data());
        std::cerr << "Program link error:\n" << log << "\n";
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

static GLuint createColorProgram(){
    const char* vsrc = R"GLSL(
        #version 330 core
        layout(location=0) in vec2 aPos;
        uniform vec4 uView; // left,right,bottom,top
        void main(){
            float x = (aPos.x - uView.x) / (uView.y - uView.x) * 2.0 - 1.0;
            float y = (aPos.y - uView.z) / (uView.w - uView.z) * 2.0 - 1.0;
            gl_Position = vec4(x,y,0,1);
        }
    )GLSL";
    const char* fsrc = R"GLSL(
        #version 330 core
        out vec4 FragColor;
        uniform vec4 uColor;
        void main(){ FragColor = uColor; }
    )GLSL";
    return linkProgram(compileShader(GL_VERTEX_SHADER, vsrc),
                       compileShader(GL_FRAGMENT_SHADER, fsrc));
}

static GLuint createTextProgram(){
    const char* vsrc = R"GLSL(
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
    const char* fsrc = R"GLSL(
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
    return linkProgram(compileShader(GL_VERTEX_SHADER, vsrc),
                       compileShader(GL_FRAGMENT_SHADER, fsrc));
}

// ----------------- Polygon triangulation (ear clipping) -----------------

static float cross2(const Vec2& a, const Vec2& b, const Vec2& c){
    // cross((b-a),(c-a))
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}
static bool pointInTri(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c){
    const float c1 = cross2(a,b,p);
    const float c2 = cross2(b,c,p);
    const float c3 = cross2(c,a,p);
    const bool hasNeg = (c1 < 0) || (c2 < 0) || (c3 < 0);
    const bool hasPos = (c1 > 0) || (c2 > 0) || (c3 > 0);
    return !(hasNeg && hasPos);
}
static bool isCCW(const std::vector<Vec2>& poly){
    double area=0;
    for(size_t i=0;i<poly.size();++i){
        const Vec2& a=poly[i];
        const Vec2& b=poly[(i+1)%poly.size()];
        area += (double)a.x*(double)b.y - (double)b.x*(double)a.y;
    }
    return area > 0;
}
static bool triangulateEarClipping(const std::vector<Vec2>& poly, std::vector<uint32_t>& outIdx){
    outIdx.clear();
    const int n = (int)poly.size();
    if(n < 3) return false;
    std::vector<int> V(n);
    if(isCCW(poly)) std::iota(V.begin(), V.end(), 0);
    else {
        for(int i=0;i<n;i++) V[i]=n-1-i;
    }

    int count = 2*n;
    int nv = n;
    int i=0;
    while(nv > 2){
        if((count--) <= 0) return false; // bad poly
        int i0 = V[(i+nv-1)%nv];
        int i1 = V[i%nv];
        int i2 = V[(i+1)%nv];
        const Vec2& a=poly[i0];
        const Vec2& b=poly[i1];
        const Vec2& c=poly[i2];

        if(cross2(a,b,c) <= 0){ i++; continue; } // reflex
        bool anyInside=false;
        for(int j=0;j<nv;j++){
            int vi = V[j];
            if(vi==i0||vi==i1||vi==i2) continue;
            if(pointInTri(poly[vi], a,b,c)){ anyInside=true; break; }
        }
        if(anyInside){ i++; continue; }

        outIdx.push_back((uint32_t)i0);
        outIdx.push_back((uint32_t)i1);
        outIdx.push_back((uint32_t)i2);
        V.erase(V.begin() + (i%nv));
        nv--;
        count = 2*nv;
        i=0;
    }
    return true;
}

// ----------------- Text atlas -----------------

static int gAtlasW = 1024;
static int gAtlasH = 1024;
static float gFontPixelSize = 32.0f;
static stbtt_bakedchar gBaked[96];
static GLuint gTextAtlasTex = 0;
static bool gTextReady = false;

static bool readFileBytes(const std::string& path, std::vector<uint8_t>& out){
    std::ifstream in(path, std::ios::binary);
    if(!in) return false;
    in.seekg(0,std::ios::end);
    size_t n = (size_t)in.tellg();
    in.seekg(0,std::ios::beg);
    out.resize(n);
    in.read((char*)out.data(), (std::streamsize)n);
    return true;
}

static std::string findDefaultFontPath(){
    // Prefer project root data/ and external/
    const std::vector<std::string> candidates = {
        "data/DejaVuSans.ttf",
        "external/fonts/DejaVuSans.ttf",
        "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    };
    for(const auto& p: candidates){
        std::ifstream f(p, std::ios::binary);
        if(f.good()) return p;
    }
    return "";
}

static bool initFontAtlas(const std::string& fontPath){
    std::vector<uint8_t> ttf;
    if(!readFileBytes(fontPath, ttf)){
        std::cerr << "Failed to read font: " << fontPath << "\n";
        return false;
    }
    std::vector<uint8_t> bitmap((size_t)gAtlasW * (size_t)gAtlasH, 0);
    int res = stbtt_BakeFontBitmap(ttf.data(), 0, gFontPixelSize,
                                   bitmap.data(), gAtlasW, gAtlasH, 32, 96, gBaked);
    if(res <= 0){
        std::cerr << "stbtt_BakeFontBitmap failed. Try a different TTF or a larger atlas.\n";
        return false;
    }

    glGenTextures(1, &gTextAtlasTex);
    glBindTexture(GL_TEXTURE_2D, gTextAtlasTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, gAtlasW, gAtlasH, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    gTextReady = true;
    std::cout << "Font loaded: " << fontPath << "\n";
    return true;
}

// ----------------- Cache tiling -----------------

static const int kTileSize = 65535;
static const int kCacheVersion = 1;

struct TileKey { int tx=0, ty=0; };
static inline uint64_t tileKey64(int tx,int ty){ return ((uint64_t)(uint32_t)tx<<32) | (uint32_t)ty; }

static inline TileKey worldToTile(float x, float y){
    return TileKey{ (int)std::floor((double)x / (double)kTileSize),
                    (int)std::floor((double)y / (double)kTileSize) };
}

static inline void tilesForAABB(const AABB& bb, std::vector<TileKey>& out){
    int tx0 = (int)std::floor((double)bb.minx / (double)kTileSize);
    int tx1 = (int)std::floor((double)bb.maxx / (double)kTileSize);
    int ty0 = (int)std::floor((double)bb.miny / (double)kTileSize);
    int ty1 = (int)std::floor((double)bb.maxy / (double)kTileSize);
    out.clear();
    for(int ty=ty0; ty<=ty1; ++ty)
        for(int tx=tx0; tx<=tx1; ++tx)
            out.push_back(TileKey{tx,ty});
}

struct CacheInfo {
    std::string cacheDir;
    std::string tilesDir;
    std::string manifestPath;

    // Packed tile store (preferred at runtime)
    std::string tilesDatPath;
    std::string tilesIdxPath;

    Bounds bounds;
    uint64_t gmSize=0, gmMtime=0;
    bool valid=false;
};

static std::string computeCacheDir(const std::string& gmPath){
    uint64_t sz=0, mt=0;
    if(!statFile(gmPath, sz, mt)) return "";
    std::string key = gmPath + "|" + std::to_string(sz) + "|" + std::to_string(mt)
                    + "|" + std::to_string(kTileSize) + "|" + std::to_string(kCacheVersion);
    uint64_t h = fnv1a64_str(key);
    return getCacheRoot() + "/" + baseName(gmPath) + "_" + hex64(h);
}

static bool writeManifest(const CacheInfo& ci){
    std::ofstream out(ci.manifestPath, std::ios::binary);
    if(!out) return false;
    out << "version " << kCacheVersion << "\n";
    out << "tileSize " << kTileSize << "\n";
    out << "gmSize " << ci.gmSize << "\n";
    out << "gmMtime " << ci.gmMtime << "\n";
    out << "bounds " << ci.bounds.minx << " " << ci.bounds.miny << " " << ci.bounds.maxx << " " << ci.bounds.maxy << "\n";
    return true;
}

static bool readManifest(CacheInfo& ci){
    std::ifstream in(ci.manifestPath, std::ios::binary);
    if(!in) return false;
    std::string tag;
    int version=0, tileSize=0;
    uint64_t gmSize=0, gmMtime=0;
    Bounds b{};
    while(in >> tag){
        if(tag=="version") in >> version;
        else if(tag=="tileSize") in >> tileSize;
        else if(tag=="gmSize") in >> gmSize;
        else if(tag=="gmMtime") in >> gmMtime;
        else if(tag=="bounds") in >> b.minx >> b.miny >> b.maxx >> b.maxy;
        else {
            std::string rest; std::getline(in, rest);
        }
    }
    if(version!=kCacheVersion || tileSize!=kTileSize) return false;
    uint64_t realSz=0, realMt=0;
    if(!statFile(ci.cacheDir.substr(0,0), realSz, realMt)){} // no-op
    uint64_t srcSz=0, srcMt=0;
    // The caller should fill srcSz/srcMt for gmPath check; we just return fields
    ci.bounds = b;
    ci.gmSize = gmSize;
    ci.gmMtime = gmMtime;
    return true;
}

struct TileWriter {
    std::string tilesDir;
    size_t maxOpen = 64;
    struct Handle { FILE* f=nullptr; uint64_t last=0; };
    std::unordered_map<uint64_t, Handle> open;
    uint64_t tick=1;

    bool init(const std::string& cacheDir){
        tilesDir = cacheDir + "/tiles";
        return ensureDir(tilesDir);
    }

    std::string pathFor(int tx,int ty) const {
        return tilesDir + "/t_" + std::to_string(tx) + "_" + std::to_string(ty) + ".bin";
    }

    FILE* get(int tx,int ty){
        uint64_t k = tileKey64(tx,ty);
        auto it = open.find(k);
        if(it!=open.end()){ it->second.last=tick++; return it->second.f; }

        if(open.size() >= maxOpen){
            auto itOld = open.end();
            for(auto it2=open.begin(); it2!=open.end(); ++it2){
                if(itOld==open.end() || it2->second.last < itOld->second.last) itOld=it2;
            }
            if(itOld!=open.end()){
                std::fclose(itOld->second.f);
                open.erase(itOld);
            }
        }

        std::string p = pathFor(tx,ty);
        FILE* f = std::fopen(p.c_str(), "ab");
        if(!f) return nullptr;
        open[k] = Handle{f, tick++};
        return f;
    }

    void writeRec(int tx,int ty, uint8_t type, const void* payload, uint32_t sz){
        FILE* f = get(tx,ty);
        if(!f) return;
        std::fwrite(&type, 1, 1, f);
        std::fwrite(&sz, 4, 1, f);
        std::fwrite(payload, 1, sz, f);
    }

    void shutdown(){
        for(auto& kv: open) std::fclose(kv.second.f);
        open.clear();
    }
};

// Record types in tile files
enum RecType : uint8_t {
    REC_EDGE = 1,
    REC_POLY = 2,
    REC_TEXT = 3,
};

// Edge payload: float x0 y0 x1 y1, int32 w, int32 c
struct EdgeRec {
    float x0,y0,x1,y1;
    int32_t w;
    int32_t c;
};

// Poly payload: int32 layer, float r g b a, uint32 n, float2*n
// Text payload: float x y angle size, float r g b a, uint32 len, bytes[len]

// Build cache by sequentially scanning the .gm file.
// Implementation policy:
// - Read nodes into a temp float array (RAM). This is required for resolving edges.
// - Stream edges/polys/texts directly into tile files (append).
static bool buildCacheFromGM(const std::string& gmPath, CacheInfo& out){
    uint64_t gmSize=0, gmMtime=0;
    if(!statFile(gmPath, gmSize, gmMtime)){
        std::cerr << "ERROR: Cannot stat gm: " << gmPath << "\n";
        return false;
    }

    out.cacheDir = computeCacheDir(gmPath);
    out.tilesDir = out.cacheDir + "/tiles";
    out.manifestPath = out.cacheDir + "/manifest.txt";
    out.tilesDatPath = out.cacheDir + "/tiles.dat";
    out.tilesIdxPath = out.cacheDir + "/tiles.idx";
    out.gmSize = gmSize;
    out.gmMtime = gmMtime;

    ensureDir(getCacheRoot());
    ensureDir(out.cacheDir);
    ensureDir(out.tilesDir);

    std::ifstream in(gmPath);
    if(!in){
        std::cerr << "ERROR: Cannot open gm: " << gmPath << "\n";
        return false;
    }
    // Larger stream buffer for faster parsing
    static std::vector<char> buf(4 * 1024 * 1024);
    in.rdbuf()->pubsetbuf(buf.data(), (std::streamsize)buf.size());

    size_t numNodes=0, numEdges=0, numPolys=0, numTexts=0;
    std::string line;

    // Read first 4 lines with counts (format: <num>//Node)
    auto readCountLine = [&](size_t& outCount)->bool{
        if(!std::getline(in, line)) return false;
        std::stringstream ss(line);
        ss >> outCount;
        return true;
    };
    if(!readCountLine(numNodes) || !readCountLine(numEdges) || !readCountLine(numPolys) || !readCountLine(numTexts)){
        std::cerr << "ERROR: Invalid header counts.\n";
        return false;
    }
    // BOUNDS line
    if(!std::getline(in, line)){ std::cerr << "ERROR: Missing BOUNDS.\n"; return false; }
    {
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if(tag != "BOUNDS"){
            std::cerr << "ERROR: Expected BOUNDS, got: " << tag << "\n";
            return false;
        }
        ss >> out.bounds.minx >> out.bounds.miny >> out.bounds.maxx >> out.bounds.maxy;
    }

    std::cout << "[Cache] Building cache...\n";
    std::cout << "  Nodes=" << numNodes << " Edges=" << numEdges << " Polys=" << numPolys << " Texts=" << numTexts << "\n";
    std::cout << "  TileSize=" << kTileSize << "\n";
    std::cout << "  CacheDir=" << out.cacheDir << "\n";

    // Phase A: read nodes
    std::vector<float> nodesXY;
    nodesXY.resize(numNodes * 2);
    for(size_t i=0;i<numNodes;i++){
        double x=0,y=0;
        if(!(in >> x >> y)){
            std::cerr << "ERROR: Failed reading node " << i << "\n";
            return false;
        }
        nodesXY[i*2+0] = (float)x;
        nodesXY[i*2+1] = (float)y;
    }
    // consume endline after last node
    std::getline(in, line);

    TileWriter writer;
    if(!writer.init(out.cacheDir)){
        std::cerr << "ERROR: Failed creating tile writer.\n";
        return false;
    }

    std::vector<TileKey> tiles;

    // Phase B1: edges
    for(size_t i=0;i<numEdges;i++){
        uint32_t u=0, v=0; int w=1, c=0;
        if(!(in >> u >> v >> w >> c)){
            std::cerr << "ERROR: Failed reading edge " << i << "\n";
            writer.shutdown();
            return false;
        }
        if(u >= numNodes || v >= numNodes) continue;
        EdgeRec rec{};
        rec.x0 = nodesXY[u*2+0]; rec.y0 = nodesXY[u*2+1];
        rec.x1 = nodesXY[v*2+0]; rec.y1 = nodesXY[v*2+1];
        rec.w = (int32_t)w;
        rec.c = (int32_t)c;

        AABB bb;
        bb.minx = std::min(rec.x0, rec.x1);
        bb.maxx = std::max(rec.x0, rec.x1);
        bb.miny = std::min(rec.y0, rec.y1);
        bb.maxy = std::max(rec.y0, rec.y1);

        tilesForAABB(bb, tiles);
        for(const auto& t : tiles){
            writer.writeRec(t.tx, t.ty, REC_EDGE, &rec, (uint32_t)sizeof(EdgeRec));
        }
    }
    std::getline(in, line);

    // Phase B2: polygons (line-based)
    for(size_t i=0;i<numPolys;i++){
        if(!std::getline(in, line)){
            std::cerr << "ERROR: Missing polygon line " << i << "\n";
            writer.shutdown();
            return false;
        }
        if(line.empty()){ i--; continue; }
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if(tag != "POLY"){ std::cerr << "ERROR: Expected POLY.\n"; writer.shutdown(); return false; }

        int layer=0; float r=0,g=0,b=0,a=0; uint32_t n=0;
        ss >> layer >> r >> g >> b >> a >> n;
        std::vector<Vec2> verts;
        verts.reserve(n);
        AABB bb; bb.minx=+1e30f; bb.miny=+1e30f; bb.maxx=-1e30f; bb.maxy=-1e30f;
        for(uint32_t k=0;k<n;k++){
            float x=0,y=0;
            ss >> x >> y;
            verts.push_back(Vec2{x,y});
            bb.minx = std::min(bb.minx, x); bb.maxx = std::max(bb.maxx, x);
            bb.miny = std::min(bb.miny, y); bb.maxy = std::max(bb.maxy, y);
        }

        // serialize payload
        const uint32_t payloadSz = (uint32_t)(sizeof(int32_t) + sizeof(float)*4 + sizeof(uint32_t) + sizeof(float)*2*n);
        std::vector<uint8_t> payload(payloadSz);
        uint8_t* p = payload.data();
        auto wr = [&](const void* src, size_t sz){ std::memcpy(p, src, sz); p += sz; };

        int32_t layer32 = (int32_t)layer;
        wr(&layer32, sizeof(layer32));
        wr(&r, sizeof(float)); wr(&g, sizeof(float)); wr(&b, sizeof(float)); wr(&a, sizeof(float));
        wr(&n, sizeof(uint32_t));
        for(uint32_t k=0;k<n;k++){
            wr(&verts[k].x, sizeof(float));
            wr(&verts[k].y, sizeof(float));
        }

        tilesForAABB(bb, tiles);
        for(const auto& t : tiles){
            writer.writeRec(t.tx, t.ty, REC_POLY, payload.data(), payloadSz);
        }
    }

    // Phase B3: texts
    for(size_t i=0;i<numTexts;i++){
        if(!std::getline(in, line)){
            std::cerr << "ERROR: Missing text line " << i << "\n";
            writer.shutdown();
            return false;
        }
        if(line.empty()){ i--; continue; }
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if(tag != "TEXT"){ std::cerr << "ERROR: Expected TEXT.\n"; writer.shutdown(); return false; }
        float x=0,y=0, angle=0, size=0;
        float r=1,g=1,b=1,a=1;
        ss >> x >> y >> angle >> size >> r >> g >> b >> a;

        // remainder of line is the text string, including spaces
        std::string rest;
        std::getline(ss, rest);
        if(!rest.empty() && rest[0]==' ') rest.erase(rest.begin());

        uint32_t len = (uint32_t)rest.size();

        const uint32_t payloadSz = (uint32_t)(sizeof(float)*4 + sizeof(float)*4 + sizeof(uint32_t) + len);
        std::vector<uint8_t> payload(payloadSz);
        uint8_t* p = payload.data();
        auto wr = [&](const void* src, size_t sz){ std::memcpy(p, src, sz); p += sz; };

        wr(&x, sizeof(float)); wr(&y, sizeof(float)); wr(&angle, sizeof(float)); wr(&size, sizeof(float));
        wr(&r, sizeof(float)); wr(&g, sizeof(float)); wr(&b, sizeof(float)); wr(&a, sizeof(float));
        wr(&len, sizeof(uint32_t));
        if(len) wr(rest.data(), len);

        AABB bb; bb.minx=x; bb.maxx=x; bb.miny=y; bb.maxy=y;
        tilesForAABB(bb, tiles);
        for(const auto& t : tiles){
            writer.writeRec(t.tx, t.ty, REC_TEXT, payload.data(), payloadSz);
        }
    }

    writer.shutdown();

    // Pack per-tile small files into a single tiles.dat + tiles.idx for fast runtime mmap/pread.
    // This avoids opening thousands of tiny files while panning/zooming.
    auto packTiles = [&](const CacheInfo& ci)->bool{
        namespace fs = std::filesystem;
        if(!fs::exists(ci.tilesDir)) return false;

        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(ci.tilesDir)) {
            if(!e.is_regular_file()) continue;
            auto name = e.path().filename().string();
            if(name.rfind("t_", 0) != 0) continue;
            if(e.path().extension() != ".bin") continue;
            files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());

        std::ofstream dat(ci.tilesDatPath, std::ios::binary);
        std::ofstream idx(ci.tilesIdxPath, std::ios::binary);
        if(!dat || !idx) return false;

        // idx format:
        // [uint32_t magic 'MVTI'][uint32_t version][uint32_t count]
        // then count entries: [uint64_t key][uint64_t offset][uint32_t size][uint32_t reserved]
        const uint32_t magic = 0x4954564d; // 'MVTI' little-endian
        const uint32_t ver = (uint32_t)kCacheVersion;
        idx.write((const char*)&magic, 4);
        idx.write((const char*)&ver, 4);
        uint32_t count = (uint32_t)files.size();
        idx.write((const char*)&count, 4);

        uint64_t offset = 0;
        for(const auto& fp : files){
            std::ifstream in(fp, std::ios::binary);
            if(!in) continue;
            in.seekg(0, std::ios::end);
            std::streamsize sz = in.tellg();
            in.seekg(0, std::ios::beg);
            if(sz <= 0) continue;

            std::vector<char> bufLocal((size_t)sz);
            in.read(bufLocal.data(), sz);
            if(!in) continue;

            // parse tx,ty from filename: t_<tx>_<ty>.bin
            int tx=0, ty=0;
            {
                auto stem = fp.stem().string(); // t_tx_ty
                // crude parse
                std::sscanf(stem.c_str(), "t_%d_%d", &tx, &ty);
            }
            uint64_t key = tileKey64(tx,ty);
            uint64_t off = offset;
            uint32_t usz = (uint32_t)sz;

            dat.write(bufLocal.data(), sz);
            // entry
            idx.write((const char*)&key, 8);
            idx.write((const char*)&off, 8);
            idx.write((const char*)&usz, 4);
            uint32_t reserved = 0;
            idx.write((const char*)&reserved, 4);

            offset += (uint64_t)sz;
        }

        dat.flush(); idx.flush();

        // Optional: keep the original small files for debugging. If you want to delete:
        // for(const auto& fp : files) fs::remove(fp);
        return true;
    };

    if(!packTiles(out)){
        std::cerr << "[Cache] Warning: failed to pack tiles into tiles.dat/tiles.idx. Runtime will fall back to per-tile files.\n";
    }


    // Write manifest last
    if(!writeManifest(out)){
        std::cerr << "ERROR: Failed writing manifest.\n";
        return false;
    }

    out.valid = true;
    std::cout << "[Cache] Build done.\n";
    return true;
}

static bool tryOpenCache(const std::string& gmPath, CacheInfo& out){
    uint64_t srcSz=0, srcMt=0;
    if(!statFile(gmPath, srcSz, srcMt)) return false;

    out.cacheDir = computeCacheDir(gmPath);
    out.tilesDir = out.cacheDir + "/tiles";
    out.manifestPath = out.cacheDir + "/manifest.txt";
    out.tilesDatPath = out.cacheDir + "/tiles.dat";
    out.tilesIdxPath = out.cacheDir + "/tiles.idx";

    // quick check manifest exists
    {
        std::ifstream f(out.manifestPath, std::ios::binary);
        if(!f.good()) return false;
    }
    CacheInfo tmp = out;
    if(!readManifest(tmp)) return false;

    if(tmp.gmSize != srcSz || tmp.gmMtime != srcMt) return false;
    // Prefer packed tile store
    tmp.tilesDatPath = tmp.cacheDir + "/tiles.dat";
    tmp.tilesIdxPath = tmp.cacheDir + "/tiles.idx";
    if(!(std::ifstream(tmp.tilesDatPath, std::ios::binary).good() && std::ifstream(tmp.tilesIdxPath, std::ios::binary).good())){
        // If packed store missing but per-tile files exist (older cache), keep valid; runtime can fall back.
        // Caller may rebuild cache if desired.
    }
    tmp.valid = true;
    out = tmp;
    std::cout << "[Cache] Hit: " << out.cacheDir << "\n";
    return true;
}

// ----------------- Runtime tile loading -----------------

// NOTE: We avoid per-tile GPU objects. Tiles are CPU containers.
// All visible tiles are merged into a small set of large GPU buffers ("merged batches").
//
// Roads: grouped by (w,c) -> one VBO per style.
// Polygons: grouped by (layer,color) -> one VBO/EBO per group, sorted by layer.
// Text: one global VBO for all visible labels (rebuilt when camera changes or tiles change).

struct TextVertex {
    float x,y;
    float u,v;
    float r,g,b,a;
};

struct TextLabelCPU {
    Vec2 pos;
    float angle=0;
    float size=400;
    float color[4]={1,1,1,1};
    std::string text;
};

// CPU-side per-tile line batch
struct LineBatchCPU { std::vector<Vec2> verts; int w=1; int c=0; };

// CPU-side per-tile polygon mesh
struct PolyMeshCPU {
    std::vector<Vec2> verts;
    std::vector<uint32_t> idx; // triangles
    std::array<float,4> color{1,1,1,1};
    int layer=0;
    AABB bb{};
};

// Merged GPU batches
struct LineBatchGpu {
    GLuint vao=0, vbo=0;
    GLsizei vertCount=0;
    float color[4] = {1,1,1,1};
    float width=1.0f;
    AABB bb{};
    int w=1, c=0;
};

struct PolyBatchGpu {
    GLuint vao=0, vbo=0, ebo=0;
    GLsizei indexCount=0;
    float color[4] = {1,1,1,1};
    int layer=0;
    AABB bb{};
};

struct TextGpu {
    GLuint vao=0, vbo=0;
    GLsizei vertCount=0;
};

struct Tile {
    int tx=0, ty=0;
    bool loaded=false;

    // CPU (kept only if needed; lines/polys are released after GPU upload)
    std::vector<LineBatchCPU> lines;   // CPU, per style within tile
    std::vector<PolyMeshCPU> polys;    // CPU meshes (triangulated)
    std::vector<TextLabelCPU> texts;   // CPU labels

    // GPU (per-tile, incremental; no merged full rebuild)
    std::vector<LineBatchGpu> gpuLines;
    std::vector<PolyBatchGpu> gpuPolys;
    TextGpu gpuText;
    bool gpuTextValid=false;
    uint64_t gpuTextRev=0;

    AABB bb{}; // tile AABB
};

// Color mapping for roads: must match main.cpp's colorFromType(c)
static std::array<float,4> roadColor(int /*w*/, int c){
    float r=1.0f,g=1.0f,b=1.0f;
    switch(c){
        case 0: r=1.0f; g=1.0f; b=1.0f; break; // white
        case 1: r=0.4f; g=0.8f; b=1.0f; break; // blue-ish
        case 2: r=1.0f; g=0.4f; b=0.4f; break; // red-ish
        case 3: r=1.0f; g=0.8f; b=0.2f; break; // yellow
        case 4: r=0.4f; g=1.0f; b=0.4f; break; // green
        case 5: r=1.0f; g=0.4f; b=1.0f; break; // magenta
        case 6: r=0.8f; g=0.8f; b=0.8f; break; // light gray
        case 7: r=0.6f; g=0.6f; b=0.6f; break; // darker gray
        default: break;
    }
    return {r,g,b,1.0f};
}

static float roadWidthPx(int w)
{
    float lw = static_cast<float>(w);
    if(lw < 1.0f) lw = 1.0f;
    return lw;
}

struct TileManager {
    CacheInfo ci;

    struct TileIndexEntry { uint64_t off=0; uint32_t sz=0; };
    std::unordered_map<uint64_t, TileIndexEntry> index;
    int datFd = -1;
    size_t datBytes = 0;
    const uint8_t* datPtr = nullptr;

    std::unordered_map<uint64_t, Tile> tiles;
    std::vector<TileKey> visible;          // tiles needed for rendering
    std::unordered_set<uint64_t> visibleSet;
    std::vector<TileKey> prefetch;         // tiles to request (visible + expanded neighborhood)
    std::unordered_set<uint64_t> prefetchSet;
    double viewCenterX = 0.0; // world units
    double viewCenterY = 0.0;
    // Async loader
    struct PendingTileCPU {
        uint64_t gen = 0;
        int tx=0, ty=0;
    std::vector<LineBatchCPU> lines;
    std::vector<PolyMeshCPU> polys;
    std::vector<TextLabelCPU> texts;
    bool empty=true;
};

    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<uint64_t> viewGen{1};
    std::queue<std::pair<uint64_t,uint64_t>> reqQ; // (tileKey, gen)
    std::unordered_set<uint64_t> inFlight;
    std::queue<PendingTileCPU> readyQ;
    std::atomic<bool> stop{false};
    int workerCount = 0;
    std::vector<std::thread> workers;

    // Tuning
    int maxResidentTiles = 512;
    int maxUploadPerFrame = 64;
    int padTiles = 1;              // preload ring
    uint64_t frameId = 0;
    uint64_t textRev = 1; // bump when view scale changes
    double lastViewArea = -1.0;
    uint64_t ttlFrames = 180;      // evict tiles not used for this many frames
    // Upload pacing
    // In normal view we cap uploads to avoid hitches; in "sprint" mode (many tiles needed) we spend more time uploading.
    int sprintThresholdTiles = 80;   // if visible tiles exceed this, enable sprint mode
    double normalUploadBudgetMs = 10.0;
    double sprintUploadBudgetMs = 28.0;
    int maxRequestEnqueue = 20000; // limit how many new tile requests we enqueue per camera change

    // Track text visibility state (per label stable across rebuilds)
    std::vector<uint8_t> textVisibleState;

    // ---- Packed store ----
    bool openPackedStore(){
        // Load idx if present
        std::ifstream idx(ci.tilesIdxPath, std::ios::binary);
        std::ifstream dat(ci.tilesDatPath, std::ios::binary);
        if(!idx.good() || !dat.good()){
            return false;
        }
        uint32_t magic=0, ver=0, count=0;
        idx.read((char*)&magic,4);
        idx.read((char*)&ver,4);
        idx.read((char*)&count,4);
        if(!idx || magic != 0x4954564d) { // 'MVTI'
            return false;
        }
        // accept older ver too, but layout must match
        index.clear();
        for(uint32_t i=0;i<count;i++){
            uint64_t key=0, off=0;
            uint32_t sz=0, rsv=0;
            idx.read((char*)&key,8);
            idx.read((char*)&off,8);
            idx.read((char*)&sz,4);
            idx.read((char*)&rsv,4);
            if(!idx) break;
            index[key] = TileIndexEntry{off, sz};
        }

        datFd = ::open(ci.tilesDatPath.c_str(), O_RDONLY);
        if(datFd < 0) return false;
        struct stat st{};
        if(fstat(datFd, &st) != 0){ ::close(datFd); datFd=-1; return false; }
        datBytes = (size_t)st.st_size;
        void* p = mmap(nullptr, datBytes, PROT_READ, MAP_PRIVATE, datFd, 0);
        if(p == MAP_FAILED){ ::close(datFd); datFd=-1; datBytes=0; return false; }
        datPtr = (const uint8_t*)p;

        // Hint the kernel we will mostly scan forward through this file (helps mmap readahead).
        (void)posix_fadvise(datFd, 0, 0, POSIX_FADV_SEQUENTIAL);
        return true;
    }

    void closePackedStore(){
        if(datPtr){ munmap((void*)datPtr, datBytes); datPtr=nullptr; }
        if(datFd>=0){ ::close(datFd); datFd=-1; }
        datBytes=0;
        index.clear();
    }

    // Fallback: per-tile file path
    std::string tilePath(int tx,int ty) const {
        return ci.tilesDir + "/t_" + std::to_string(tx) + "_" + std::to_string(ty) + ".bin";
    }

    bool readTileBlob(int tx,int ty, std::vector<uint8_t>& outBlob){
        outBlob.clear();
        uint64_t key = tileKey64(tx,ty);

        if(datPtr && !index.empty()){
            auto it = index.find(key);
            if(it == index.end()) return true; // empty tile
            const auto off = it->second.off;
            const auto sz  = it->second.sz;
            if(off + sz > datBytes) return false;
            outBlob.resize(sz);
            std::memcpy(outBlob.data(), datPtr + off, sz);
            return true;
        }

        // fallback: small tile file
        std::ifstream in(tilePath(tx,ty), std::ios::binary);
        if(!in) return true;
        in.seekg(0, std::ios::end);
        std::streamsize sz = in.tellg();
        in.seekg(0, std::ios::beg);
        if(sz <= 0) return true;
        outBlob.resize((size_t)sz);
        in.read((char*)outBlob.data(), sz);
        return (bool)in;
    }

    static bool parseTileBlob(const std::vector<uint8_t>& blob, PendingTileCPU& out){
        out.empty = true;
        if(blob.empty()) return true;

        // temp batches keyed by style
        struct TmpBatch { std::vector<Vec2> verts; int w=1; int c=0; };
        std::unordered_map<int, TmpBatch> batches;

        size_t pos = 0;
        const uint8_t* data = blob.data();
        const size_t N = blob.size();

        while(pos + 5 <= N){
            uint8_t type = data[pos]; pos += 1;
            uint32_t sz = 0;
            std::memcpy(&sz, data + pos, 4); pos += 4;
            if(pos + sz > N) break;
            const uint8_t* p = data + pos;
            pos += sz;

            if(type == REC_EDGE){
                if(sz != sizeof(EdgeRec)) continue;
                EdgeRec rec{};
                std::memcpy(&rec, p, sizeof(EdgeRec));
                int keyStyle = (rec.w<<16) ^ (rec.c & 0xffff);
                auto& b = batches[keyStyle];
                b.w = rec.w; b.c = rec.c;
                b.verts.push_back(v2(rec.x0, rec.y0));
                b.verts.push_back(v2(rec.x1, rec.y1));
                out.empty = false;
            } 
            else if(type == REC_POLY){
                const uint8_t* q = p;
                int32_t layer=0; float r,g,b,a; uint32_t n=0;
                if(sz < 4+4*4+4) continue;
                std::memcpy(&layer, q, 4); q+=4;
                std::memcpy(&r, q, 4); q+=4;
                std::memcpy(&g, q, 4); q+=4;
                std::memcpy(&b, q, 4); q+=4;
                std::memcpy(&a, q, 4); q+=4;
                std::memcpy(&n, q, 4); q+=4;
                if(n < 3) continue;
                if((size_t)(q - p) + (size_t)n*8 > (size_t)sz) continue;

                PolyMeshCPU pm{};
                pm.layer = (int)layer;
                pm.color = {r,g,b,a};
                pm.verts.resize(n);

                AABB bb{+1e30f,+1e30f,-1e30f,-1e30f};
                for(uint32_t i=0;i<n;i++){
                    float x,y;
                    std::memcpy(&x,q,4); q+=4;
                    std::memcpy(&y,q,4); q+=4;
                    pm.verts[i]=v2(x,y);
                    bb.minx = std::min(bb.minx, x);
                    bb.miny = std::min(bb.miny, y);
                    bb.maxx = std::max(bb.maxx, x);
                    bb.maxy = std::max(bb.maxy, y);
                }
                pm.bb = bb;

                if(!triangulateEarClipping(pm.verts, pm.idx)) continue;

                out.polys.push_back(std::move(pm));
                out.empty = false;
            } else if(type == REC_TEXT){
                const uint8_t* q = p;
                float x=0,y=0,angle=0,sizePt=0,r=1,g=1,b=1,a=1;
                uint32_t len=0;
                if(sz < 4*10 + 4) continue;
                std::memcpy(&x,q,4); q+=4;
                std::memcpy(&y,q,4); q+=4;
                std::memcpy(&angle,q,4); q+=4;
                std::memcpy(&sizePt,q,4); q+=4;
                std::memcpy(&r,q,4); q+=4;
                std::memcpy(&g,q,4); q+=4;
                std::memcpy(&b,q,4); q+=4;
                std::memcpy(&a,q,4); q+=4;
                std::memcpy(&len,q,4); q+=4;
                if((size_t)(q - p) + (size_t)len > (size_t)sz) continue;
                TextLabelCPU t{};
                t.pos = v2(x,y);
                t.angle = angle;
                t.size = sizePt;
                t.color[0]=r; t.color[1]=g; t.color[2]=b; t.color[3]=a;
                t.text.assign((const char*)q, (size_t)len);
                out.texts.push_back(std::move(t));
                out.empty = false;
            }
        }

        // convert batches map to vector
        out.lines.clear();
        out.lines.reserve(batches.size());
        for(auto& kv : batches){
            LineBatchCPU b{};
            b.w = kv.second.w;
            b.c = kv.second.c;
            b.verts = std::move(kv.second.verts);
            out.lines.push_back(std::move(b));
        }
        return true;
    }
    // ---- Per-tile GPU helpers (incremental, no full rebuild) ----

    bool gpuDirty = false; // kept for compatibility; no longer triggers full rebuild

    bool anyTextLoaded = false;
    bool anyPolyLoaded = false;

    static void destroyLineGpu(LineBatchGpu& l){
        if(l.vbo){ glDeleteBuffers(1, &l.vbo); l.vbo=0; }
        if(l.vao){ glDeleteVertexArrays(1, &l.vao); l.vao=0; }
        l.vertCount = 0;
    }
    static void destroyPolyGpu(PolyBatchGpu& p){
        if(p.ebo){ glDeleteBuffers(1, &p.ebo); p.ebo=0; }
        if(p.vbo){ glDeleteBuffers(1, &p.vbo); p.vbo=0; }
        if(p.vao){ glDeleteVertexArrays(1, &p.vao); p.vao=0; }
        p.indexCount = 0;
    }
    static void destroyTextGpu(TextGpu& t){
        if(t.vbo){ glDeleteBuffers(1, &t.vbo); t.vbo=0; }
        if(t.vao){ glDeleteVertexArrays(1, &t.vao); t.vao=0; }
        t.vertCount=0;
    }

    void destroyTileGpu(Tile& t){
        for(auto& l : t.gpuLines) destroyLineGpu(l);
        t.gpuLines.clear();
        for(auto& p : t.gpuPolys) destroyPolyGpu(p);
        t.gpuPolys.clear();
        if(t.gpuTextValid) destroyTextGpu(t.gpuText);
        t.gpuTextValid = false;
        t.gpuText = {};
        t.gpuTextRev = 0;
    }
    static float estimateTextWidthPx(const std::string& s){
        float x = 0.0f;
        for(unsigned char ch : s){
            if(ch < 32 || ch > 126) continue;
            const stbtt_bakedchar& bc = gBaked[ch - 32];
            x += bc.xadvance;
        }
        return x;
    }

    void rebuildTileTextGpu(Tile& t, double viewArea){
        // Destroy previous text GPU (if any)
        if(t.gpuTextValid) destroyTextGpu(t.gpuText);
        t.gpuTextValid = false;
        t.gpuTextRev = textRev;

        if(!gTextReady || t.texts.empty()) return;

        const double minRatio = 0.01; // 1%
        const double maxRatio = 0.15; // 15%

        std::vector<TextVertex> verts;
        verts.reserve(t.texts.size() * 6 * 8); // rough

        for(const auto& lbl : t.texts){
            // Visibility rule based on area ratio
            float wpx = estimateTextWidthPx(lbl.text);
            float hpx = gFontPixelSize;
            if(wpx <= 0.0f) continue;

            // scale: baked pixels -> world units
            float scale = (lbl.size > 1e-3f) ? (lbl.size / gFontPixelSize) : 1.0f;
            double worldW = (double)wpx * (double)scale;
            double worldH = (double)lbl.size;
            double area = worldW * worldH;
            double ratio = (viewArea > 1e-12) ? (area / viewArea) : 0.0;
            if(ratio < minRatio || ratio > maxRatio) continue;

	            // Angle unit note:
	            // The GM TEXT record's angle may be radians or degrees depending on the source.
	            // If we always assume degrees, radian inputs (e.g., ~1.57) become tiny (~0.027rad)
	            // and everything looks almost horizontal. Use a simple heuristic:
	            // - |angle| > 2*pi (~6.283) -> treat as degrees
	            // - otherwise -> treat as radians
	            const float twoPi = 6.2831853071795864769f;
	            const float ang = (std::fabs(lbl.angle) > twoPi)
	                ? (lbl.angle * 3.14159265358979323846f / 180.0f)
	                : lbl.angle;
	            const float cs = std::cos(ang);
	            const float sn = std::sin(ang);

            float penX = 0.0f;
            float penY = 0.0f;

            for(unsigned char ch : lbl.text){
                if(ch < 32 || ch > 126) continue;
                stbtt_aligned_quad q{};
                stbtt_GetBakedQuad(gBaked, gAtlasW, gAtlasH, (int)ch - 32, &penX, &penY, &q, 1);

                // local coords (pixels) -> world (scale), flip Y so up is positive
                float lx0 = q.x0 * scale;
                float ly0 = -q.y0 * scale;
                float lx1 = q.x1 * scale;
                float ly1 = -q.y1 * scale;

                // rotate around label.pos
                auto xform = [&](float lx, float ly)->Vec2{
                    float rx = lx*cs - ly*sn;
                    float ry = lx*sn + ly*cs;
                    return Vec2{ lbl.pos.x + rx, lbl.pos.y + ry };
                };

                Vec2 p0 = xform(lx0, ly0);
                Vec2 p1 = xform(lx1, ly0);
                Vec2 p2 = xform(lx1, ly1);
                Vec2 p3 = xform(lx0, ly1);

                TextVertex v0{p0.x,p0.y, q.s0,q.t0, lbl.color[0],lbl.color[1],lbl.color[2],lbl.color[3]};
                TextVertex v1{p1.x,p1.y, q.s1,q.t0, lbl.color[0],lbl.color[1],lbl.color[2],lbl.color[3]};
                TextVertex v2{p2.x,p2.y, q.s1,q.t1, lbl.color[0],lbl.color[1],lbl.color[2],lbl.color[3]};
                TextVertex v3{p3.x,p3.y, q.s0,q.t1, lbl.color[0],lbl.color[1],lbl.color[2],lbl.color[3]};

                // two triangles
                verts.push_back(v0); verts.push_back(v1); verts.push_back(v2);
                verts.push_back(v0); verts.push_back(v2); verts.push_back(v3);
            }
        }

        if(verts.empty()) return;

        glGenVertexArrays(1, &t.gpuText.vao);
        glGenBuffers(1, &t.gpuText.vbo);
        glBindVertexArray(t.gpuText.vao);
        glBindBuffer(GL_ARRAY_BUFFER, t.gpuText.vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size()*sizeof(TextVertex)), verts.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)(2*sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)(4*sizeof(float)));

        glBindVertexArray(0);

        t.gpuText.vertCount = (GLsizei)verts.size();
        t.gpuTextValid = true;
    }

    void uploadTileCPU(PendingTileCPU&& cpu){
        uint64_t key = tileKey64(cpu.tx, cpu.ty);
        auto& t = tiles[key];

        // If reloading the same tile (rare), clean up old GPU first.
        if(t.loaded){
            destroyTileGpu(t);
        }

        t.tx = cpu.tx; t.ty = cpu.ty;
        t.loaded = true;

        // Tile bounding box (world)
        const float x0 = (float)(t.tx * kTileSize);
        const float y0 = (float)(t.ty * kTileSize);
        t.bb = AABB{ x0, y0, x0 + (float)kTileSize, y0 + (float)kTileSize };

        // Store texts on CPU (for now), but build roads/polys into per-tile GPU buffers.
        t.texts = std::move(cpu.texts);
        if(!t.texts.empty()) anyTextLoaded = true;

        // ---- Roads ----
        t.gpuLines.clear();
        t.gpuLines.reserve(cpu.lines.size());
        for(const auto& b : cpu.lines){
            if(b.verts.empty()) continue;

            LineBatchGpu g{};
            g.w = b.w; g.c = b.c;
            g.width = roadWidthPx(b.w);

            auto col = roadColor(b.w, b.c);
            g.color[0]=col[0]; g.color[1]=col[1]; g.color[2]=col[2]; g.color[3]=col[3];

            glGenVertexArrays(1, &g.vao);
            glGenBuffers(1, &g.vbo);
            glBindVertexArray(g.vao);
            glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(b.verts.size()*sizeof(Vec2)), b.verts.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), (void*)0);
            glBindVertexArray(0);

            g.vertCount = (GLsizei)b.verts.size();
            t.gpuLines.push_back(g);
        }

        // ---- Polygons (optional) ----
        t.gpuPolys.clear();
        if(!cpu.polys.empty()){
            anyPolyLoaded = true;
            t.gpuPolys.reserve(cpu.polys.size());
            for(const auto& pm : cpu.polys){
                if(pm.idx.empty() || pm.verts.empty()) continue;
                PolyBatchGpu pg{};
                pg.layer = pm.layer;
                pg.color[0]=pm.color[0]; pg.color[1]=pm.color[1]; pg.color[2]=pm.color[2]; pg.color[3]=pm.color[3];

                glGenVertexArrays(1, &pg.vao);
                glGenBuffers(1, &pg.vbo);
                glGenBuffers(1, &pg.ebo);

                glBindVertexArray(pg.vao);
                glBindBuffer(GL_ARRAY_BUFFER, pg.vbo);
                glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(pm.verts.size()*sizeof(Vec2)), pm.verts.data(), GL_STATIC_DRAW);
                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), (void*)0);

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pg.ebo);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(pm.idx.size()*sizeof(uint32_t)), pm.idx.data(), GL_STATIC_DRAW);

                glBindVertexArray(0);

                pg.indexCount = (GLsizei)pm.idx.size();
                t.gpuPolys.push_back(pg);
            }
        }

        // Release heavy CPU geometry right after upload (keeps memory low).
        // Texts stay on CPU for now (visibility depends on camera scale).
        t.lines.clear();
        t.lines.shrink_to_fit();
        t.polys.clear();
        t.polys.shrink_to_fit();

        gpuDirty = false;
    }

    // Usage tracking
    std::unordered_map<uint64_t, uint64_t> lastUsed;

    void markUsed(uint64_t key){
        lastUsed[key] = frameId;
    }

    // ---- Visible tiles ----
    // Compute render-visible tiles and a larger prefetch neighborhood.
    // Returns true if the render-visible set changed.
    bool computeVisible(double left,double right,double bottom,double top){
        // Core (render) bounds: just the view plus a small fixed pad to avoid edge popping.
        const int corePad = 1;

        const int vtx0 = (int)std::floor(left  / (double)kTileSize);
        const int vtx1 = (int)std::floor(right / (double)kTileSize);
        const int vty0 = (int)std::floor(bottom/ (double)kTileSize);
        const int vty1 = (int)std::floor(top   / (double)kTileSize);

        int tx0 = vtx0 - corePad;
        int tx1 = vtx1 + corePad;
        int ty0 = vty0 - corePad;
        int ty1 = vty1 + corePad;

        // Render-visible set
        std::unordered_set<uint64_t> newVisibleSet;
        newVisibleSet.reserve((size_t)std::max(1, (tx1-tx0+1)*(ty1-ty0+1)));

        std::vector<TileKey> newVisible;
        newVisible.reserve((size_t)std::max(1, (tx1-tx0+1)*(ty1-ty0+1)));

        for(int ty=ty0; ty<=ty1; ++ty){
            for(int tx=tx0; tx<=tx1; ++tx){
                newVisible.push_back(TileKey{tx,ty});
                newVisibleSet.insert(tileKey64(tx,ty));
            }
        }

        // Prefetch neighborhood: expand by 1x of the current view tile-span in each dimension.
// This means the prefetch rectangle is ~3x wider and ~3x taller than the view (i.e., ~9x area),
// which matches the requested "9x9-ish" neighborhood rule for typical view sizes.
const int nx = std::max(1, vtx1 - vtx0 + 1);
const int ny = std::max(1, vty1 - vty0 + 1);

int padX = nx; // 1x view span on each side
int padY = ny;

// Always respect the legacy minimum padTiles (compat)
padX = std::max(padX, padTiles);
padY = std::max(padY, padTiles);

        int ptx0 = vtx0 - padX;
        int ptx1 = vtx1 + padX;
        int pty0 = vty0 - padY;
        int pty1 = vty1 + padY;

        // Hard cap total prefetch tiles to keep things sane in global view.
        // We still request tiles by priority (center-first), so the user-visible effect remains.
        const int64_t maxPrefetchTotal = 20000; // can be tuned
        int64_t totalPref = (int64_t)(ptx1-ptx0+1) * (int64_t)(pty1-pty0+1);
        if(totalPref > maxPrefetchTotal){
            // Shrink pads proportionally.
            double scale = std::sqrt((double)maxPrefetchTotal / (double)std::max<int64_t>(1,totalPref));
            int newPadX = std::max(padTiles, (int)std::floor((double)padX * scale));
            int newPadY = std::max(padTiles, (int)std::floor((double)padY * scale));
            padX = std::max(newPadX, corePad);
            padY = std::max(newPadY, corePad);
            ptx0 = vtx0 - padX; ptx1 = vtx1 + padX;
            pty0 = vty0 - padY; pty1 = vty1 + padY;
        }

        std::unordered_set<uint64_t> newPrefetchSet;
        const int64_t prefCountEst = (int64_t)(ptx1-ptx0+1) * (int64_t)(pty1-pty0+1);
        newPrefetchSet.reserve((size_t)std::max<int64_t>(1, std::min<int64_t>(prefCountEst, 200000)));

        std::vector<TileKey> newPrefetch;
        newPrefetch.reserve((size_t)std::max<int64_t>(1, std::min<int64_t>(prefCountEst, 200000)));

        for(int ty=pty0; ty<=pty1; ++ty){
            for(int tx=ptx0; tx<=ptx1; ++tx){
                newPrefetch.push_back(TileKey{tx,ty});
                newPrefetchSet.insert(tileKey64(tx,ty));
            }
        }

        bool changed = (newVisibleSet.size() != visibleSet.size());
        if(!changed){
            // cheap equality check by probing existing set
            for(uint64_t k : newVisibleSet){
                if(visibleSet.find(k) == visibleSet.end()){ changed = true; break; }
            }
        }

        
        bool prefetchChanged = false;
        {
            std::lock_guard<std::mutex> lk(mtx);
            if(newPrefetchSet.size() != prefetchSet.size()){
                prefetchChanged = true;
            }else{
                for(uint64_t k : newPrefetchSet){
                    if(prefetchSet.find(k) == prefetchSet.end()){ prefetchChanged = true; break; }
                }
            }
        }

viewCenterX = 0.5*(left+right);
        viewCenterY = 0.5*(bottom+top);

        visible = std::move(newVisible);
        visibleSet = std::move(newVisibleSet);
        prefetch = std::move(newPrefetch);
        {
            std::lock_guard<std::mutex> lk(mtx);
            prefetchSet = std::move(newPrefetchSet);
        }
if(changed || prefetchChanged){
            viewGen.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(mtx);
            while(!reqQ.empty()) reqQ.pop();
            while(!readyQ.empty()) readyQ.pop();
            inFlight.clear();
        }
        return (changed || prefetchChanged);
    }


    void start(){
        // Open packed store if present
        openPackedStore();

        stop = false;

        // Spawn a small thread pool for tile I/O + CPU parsing.
        // Using multiple workers avoids the "one tile at a time" behavior in global view.
        if(workerCount <= 0){
            unsigned hc = std::thread::hardware_concurrency();
            if(hc == 0) hc = 4;
            workerCount = (int)std::max(2u, hc > 1 ? (hc - 1) : 2u);
        }

        workers.clear();
        workers.reserve((size_t)workerCount);

        for(int wi=0; wi<workerCount; ++wi){
            workers.emplace_back([this]{
                std::vector<uint8_t> blob;
                blob.reserve(1<<20);
                while(true){
                    uint64_t key = 0;
                    uint64_t gen = 0;
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv.wait(lk, [&]{ return stop.load() || !reqQ.empty(); });
                        if(stop.load() && reqQ.empty()) break;
                        auto item = reqQ.front();
                        reqQ.pop();
                        key = item.first;
                        gen = item.second;
                        // Skip stale request if view has changed.
                        if(gen != viewGen.load(std::memory_order_relaxed)) {
                            continue;
                        }

                    }

                    int tx = (int)(int32_t)(key >> 32);
                    int ty = (int)(int32_t)(key & 0xffffffffu);

                    PendingTileCPU cpu{};
                    cpu.tx=tx; cpu.ty=ty; cpu.gen=gen;

                    bool ok = readTileBlob(tx,ty, blob);
                    if(ok){
                        parseTileBlob(blob, cpu);
                    }

                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        
        const uint64_t curGen = viewGen.load(std::memory_order_relaxed);
inFlight.erase(key);
                        // Drop stale tiles if view changed or tile no longer requested.
                        if(cpu.gen != viewGen.load(std::memory_order_relaxed)) {
                            continue;
                        }
                        if(prefetchSet.find(key) == prefetchSet.end()) {
                            continue;
                        }
                        readyQ.push(std::move(cpu));
                    }
                }
            });
        }
    }
    void shutdown(){
        stop = true;
        cv.notify_all();
        for(auto& th : workers){ if(th.joinable()) th.join(); }
        workers.clear();
        closePackedStore();

        // destroy per-tile GPU
        for(auto& kv : tiles){
            destroyTileGpu(kv.second);
        }
        tiles.clear();
        inFlight.clear();
        while(!reqQ.empty()) reqQ.pop();
        while(!readyQ.empty()) readyQ.pop();
    }

    void requestVisibleTiles(){
        // Build a prioritized request list outside the lock to minimize contention.
        struct ReqItem { uint64_t key; int tx; int ty; float dist2; bool isCore; uint64_t off; };
        std::vector<ReqItem> items;
        items.reserve(prefetch.size());

        const double cx = viewCenterX;
        const double cy = viewCenterY;

        for(const auto& tk : prefetch){
            uint64_t key = tileKey64(tk.tx, tk.ty);
            double txc = ((double)tk.tx + 0.5) * (double)kTileSize;
            double tyc = ((double)tk.ty + 0.5) * (double)kTileSize;
            float dx = (float)(txc - cx);
            float dy = (float)(tyc - cy);
            float d2 = dx*dx + dy*dy;
            bool core = (visibleSet.find(key) != visibleSet.end());
            uint64_t off = 0;
            auto iit = index.find(key);
            if(iit != index.end()) off = iit->second.off;
            else off = UINT64_MAX;
            items.push_back(ReqItem{key, tk.tx, tk.ty, d2, core, off});
        }

        // Sort: core-visible first, then nearest to center.
        std::sort(items.begin(), items.end(), [](const ReqItem& a, const ReqItem& b){
            if(a.isCore != b.isCore) return a.isCore > b.isCore;
            if(a.dist2 != b.dist2) return a.dist2 < b.dist2;
            return a.off < b.off;
        });

        // Improve IO locality without destroying priority:
        // within small priority windows, sort by file offset (sequential-ish mmap page-in).
        const size_t kWindow = 256;
        for(size_t i=0; i<items.size(); i+=kWindow){
            size_t j = std::min(items.size(), i + kWindow);
            std::stable_sort(items.begin()+i, items.begin()+j, [](const ReqItem& a, const ReqItem& b){
                return a.off < b.off;
            });
        }

                const uint64_t genSnap = viewGen.load(std::memory_order_relaxed);

{
            std::lock_guard<std::mutex> lk(mtx);
            int enq = 0;
            for(const auto& it : items){
                auto tIt = tiles.find(it.key);
                bool alreadyLoaded = (tIt != tiles.end() && tIt->second.loaded);
                if(alreadyLoaded) continue;
                if(inFlight.find(it.key) != inFlight.end()) continue;
                inFlight.insert(it.key);
                reqQ.push(std::make_pair(it.key, genSnap));
                enq++;
                if(enq >= maxRequestEnqueue) break;
            }
        }
        cv.notify_all();
    }

        void pumpUploads(){
        // Use a time budget instead of a fixed count so we can "sprint" when many tiles are needed.
        const bool sprint = (int)prefetch.size() >= sprintThresholdTiles;
        const double budgetMs = sprint ? sprintUploadBudgetMs : normalUploadBudgetMs;

        using Clock = std::chrono::high_resolution_clock;
        auto t0 = Clock::now();

        int uploaded = 0;
        while(true){
            PendingTileCPU cpu{};
            {
                std::lock_guard<std::mutex> lk(mtx);
                if(readyQ.empty()) break;
                cpu = std::move(readyQ.front());
                readyQ.pop();
            }

            uint64_t key = tileKey64(cpu.tx, cpu.ty);
            uploadTileCPU(std::move(cpu));
            markUsed(key);
            uploaded++;

            // Also cap by count as a safety net (prevents pathological stalls on slow drivers).
            if(uploaded >= maxUploadPerFrame && !sprint) break;

            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if(ms >= budgetMs) break;
        }
    }
    void evictIfNeeded(){
        int freedTiles = 0;
        // If too many, evict least recently used that are not visible.
        if((int)tiles.size() <= maxResidentTiles){
            // also evict by TTL
        }

        // Build list of candidates
        std::vector<std::pair<uint64_t,uint64_t>> candidates; // (key,lastUsed)
        candidates.reserve(tiles.size());
        for(const auto& kv : tiles){
            uint64_t key = kv.first;
            if(visibleSet.find(key) != visibleSet.end()) continue; // keep visible
            uint64_t lu = 0;
            auto it = lastUsed.find(key);
            if(it != lastUsed.end()) lu = it->second;
            candidates.push_back({key, lu});
        }

        // TTL evict first
        for(const auto& c : candidates){
            if(frameId > c.second && (frameId - c.second) > ttlFrames){
                auto it = tiles.find(c.first);
                if(it != tiles.end()){
                    it->second.lines.clear(); it->second.polys.clear(); it->second.texts.clear();
                    destroyTileGpu(it->second);
                    tiles.erase(it); freedTiles++;
                }
                lastUsed.erase(c.first);
            }
        }

        if((int)tiles.size() <= maxResidentTiles) return;

        // LRU sort
        std::sort(candidates.begin(), candidates.end(),
                  [](auto a, auto b){ return a.second < b.second; });

        int toEvict = (int)tiles.size() - maxResidentTiles;
        for(int i=0;i<(int)candidates.size() && toEvict>0;i++){
            uint64_t key = candidates[i].first;
            auto it = tiles.find(key);
            if(it != tiles.end()){
                it->second.lines.clear(); it->second.polys.clear(); it->second.texts.clear();
                    destroyTileGpu(it->second);
                    tiles.erase(it); freedTiles++;
                lastUsed.erase(key);
                toEvict--;
            }
        }
#ifdef __GLIBC__
        if(freedTiles > 0) malloc_trim(0);
#endif
    }

        // ---- Merged geometry rebuild ----
    // (removed) Incremental per-tile GPU is used instead.

};

// ----------------- GM bounds-only reader -----------------

static bool readBoundsOnly(const std::string& gmPath, Bounds& outB){
    std::ifstream in(gmPath);
    if(!in) return false;
    std::string line;
    // first 4 lines skip
    for(int i=0;i<4;i++){
        if(!std::getline(in,line)) return false;
    }
    if(!std::getline(in,line)) return false;
    std::stringstream ss(line);
    std::string tag;
    ss >> tag;
    if(tag!="BOUNDS") return false;
    ss >> outB.minx >> outB.miny >> outB.maxx >> outB.maxy;
    return true;
}

// ----------------- Input callbacks -----------------

static void framebufferSizeCallback(GLFWwindow* window, int w, int h){
    (void)window;
    gCam.fbW = w;
    gCam.fbH = h;
    glViewport(0,0,w,h);
    gDirty = true;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods){
    (void)window; (void)mods;
    if(button == GLFW_MOUSE_BUTTON_LEFT){
        if(action == GLFW_PRESS){
            gMouseLeft = true;
            glfwGetCursorPos(window, &gLastX, &gLastY);
        } else if(action == GLFW_RELEASE){
            gMouseLeft = false;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double x, double y){
    if(!gMouseLeft) return;

    double dx = x - gLastX;
    double dy = y - gLastY;
    gLastX = x; gLastY = y;

    int winW=0, winH=0;
    glfwGetWindowSize(window, &winW, &winH);
    if(winW<=0 || winH<=0) return;

    // Convert window pixel delta to world delta
    double left,right,bottom,top;
    getViewRect(left,right,bottom,top);

    double ndc_dx = 2.0 * dx / (double)winW;
    double ndc_dy = 2.0 * dy / (double)winH;
    double world_dx = ndc_dx * (right-left) * 0.5;
    double world_dy = -ndc_dy * (top-bottom) * 0.5;

    gCam.centerX -= world_dx;
    gCam.centerY -= world_dy;
    gDirty = true;
}

static void scrollCallback(GLFWwindow* window, double xoff, double yoff){
    (void)xoff;
    if(yoff == 0.0) return;

    // Cursor anchored zoom: keep world point under cursor fixed.
    double cx=0, cy=0;
    glfwGetCursorPos(window, &cx, &cy);
    int winW=0, winH=0;
    glfwGetWindowSize(window, &winW, &winH);
    if(winW<=0 || winH<=0) return;

    double left,right,bottom,top;
    getViewRect(left,right,bottom,top);

    double nx = cx / (double)winW;
    double ny = cy / (double)winH;

    double worldX = left + nx * (right-left);
    double worldY = top - ny * (top-bottom);

    double zoomFactor = std::pow(1.1, yoff);
    double newZoom = gCam.zoom * zoomFactor;
    newZoom = std::max(0.05, std::min(200.0, newZoom));

    // Update zoom
    gCam.zoom = newZoom;

    // Recompute rect and re-center to keep (worldX,worldY) at cursor
    getViewRect(left,right,bottom,top);
    double newWorldX = left + nx * (right-left);
    double newWorldY = top - ny * (top-bottom);

    gCam.centerX += (worldX - newWorldX);
    gCam.centerY += (worldY - newWorldY);

    gDirty = true;
}

// ----------------- Main -----------------

int main(int argc, char** argv){
    std::string gmPath;
    if(argc >= 2) gmPath = argv[1];
    else {
        std::cerr << "Usage: " << argv[0] << " <map.gm> [font.ttf]\n";
        return 1;
    }

    // Default: data/ is relative to project root; run from repo root or pass absolute path.
    std::string fontPath;
    if(argc >= 3) fontPath = argv[2];
    else fontPath = findDefaultFontPath();

    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "map_viewer", nullptr, nullptr);
    if(!window){
        glfwTerminate();
        std::cerr << "Failed to create window\n";
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "Failed to load GL\n";
        return 1;
    }

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    int fbW=0, fbH=0;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    framebufferSizeCallback(window, fbW, fbH);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint colorProg = createColorProgram();
    GLuint textProg = createTextProgram();

    if(!fontPath.empty()){
        if(!initFontAtlas(fontPath)){
            std::cerr << "Font init failed. Text rendering disabled.\n";
            gTextReady = false;
        }
    } else {
        std::cerr << "No font file found. Put a TTF at data/DejaVuSans.ttf or external/fonts/DejaVuSans.ttf or pass it as argv[2]. Text rendering disabled.\n";
        gTextReady = false;
    }

    CacheInfo ci{};
    bool cacheHit = tryOpenCache(gmPath, ci);
    if(!cacheHit){
        // Need bounds for camera while building cache? We'll read bounds first (cheap)
        Bounds b{};
        if(!readBoundsOnly(gmPath, b)){
            std::cerr << "ERROR: Cannot read BOUNDS from gm.\n";
            return 1;
        }
        initCameraFromBounds(b, fbW, fbH);

        if(!buildCacheFromGM(gmPath, ci)){
            std::cerr << "ERROR: cache build failed.\n";
            return 1;
        }
    }

    // Initialize camera from cached bounds
    initCameraFromBounds(ci.bounds, fbW, fbH);

    TileManager tm{};
    tm.ci = ci;
    tm.start();

    // Main loop
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();

        double left,right,bottom,top;
        getViewRect(left,right,bottom,top);

        // Tile streaming tick (async I/O + bounded GPU uploads)
        tm.frameId++;
        if(gDirty){
            bool visChanged = tm.computeVisible(left,right,bottom,top);
            tm.requestVisibleTiles();

            // Bump text revision only when view area (scale) changes enough.
            const double viewArea = (right-left) * (top-bottom);
            if(tm.lastViewArea < 0.0){
                tm.lastViewArea = viewArea;
                tm.textRev++;
            } else {
                const double denom = std::max(1.0, tm.lastViewArea);
                const double rel = std::abs(viewArea - tm.lastViewArea) / denom;
                if(rel > 0.002) { // ~0.2% area change
                    tm.lastViewArea = viewArea;
                    tm.textRev++;
                } else if(visChanged){
                    // Visible set changed -> labels might enter/leave, rebuild their GPU.
                    tm.textRev++;
                }
            }

            gDirty = false;
        }
        // Upload a small number of tiles per frame to avoid hitches (especially in global view).
        tm.pumpUploads();
        // Evict tiles not visible / least-recently-used to keep RSS/VRAM bounded.
        tm.evictIfNeeded();
        glClearColor(0.08f, 0.09f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw order: polygons (by layer asc), roads, texts
        glUseProgram(colorProg);
        glUniform4f(glGetUniformLocation(colorProg, "uView"),
                    (float)left, (float)right, (float)bottom, (float)top);

        AABB viewBB{ (float)left, (float)bottom, (float)right, (float)top };

        // --------- Polygons (per-tile GPU; optional) ---------
        if(tm.anyPolyLoaded){
            // Collect visible polys and sort by layer (only if polygons exist).
            std::vector<const PolyBatchGpu*> drawPolys;
            drawPolys.reserve(1024);
            for(const auto& vk : tm.visible){
                uint64_t k64 = tileKey64(vk.tx, vk.ty);
                auto it = tm.tiles.find(k64);
                if(it == tm.tiles.end() || !it->second.loaded) continue;
                const Tile& t = it->second;
                for(const auto& pg : t.gpuPolys){
                    drawPolys.push_back(&pg);
                }
            }
            std::sort(drawPolys.begin(), drawPolys.end(),
                      [](const PolyBatchGpu* a, const PolyBatchGpu* b){ return a->layer < b->layer; });

            for(const PolyBatchGpu* p : drawPolys){
                glUniform4f(glGetUniformLocation(colorProg, "uColor"),
                            p->color[0], p->color[1], p->color[2], p->color[3]);
                glBindVertexArray(p->vao);
                glDrawElements(GL_TRIANGLES, p->indexCount, GL_UNSIGNED_INT, (void*)0);
            }
            glBindVertexArray(0);
        }

        // --------- Roads (per-tile GPU; incremental) ---------
        for(const auto& vk : tm.visible){
            uint64_t k64 = tileKey64(vk.tx, vk.ty);
            auto it = tm.tiles.find(k64);
            if(it == tm.tiles.end() || !it->second.loaded) continue;
            const Tile& t = it->second;

            // Optional: coarse culling by tile AABB
            if(!aabbIntersects(t.bb, viewBB)) continue;

            for(const auto& l : t.gpuLines){
                glUniform4f(glGetUniformLocation(colorProg, "uColor"),
                            l.color[0], l.color[1], l.color[2], l.color[3]);
                glLineWidth(l.width);
                glBindVertexArray(l.vao);
                glDrawArrays(GL_LINES, 0, l.vertCount);
            }
        }
        glBindVertexArray(0);

        // --------- Text ---------
                if(gTextReady && tm.anyTextLoaded){
                    const double viewArea = (right-left) * (top-bottom);
        
                    glUseProgram(textProg);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, gTextAtlasTex);
                    glUniform1i(glGetUniformLocation(textProg, "uTex"), 0);
                    glUniform4f(glGetUniformLocation(textProg, "uView"), (float)left,(float)right,(float)bottom,(float)top);
        
                    for(const auto& vk : tm.visible){
                        uint64_t k64 = tileKey64(vk.tx, vk.ty);
                        auto it = tm.tiles.find(k64);
                        if(it == tm.tiles.end() || !it->second.loaded) continue;
                        Tile& t = it->second;
                        if(t.texts.empty()) continue;
        
                        // Rebuild tile text VBO if scale changed enough.
                        if(t.gpuTextRev != tm.textRev){
                            tm.rebuildTileTextGpu(t, viewArea);
                        }
                        if(!t.gpuTextValid || t.gpuText.vertCount == 0) continue;
        
                        glBindVertexArray(t.gpuText.vao);
                        glDrawArrays(GL_TRIANGLES, 0, t.gpuText.vertCount);
                    }
                    glBindVertexArray(0);
                    glBindTexture(GL_TEXTURE_2D, 0);
                }
        
        glfwSwapBuffers(window);

    }

    tm.shutdown();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
