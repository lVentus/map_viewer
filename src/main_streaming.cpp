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

#include <sys/mman.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <unistd.h>

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

static const int kTileSize = 4096;
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

    std::vector<LineBatchCPU> lines;   // CPU, per style within tile
    std::vector<PolyMeshCPU> polys;    // CPU meshes (triangulated)
    std::vector<TextLabelCPU> texts;   // CPU labels

    AABB bb{}; // tile AABB
};

// Color mapping for roads (w,c)
static std::array<float,4> roadColor(int w,int c){
    (void)c;
    // simple palette by width
    if(w>=4) return {0.95f,0.85f,0.35f,1.0f};
    if(w==3) return {0.95f,0.95f,0.95f,1.0f};
    if(w==2) return {0.85f,0.85f,0.85f,1.0f};
    return {0.7f,0.7f,0.7f,1.0f};
}

static float roadWidthPx(int w)
{
    switch (w)
    {
        case 0:  return 1.0f;
        case 1:  return 1.2f;
        case 2:  return 1.6f;
        case 3:  return 2.2f;
        case 4:  return 3.0f;
        case 5:  return 4.0f;
        default: return 1.0f;
    }
}

struct TileManager {
    CacheInfo ci;

    struct TileIndexEntry { uint64_t off=0; uint32_t sz=0; };
    std::unordered_map<uint64_t, TileIndexEntry> index;
    int datFd = -1;
    size_t datBytes = 0;
    const uint8_t* datPtr = nullptr;

    std::unordered_map<uint64_t, Tile> tiles;
    std::vector<TileKey> visible;
    std::unordered_set<uint64_t> visibleSet;
    // Async loader
    struct PendingTileCPU {
    int tx=0, ty=0;
    std::vector<LineBatchCPU> lines;
    std::vector<PolyMeshCPU> polys;
    std::vector<TextLabelCPU> texts;
    bool empty=true;
};

    std::mutex mtx;
    std::condition_variable cv;
    std::queue<uint64_t> reqQ;
    std::unordered_set<uint64_t> inFlight;
    std::queue<PendingTileCPU> readyQ;
    std::atomic<bool> stop{false};
    int workerCount = 0;
    std::vector<std::thread> workers;

    // Tuning
    int maxResidentTiles = 512;
    int maxUploadPerFrame = 8;
    int padTiles = 1;              // preload ring
    uint64_t frameId = 0;
    uint64_t ttlFrames = 600;      // evict tiles not used for this many frames
    // Upload pacing
    // In normal view we cap uploads to avoid hitches; in "sprint" mode (many tiles needed) we spend more time uploading.
    int sprintThresholdTiles = 400;   // if visible tiles exceed this, enable sprint mode
    double normalUploadBudgetMs = 4.0;
    double sprintUploadBudgetMs = 20.0;

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

    // ---- Merged GPU batches ----
std::vector<LineBatchGpu> mergedLines;
std::vector<PolyBatchGpu> mergedPolys;
TextGpu mergedText;

bool gpuDirty = true; // set when tiles are added/removed or camera changes

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
void destroyMergedGpu(){
    for(auto& l : mergedLines) destroyLineGpu(l);
    for(auto& p : mergedPolys) destroyPolyGpu(p);
    destroyTextGpu(mergedText);
    mergedLines.clear();
    mergedPolys.clear();
}

void uploadTileCPU(PendingTileCPU&& cpu){
    uint64_t key = tileKey64(cpu.tx, cpu.ty);
    auto& t = tiles[key];
    t.tx = cpu.tx; t.ty = cpu.ty;
    t.loaded = true;

    // Store CPU geometry
    t.lines = std::move(cpu.lines);
    t.polys = std::move(cpu.polys);
    t.texts = std::move(cpu.texts);

    // Tile bounding box (world)
    const float x0 = (float)(t.tx * kTileSize);
    const float y0 = (float)(t.ty * kTileSize);
    t.bb = AABB{ x0, y0, x0 + (float)kTileSize, y0 + (float)kTileSize };

    gpuDirty = true;
}

// Usage tracking
    std::unordered_map<uint64_t, uint64_t> lastUsed;

    void markUsed(uint64_t key){
        lastUsed[key] = frameId;
    }

    // ---- Visible tiles ----
    void computeVisible(double left,double right,double bottom,double top){
        visible.clear();
        visibleSet.clear();

        int tx0 = (int)std::floor(left  / (double)kTileSize) - padTiles;
        int tx1 = (int)std::floor(right / (double)kTileSize) + padTiles;
        int ty0 = (int)std::floor(bottom/ (double)kTileSize) - padTiles;
        int ty1 = (int)std::floor(top   / (double)kTileSize) + padTiles;

        for(int ty=ty0; ty<=ty1; ++ty){
            for(int tx=tx0; tx<=tx1; ++tx){
                TileKey tk{tx,ty};
                visible.push_back(tk);
                visibleSet.insert(tileKey64(tx,ty));
            }
        }
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
                    uint64_t key=0;
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv.wait(lk, [&]{ return stop.load() || !reqQ.empty(); });
                        if(stop.load() && reqQ.empty()) break;
                        key = reqQ.front();
                        reqQ.pop();
                    }

                    int tx = (int)(int32_t)(key >> 32);
                    int ty = (int)(int32_t)(key & 0xffffffffu);

                    PendingTileCPU cpu{};
                    cpu.tx=tx; cpu.ty=ty;

                    bool ok = readTileBlob(tx,ty, blob);
                    if(ok){
                        parseTileBlob(blob, cpu);
                    }

                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        inFlight.erase(key);
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

        // destroy all merged GPU batches
        destroyMergedGpu();
        tiles.clear();
        inFlight.clear();
        while(!reqQ.empty()) reqQ.pop();
        while(!readyQ.empty()) readyQ.pop();
    }

    void requestVisibleTiles(){
        std::lock_guard<std::mutex> lk(mtx);
        for(const auto& tk : visible){
            uint64_t key = tileKey64(tk.tx, tk.ty);
            auto it = tiles.find(key);
            bool alreadyLoaded = (it != tiles.end() && it->second.loaded);
            if(alreadyLoaded) continue;
            if(inFlight.find(key) != inFlight.end()) continue;
            inFlight.insert(key);
            reqQ.push(key);
        }
        cv.notify_all();
    }

        void pumpUploads(){
        // Use a time budget instead of a fixed count so we can "sprint" when many tiles are needed.
        const bool sprint = (int)visible.size() >= sprintThresholdTiles;
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
                    tiles.erase(it);
                    gpuDirty = true;
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
                    tiles.erase(it);
                    gpuDirty = true;
                lastUsed.erase(key);
                toEvict--;
            }
        }
    }

    // ---- Merged geometry rebuild ----
// Rebuild merged GPU buffers for currently visible tiles.
// Called when camera changes or when tiles are added/evicted.
void rebuildMergedGeometry(double left,double right,double bottom,double top){
    destroyMergedGpu();

    AABB viewBB{ (float)left, (float)bottom, (float)right, (float)top };

    // --------- Polygons (group by layer + color) ----------
    struct PolyKey {
        int layer;
        uint64_t colorBits;
    };
    struct PolyKeyHash {
        size_t operator()(const PolyKey& k) const noexcept {
            return std::hash<uint64_t>()(((uint64_t)(uint32_t)k.layer<<32) ^ k.colorBits);
        }
    };
    struct PolyKeyEq {
        bool operator()(const PolyKey& a, const PolyKey& b) const noexcept {
            return a.layer==b.layer && a.colorBits==b.colorBits;
        }
    };
    struct PolyGroupCPU {
        int layer=0;
        float color[4]={1,1,1,1};
        std::vector<Vec2> verts;
        std::vector<uint32_t> idx;
        AABB bb{+1e30f,+1e30f,-1e30f,-1e30f};
    };

    std::unordered_map<PolyKey, PolyGroupCPU, PolyKeyHash, PolyKeyEq> polyGroups;

    auto packColorBits = [](const std::array<float,4>& c)->uint64_t{
        uint32_t r = (uint32_t)std::round(std::clamp(c[0],0.f,1.f)*255.f);
        uint32_t g = (uint32_t)std::round(std::clamp(c[1],0.f,1.f)*255.f);
        uint32_t b = (uint32_t)std::round(std::clamp(c[2],0.f,1.f)*255.f);
        uint32_t a = (uint32_t)std::round(std::clamp(c[3],0.f,1.f)*255.f);
        return (uint64_t)r | ((uint64_t)g<<8) | ((uint64_t)b<<16) | ((uint64_t)a<<24);
    };

    for(const auto& vk : visible){
        uint64_t k64 = tileKey64(vk.tx, vk.ty);
        auto it = tiles.find(k64);
        if(it==tiles.end()) continue;
        Tile& t = it->second;
        if(!t.loaded) continue;

        for(const auto& pm : t.polys){
            if(!aabbIntersects(pm.bb, viewBB)) continue;
            PolyKey pk{ pm.layer, packColorBits(pm.color) };
            auto& g = polyGroups[pk];
            if(g.verts.empty() && g.idx.empty()){
                g.layer = pm.layer;
                g.color[0]=pm.color[0]; g.color[1]=pm.color[1]; g.color[2]=pm.color[2]; g.color[3]=pm.color[3];
            }
            uint32_t base = (uint32_t)g.verts.size();
            g.verts.insert(g.verts.end(), pm.verts.begin(), pm.verts.end());
            g.idx.reserve(g.idx.size() + pm.idx.size());
            for(uint32_t id : pm.idx) g.idx.push_back(base + id);

            g.bb.minx = std::min(g.bb.minx, pm.bb.minx);
            g.bb.miny = std::min(g.bb.miny, pm.bb.miny);
            g.bb.maxx = std::max(g.bb.maxx, pm.bb.maxx);
            g.bb.maxy = std::max(g.bb.maxy, pm.bb.maxy);
        }
    }

    // Upload poly groups (sorted by layer)
    std::vector<PolyGroupCPU*> polyList;
    polyList.reserve(polyGroups.size());
    for(auto& kv : polyGroups) polyList.push_back(&kv.second);
    std::sort(polyList.begin(), polyList.end(), [](const PolyGroupCPU* a, const PolyGroupCPU* b){
        return a->layer < b->layer;
    });

    for(auto* g : polyList){
        if(g->verts.empty() || g->idx.empty()) continue;
        PolyBatchGpu pg{};
        pg.layer = g->layer;
        pg.color[0]=g->color[0]; pg.color[1]=g->color[1]; pg.color[2]=g->color[2]; pg.color[3]=g->color[3];
        pg.bb = g->bb;

        glGenVertexArrays(1, &pg.vao);
        glBindVertexArray(pg.vao);

        glGenBuffers(1, &pg.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, pg.vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(g->verts.size()*sizeof(Vec2)), g->verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), (void*)0);

        glGenBuffers(1, &pg.ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pg.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(g->idx.size()*sizeof(uint32_t)), g->idx.data(), GL_STATIC_DRAW);

        glBindVertexArray(0);

        pg.indexCount = (GLsizei)g->idx.size();
        mergedPolys.push_back(pg);
    }

    // --------- Roads (group by w,c) ----------
    struct RoadKey { int w=1; int c=0; };
    struct RoadKeyHash { size_t operator()(const RoadKey& k) const noexcept { return (size_t)((k.w<<16) ^ (k.c & 0xffff)); } };
    struct RoadKeyEq { bool operator()(const RoadKey& a, const RoadKey& b) const noexcept { return a.w==b.w && a.c==b.c; } };
    struct RoadGroupCPU {
        int w=1,c=0;
        std::vector<Vec2> verts;
        AABB bb{+1e30f,+1e30f,-1e30f,-1e30f};
    };
    std::unordered_map<RoadKey, RoadGroupCPU, RoadKeyHash, RoadKeyEq> roadGroups;

    for(const auto& vk : visible){
        uint64_t k64 = tileKey64(vk.tx, vk.ty);
        auto it = tiles.find(k64);
        if(it==tiles.end()) continue;
        Tile& t = it->second;
        if(!t.loaded) continue;

        for(const auto& lb : t.lines){
            if(lb.verts.empty()) continue;
            RoadKey rk{lb.w, lb.c};
            auto& g = roadGroups[rk];
            g.w = lb.w; g.c = lb.c;
            g.verts.insert(g.verts.end(), lb.verts.begin(), lb.verts.end());
            for(const auto& v : lb.verts){
                g.bb.minx = std::min(g.bb.minx, v.x);
                g.bb.miny = std::min(g.bb.miny, v.y);
                g.bb.maxx = std::max(g.bb.maxx, v.x);
                g.bb.maxy = std::max(g.bb.maxy, v.y);
            }
        }
    }

    for(auto& kv : roadGroups){
        auto& g = kv.second;
        if(g.verts.empty()) continue;

        LineBatchGpu lg{};
        lg.w = g.w; lg.c = g.c;
        auto col = roadColor(g.w, g.c);
        lg.color[0]=col[0]; lg.color[1]=col[1]; lg.color[2]=col[2]; lg.color[3]=col[3];
        lg.width = roadWidthPx(g.w);
        lg.bb = g.bb;

        glGenVertexArrays(1, &lg.vao);
        glBindVertexArray(lg.vao);

        glGenBuffers(1, &lg.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, lg.vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(g.verts.size()*sizeof(Vec2)), g.verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2), (void*)0);

        glBindVertexArray(0);

        lg.vertCount = (GLsizei)g.verts.size();
        mergedLines.push_back(lg);
    }

    // --------- Text (single global mesh) ----------
    rebuildMergedText(left,right,bottom,top);

    gpuDirty = false;
}

void rebuildMergedText(double left,double right,double bottom,double top){
    destroyTextGpu(mergedText);
    if(!gTextReady) return;

    const double viewW = (right-left);
    const double viewH = (top-bottom);
    const double viewArea = viewW*viewH;

    const double minOn  = 0.01;
    const double minOff = 0.008;
    const double maxOn  = 0.15;
    const double maxOff = 0.18;

    std::vector<TextVertex> verts;
    verts.reserve(1<<16);

    // persistent hysteresis per tile
    static std::unordered_map<uint64_t, std::vector<uint8_t>> visState;

    for(const auto& vk : visible){
        uint64_t k64 = tileKey64(vk.tx, vk.ty);
        auto it = tiles.find(k64);
        if(it==tiles.end()) continue;
        Tile& t = it->second;
        if(!t.loaded) continue;
        if(t.texts.empty()) continue;

        auto& vs = visState[k64];
        if(vs.size() != t.texts.size()) vs.assign(t.texts.size(), 1);

        for(size_t i=0;i<t.texts.size();++i){
            const auto& lbl = t.texts[i];

            if(lbl.pos.x < (float)left || lbl.pos.x > (float)right || lbl.pos.y < (float)bottom || lbl.pos.y > (float)top){
                vs[i]=0;
                continue;
            }

            float minX=+1e9f, minY=+1e9f, maxX=-1e9f, maxY=-1e9f;
            float penX=0.f, penY=0.f;
            for(unsigned char uc : lbl.text){
                if(uc < 32 || uc > 126) continue;
                stbtt_aligned_quad q;
                stbtt_GetBakedQuad(gBaked, gAtlasW, gAtlasH, uc-32, &penX, &penY, &q, 1);
                float qx0=q.x0, qx1=q.x1;
                float qy0=-q.y0, qy1=-q.y1;
                minX = std::min(minX, qx0);
                maxX = std::max(maxX, qx1);
                minY = std::min(minY, qy1);
                maxY = std::max(maxY, qy0);
            }
            if(minX > maxX || minY > maxY){
                vs[i]=0;
                continue;
            }

            const float bboxWpx = (maxX-minX);
            const float bboxHpx = (maxY-minY);
            const float scale = lbl.size / gFontPixelSize;
            const double bboxW = (double)bboxWpx * (double)scale;
            const double bboxH = (double)bboxHpx * (double)scale;
            const double areaRatio = (bboxW*bboxH)/viewArea;

            bool was = vs[i]!=0;
            double minT = was ? minOff : minOn;
            double maxT = was ? maxOff : maxOn;
            bool pass = (areaRatio >= minT) && (areaRatio <= maxT);
            vs[i] = pass ? 1 : 0;
            if(!pass) continue;

            const float cs = std::cos(lbl.angle);
            const float sn = std::sin(lbl.angle);

            auto emit = [&](float lx, float ly, float u, float v){
                float wx = lx * scale;
                float wy = ly * scale;
                float rx = wx*cs - wy*sn;
                float ry = wx*sn + wy*cs;
                TextVertex tv{};
                tv.x = lbl.pos.x + rx;
                tv.y = lbl.pos.y + ry;
                tv.u = u; tv.v = v;
                tv.r = lbl.color[0]; tv.g = lbl.color[1]; tv.b = lbl.color[2]; tv.a = lbl.color[3];
                verts.push_back(tv);
            };

            // Generate quads
            float x=0.f, y=0.f;
            for(unsigned char uc : lbl.text){
                if(uc < 32 || uc > 126) continue;
                stbtt_aligned_quad q;
                stbtt_GetBakedQuad(gBaked, gAtlasW, gAtlasH, uc-32, &x, &y, &q, 1);

                // q positions are in pixels; y down. We'll flip y for local.
                float x0=q.x0, x1=q.x1;
                float y0=-q.y0, y1=-q.y1;

                // 2 triangles
                emit(x0, y0, q.s0, q.t0);
                emit(x1, y0, q.s1, q.t0);
                emit(x1, y1, q.s1, q.t1);

                emit(x0, y0, q.s0, q.t0);
                emit(x1, y1, q.s1, q.t1);
                emit(x0, y1, q.s0, q.t1);
            }
        }
    }

    if(verts.empty()) return;

    glGenVertexArrays(1, &mergedText.vao);
    glBindVertexArray(mergedText.vao);
    glGenBuffers(1, &mergedText.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, mergedText.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size()*sizeof(TextVertex)), verts.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)offsetof(TextVertex,x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)offsetof(TextVertex,u));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(TextVertex), (void*)offsetof(TextVertex,r));
    glBindVertexArray(0);

    mergedText.vertCount = (GLsizei)verts.size();
}
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
            tm.computeVisible(left,right,bottom,top);
            tm.requestVisibleTiles();
            tm.gpuDirty = true;
            gDirty = false;
        }
        // Upload a small number of tiles per frame to avoid hitches (especially in global view).
        tm.pumpUploads();
        // Evict tiles not visible / least-recently-used to keep RSS/VRAM bounded.
        tm.evictIfNeeded();
        if(tm.gpuDirty){
            tm.rebuildMergedGeometry(left,right,bottom,top);
        }

        glClearColor(0.08f, 0.09f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw order: polygons (by layer asc), roads, texts
glUseProgram(colorProg);
glUniform4f(glGetUniformLocation(colorProg, "uView"),
            (float)left, (float)right, (float)bottom, (float)top);

AABB viewBB{ (float)left, (float)bottom, (float)right, (float)top };

// Polygons (already grouped & sorted by layer)
for(const auto& p : tm.mergedPolys){
    if(!aabbIntersects(p.bb, viewBB)) continue;
    glUniform4f(glGetUniformLocation(colorProg, "uColor"),
                p.color[0], p.color[1], p.color[2], p.color[3]);
    glBindVertexArray(p.vao);
    glDrawElements(GL_TRIANGLES, p.indexCount, GL_UNSIGNED_INT, (void*)0);
}
glBindVertexArray(0);

// Roads (grouped by style)
for(const auto& l : tm.mergedLines){
    if(!aabbIntersects(l.bb, viewBB)) continue;
    glUniform4f(glGetUniformLocation(colorProg, "uColor"),
                l.color[0], l.color[1], l.color[2], l.color[3]);
    glLineWidth(l.width);
    glBindVertexArray(l.vao);
    glDrawArrays(GL_LINES, 0, l.vertCount);
}
glBindVertexArray(0);

// Text (single merged draw)
if(gTextReady && tm.mergedText.vertCount > 0){
    glUseProgram(textProg);
    glUniform4f(glGetUniformLocation(textProg, "uView"),
                (float)left, (float)right, (float)bottom, (float)top);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gTextAtlasTex);
    glUniform1i(glGetUniformLocation(textProg, "uTex"), 0);

    glBindVertexArray(tm.mergedText.vao);
    glDrawArrays(GL_TRIANGLES, 0, tm.mergedText.vertCount);
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
