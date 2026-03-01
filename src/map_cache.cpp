// src/map_cache.cpp
#include "map_cache.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __GLIBC__
#include <malloc.h>
#endif

using Clock = std::chrono::high_resolution_clock;

// ---- cache dir key ----

static std::string computeCacheDir(const std::string& gmPath) {
    uint64_t sz = 0, mt = 0;
    if (!statFile(gmPath, sz, mt)) return "";

    std::string key = gmPath + "|" + std::to_string(sz) + "|" + std::to_string(mt)
                    + "|" + std::to_string(kTileSize) + "|" + std::to_string(kCacheVersion);

    const uint64_t h = fnv1a64_str(key);
    return getCacheRoot() + "/" + baseName(gmPath) + "_" + hex64(h);
}

// ---- manifest ----

static bool writeManifest(const CacheInfo& ci) {
    std::ofstream out(ci.manifestPath, std::ios::binary);
    if (!out) return false;

    out << "version " << kCacheVersion << "\n";
    out << "tileSize " << kTileSize << "\n";
    out << "gmSize " << ci.gmSize << "\n";
    out << "gmMtime " << ci.gmMtime << "\n";
    out << "bounds " << ci.bounds.minx << " " << ci.bounds.miny << " " << ci.bounds.maxx << " " << ci.bounds.maxy << "\n";
    return true;
}

static bool readManifest(CacheInfo& ci) {
    std::ifstream in(ci.manifestPath, std::ios::binary);
    if (!in) return false;

    std::string tag;
    int version = 0;
    int tileSize = 0;
    uint64_t gmSize = 0, gmMtime = 0;
    Bounds b{};

    while (in >> tag) {
        if (tag == "version") in >> version;
        else if (tag == "tileSize") in >> tileSize;
        else if (tag == "gmSize") in >> gmSize;
        else if (tag == "gmMtime") in >> gmMtime;
        else if (tag == "bounds") in >> b.minx >> b.miny >> b.maxx >> b.maxy;
        else { std::string rest; std::getline(in, rest); }
    }

    if (version != kCacheVersion || tileSize != kTileSize) return false;

    ci.bounds = b;
    ci.gmSize = gmSize;
    ci.gmMtime = gmMtime;
    return true;
}

// ---- tile writer (per-tile small files) ----

struct TileWriter {
    std::string tilesDir;
    size_t maxOpen = 64;

    struct Handle { FILE* f = nullptr; uint64_t last = 0; };
    std::unordered_map<uint64_t, Handle> open;
    uint64_t tick = 1;

    bool init(const std::string& cacheDir) {
        tilesDir = cacheDir + "/tiles";
        return ensureDir(tilesDir);
    }

    std::string pathFor(int tx, int ty) const {
        return tilesDir + "/t_" + std::to_string(tx) + "_" + std::to_string(ty) + ".bin";
    }

    FILE* get(int tx, int ty) {
        const uint64_t k = tileKey64(tx, ty);
        auto it = open.find(k);
        if (it != open.end()) { it->second.last = tick++; return it->second.f; }

        if (open.size() >= maxOpen) {
            auto itOld = open.end();
            for (auto it2 = open.begin(); it2 != open.end(); ++it2) {
                if (itOld == open.end() || it2->second.last < itOld->second.last) itOld = it2;
            }
            if (itOld != open.end()) {
                std::fclose(itOld->second.f);
                open.erase(itOld);
            }
        }

        const std::string p = pathFor(tx, ty);
        FILE* f = std::fopen(p.c_str(), "ab");
        if (!f) return nullptr;

        open[k] = Handle{f, tick++};
        return f;
    }

    void writeRec(int tx, int ty, uint8_t type, const void* payload, uint32_t sz) {
        FILE* f = get(tx, ty);
        if (!f) return;
        std::fwrite(&type, 1, 1, f);
        std::fwrite(&sz, 4, 1, f);
        std::fwrite(payload, 1, sz, f);
    }

    void shutdown() {
        for (auto& kv : open) std::fclose(kv.second.f);
        open.clear();
    }
};

// ---- cache pack (tiles.dat + tiles.idx) ----

static bool packTiles(const CacheInfo& ci) {
    namespace fs = std::filesystem;
    if (!fs::exists(ci.tilesDir)) return false;

    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(ci.tilesDir)) {
        if (!e.is_regular_file()) continue;
        auto name = e.path().filename().string();
        if (name.rfind("t_", 0) != 0) continue;
        if (e.path().extension() != ".bin") continue;
        files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    std::ofstream dat(ci.tilesDatPath, std::ios::binary);
    std::ofstream idx(ci.tilesIdxPath, std::ios::binary);
    if (!dat || !idx) return false;

    // idx:
    // magic 'MVTI' + version + count
    // entries: key, offset, size, reserved
    const uint32_t magic = 0x4954564d; // 'MVTI'
    const uint32_t ver = (uint32_t)kCacheVersion;
    const uint32_t count = (uint32_t)files.size();
    idx.write((const char*)&magic, 4);
    idx.write((const char*)&ver, 4);
    idx.write((const char*)&count, 4);

    uint64_t offset = 0;
    for (const auto& fp : files) {
        std::ifstream in(fp, std::ios::binary);
        if (!in) continue;
        in.seekg(0, std::ios::end);
        const std::streamsize sz = in.tellg();
        in.seekg(0, std::ios::beg);
        if (sz <= 0) continue;

        std::vector<char> buf((size_t)sz);
        in.read(buf.data(), sz);
        if (!in) continue;

        int tx = 0, ty = 0;
        {
            const auto stem = fp.stem().string(); // t_tx_ty
            std::sscanf(stem.c_str(), "t_%d_%d", &tx, &ty);
        }

        const uint64_t key = tileKey64(tx, ty);
        const uint64_t off = offset;
        const uint32_t usz = (uint32_t)sz;

        dat.write(buf.data(), sz);

        idx.write((const char*)&key, 8);
        idx.write((const char*)&off, 8);
        idx.write((const char*)&usz, 4);
        uint32_t rsv = 0;
        idx.write((const char*)&rsv, 4);

        offset += (uint64_t)sz;
    }

    dat.flush();
    idx.flush();
    return true;
}

// ---- bounds-only reader ----

static bool readBoundsOnly(const std::string& gmPath, Bounds& outB) {
    std::ifstream in(gmPath);
    if (!in) return false;

    std::string line;
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(in, line)) return false;
    }
    if (!std::getline(in, line)) return false;

    std::stringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag != "BOUNDS") return false;

    ss >> outB.minx >> outB.miny >> outB.maxx >> outB.maxy;
    return true;
}

// ---- cache build ----

bool MapCache::openOrBuild(const std::string& gmPath, CacheInfo& outInfo) {
    CacheInfo ci{};
    if (tryOpenCache_(gmPath, ci)) {
        ci_ = ci;
        outInfo = ci_;
        return true;
    }

    // bounds must be readable even before cache exists (camera init uses it).
    Bounds b{};
    if (!readBoundsOnly(gmPath, b)) {
        std::cerr << "ERROR: cannot read BOUNDS from gm.\n";
        return false;
    }

    if (!buildCache_(gmPath, ci)) {
        return false;
    }

    ci_ = ci;
    outInfo = ci_;
    return true;
}

bool MapCache::tryOpenCache_(const std::string& gmPath, CacheInfo& out) {
    uint64_t srcSz = 0, srcMt = 0;
    if (!statFile(gmPath, srcSz, srcMt)) return false;

    out.cacheDir = computeCacheDir(gmPath);
    out.tilesDir = out.cacheDir + "/tiles";
    out.manifestPath = out.cacheDir + "/manifest.txt";
    out.tilesDatPath = out.cacheDir + "/tiles.dat";
    out.tilesIdxPath = out.cacheDir + "/tiles.idx";

    {
        std::ifstream f(out.manifestPath, std::ios::binary);
        if (!f.good()) return false;
    }

    CacheInfo tmp = out;
    if (!readManifest(tmp)) return false;

    if (tmp.gmSize != srcSz || tmp.gmMtime != srcMt) return false;

    // packed store optional; fallback to small tile files if missing
    tmp.valid = true;
    out = tmp;

    std::cout << "[Cache] Hit: " << out.cacheDir << "\n";
    return true;
}

bool MapCache::buildCache_(const std::string& gmPath, CacheInfo& out) {
    uint64_t gmSize = 0, gmMtime = 0;
    if (!statFile(gmPath, gmSize, gmMtime)) {
        std::cerr << "ERROR: cannot stat gm: " << gmPath << "\n";
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
    if (!in) {
        std::cerr << "ERROR: cannot open gm: " << gmPath << "\n";
        return false;
    }

    // bigger stream buf for text parsing
    static std::vector<char> buf(4 * 1024 * 1024);
    in.rdbuf()->pubsetbuf(buf.data(), (std::streamsize)buf.size());

    size_t numNodes = 0, numEdges = 0, numPolys = 0, numTexts = 0;
    std::string line;

    auto readCountLine = [&](size_t& outCount) -> bool {
        if (!std::getline(in, line)) return false;
        std::stringstream ss(line);
        ss >> outCount;
        return true;
    };

    if (!readCountLine(numNodes) || !readCountLine(numEdges) || !readCountLine(numPolys) || !readCountLine(numTexts)) {
        std::cerr << "ERROR: invalid header counts.\n";
        return false;
    }

    if (!std::getline(in, line)) {
        std::cerr << "ERROR: missing BOUNDS.\n";
        return false;
    }

    {
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag != "BOUNDS") {
            std::cerr << "ERROR: expected BOUNDS.\n";
            return false;
        }
        ss >> out.bounds.minx >> out.bounds.miny >> out.bounds.maxx >> out.bounds.maxy;
    }

    std::cout << "[Cache] Building cache...\n";
    std::cout << "  Nodes=" << numNodes << " Edges=" << numEdges << " Polys=" << numPolys << " Texts=" << numTexts << "\n";
    std::cout << "  TileSize=" << kTileSize << "\n";
    std::cout << "  CacheDir=" << out.cacheDir << "\n";

    // nodes in RAM (needed for edge resolve)
    std::vector<float> nodesXY(numNodes * 2);
    for (size_t i = 0; i < numNodes; ++i) {
        double x = 0, y = 0;
        if (!(in >> x >> y)) {
            std::cerr << "ERROR: failed reading node " << i << "\n";
            return false;
        }
        nodesXY[i * 2 + 0] = (float)x;
        nodesXY[i * 2 + 1] = (float)y;
    }
    std::getline(in, line);

    TileWriter writer;
    if (!writer.init(out.cacheDir)) {
        std::cerr << "ERROR: tile writer init failed.\n";
        return false;
    }

    std::vector<TileKey> tiles;

    // edges
    for (size_t i = 0; i < numEdges; ++i) {
        uint32_t u = 0, v = 0;
        int w = 1, c = 0;
        if (!(in >> u >> v >> w >> c)) {
            std::cerr << "ERROR: failed reading edge " << i << "\n";
            writer.shutdown();
            return false;
        }
        if (u >= numNodes || v >= numNodes) continue;

        EdgeRec rec{};
        rec.x0 = nodesXY[u * 2 + 0];
        rec.y0 = nodesXY[u * 2 + 1];
        rec.x1 = nodesXY[v * 2 + 0];
        rec.y1 = nodesXY[v * 2 + 1];
        rec.w = (int32_t)w;
        rec.c = (int32_t)c;

        AABB bb{};
        bb.minx = std::min(rec.x0, rec.x1);
        bb.maxx = std::max(rec.x0, rec.x1);
        bb.miny = std::min(rec.y0, rec.y1);
        bb.maxy = std::max(rec.y0, rec.y1);

        tilesForAABB(bb, tiles);
        for (const auto& t : tiles) {
            writer.writeRec(t.tx, t.ty, REC_EDGE, &rec, (uint32_t)sizeof(EdgeRec));
        }
    }
    std::getline(in, line);

    // polys (line based)
    for (size_t i = 0; i < numPolys; ++i) {
        if (!std::getline(in, line)) {
            std::cerr << "ERROR: missing polygon line " << i << "\n";
            writer.shutdown();
            return false;
        }
        if (line.empty()) { --i; continue; }

        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag != "POLY") {
            std::cerr << "ERROR: expected POLY.\n";
            writer.shutdown();
            return false;
        }

        int layer = 0;
        float r = 0, g = 0, b = 0, a = 0;
        uint32_t n = 0;
        ss >> layer >> r >> g >> b >> a >> n;

        std::vector<Vec2> verts;
        verts.reserve(n);

        AABB bb{+1e30f, +1e30f, -1e30f, -1e30f};
        for (uint32_t k = 0; k < n; ++k) {
            float x = 0, y = 0;
            ss >> x >> y;
            verts.push_back(v2(x, y));
            bb.minx = std::min(bb.minx, x);
            bb.maxx = std::max(bb.maxx, x);
            bb.miny = std::min(bb.miny, y);
            bb.maxy = std::max(bb.maxy, y);
        }

        const uint32_t payloadSz = (uint32_t)(sizeof(int32_t) + sizeof(float) * 4 + sizeof(uint32_t) + sizeof(float) * 2 * n);
        std::vector<uint8_t> payload(payloadSz);
        uint8_t* p = payload.data();

        auto wr = [&](const void* src, size_t sz) {
            std::memcpy(p, src, sz);
            p += sz;
        };

        const int32_t layer32 = (int32_t)layer;
        wr(&layer32, sizeof(layer32));
        wr(&r, sizeof(float)); wr(&g, sizeof(float)); wr(&b, sizeof(float)); wr(&a, sizeof(float));
        wr(&n, sizeof(uint32_t));
        for (uint32_t k = 0; k < n; ++k) {
            wr(&verts[k].x, sizeof(float));
            wr(&verts[k].y, sizeof(float));
        }

        tilesForAABB(bb, tiles);
        for (const auto& t : tiles) {
            writer.writeRec(t.tx, t.ty, REC_POLY, payload.data(), payloadSz);
        }
    }

    // texts
    for (size_t i = 0; i < numTexts; ++i) {
        if (!std::getline(in, line)) {
            std::cerr << "ERROR: missing text line " << i << "\n";
            writer.shutdown();
            return false;
        }
        if (line.empty()) { --i; continue; }

        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag != "TEXT") {
            std::cerr << "ERROR: expected TEXT.\n";
            writer.shutdown();
            return false;
        }

        float x = 0, y = 0, angle = 0, size = 0;
        float r = 1, g = 1, b = 1, a = 1;
        ss >> x >> y >> angle >> size >> r >> g >> b >> a;

        std::string rest;
        std::getline(ss, rest);
        if (!rest.empty() && rest[0] == ' ') rest.erase(rest.begin());

        const uint32_t len = (uint32_t)rest.size();
        const uint32_t payloadSz = (uint32_t)(sizeof(float) * 4 + sizeof(float) * 4 + sizeof(uint32_t) + len);
        std::vector<uint8_t> payload(payloadSz);
        uint8_t* p = payload.data();

        auto wr = [&](const void* src, size_t sz) {
            std::memcpy(p, src, sz);
            p += sz;
        };

        wr(&x, sizeof(float)); wr(&y, sizeof(float)); wr(&angle, sizeof(float)); wr(&size, sizeof(float));
        wr(&r, sizeof(float)); wr(&g, sizeof(float)); wr(&b, sizeof(float)); wr(&a, sizeof(float));
        wr(&len, sizeof(uint32_t));
        if (len) wr(rest.data(), len);

        AABB bb{x, y, x, y};
        tilesForAABB(bb, tiles);
        for (const auto& t : tiles) {
            writer.writeRec(t.tx, t.ty, REC_TEXT, payload.data(), payloadSz);
        }
    }

    writer.shutdown();

    if (!packTiles(out)) {
        std::cerr << "[Cache] Warning: pack tiles failed (will fall back to per-tile files).\n";
    }

    if (!writeManifest(out)) {
        std::cerr << "ERROR: failed writing manifest.\n";
        return false;
    }

    out.valid = true;
    std::cout << "[Cache] Build done.\n";
    return true;
}

// ---- packed store ----

bool MapCache::openPackedStore_() {
    std::ifstream idx(ci_.tilesIdxPath, std::ios::binary);
    std::ifstream dat(ci_.tilesDatPath, std::ios::binary);
    if (!idx.good() || !dat.good()) return false;

    uint32_t magic = 0, ver = 0, count = 0;
    idx.read((char*)&magic, 4);
    idx.read((char*)&ver, 4);
    idx.read((char*)&count, 4);
    if (!idx || magic != 0x4954564d) return false; // 'MVTI'

    index_.clear();
    index_.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        uint64_t key = 0, off = 0;
        uint32_t sz = 0, rsv = 0;
        idx.read((char*)&key, 8);
        idx.read((char*)&off, 8);
        idx.read((char*)&sz, 4);
        idx.read((char*)&rsv, 4);
        if (!idx) break;
        index_[key] = TileIndexEntry{off, sz};
    }

    datFd_ = ::open(ci_.tilesDatPath.c_str(), O_RDONLY);
    if (datFd_ < 0) return false;

    struct stat st {};
    if (fstat(datFd_, &st) != 0) {
        ::close(datFd_);
        datFd_ = -1;
        return false;
    }

    datBytes_ = (size_t)st.st_size;
    void* p = mmap(nullptr, datBytes_, PROT_READ, MAP_PRIVATE, datFd_, 0);
    if (p == MAP_FAILED) {
        ::close(datFd_);
        datFd_ = -1;
        datBytes_ = 0;
        return false;
    }

    datPtr_ = (const uint8_t*)p;
    (void)posix_fadvise(datFd_, 0, 0, POSIX_FADV_SEQUENTIAL);
    return true;
}

void MapCache::closePackedStore_() {
    if (datPtr_) {
        munmap((void*)datPtr_, datBytes_);
        datPtr_ = nullptr;
    }
    if (datFd_ >= 0) {
        ::close(datFd_);
        datFd_ = -1;
    }
    datBytes_ = 0;
    index_.clear();
}

static std::string tilePathFallback(const CacheInfo& ci, int tx, int ty) {
    return ci.tilesDir + "/t_" + std::to_string(tx) + "_" + std::to_string(ty) + ".bin";
}

bool MapCache::readTileBlob_(int tx, int ty, std::vector<uint8_t>& outBlob) {
    outBlob.clear();
    const uint64_t key = tileKey64(tx, ty);

    if (datPtr_ && !index_.empty()) {
        auto it = index_.find(key);
        if (it == index_.end()) return true; // empty
        const uint64_t off = it->second.off;
        const uint32_t sz = it->second.sz;
        if (off + sz > datBytes_) return false;
        outBlob.resize(sz);
        std::memcpy(outBlob.data(), datPtr_ + off, sz);
        return true;
    }

    // fallback: small file
    std::ifstream in(tilePathFallback(ci_, tx, ty), std::ios::binary);
    if (!in) return true;
    in.seekg(0, std::ios::end);
    const std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);
    if (sz <= 0) return true;

    outBlob.resize((size_t)sz);
    in.read((char*)outBlob.data(), sz);
    return (bool)in;
}

// ---- tile parse ----

bool MapCache::parseTileBlob_(const std::vector<uint8_t>& blob, PendingTileCPU& out) {
    out.empty = true;
    out.lines.clear();
    out.polys.clear();
    out.texts.clear();

    if (blob.empty()) return true;

    struct TmpBatch { std::vector<Vec2> verts; int w = 1; int c = 0; };
    std::unordered_map<int, TmpBatch> batches;

    size_t pos = 0;
    const uint8_t* data = blob.data();
    const size_t N = blob.size();

    while (pos + 5 <= N) {
        const uint8_t type = data[pos]; pos += 1;
        uint32_t sz = 0;
        std::memcpy(&sz, data + pos, 4); pos += 4;
        if (pos + sz > N) break;

        const uint8_t* p = data + pos;
        pos += sz;

        if (type == REC_EDGE) {
            if (sz != sizeof(EdgeRec)) continue;
            EdgeRec rec{};
            std::memcpy(&rec, p, sizeof(EdgeRec));

            const int keyStyle = (rec.w << 16) ^ (rec.c & 0xffff);
            auto& b = batches[keyStyle];
            b.w = rec.w;
            b.c = rec.c;
            b.verts.push_back(v2(rec.x0, rec.y0));
            b.verts.push_back(v2(rec.x1, rec.y1));
            out.empty = false;
        }
        else if (type == REC_POLY) {
            const uint8_t* q = p;
            int32_t layer = 0;
            float r = 0, g = 0, b = 0, a = 0;
            uint32_t n = 0;

            if (sz < 4 + 4 * 4 + 4) continue;
            std::memcpy(&layer, q, 4); q += 4;
            std::memcpy(&r, q, 4); q += 4;
            std::memcpy(&g, q, 4); q += 4;
            std::memcpy(&b, q, 4); q += 4;
            std::memcpy(&a, q, 4); q += 4;
            std::memcpy(&n, q, 4); q += 4;

            if (n < 3) continue;
            if ((size_t)(q - p) + (size_t)n * 8 > (size_t)sz) continue;

            PolyMeshCPU pm{};
            pm.layer = (int)layer;
            pm.color = {r, g, b, a};
            pm.verts.resize(n);

            AABB bb{+1e30f, +1e30f, -1e30f, -1e30f};
            for (uint32_t i = 0; i < n; ++i) {
                float x = 0, y = 0;
                std::memcpy(&x, q, 4); q += 4;
                std::memcpy(&y, q, 4); q += 4;
                pm.verts[i] = v2(x, y);
                bb.minx = std::min(bb.minx, x);
                bb.miny = std::min(bb.miny, y);
                bb.maxx = std::max(bb.maxx, x);
                bb.maxy = std::max(bb.maxy, y);
            }
            pm.bb = bb;

            if (!triangulateEarClipping(pm.verts, pm.idx)) continue;

            out.polys.push_back(std::move(pm));
            out.empty = false;
        }
        else if (type == REC_TEXT) {
            const uint8_t* q = p;
            float x = 0, y = 0, angle = 0, sizePt = 0;
            float r = 1, g = 1, b = 1, a = 1;
            uint32_t len = 0;

            if (sz < 4 * 10 + 4) continue;
            std::memcpy(&x, q, 4); q += 4;
            std::memcpy(&y, q, 4); q += 4;
            std::memcpy(&angle, q, 4); q += 4;
            std::memcpy(&sizePt, q, 4); q += 4;
            std::memcpy(&r, q, 4); q += 4;
            std::memcpy(&g, q, 4); q += 4;
            std::memcpy(&b, q, 4); q += 4;
            std::memcpy(&a, q, 4); q += 4;
            std::memcpy(&len, q, 4); q += 4;

            if ((size_t)(q - p) + (size_t)len > (size_t)sz) continue;

            TextLabelCPU t{};
            t.pos = v2(x, y);
            t.angle = angle;
            t.size = sizePt;
            t.color[0] = r; t.color[1] = g; t.color[2] = b; t.color[3] = a;
            t.text.assign((const char*)q, (size_t)len);

            out.texts.push_back(std::move(t));
            out.empty = false;
        }
    }

    out.lines.reserve(batches.size());
    for (auto& kv : batches) {
        LineBatchCPU b{};
        b.w = kv.second.w;
        b.c = kv.second.c;
        b.verts = std::move(kv.second.verts);
        out.lines.push_back(std::move(b));
    }
    return true;
}

// ---- road style ----

static std::array<float, 4> roadColor(int /*w*/, int c) {
    float r = 1.0f, g = 1.0f, b = 1.0f;
    switch (c) {
        case 0: r = 1.0f; g = 1.0f; b = 1.0f; break;
        case 1: r = 0.4f; g = 0.8f; b = 1.0f; break;
        case 2: r = 1.0f; g = 0.4f; b = 0.4f; break;
        case 3: r = 1.0f; g = 0.8f; b = 0.2f; break;
        case 4: r = 0.4f; g = 1.0f; b = 0.4f; break;
        case 5: r = 1.0f; g = 0.4f; b = 1.0f; break;
        case 6: r = 0.8f; g = 0.8f; b = 0.8f; break;
        case 7: r = 0.6f; g = 0.6f; b = 0.6f; break;
        default: break;
    }
    return {r, g, b, 1.0f};
}

static float roadWidthPx(int w) {
    float lw = (float)w;
    if (lw < 1.0f) lw = 1.0f;
    return lw;
}

// ---- lifecycle ----

void MapCache::start() {
    // packed store is optional
    openPackedStore_();

    stop_ = false;

    if (workerCount_ <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 4;
        workerCount_ = (int)std::max(2u, hc > 1 ? (hc - 1) : 2u);
    }

    workers_.clear();
    workers_.reserve((size_t)workerCount_);

    for (int wi = 0; wi < workerCount_; ++wi) {
        workers_.emplace_back([this] {
            std::vector<uint8_t> blob;
            blob.reserve(1 << 20);

            while (true) {
                uint64_t key = 0;
                uint64_t gen = 0;

                {
                    std::unique_lock<std::mutex> lk(mtx_);
                    cv_.wait(lk, [&] { return stop_.load() || !reqQ_.empty(); });
                    if (stop_.load() && reqQ_.empty()) break;

                    auto item = reqQ_.front();
                    reqQ_.pop();
                    key = item.first;
                    gen = item.second;

                    if (gen != viewGen_.load(std::memory_order_relaxed)) {
                        continue;
                    }
                }

                const int tx = (int)(int32_t)(key >> 32);
                const int ty = (int)(int32_t)(key & 0xffffffffu);

                PendingTileCPU cpu{};
                cpu.tx = tx;
                cpu.ty = ty;
                cpu.gen = gen;

                const bool ok = readTileBlob_(tx, ty, blob);
                if (ok) parseTileBlob_(blob, cpu);

                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    inFlight_.erase(key);

                    if (cpu.gen != viewGen_.load(std::memory_order_relaxed)) continue;
                    if (prefetchSet_.find(key) == prefetchSet_.end()) continue;

                    readyQ_.push(std::move(cpu));
                }
            }
        });
    }
}

void MapCache::shutdown() {
    stop_ = true;
    cv_.notify_all();

    for (auto& th : workers_) {
        if (th.joinable()) th.join();
    }
    workers_.clear();

    closePackedStore_();

    // free gpu
    // caller owns RendererGL; we only delete GL objects when we have renderer in upload path,
    // so here we just drop CPU+handles. (RendererGL shutdown will clean program/atlas.)
    tiles_.clear();
    lastUsed_.clear();
    inFlight_.clear();
    while (!std::queue<std::pair<uint64_t, uint64_t>>().empty()) {}

    while (!reqQ_.empty()) reqQ_.pop();
    while (!readyQ_.empty()) readyQ_.pop();
}

// ---- visibility ----

bool MapCache::updateVisible(const AABB& viewWorld) {
    return computeVisible_(viewWorld);
}

bool MapCache::computeVisible_(const AABB& viewWorld) {
    const double left = viewWorld.minx;
    const double right = viewWorld.maxx;
    const double bottom = viewWorld.miny;
    const double top = viewWorld.maxy;

    const int corePad = 1;

    const int vtx0 = (int)std::floor(left / (double)kTileSize);
    const int vtx1 = (int)std::floor(right / (double)kTileSize);
    const int vty0 = (int)std::floor(bottom / (double)kTileSize);
    const int vty1 = (int)std::floor(top / (double)kTileSize);

    int tx0 = vtx0 - corePad;
    int tx1 = vtx1 + corePad;
    int ty0 = vty0 - corePad;
    int ty1 = vty1 + corePad;

    std::unordered_set<uint64_t> newVisibleSet;
    newVisibleSet.reserve((size_t)std::max(1, (tx1 - tx0 + 1) * (ty1 - ty0 + 1)));

    std::vector<TileKey> newVisible;
    newVisible.reserve((size_t)std::max(1, (tx1 - tx0 + 1) * (ty1 - ty0 + 1)));

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            newVisible.push_back(TileKey{tx, ty});
            newVisibleSet.insert(tileKey64(tx, ty));
        }
    }

    // prefetch: expand by 1x view span (clamped)
    const int nx = std::max(1, vtx1 - vtx0 + 1);
    const int ny = std::max(1, vty1 - vty0 + 1);

    int padX = std::max(nx, padTiles);
    int padY = std::max(ny, padTiles);

    int ptx0 = vtx0 - padX;
    int ptx1 = vtx1 + padX;
    int pty0 = vty0 - padY;
    int pty1 = vty1 + padY;

    const int64_t maxPrefetchTotal = 20000;
    int64_t totalPref = (int64_t)(ptx1 - ptx0 + 1) * (int64_t)(pty1 - pty0 + 1);
    if (totalPref > maxPrefetchTotal) {
        const double scale = std::sqrt((double)maxPrefetchTotal / (double)std::max<int64_t>(1, totalPref));
        const int newPadX = std::max(padTiles, (int)std::floor((double)padX * scale));
        const int newPadY = std::max(padTiles, (int)std::floor((double)padY * scale));
        padX = std::max(newPadX, corePad);
        padY = std::max(newPadY, corePad);
        ptx0 = vtx0 - padX; ptx1 = vtx1 + padX;
        pty0 = vty0 - padY; pty1 = vty1 + padY;
    }

    std::unordered_set<uint64_t> newPrefetchSet;
    const int64_t prefCountEst = (int64_t)(ptx1 - ptx0 + 1) * (int64_t)(pty1 - pty0 + 1);
    newPrefetchSet.reserve((size_t)std::max<int64_t>(1, std::min<int64_t>(prefCountEst, 200000)));

    std::vector<TileKey> newPrefetch;
    newPrefetch.reserve((size_t)std::max<int64_t>(1, std::min<int64_t>(prefCountEst, 200000)));

    for (int ty = pty0; ty <= pty1; ++ty) {
        for (int tx = ptx0; tx <= ptx1; ++tx) {
            newPrefetch.push_back(TileKey{tx, ty});
            newPrefetchSet.insert(tileKey64(tx, ty));
        }
    }

    const bool changedVisible =
        (newVisibleSet.size() != visibleSet_.size()) ||
        std::any_of(newVisibleSet.begin(), newVisibleSet.end(), [&](uint64_t k) { return visibleSet_.find(k) == visibleSet_.end(); });

    bool changedPrefetch = false;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        changedPrefetch =
            (newPrefetchSet.size() != prefetchSet_.size()) ||
            std::any_of(newPrefetchSet.begin(), newPrefetchSet.end(), [&](uint64_t k) { return prefetchSet_.find(k) == prefetchSet_.end(); });
    }

    viewCenterX_ = 0.5 * (left + right);
    viewCenterY_ = 0.5 * (bottom + top);

    visible_ = std::move(newVisible);
    visibleSet_ = std::move(newVisibleSet);
    prefetch_ = std::move(newPrefetch);

    {
        std::lock_guard<std::mutex> lk(mtx_);
        prefetchSet_ = std::move(newPrefetchSet);
    }

    if (changedVisible || changedPrefetch) {
        viewGen_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mtx_);
        while (!reqQ_.empty()) reqQ_.pop();
        while (!readyQ_.empty()) readyQ_.pop();
        inFlight_.clear();
    }

    return (changedVisible || changedPrefetch);
}

// ---- request enqueue ----

void MapCache::enqueueRequests() {
    struct ReqItem { uint64_t key; int tx; int ty; float dist2; bool core; uint64_t off; };
    std::vector<ReqItem> items;
    items.reserve(prefetch_.size());

    const double cx = viewCenterX_;
    const double cy = viewCenterY_;

    for (const auto& tk : prefetch_) {
        const uint64_t key = tileKey64(tk.tx, tk.ty);

        const double txc = ((double)tk.tx + 0.5) * (double)kTileSize;
        const double tyc = ((double)tk.ty + 0.5) * (double)kTileSize;
        const float dx = (float)(txc - cx);
        const float dy = (float)(tyc - cy);

        const float d2 = dx * dx + dy * dy;
        const bool isCore = (visibleSet_.find(key) != visibleSet_.end());

        uint64_t off = UINT64_MAX;
        auto iit = index_.find(key);
        if (iit != index_.end()) off = iit->second.off;

        items.push_back(ReqItem{key, tk.tx, tk.ty, d2, isCore, off});
    }

    std::sort(items.begin(), items.end(), [](const ReqItem& a, const ReqItem& b) {
        if (a.core != b.core) return a.core > b.core;
        if (a.dist2 != b.dist2) return a.dist2 < b.dist2;
        return a.off < b.off;
    });

    const size_t kWindow = 256;
    for (size_t i = 0; i < items.size(); i += kWindow) {
        const size_t j = std::min(items.size(), i + kWindow);
        std::stable_sort(items.begin() + i, items.begin() + j, [](const ReqItem& a, const ReqItem& b) {
            return a.off < b.off;
        });
    }

    const uint64_t genSnap = viewGen_.load(std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(mtx_);
        int enq = 0;

        for (const auto& it : items) {
            auto tIt = tiles_.find(it.key);
            const bool alreadyLoaded = (tIt != tiles_.end() && tIt->second.loaded);
            if (alreadyLoaded) continue;
            if (inFlight_.find(it.key) != inFlight_.end()) continue;

            inFlight_.insert(it.key);
            reqQ_.push(std::make_pair(it.key, genSnap));
            if (++enq >= maxRequestEnqueue) break;
        }
    }

    cv_.notify_all();
}

// ---- tile lookup ----

const MapCache::Tile* MapCache::findTile(int tx, int ty) const {
    const uint64_t key = tileKey64(tx, ty);
    auto it = tiles_.find(key);
    if (it == tiles_.end()) return nullptr;
    return &it->second;
}

MapCache::Tile* MapCache::findTile(int tx, int ty) {
    const uint64_t key = tileKey64(tx, ty);
    auto it = tiles_.find(key);
    if (it == tiles_.end()) return nullptr;
    return &it->second;
}

// ---- gpu helpers ----

void MapCache::destroyTileGpu_(RendererGL& r, Tile& t) {
    for (auto& l : t.gpuLines) r.destroy(l);
    t.gpuLines.clear();
    for (auto& p : t.gpuPolys) r.destroy(p);
    t.gpuPolys.clear();
    if (t.gpuTextValid) r.destroy(t.gpuText);
    t.gpuTextValid = false;
    t.gpuText = {};
    t.gpuTextRev = 0;
}

void MapCache::rebuildTileText_(RendererGL& r, Tile& t, double viewArea) {
    if (t.gpuTextValid) r.destroy(t.gpuText);
    t.gpuTextValid = false;
    t.gpuTextRev = textRev_;

    if (!r.textReady() || t.texts.empty()) return;

    const double minRatio = 0.01;
    const double maxRatio = 0.15;

    // Visibility filter stays here (same rule as before).
    std::vector<TextLabelCPU> filtered;
    filtered.reserve(t.texts.size());

    for (const auto& lbl : t.texts) {
        const float wpx = r.estimateTextWidthPx(lbl.text);
        if (wpx <= 0.0f) continue;

        const float scale = (lbl.size > 1e-3f) ? (lbl.size / r.fontPixelSize()) : 1.0f;
        const double worldW = (double)wpx * (double)scale;
        const double worldH = (double)lbl.size;
        const double ratio = (viewArea > 1e-12) ? ((worldW * worldH) / viewArea) : 0.0;

        if (ratio < minRatio || ratio > maxRatio) continue;
        filtered.push_back(lbl);
    }

    if (filtered.empty()) return;

    std::vector<TextVertex> verts;
    r.buildTextVerts(filtered, verts);
    if (verts.empty()) return;

    r.uploadTextQuads(verts, t.gpuText);
    t.gpuTextValid = (t.gpuText.vertCount > 0);
}

void MapCache::uploadTile_(RendererGL& r, PendingTileCPU&& cpu, double viewArea) {
    const uint64_t key = tileKey64(cpu.tx, cpu.ty);
    auto& t = tiles_[key];

    if (t.loaded) {
        destroyTileGpu_(r, t);
    }

    t.tx = cpu.tx;
    t.ty = cpu.ty;
    t.loaded = true;

    const float x0 = (float)(t.tx * kTileSize);
    const float y0 = (float)(t.ty * kTileSize);
    t.bb = AABB{x0, y0, x0 + (float)kTileSize, y0 + (float)kTileSize};

    t.texts = std::move(cpu.texts);
    if (!t.texts.empty()) anyTextLoaded_ = true;

    // roads
    t.gpuLines.clear();
    t.gpuLines.reserve(cpu.lines.size());
    for (const auto& b : cpu.lines) {
        if (b.verts.empty()) continue;

        LineBatchGpu g{};
        g.w = b.w;
        g.c = b.c;
        g.width = roadWidthPx(b.w);

        auto col = roadColor(b.w, b.c);
        g.color[0] = col[0]; g.color[1] = col[1]; g.color[2] = col[2]; g.color[3] = col[3];

        r.uploadLines(b, g);
        t.gpuLines.push_back(g);
    }

    // polys
    t.gpuPolys.clear();
    if (!cpu.polys.empty()) {
        anyPolyLoaded_ = true;
        t.gpuPolys.reserve(cpu.polys.size());
        for (const auto& pm : cpu.polys) {
            if (pm.idx.empty() || pm.verts.empty()) continue;

            PolyBatchGpu pg{};
            r.uploadPoly(pm, pg);
            t.gpuPolys.push_back(pg);
        }
    }

    // text (build on demand by rev)
    t.gpuTextValid = false;
    t.gpuTextRev = 0;

    // Optional eager build if you want: keep identical behavior (lazy via rev check in main).
    (void)viewArea;
}

// ---- upload pump ----

void MapCache::uploadReady(RendererGL& r, double viewArea) {
    const bool sprint = (int)prefetch_.size() >= sprintThresholdTiles;
    const double budgetMs = sprint ? sprintUploadBudgetMs : normalUploadBudgetMs;

    auto t0 = Clock::now();
    int uploaded = 0;

    while (true) {
        PendingTileCPU cpu{};
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (readyQ_.empty()) break;
            cpu = std::move(readyQ_.front());
            readyQ_.pop();
        }

        const uint64_t key = tileKey64(cpu.tx, cpu.ty);
        uploadTile_(r, std::move(cpu), viewArea);
        markUsed_(key);
        ++uploaded;

        if (uploaded >= maxUploadPerFrame && !sprint) break;

        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms >= budgetMs) break;
    }
}

// ---- eviction ----

void MapCache::evict() {
    int freedTiles = 0;

    std::vector<std::pair<uint64_t, uint64_t>> candidates;
    candidates.reserve(tiles_.size());

    for (const auto& kv : tiles_) {
        const uint64_t key = kv.first;
        if (visibleSet_.find(key) != visibleSet_.end()) continue;

        uint64_t lu = 0;
        auto it = lastUsed_.find(key);
        if (it != lastUsed_.end()) lu = it->second;
        candidates.push_back({key, lu});
    }

    // TTL first
    for (const auto& c : candidates) {
        if (frameId_ > c.second && (frameId_ - c.second) > ttlFrames) {
            auto it = tiles_.find(c.first);
            if (it != tiles_.end()) {
                // GL objects are owned; in this minimal split, we only destroy with renderer during upload path.
                // To keep behavior predictable, we just drop CPU + handles here.
                it->second.texts.clear();
                it->second.gpuLines.clear();
                it->second.gpuPolys.clear();
                it->second.gpuTextValid = false;
                tiles_.erase(it);
                ++freedTiles;
            }
            lastUsed_.erase(c.first);
        }
    }

    if ((int)tiles_.size() <= maxResidentTiles) {
#ifdef __GLIBC__
        if (freedTiles > 0) malloc_trim(0);
#endif
        return;
    }

    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.second < b.second; });

    int toEvict = (int)tiles_.size() - maxResidentTiles;
    for (int i = 0; i < (int)candidates.size() && toEvict > 0; ++i) {
        const uint64_t key = candidates[i].first;
        auto it = tiles_.find(key);
        if (it != tiles_.end()) {
            it->second.texts.clear();
            it->second.gpuLines.clear();
            it->second.gpuPolys.clear();
            it->second.gpuTextValid = false;
            tiles_.erase(it);
            ++freedTiles;
        }
        lastUsed_.erase(key);
        --toEvict;
    }

#ifdef __GLIBC__
    if (freedTiles > 0) malloc_trim(0);
#endif
}

void MapCache::ensureTileText(RendererGL& r, int tx, int ty, double viewArea) {
    const uint64_t key = tileKey64(tx, ty);
    auto it = tiles_.find(key);
    if (it == tiles_.end()) return;

    Tile& t = it->second;
    if (!t.loaded || t.texts.empty()) return;
    if (!r.textReady()) return;

    if (t.gpuTextRev != textRev_) {
        rebuildTileText_(r, t, viewArea);
    }
}