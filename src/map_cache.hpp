// src/map_cache.hpp
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "app.hpp"
#include "renderer_gl.hpp"

class MapCache {
public:
    MapCache() = default;
    ~MapCache() = default;

    // Cache lifecycle
    bool openOrBuild(const std::string& gmPath, CacheInfo& outInfo);
    const CacheInfo& info() const { return ci_; }

    // Streaming lifecycle
    void start();
    void shutdown();

    // View update (call when camera changes). Returns true if visible/prefetch set changed.
    bool updateVisible(const AABB& viewWorld);

    // Enqueue I/O for current prefetch set (call after updateVisible()).
    void enqueueRequests();

    // Upload ready tiles to GPU within a time budget.
    void uploadReady(RendererGL& r, double viewArea);

    // Evict old / unused tiles.
    void evict();

    // Tick counters
    void nextFrame() { ++frameId_; }

    // Draw helpers (iterate visible tiles)
    const std::vector<TileKey>& visibleTiles() const { return visible_; }

    // Access tile for drawing (may be missing/unloaded).
    struct Tile {
        int tx = 0;
        int ty = 0;
        bool loaded = false;

        // CPU-side labels kept (text rebuild depends on scale).
        std::vector<TextLabelCPU> texts;

        // GPU resources (per tile)
        std::vector<LineBatchGpu> gpuLines;
        std::vector<PolyBatchGpu> gpuPolys;

        TextGpu gpuText{};
        bool gpuTextValid = false;
        uint64_t gpuTextRev = 0;

        AABB bb{};
    };

    const Tile* findTile(int tx, int ty) const;
    Tile* findTile(int tx, int ty);

    // Text revision: bump when scale/visible set changes enough.
    uint64_t textRev() const { return textRev_; }
    void bumpTextRev() { ++textRev_; }

    // Flags (same semantics as your original)
    bool anyTextLoaded() const { return anyTextLoaded_; }
    bool anyPolyLoaded() const { return anyPolyLoaded_; }

    // Tuning knobs (keep defaults consistent with your original)
    int maxResidentTiles = 512;
    int maxUploadPerFrame = 64;
    int padTiles = 1;
    uint64_t ttlFrames = 180;

    int sprintThresholdTiles = 80;
    double normalUploadBudgetMs = 10.0;
    double sprintUploadBudgetMs = 28.0;
    int maxRequestEnqueue = 20000;

    void ensureTileText(RendererGL& r, int tx, int ty, double viewArea);

private:
    // ---- cache (build/open) ----
    bool tryOpenCache_(const std::string& gmPath, CacheInfo& out);
    bool buildCache_(const std::string& gmPath, CacheInfo& out);

private:
    // ---- packed store ----
    struct TileIndexEntry { uint64_t off = 0; uint32_t sz = 0; };
    bool openPackedStore_();
    void closePackedStore_();
    bool readTileBlob_(int tx, int ty, std::vector<uint8_t>& outBlob);

    // ---- parsing ----
    struct PendingTileCPU {
        uint64_t gen = 0;
        int tx = 0;
        int ty = 0;
        std::vector<LineBatchCPU> lines;
        std::vector<PolyMeshCPU> polys;
        std::vector<TextLabelCPU> texts;
        bool empty = true;
    };
    static bool parseTileBlob_(const std::vector<uint8_t>& blob, PendingTileCPU& out);

    // ---- per-tile GPU ----
    void destroyTileGpu_(RendererGL& r, Tile& t);
    void uploadTile_(RendererGL& r, PendingTileCPU&& cpu, double viewArea);
    void rebuildTileText_(RendererGL& r, Tile& t, double viewArea);

private:
    // ---- visibility / requests ----
    bool computeVisible_(const AABB& viewWorld);

    void markUsed_(uint64_t key) { lastUsed_[key] = frameId_; }

private:
    // ---- state ----
    CacheInfo ci_{};

    // Packed store
    std::unordered_map<uint64_t, TileIndexEntry> index_;
    int datFd_ = -1;
    size_t datBytes_ = 0;
    const uint8_t* datPtr_ = nullptr;

    // Tiles
    std::unordered_map<uint64_t, Tile> tiles_;
    std::unordered_map<uint64_t, uint64_t> lastUsed_;

    // Visible / prefetch
    std::vector<TileKey> visible_;
    std::unordered_set<uint64_t> visibleSet_;

    std::vector<TileKey> prefetch_;
    std::unordered_set<uint64_t> prefetchSet_;

    double viewCenterX_ = 0.0;
    double viewCenterY_ = 0.0;

    // Async loader
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<uint64_t> viewGen_{1};

    std::queue<std::pair<uint64_t, uint64_t>> reqQ_; // (tileKey, gen)
    std::unordered_set<uint64_t> inFlight_;
    std::queue<PendingTileCPU> readyQ_;

    std::atomic<bool> stop_{false};
    int workerCount_ = 0;
    std::vector<std::thread> workers_;

    // Revision / frame
    uint64_t frameId_ = 0;
    uint64_t textRev_ = 1;

    bool anyTextLoaded_ = false;
    bool anyPolyLoaded_ = false;
};