#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <iomanip>

#define M_PI       3.14159265358979323846

struct NodeXY {
    double x;
    double y;
};

struct Edge {
    int u;
    int v;
    int w;
    int c;
};

constexpr double EARTH_RADIUS = 6378137.0;
constexpr double MAX_LAT = 85.05112878;

static inline void mercatorProject(double latDeg, double lonDeg, double& outX, double& outY)
{
    if (latDeg > MAX_LAT)  latDeg = MAX_LAT;
    if (latDeg < -MAX_LAT) latDeg = -MAX_LAT;

    double latRad = latDeg * M_PI / 180.0;
    double lonRad = lonDeg * M_PI / 180.0;

    outX = EARTH_RADIUS * lonRad;
    outY = EARTH_RADIUS * std::log(std::tan((M_PI / 4.0) + (latRad / 2.0)));
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: gl_converter <input.gl> <output.gm>\n";
        return 1;
    }

    const std::string inputPath  = argv[1];
    const std::string outputPath = argv[2];

    std::ifstream in(inputPath);
    if (!in)
    {
        std::cerr << "Failed to open input file " << inputPath << "\n";
        return 1;
    }

    std::size_t numNodes = 0;
    std::size_t numEdges = 0;

    if (!(in >> numNodes))
    {
        std::cerr << "Failed to read node count.\n";
        return 1;
    }
    if (!(in >> numEdges))
    {
        std::cerr << "Failed to read edge count.\n";
        return 1;
    }

    std::vector<NodeXY> nodes;
    nodes.reserve(numNodes);

    double minX =  std::numeric_limits<double>::infinity();
    double minY =  std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < numNodes; ++i)
    {
        double lat = 0.0;
        double lon = 0.0;

        if (!(in >> lat >> lon))
        {
            std::cerr << "Error reading node " << i << " (lat lon).\n";
            return 1;
        }

        double x, y;
        mercatorProject(lat, lon, x, y);

        nodes.push_back({x, y});

        // 更新 BOUNDS
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    std::vector<Edge> edges;
    edges.reserve(numEdges);

    for (std::size_t i = 0; i < numEdges; ++i)
    {
        Edge e{};
        if (!(in >> e.u >> e.v >> e.w >> e.c))
        {
            std::cerr << "Warning: stopping at edge " << i
                      << ", malformed line or EOF.\n";
            break;
        }

        if (e.u < 0 || e.v < 0 ||
            static_cast<std::size_t>(e.u) >= numNodes ||
            static_cast<std::size_t>(e.v) >= numNodes)
        {
            std::cerr << "Warning: edge (" << e.u << "," << e.v
                      << ") references invalid node index. Skipped.\n";
        }
        else
        {
            edges.push_back(e);
        }
    }

    in.close();

    std::cout << "Loaded from GL: " << numNodes << " nodes, "
              << edges.size() << " edges (valid).\n";


    std::ofstream out(outputPath);
    if (!out)
    {
        std::cerr << "Failed to open output file " << outputPath << "\n";
        return 1;
    }

    out << std::fixed << std::setprecision(15);

    std::size_t polygonCount = 0;
    std::size_t textCount    = 0;

    out << nodes.size()  << "//Node\n";         // node_count
    out << edges.size()  << "//Edge\n";         // edge_count
    out << polygonCount  << "//Polygon\n";      // polygon_count = 0
    out << textCount     << "//Text\n";         // text_count   = 0

    const double TILE_SIZE = 4096.0;
    const int PAD_TILES = 2;

    const double pad = TILE_SIZE * PAD_TILES;
    minX -= pad;
    minY -= pad;
    maxX += pad;
    maxY += pad;

    out << "BOUNDS "
        << minX << " " << minY << " "
        << maxX << " " << maxY << "\n";

    for (const auto& n : nodes)
    {
        out << n.x << " " << n.y << "\n";
    }

    for (const auto& e : edges)
    {
        out << e.u << " " << e.v << " " << e.w << " " << e.c << "\n";
    }

    out.close();

    std::cout << "Wrote GM file: " << outputPath << "\n";
    std::cout << "Projected bounds (meters): (" << minX << ", " << minY
              << ") - (" << maxX << ", " << maxY << ")\n";

    return 0;
}
