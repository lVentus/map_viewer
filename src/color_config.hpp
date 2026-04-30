#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

class ColorConfig {
public:
    std::array<float, 4> background = {0.06f, 0.065f, 0.075f, 1.0f};

    ColorConfig() {
        colors_[0] = {0.08f, 0.08f, 0.08f, 1.0f};
        colors_[1] = {0.20f, 0.20f, 0.20f, 1.0f};
        colors_[2] = {0.35f, 0.35f, 0.35f, 1.0f};
        colors_[3] = {0.70f, 0.70f, 0.70f, 1.0f};
        colors_[4] = {0.18f, 0.28f, 0.75f, 0.35f};
        colors_[5] = {0.28f, 0.62f, 0.28f, 0.35f};
        colors_[6] = {0.92f, 0.92f, 0.92f, 1.0f};
        colors_[7] = {0.90f, 0.64f, 0.18f, 1.0f};
    }

    void set(int index, const std::array<float, 4>& rgba) {
        colors_[index] = rgba;
    }

    std::array<float, 4> get(int index) const {
        auto it = colors_.find(index);
        if (it != colors_.end()) return it->second;
        auto fallback = colors_.find(0);
        if (fallback != colors_.end()) return fallback->second;
        return {1.0f, 1.0f, 1.0f, 1.0f};
    }

private:
    std::unordered_map<int, std::array<float, 4>> colors_;
};

inline std::string stripConfigComment(std::string line) {
    const std::size_t hashPos = line.find('#');
    const std::size_t slashPos = line.find("//");
    std::size_t cut = std::string::npos;
    if (hashPos != std::string::npos) cut = hashPos;
    if (slashPos != std::string::npos) cut = (cut == std::string::npos) ? slashPos : std::min(cut, slashPos);
    if (cut != std::string::npos) line.resize(cut);

    auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    while (!line.empty() && !notSpace(static_cast<unsigned char>(line.back()))) line.pop_back();
    std::size_t first = 0;
    while (first < line.size() && !notSpace(static_cast<unsigned char>(line[first]))) ++first;
    return line.substr(first);
}

inline bool loadColorConfig(const std::string& path, ColorConfig& config) {
    std::ifstream in(path);
    if (!in) return false;

    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        line = stripConfigComment(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string first;
        iss >> first;

        float r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;
        if (first == "background") {
            if (!(iss >> r >> g >> b >> a)) {
                std::cerr << "Warning: invalid background in color config line " << lineNo << "\n";
                continue;
            }
            config.background = {r, g, b, a};
            continue;
        }

        int index = 0;
        if (first == "color") {
            if (!(iss >> index >> r >> g >> b >> a)) {
                std::cerr << "Warning: invalid color in color config line " << lineNo << "\n";
                continue;
            }
        } else {
            std::istringstream firstAsIndex(first);
            if (!(firstAsIndex >> index) || !(iss >> r >> g >> b >> a)) {
                std::cerr << "Warning: invalid color config line " << lineNo << "\n";
                continue;
            }
        }

        config.set(index, {r, g, b, a});
    }

    return true;
}
