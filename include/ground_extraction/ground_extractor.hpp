#pragma once

#include <array>
#include <string>
#include <sys/types.h>
#include <vector>
#include <cstdint>

#include <pcl/common/common_headers.h>
#include <opencv2/core.hpp>

namespace GroundExtraction 
{

using PointT = pcl::PointXYZL;

// Row Major. Usual robot coordinate frame (x positive towards the top [row 0], y positive to the right)
class Grid2D
{
public:
    enum class Labels : std::uint8_t
    {
        Unknown,
        Obstacle,
        Unoccupied
    };
    struct GridParameters
    {
        std::size_t cols;
        std::size_t rows;
        float reso;
        std::array<float,2> origin; // (x,y), this origin is placed at xmin, ymax
        std::array<float,2> height_bounds; // (zmin, zmax) for gradient calibration of height map
    };

    GridParameters m_parameters;
    std::vector<Labels> m_grid;
    std::vector<float> height_grid;
};


class GroundExtractor
{

public:

struct ExtractionSettings
{
    std::string terrain_type; // Should be an image output file
    float m_resolution{0.1f};
    float zaxis_ground{0.2f};
    float zaxis_ceil{1.2f};
    float MSEmax{0.03f};
    float plane_ground{0.0f};
    float plane_offset{1.0f};
    float plane_resolution{0.8f}; 
    int confidence_label{1};
    int confidence_zaxis{1};
    int confidence_grid_plane{1};
    int confidence_full_plane{1};
    float confidence_threshold{18.0f}; 
    float m_inflation_dis{0.5f};
};

enum class ExtractionSettingPreset
{
    DEFAULT,
    FLAT_TERRAIN,
    SLOPEY_TERRAIN,
    UNLABELLED_CLOUD
};

Grid2D Extract(pcl::PointCloud<PointT>::Ptr& labelled_cloud, ExtractionSettings& input_param);

};




}