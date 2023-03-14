#include "ground_extraction/ground_extractor.hpp"
#include "internal_utils.hpp"
#include <limits>
#include <vector>
#include <chrono>

namespace GroundExtraction
{

typedef pcl::PointXYZL PointT;

Grid2D GroundExtractor::Extract(pcl::PointCloud<PointT>::Ptr& labelled_cloud, GroundExtractor::ExtractionSettings& input_param)
{      
    // OutlierRemoval(labelled_cloud);
    Grid2D gridOut;
    GroundExtractor::ExtractionSettingPreset preset;

    if (input_param.terrain_type == "slopey")
    {
        preset = GroundExtractor::ExtractionSettingPreset::SLOPEY_TERRAIN;
    }
    else if (input_param.terrain_type == "flat")
    {
        preset = GroundExtractor::ExtractionSettingPreset::FLAT_TERRAIN;
    }
    else if (input_param.terrain_type == "default")
    {
        preset = GroundExtractor::ExtractionSettingPreset::DEFAULT;
    }

    defineGridBounds(labelled_cloud, gridOut.m_parameters, input_param, preset);
    input_param = GenerateSettingsFromPreset(preset);

    std::size_t grid_size = gridOut.m_parameters.rows*gridOut.m_parameters.cols;
    std::vector<std::uint8_t> num_obstacle_labels(grid_size, 0);
    std::vector<std::uint8_t> num_obstacle_zaxis(grid_size, 0);
    std::vector<std::uint8_t> num_obstacle_grid_plane(grid_size, 0);
    std::vector<std::uint8_t> num_obstacle_full_plane(grid_size, 0);
    std::vector<std::uint8_t> num_points(grid_size, 0);

    gridOut.height_grid.resize(grid_size,0);
    
    // std::cout << input_param.confidence_label << ", "
    //           << input_param.confidence_grid_plane << ", "
    //           << input_param.confidence_full_plane << ", "
    //           << input_param.confidence_zaxis << std::endl;

    ExtractionMethods(labelled_cloud, num_obstacle_labels, num_obstacle_zaxis, num_obstacle_grid_plane, num_obstacle_full_plane, num_points, gridOut.m_parameters, input_param, gridOut.height_grid);

    ConfidenceExtraction(labelled_cloud, input_param, gridOut.m_grid, num_points, num_obstacle_labels, num_obstacle_grid_plane, num_obstacle_full_plane, num_obstacle_zaxis);

    HeightVisualisation(gridOut.height_grid, gridOut.m_parameters, gridOut.m_grid);

    // LabelnZaxisMethod(labelled_cloud, num_obstacle_labels, num_obstacle_zaxis, num_points, gridOut.m_parameters, input_param);
    // GridPlaneMethod(labelled_cloud, num_obstacle_plane, num_points, gridOut.m_parameters, input_param);
    // FullPlaneMethod(labelled_cloud, num_obstacle_plane, gridOut.m_parameters, input_param);
    // GridDilation(gridOut.m_grid, gridOut.m_parameters);
    // ExportPNG(gridOut.m_grid, gridOut.m_parameters, input_param.output_filename);

    return gridOut;
}



}