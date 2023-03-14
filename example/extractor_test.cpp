#include "ground_extraction/ground_extractor.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>
#include <limits>
#include <string>
#include <cmath>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/point_types.h>
#include <nlohmann/json.hpp>

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


unsigned int text_id = 0;
using PointT = pcl::PointXYZL;

float x_min = std::numeric_limits<float>::max();
float x_max = std::numeric_limits<float>::lowest();
float y_min = std::numeric_limits<float>::max();
float y_max = std::numeric_limits<float>::lowest();


namespace GroundExtraction
{

void getMaxRange (const pcl::PointCloud<PointT>::Ptr labelled_cloud, const GroundExtractor::ExtractionSettings& input_param)
{
    for (const auto& point: *labelled_cloud)   
    {
        x_min = std::min(x_min, point.x);
        x_max = std::max(x_max, point.x);
        y_min = std::min(y_min, point.y);
        y_max = std::max(y_max, point.y);                
    }
}


void GridConversion(const std::vector<Grid2D::Labels>& m_grid, 
                    pcl::PointCloud<PointT>::Ptr &ground_map, 
                    pcl::PointCloud<PointT>::Ptr &obstacle_map,  
                    const GroundExtractor::ExtractionSettings& input_param)
{               
    float reso = input_param.m_resolution;

    std::size_t cols = static_cast<std::size_t>(std::ceil((x_max-x_min)/(reso)));
    for (std::size_t i = 0; i < m_grid.size(); i++)
    {
        PointT ground, obstacle;
        if (m_grid[i] == Grid2D::Labels::Unoccupied)
        {
            std::size_t y = static_cast<std::size_t>(floor(i/cols));
            ground.y = y_max - (y+0.5)*reso;
            ground.x = ((i-y*cols)+0.5)*reso + x_min;
            ground.z = input_param.zaxis_ceil;
            ground.label = 1;
            ground_map->points.push_back(ground);
        }
        else if (m_grid[i] == Grid2D::Labels::Obstacle)
        {
            std::size_t y = static_cast<std::size_t>(floor(i/cols));
            obstacle.y = y_max - (y+0.5)*reso;
            obstacle.x = ((i-y*cols)+0.5)*reso + x_min;
            obstacle.z = input_param.zaxis_ceil;
            obstacle.label = 0;
            obstacle_map->points.push_back(obstacle);
        }
    }
}

nlohmann::json CreateMapconfig (const GroundExtractor::ExtractionSettings& input_param, const cv::Size& size)
{
    nlohmann::json output;

    int origin_x = static_cast<int>(std::floor(y_max/input_param.m_resolution)); // x
    int origin_y = static_cast<int>(std::floor(x_max/input_param.m_resolution)); // y

    output["name"] = "test";
    output["img_height_px"] = size.height;
    output["img_width_px"] = size.width;
    output["origin_x_px"] = origin_x;
    output["origin_y_px"] = origin_y;
    output["resolution"] = input_param.m_resolution;

    return output;
}

bool checkSurroundingFree (const std::vector<Grid2D::Labels>& m_grid, int pixel_x, int pixel_y, int col_max, int row_max, int search)
{
    int min_x_check = std::max(pixel_x - search, 0);
    int max_x_check = std::min(pixel_x + search, col_max);
    int min_y_check = std::max(pixel_y - search, 0);
    int max_y_check = std::min(pixel_y + search, row_max);

    for (int i = min_x_check; i <= max_x_check; i ++)
        for (int j = min_y_check; j <= max_y_check; j ++)
        {
            const int grid_id = i + j * col_max;
            if (m_grid[grid_id] == Grid2D::Labels::Unoccupied){
                return true;
            }
        }   

    return false;
}

cv::Mat CreateImage (const pcl::PointCloud<PointT>::Ptr labelled_cloud, const std::vector<Grid2D::Labels>& m_grid, const GroundExtractor::ExtractionSettings& input_param)
{
    float reso = input_param.m_resolution;
    int cols = static_cast<int>(std::ceil((x_max-x_min)/reso)); 
    int rows = static_cast<int>(std::ceil((y_max-y_min)/reso));
    std::cout << "cols : " << cols << " rows : " << rows << std::endl;
    cv::Mat output(rows + 1, cols + 1 , CV_8U);
    std::cout << " Image size " << output.size[0] << " * " << output.size[1] << " pixels" << std::endl;
    int inflation_ratio = static_cast<int>(std::ceil(input_param.m_inflation_dis/input_param.m_resolution));
    std::cout << " Inflation ratio " << inflation_ratio << std::endl;
    for (std::size_t i = 0; i < m_grid.size(); i++)
    {
        int y = static_cast<int>(floor(i/cols));
        int x = i - y*cols;
        assert(y < output.size[1] && x < output.size[0]);
        if (m_grid[i] == Grid2D::Labels::Unoccupied){
            cv::circle(output, cv::Point(x, y), 1, uchar(255), -1);
        } else if (m_grid[i] == Grid2D::Labels::Obstacle){
            cv::circle(output, cv::Point(x, y), 1, uchar(0), inflation_ratio);
        }  else if (m_grid[i] == Grid2D::Labels::Unknown){
            if (checkSurroundingFree(m_grid, x, y, cols, rows, 4)) {
                cv::circle(output, cv::Point(x, y), 1, uchar(255), -1);
            } else {
                cv::circle(output, cv::Point(x, y), 1, uchar(100), -1);
            }
        }
    }

    cv::rotate(output, output, cv::ROTATE_90_COUNTERCLOCKWISE);
    return output;
}

pcl::visualization::PCLVisualizer::Ptr GridVis (pcl::PointCloud<PointT>::ConstPtr labelled_cloud, pcl::PointCloud<PointT>::ConstPtr ground_map, pcl::PointCloud<PointT>::ConstPtr obstacle_map)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    // viewer->addPointCloud<PointT> (labelled_cloud, filename+"cloud");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> blue_color(obstacle_map, 0, 0, 255);
    viewer->addPointCloud<PointT> (obstacle_map, blue_color, "obstacle");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> red_color(ground_map, 255, 0, 0);
    viewer->addPointCloud<PointT> (ground_map, red_color, "ground");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "ground");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "obstacle");
    viewer->addCoordinateSystem (1.0,"cloud");
    viewer->initCameraParameters ();
    return (viewer);
}


} // namespace GroundExtraction

using namespace GroundExtraction;

int main(int argc, const char* argv[])
{    
	if(argc != 2){
        std::cout << "Arguments incomplets. ./ground_extractor <Pointcloud-Path>" << std::endl;
        return 0;
    }
    GroundExtractor::ExtractionSettings input_param;
    pcl::PointCloud<PointT>::Ptr labelled_cloud (new pcl::PointCloud<PointT>);

    std::string addr = argv[1];
    if (pcl::io::loadPCDFile<PointT> (addr, *labelled_cloud) == -1)
    {
        return 0;
    }
    
    std::ifstream settingsF("/home/joel/gridmap_conversion_lib/example/settings.json");
    if (!settingsF) 
    {
        std::cout << "Cannot load the setting file" << std::endl;
        return 0;
    }

    nlohmann::json settingsJson = nlohmann::json::parse(settingsF);
    input_param.m_inflation_dis = settingsJson["inflation_dis"].get<float>();
    input_param.terrain_type = settingsJson["terrain_type"].get<std::string>();

    // input_param.confidence_label = settingsJson["confidence_label"].get<float>();
    // input_param.confidence_zaxis = settingsJson["confidence_zaxis"].get<float>();   
    // input_param.confidence_grid_plane = settingsJson["confidence_plane"].get<float>()/2;
    // input_param.confidence_full_plane = settingsJson["confidence_plane"].get<float>()/2;
    // input_param.confidence_threshold = settingsJson["probability_threshold"].get<float>();
    std::cout << "Read settings json done" << std::endl;

    pcl::PointCloud<PointT>::Ptr ground_map (new pcl::PointCloud<PointT>);
            pcl::PointCloud<PointT>::Ptr obstacle_map (new pcl::PointCloud<PointT>);
    getMaxRange(labelled_cloud, input_param);
    GroundExtractor extractor;
    const auto start = std::chrono::high_resolution_clock::now();
    const auto grid_2d = extractor.Extract(labelled_cloud, input_param);
    GridConversion(grid_2d.m_grid, ground_map, obstacle_map, input_param);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto time_used = end - start;

    std::cout<<"Time used for grid conversion (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(time_used).count() << std::endl;
    
    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = GridVis(labelled_cloud, ground_map, obstacle_map);

    cv::Mat MapImage = CreateImage(labelled_cloud, grid_2d.m_grid, input_param);
    cv::imwrite("../output_images/2D_Grid_Map_.png", MapImage);

    nlohmann::json map2DConfig = CreateMapconfig(input_param, MapImage.size());

    std::ofstream map2DConfigFile("map_config.json");
	map2DConfigFile<<std::setw(4)<<map2DConfig<<std::endl;
	map2DConfigFile.close();

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}