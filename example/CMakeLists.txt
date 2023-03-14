cmake_minimum_required(VERSION 3.15)
project(ground_extractor)

find_package(PCL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

if (NOT TARGET opencv_world)
  find_package(OpenCV CONFIG REQUIRED)
endif()

add_executable(ground_extractor
  extractor_test.cpp
)
target_link_libraries(ground_extractor
  dash::labelled_ground_extraction
  ${PCL_LIBRARIES}
  nlohmann_json::nlohmann_json  
)