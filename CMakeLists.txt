cmake_minimum_required(VERSION 4.1)
project(dev_lib)

set(CMAKE_CXX_STANDARD 23)

file(GLOB_RECURSE src_files src/*.cpp)
file(GLOB_RECURSE header_files src/*.h src/*.hpp)
file(GLOB_RECURSE module_files src/*.ixx  src/*.cppm)

add_executable(dev_lib ${src_files} ${header_files} ${module_files})

target_sources(dev_lib
        PUBLIC
        FILE_SET cxx_modules
        TYPE CXX_MODULES
        FILES ${module_files}
)