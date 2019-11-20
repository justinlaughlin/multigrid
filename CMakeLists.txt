cmake_minimum_required(VERSION 3.0)
# Disable in source builds
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# Add path to custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

project(multigrid)

# Require c++14 and standard libraries
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add -fPIC to all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Define where to put the libraries and binaries
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
add_subdirectory(libraries)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

list(APPEND SOURCES
        "src/linearalgebra.cpp"
        "src/multigrid.cpp"
        "src/initializeProblem.cpp"
        )

# MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
add_executable(main src/main.cpp src/linearalgebra.cpp src/multigrid.cpp src/initializeProblem.cpp)

target_include_directories(main PRIVATE ${MPI_CXX_INCLUDE_PATH} include/linearalgebra.h include/multigrid.h include/initializeProblem.h)
add_library(multigridshared SHARED ${SOURCES})
target_link_libraries(main ${MPI_CXX_LIBRARIES})


install(DIRECTORY include DESTINATION include FILES_MATCHING PATTERN "*.h")
install(TARGETS multigridshared DESTINATION lib)

##include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
#
## Add -fPIC to all targets
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#
## Define where to put the libraries and binaries
#set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#
#add_subdirectory(libraries)
#
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
##target_include_directories(multigridshared PRIVATE ${MPI_CXX_INCLUDE_PATH})
#
#list(APPEND SOURCES
#        "src/linearalgebra.cpp"
#        "src/multigrid.cpp"
#        "src/initializeProblem.cpp"
#        )
#
#
#add_executable(main src/main.cpp src/linearalgebra.cpp src/multigrid.cpp src/initializeProblem.cpp)
#
#add_library(multigridshared SHARED ${SOURCES})
##target_link_libraries(multigridshared ${MPI_CXX_LIBRARIES})
#
#
#install(DIRECTORY include DESTINATION include FILES_MATCHING PATTERN "*.h")
#install(TARGETS multigridshared DESTINATION lib)
#

