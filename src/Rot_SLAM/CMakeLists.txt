cmake_minimum_required(VERSION 3.0.0)
project(rigelslam_rot)

# SET(CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
endif()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
ADD_COMPILE_OPTIONS(-std=c++17)

if(UNIX)
    message("current platform: Linux ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -Wall -march=native")
    set(PCAP_LIBRARIES pcap)

    find_package(catkin REQUIRED COMPONENTS
            rosbag
            roscpp
            std_msgs
            sensor_msgs
            geometry_msgs
            nav_msgs
            tf
            )
    include_directories(
            ${catkin_INCLUDE_DIRS}
    )

    add_message_files(
            DIRECTORY msg
            FILES
            motor.msg
    )
    generate_messages(
            DEPENDENCIES
            geometry_msgs
            std_msgs
            nav_msgs
            sensor_msgs
    )
    catkin_package()

elseif(WIN32)
    message("current platform: Windows ")
    add_definitions(-D__SHARED_LIBS__)
    #add_definitions(-DCV_IGNORE_DEBUG_BUILD_GUARD)
    ADD_COMPILE_OPTIONS(/bigobj)
    #ADD_COMPILE_OPTIONS(/arch=AVX512)
    #add_definitions(-DEIGEN_USE_MKL_ALL)
    #add_definitions(-D__AVX__)
    #add_definitions(-D__AVX2__)
    #add_definitions(-D__AVX512BW__)
    #add_definitions(-D__AVX512CD__)
    #add_definitions(-D__AVX512DQ__)
    #add_definitions(-D__AVX512F__)
    #add_definitions(-D__AVX512VL__)
  
    #add_definitions(-DEIGEN_VECTORIZE_SSE4_2)

    set(EIGEN_DIR ${PROJECT_SOURCE_DIR}/3rdParty/eigen3.4.0/share/eigen3/cmake)
    set(PCL_DIR ${PROJECT_SOURCE_DIR}/3rdParty/PCL_1_12_1/cmake)
    set(gflags_DIR ${PROJECT_SOURCE_DIR}/3rdParty/gflags/lib/cmake/gflags)
    set(glog_DIR ${PROJECT_SOURCE_DIR}/3rdParty/glog/lib/cmake/glog)
    set(Ceres_DIR ${PROJECT_SOURCE_DIR}/3rdParty/Ceres/lib/cmake/Ceres)
    set(VTK_DIR ${PROJECT_SOURCE_DIR}/3rdParty/PCL_1_12_1/3rdParty/VTK/lib/cmake/vtk-9.1)
    set(yaml-cpp_DIR ${PROJECT_SOURCE_DIR}/3rdParty/YAML_CPP/lib/cmake/yaml-cpp)

    set(PCAP_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdParty/npcap-sdk-1.13/Include)
    set(PCAP_LIBRARY_DIRS ${PROJECT_SOURCE_DIR}/3rdParty/npcap-sdk-1.13/Lib/x64)
    set(PCAP_LIBRARIES wpcap)

    set(PTHREADS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdParty/pthreads.2.9.1.4/build/native/include)
    set(PTHREADS_LIBRARY_DIRS ${PROJECT_SOURCE_DIR}/3rdParty/pthreads.2.9.1.4/build/native/lib)
    set(PTHREADS_LIBRARY ${PROJECT_SOURCE_DIR}/3rdParty/pthreads.2.9.1.4/build/native/lib/pthreadVC2.lib)

    set(OPENNI2_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdParty/OpenNI2/Include)
    set(OPENNI2_LIBRARY ${PROJECT_SOURCE_DIR}/3rdParty/OpenNI2/Lib/OpenNI2.lib)

    set(GLEW_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rdParty/glew/include)
    set(GLEW_GLEW_LIBRARY ${PROJECT_SOURCE_DIR}/3rdParty/glew/lib/Release/x64/glew32.lib)

    set(OPENCV_LIBRARY_DIRS ${PROJECT_SOURCE_DIR}/3rdParty/opencv-4.5.5)

    include_directories(
            ${PCAP_INCLUDE_DIRS}
            ${PTHREADS_INCLUDE_DIRS}
    )

    link_directories(
            ${PCAP_LIBRARY_DIRS}
            ${PTHREADS_LIBRARY_DIRS}
    )
else()
    message(FATAL_ERROR "current platform: unkonw ")
endif()

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
add_definitions(-DNOMINMAX)

option(BA_ENABLE "Enable mapping with bundle adjustment" OFF)
option(TEASER_ON "Enable teaser registration" OFF)
option(CERES_ON "Enable ceres optimization" ON)
option(G2O_ON "Enable g2o optimization" OFF)
option(GTSAM_ON "Enable gtsam optimization" OFF)

find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif(CCACHE_PROGRAM)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    include(ProcessorCount)
    ProcessorCount(N)
    message("Processer number:  ${N}")

    if(N GREATER 4)
        add_definitions(-DMP_EN)
#        math(EXPR PROC_NUM "(${N} / 2)")
#        add_definitions(-DMP_PROC_NUM=${PROC_NUM})
        add_definitions(-DMP_PROC_NUM=3)
        message("core for MP: 3")
    elseif(N GREATER 3)
        add_definitions(-DMP_EN)
        add_definitions(-DMP_PROC_NUM=2)
        message("core for MP: 2")
    else()
        add_definitions(-DMP_PROC_NUM=1)
    endif()
else()
    add_definitions(-DMP_PROC_NUM=1)
endif()

if(BA_ENABLE)
    add_definitions(-DBA_ENABLE)
    message("Enable voxel mapping with bundle adjustment")
else()
    message("voxel mapping without bundle adjustment")
endif()

if(TEASER_ON)
    add_definitions(-DTEASER_ON)
    find_package(teaserpp REQUIRED QUIET)
    include_directories(${teaserpp_INCLUDE_DIRS})
    message("Teaser++ [OK]")
else()
    message("Disable teaser registration")
endif()

if(CERES_ON)
    add_definitions(-DCERES_ON)
    message("Ceres [Enable]")
endif()

if(G2O_ON)
    add_definitions(-DG2O_ON)
    message("G2O [Enable]")
endif()

if(GTSAM_ON)
    add_definitions(-DGTSAM_ON)
    message("GTSAM [Enable]")
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(re2 QUIET)
if(NOT re2_FOUND)
  find_library(RE2_LIBRARY re2)
endif()

macro(source_group_by_dir source_files)
    if(MSVC)
        set(sgbd_cur_dir ${CMAKE_CURRENT_SOURCE_DIR})

        foreach(sgbd_file ${${source_files}})
            string(REGEX REPLACE ${sgbd_cur_dir}/\(.*\) \\1 sgbd_fpath ${sgbd_file})
            string(REGEX REPLACE "\(.*\)/.*" \\1 sgbd_group_name ${sgbd_fpath})
            string(COMPARE EQUAL ${sgbd_fpath} ${sgbd_group_name} sgbd_nogroup)
            string(REPLACE "/" "\\" sgbd_group_name ${sgbd_group_name})

            if(sgbd_nogroup)
                set(sgbd_group_name "\\")
            endif(sgbd_nogroup)

            source_group(${sgbd_group_name} FILES ${sgbd_file})
        endforeach(sgbd_file)
    endif(MSVC)
endmacro(source_group_by_dir)
set(cv_bridge_DIR "/opt/ros/noetic/share/cv_bridge/cmake")
find_package(cv_bridge REQUIRED)
find_package(PythonLibs REQUIRED)
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
find_package(PCL 1.7 REQUIRED)
message(STATUS "Found PCL version: ${PCL_VERSION}")

find_package(Ceres REQUIRED)
message(STATUS "Found Ceres version: ${Ceres_VERSION}")

find_package(yaml-cpp REQUIRED)
message(STATUS "Found yaml-cpp version: ${YAML_CPP_VERSION}")

find_package(Eigen3 REQUIRED)
message(STATUS "Found Eigen3 version: ${EIGEN3_VERSION}")

find_package(GTSAM REQUIRED QUIET)
message(STATUS "Found GTSAM version: ${GTSAM_VERSION}")
#set(CMAKE_PREFIX_PATH "/home/zzy/SLAM/xfeat_cpp/thirdparty/pytorch/torch")
message(111: ${cv_bridge_DIR})
#find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(CMAKE_PREFIX_PATH "/home/zg/depends/opencv-4.5.5/install")
find_package(OpenCV REQUIRED QUIET)

set(CMAKE_PREFIX_PATH "/home/zzy/Downloads/pytorch/torch")
set(Torch_DIR /home/zzy/Downloads/pytorch/torch)
find_package(Torch REQUIRED)




add_subdirectory(3rdParty/HesaiLidar_General_SDK)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

if(GTSAM_ON)
    link_directories(${GTSAM_LIBRARY_DIRS})
    set(GTSAM_LIBRARIES gtsam)
endif()

set(ONNXRUNTIME_DIR  "/home/zzy/Downloads/onnxruntime")

include_directories(
        include/DataIO
        include/Feature
        include/Initializer
        include/LoopCloser
        include/LoopCloser/ContourContext
        include/Mapper
        include/Matcher
        include/Misc
        include/Optimizer
        include/Registrator
        include/Registrator/super4pcs
        include/Registrator/fast_gicp
        include/SensorProcess
        include/Viewer
        include/xfeat
        include/ImageProcess
        include/MLSD
        include/BEVProcess
        include/patchwork
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
        ${ONNXRUNTIME_DIR}/include
        ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session
        ${ONNXRUNTIME_DIR}/include/onnxruntime/core/providers/cuda
        3rdParty/HesaiLidar_General_SDK/include
        ${TORCH_INCLUDE_DIRS}
        include

)

#file(GLOB_RECURSE CORE_SRC "src/*.cpp" "src/*.cc" "src/*.c")
link_directories("${ONNXRUNTIME_ROOT}/lib")



file(GLOB_RECURSE DataIO_SRC "src/DataIO/*.cpp")
add_library(DataIO SHARED ${DataIO_SRC})
target_link_libraries(DataIO ${PCL_LIBRARIES} ${PCAP_LIBRARIES} ${PTHREADS_LIBRARY} PandarGeneralSDK)
target_compile_definitions(DataIO PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Feature_SRC "src/Feature/*.cpp")
add_library(Feature SHARED ${Feature_SRC})
target_link_libraries(Feature ${PCL_LIBRARIES} ${PTHREADS_LIBRARY})
target_compile_definitions(Feature PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Initializer_SRC "src/Initializer/*.cpp")
add_library(Initializer SHARED ${Initializer_SRC})
target_link_libraries(Initializer ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES} ${PYTHON_LIBRARIES})
target_compile_definitions(Initializer PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Mapper_SRC "src/Mapper/*.cpp")
add_library(Mapper SHARED ${Mapper_SRC})
target_link_libraries(Mapper Misc ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES})
target_compile_definitions(Mapper PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Matcher_SRC "src/Matcher/*.cpp")
add_library(Matcher SHARED ${Matcher_SRC})
target_link_libraries(Matcher ${PCL_LIBRARIES} ${PCAP_LIBRARIES} ${CERES_LIBRARIES})
target_compile_definitions(Matcher PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Misc_SRC "src/Misc/*.cpp")
add_library(Misc SHARED ${Misc_SRC})
target_link_libraries(Misc ${PCL_LIBRARIES})
target_compile_definitions(Misc PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Optimizer_SRC "src/Optimizer/*.cpp")
add_library(Optimizer SHARED ${Optimizer_SRC})
target_link_libraries(Optimizer Mapper Misc ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES} gtsam)
target_compile_definitions(Optimizer PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Registrator_SRC "src/Registrator/*.cpp" "src/Registrator/*.cc")
add_library(Registrator SHARED ${Registrator_SRC})
target_link_libraries(Registrator Misc SensorProcess ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES}  ${OpenCV_LIBRARIES} gtsam)
target_compile_definitions(Registrator PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)
if(TEASER_ON)
    target_link_libraries(Registrator teaserpp::teaser_registration teaserpp::teaser_io)
endif()

file(GLOB_RECURSE SensorProcess_SRC "src/SensorProcess/*.cpp")
add_library(SensorProcess SHARED ${SensorProcess_SRC})
target_link_libraries(SensorProcess Mapper Misc ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES} gtsam)
target_compile_definitions(SensorProcess PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE BEVProcess_SRC "src/BEVProcess/*.cpp")
add_library(BEVProcess SHARED ${BEVProcess_SRC})
target_link_libraries(BEVProcess Mapper Misc ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} gtsam)
target_compile_definitions(BEVProcess PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE Viewer_SRC "src/Viewer/*.cpp")
add_library(Viewer SHARED ${Viewer_SRC})
target_link_libraries(Viewer Misc Mapper ${PCL_LIBRARIES} ${PTHREADS_LIBRARY})
target_compile_definitions(Viewer PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE LoopCloser_SRC "src/LoopCloser/*.cpp")
add_library(LoopCloser SHARED ${LoopCloser_SRC})
target_link_libraries(LoopCloser Misc Mapper Registrator Feature Optimizer DataIO ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES} gtsam)
target_compile_definitions(LoopCloser PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE xfeat_SRC "src/xfeat/*.cpp")
add_library(xfeat SHARED ${xfeat_SRC})
target_link_libraries(xfeat ${OpenCV_LIBS} ${TORCH_LIBRARIES} Misc Mapper Registrator Feature Optimizer DataIO ${PCL_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES} gtsam)
target_compile_definitions(xfeat PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE mlsd_SRC "src/MLSD/*.cpp")
add_library(mlsd SHARED ${mlsd_SRC})
target_link_libraries(mlsd ${OpenCV_LIBS} onnxruntime re2 ${PCL_LIBRARIES})
target_compile_definitions(mlsd PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE imgp_SRC "src/ImageProcess/*.cpp")
add_library(imgp SHARED ${imgp_SRC})
target_link_libraries(imgp ${OpenCV_LIBS} ${PCL_LIBRARIES} Mapper mlsd) 
target_compile_definitions(imgp PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

file(GLOB_RECURSE imgp_SRC "src/patchwork/*.cpp")
add_library(patchwork SHARED ${imgp_SRC})
target_link_libraries(patchwork ${OpenCV_LIBS} ${PCL_LIBRARIES} Mapper mlsd) 
target_compile_definitions(patchwork PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

SET(cv_bridge_LIBRARIES "/opt/ros/noetic/lib/libcv_bridge.so")
#file(GLOB_RECURSE SlamSys_SRC "src/System.cpp")
if(MSVC)
    file(GLOB_RECURSE CORE_HDR "include/*.hpp" "include/*.h")
    source_group_by_dir(CORE_HDR)
    add_library(SlamCore
            SHARED
            "src/System/System.cpp"
            "src/System/System-visualization.cpp"
            "src/System/System-process.cpp"
            "src/System/System-utils.cpp"
            "src/SensorProcess/LidarProcess.cpp"
            "src/SensorProcess/IMUPreintegration.cpp"
            "src/Registrator/RegistrationLoam.cpp"
            ${Misc_SRC}
            ${LoopCloser_SRC}
            ${CORE_HDR}
            ${cv_bridge_LIBRARIES})
            
            
else()
    add_library(SlamCore
            SHARED
            "src/System/System.cpp"
            "src/System/System-visualization.cpp"
            "src/System/System-process.cpp"
            "src/System/System-utils.cpp")
endif()

target_link_libraries(SlamCore mlsd  xfeat imgp SensorProcess patchwork DataIO Feature BEVProcess Initializer LoopCloser Mapper Matcher Misc Optimizer Registrator Viewer 
${PCL_LIBRARIES} ${PCAP_LIBRARIES} ${TORCH_LIBRARIES} ${PTHREADS_LIBRARY} ${CERES_LIBRARIES} ${GTSAM_LIBRARIES} ${YAML_CPP_LIBRARIES} ${OpenCV_LIBRARIES}  ${cv_bridge_LIBRARIES} PandarGeneralSDK )

target_compile_definitions(SlamCore PRIVATE $<$<CXX_COMPILER_ID:MSVC>:__DLL_EXPORTS__>)

#
#add_executable(SlamTest Examples/SlamTest.cpp)
#target_link_libraries(SlamTest SlamCore ${PTHREADS_LIBRARY})
#
#add_executable(JiShangSlamTest Examples/JiShangSlamTest.cpp)
#target_link_libraries(JiShangSlamTest SlamCore ${PTHREADS_LIBRARY})

#add_executable(HiltiSlamTest Examples/HiltiSlamTest.cpp)
#target_link_libraries(HiltiSlamTest SlamCore ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})

# add_executable(FeimaSlamTest Examples/FeimaSlamTest.cpp)
# target_link_libraries(FeimaSlamTest SlamCore ${PTHREADS_LIBRARY})

#add_executable(DasSlamTest Examples/DasSlamTest.cpp)
#target_link_libraries(DasSlamTest SlamCore ${PTHREADS_LIBRARY})

#add_executable(OfflineLoopTest Examples/OfflineLoopTest.cpp)
#target_link_libraries(OfflineLoopTest SlamCore ${PTHREADS_LIBRARY})

#add_executable(XiangyinSlamTest Examples/XiangyinSlamTest.cpp)
#target_link_libraries(XiangyinSlamTest SlamCore ${PTHREADS_LIBRARY})

#add_executable(ContourContextTest Examples/ContourContextTest.cpp)
#target_link_libraries(ContourContextTest SlamCore ${PTHREADS_LIBRARY})

#add_executable(ForceLoopTest Examples/ForceLoopTest.cpp)
#target_link_libraries(ForceLoopTest SlamCore ${PTHREADS_LIBRARY})
if(UNIX)
#add_executable(ZGSlamTest Examples/ZGSlamTest.cpp)
#target_link_libraries(ZGSlamTest SlamCore ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})

add_executable(ZGSlamRosTest Examples/ZGSlamRosTest.cpp)
target_link_libraries(ZGSlamRosTest SlamCore Misc ${PTHREADS_LIBRARY} ${catkin_LIBRARIES} ${cv_bridge_LIBRARIES})

add_executable(UndistTest Examples/undis.cpp)
target_link_libraries(UndistTest SlamCore Misc ${PTHREADS_LIBRARY} ${catkin_LIBRARIES} ${cv_bridge_LIBRARIES})

# add_executable(CalibTest Examples/calibtest.cpp)
# target_link_libraries(CalibTest SlamCore Misc ${PTHREADS_LIBRARY} ${catkin_LIBRARIES} ${cv_bridge_LIBRARIES})

add_executable(Sim Examples/Sim.cpp)
target_link_libraries(Sim SlamCore Misc ${PTHREADS_LIBRARY} ${catkin_LIBRARIES} ${cv_bridge_LIBRARIES})
#add_executable(RosSlamTest Examples/RosSlamTest.cpp)
#target_link_libraries(RosSlamTest SlamCore ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})

add_executable(Reloc Examples/Reloc.cpp)
target_link_libraries(Reloc SlamCore Misc  ${PTHREADS_LIBRARY} ${catkin_LIBRARIES} ${cv_bridge_LIBRARIES})
endif()



#add_executable(RosSlamTest Examples/RosSlamTest.cpp)
#target_link_libraries(RosSlamTest SlamCore ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})


#add_executable(AlignTest Examples/AlignTest.cpp)
#target_link_libraries(AlignTest SlamCore Misc ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})

#add_executable(ZGPostOptimizeTest "Examples/ZGPostOptimizeTest.cpp")
#target_link_libraries(ZGPostOptimizeTest SlamCore Misc ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})

#add_executable(ZGRawReader "Examples/ZGRawReader.cpp")
#target_link_libraries(ZGRawReader SlamCore Misc ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})

#add_executable(EigenTest "Examples/EigenTest.cpp")
#target_link_libraries(EigenTest ${PTHREADS_LIBRARY} ${catkin_LIBRARIES})