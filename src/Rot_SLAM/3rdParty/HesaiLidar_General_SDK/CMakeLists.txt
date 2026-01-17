cmake_minimum_required(VERSION 2.8)
project(PandarGeneralSDK)

find_package(PCL REQUIRED COMPONENTS common)

ADD_COMPILE_OPTIONS(-std=c++11)
ADD_COMPILE_OPTIONS(-std=c++14)

if(UNIX)
    message("current platform: Linux ")
    set(CMAKE_CXX_FLAGS "-fPIC --std=c++11 -pthread")
    set(PCAP_LIBRARIES pcap)
elseif(WIN32)
    message("current platform: Windows ")
    add_definitions(-D__DLL_EXPORTS__)

    if(NOT DEFINED PCAP_INCLUDE_DIRS)
        set(PCAP_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../npcap-sdk-1.13/Include)
    endif()

    if(NOT DEFINED PCAP_LIBRARY_DIRS)
        set(PCAP_LIBRARY_DIRS ${PROJECT_SOURCE_DIR}/../npcap-sdk-1.13/Lib/x64)
    endif()

    set(PCAP_LIBRARIES wpcap)
    include_directories(
        ${PCAP_INCLUDE_DIRS}
    )
    link_directories(
        ${PCAP_LIBRARY_DIRS}
    )
else()
    message(FATAL_ERROR "current platform: unkonw ")
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

include_directories(
    .
    include
    ${PCL_INCLUDE_DIRS}
)

file(GLOB_RECURSE CORE_SRC "src/*.cpp" "src/*.cc" "src/*.c")

if(MSVC)
    file(GLOB_RECURSE CORE_HDR "include/*.hpp" "include/*.h")
    source_group_by_dir(CORE_SRC)
    source_group_by_dir(CORE_HDR)
    add_library(PandarGeneralSDK STATIC ${CORE_SRC} ${CORE_HDR})
else()
    add_library(PandarGeneralSDK STATIC ${CORE_SRC})
endif()

target_link_libraries(PandarGeneralSDK
    ${PCL_IO_LIBRARIES}
    ${PCAP_LIBRARIES}
)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    find_package(PCL REQUIRED)
    add_executable(PandarGeneralTest
        test/test.cc
    )

    target_link_libraries(PandarGeneralTest
        PandarGeneralSDK
        ${PCL_IO_LIBRARIES}
    )
endif(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
