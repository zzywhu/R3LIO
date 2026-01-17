/******************************************************************************
 * Copyright 2019 The Hesai Technology Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "pandarGeneral/pandarGeneral.h"
#include <fstream>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif // _WIN32

#define PRINT_FLAG
// #define PCD_FILE_WRITE_FLAG

int frameItem = 0;

void gpsCallback(int timestamp) {
#ifdef PRINT_FLAG
    printf("gps: %d\n", timestamp);
#endif
}

void lidarCallback(PPointCloud::Ptr cld, double timestamp) {
#ifdef PRINT_FLAG
    printf("timestamp: %lf,point_size: %ld\n", timestamp, cld->points.size());
#endif
#ifdef PCD_FILE_WRITE_FLAG
    frameItem++;
    if (10 == frameItem) {
        printf("write pcd file\n");
        pcl::PCDWriter writer;
        writer.write("PandarPointXYZI.pcd", *cld);
    }
#endif
}

int main(int argc, char **argv) {

    PandarGeneral *pandarGeneral = new PandarGeneral(
        std::string("/media/pz/Data/Dataset/FM_LiDAR/data/raw/SN_00036/dbea544d-3b92-4687-8715-e1f5ed6d1f9a/"
                    "20220415-023457_00036_SLAM100_Pandar_0001.pcap"),
        lidarCallback, 0, 0, 1, std::string("PandarXT-16"), "", "", false);
    std::string filePath = "/home/pz/cmake_ws/DAS_Lidar_Reader/config/pandar/PandarXT-16.csv";
    std::ifstream fin(filePath);
    if (fin.is_open()) {
        std::cout << "Open correction file " << filePath << " succeed" << std::endl;
        int length = 0;
        std::string strlidarCalibration;
        fin.seekg(0, std::ios::end);
        length = fin.tellg();
        fin.seekg(0, std::ios::beg);
        char *buffer = new char[length];
        fin.read(buffer, length);
        fin.close();
        strlidarCalibration = buffer;
        int ret = pandarGeneral->LoadCorrectionFile(strlidarCalibration);
        if (ret != 0) {
            std::cout << "Load correction file from " << filePath << " failed" << std::endl;
        } else {
            std::cout << "Load correction file from " << filePath << " succeed" << std::endl;
        }
    } else {
        std::cout << "Open correction file " << filePath << " failed" << std::endl;
    }

    pandarGeneral->Start();

    while (!pandarGeneral->isFinished()) {
#ifdef _WIN32
        Sleep(1); // 1 ms
#else
        usleep(1 * 1000); // 1000 us
#endif // _WIN32
        static int i = 0;
        if (++i > 1000) {
            pandarGeneral->Stop();
            std::cout << "stoped" << std::endl;
        }
    }

    delete pandarGeneral;
    pandarGeneral = nullptr;

    // show whether pthread_create is compiled
#ifdef GTHR_ACTIVE_PROXY
    std::cout << "defined GTHR_ACTIVE_PROXY" << std::endl;
#else
    std::cout << "no define GTHR_ACTIVE_PROXY" << std::endl;
#endif

    return 0;
}
