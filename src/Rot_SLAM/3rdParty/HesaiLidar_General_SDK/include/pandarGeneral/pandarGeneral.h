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

#ifndef SRC_PANDARGENERAL_H_
#define SRC_PANDARGENERAL_H_

#include <array>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

#include "point_types.h"
#include "pandarXT.h"
#include "pcap_reader.h"

// each block have  64 Unit
#define HS_LIDAR_L64_UNIT_NUM (64)
// each Unit have 3 byte :2 bytes(distance) + 1 byte(intensity)
#define HS_LIDAR_L64_UNIT_SIZE (3)

#define GPS_PACKET_SIZE (512)

#define MAX_LASER_NUM (256)
#define MAX_POINT_CLOUD_NUM (1000000)
#define MAX_POINT_CLOUD_NUM_PER_CHANNEL (10000)
#define MAX_AZIMUTH_DEGREE_NUM (36000)
#define HS_LIDAR_XT_COORDINATE_CORRECTION_H (0.0315)
#define HS_LIDAR_XT_COORDINATE_CORRECTION_B (0.013)
#define COORDINATE_CORRECTION_CHECK (false)

#define ROTATION_MAX_UNITS (36001)
#define ETHERNET_MTU (1500)

typedef struct PandarPacket_s {
  uint8_t data[ETHERNET_MTU];
  uint32_t size;
} PandarPacket;

typedef std::array<PandarPacket, 36000> PktArray;

typedef struct PacketsBuffer_s {
  PktArray m_buffers{};
  PktArray::iterator m_iterPush;
  PktArray::iterator m_iterCalc;
  bool m_startFlag;
  inline PacketsBuffer_s() {
    m_iterPush = m_buffers.begin();
    m_iterCalc = m_buffers.begin();
    m_startFlag = false;
  }
  inline int push_back(PandarPacket pkt) {
    if (!m_startFlag) {
      *m_iterPush = pkt;
      m_startFlag = true;
      return 1;
    } 
    m_iterPush++;

    if (m_iterPush == m_iterCalc) {
      printf("buffer don't have space!, %ld\n", m_iterPush - m_buffers.begin());
      return 0;
    }

    if (m_buffers.end() == m_iterPush) {
      m_iterPush = m_buffers.begin();
      *m_iterPush = pkt;
    }
    *m_iterPush = pkt;
    return 1;
    
  }
  inline bool hasEnoughPackets() {
    return ((m_iterPush - m_iterCalc > 0 ) ||
            ((m_iterPush - m_iterCalc + 36000 > 0 ) && (m_buffers.end() - m_iterCalc < 1000) && (m_iterPush - m_buffers.begin() < 1000)));
  }
  inline PktArray::iterator getIterCalc() { return m_iterCalc;}
  inline void moveIterCalc() {
    m_iterCalc++;
    if (m_buffers.end() == m_iterCalc) {
      m_iterCalc = m_buffers.begin();
    }
  }
} PacketsBuffer;


#ifdef __DLL_EXPORTS__
#define PANDAR_GENERAL_API __declspec(dllexport)
#else
#define PANDAR_GENERAL_API
#endif

class PANDAR_GENERAL_API PandarGeneral {
 public:
  /**
   * @brief Constructor
   * @param pcap_path         The path of pcap file
   *        pcl_callback      The callback of PCL data structure
   *        start_angle       The start angle of frame
   *        tz                The timezone
   *        pcl_type          Structured Pointcloud
   *        frame_id          The frame id of pcd
   */
  PandarGeneral(
      std::string pcap_path, \
      std::function<void(PPointCloud::Ptr, double)> \
      pcl_callback, uint16_t start_angle, int tz, int pcl_type, \
      std::string lidar_type, std::string frame_id, std::string timestampType, bool coordinate_correction_flag);// the default timestamp type is LiDAR time
  ~PandarGeneral();

  /**
   * @brief load the correction file
   * @param correction The path of correction file
   */
  int LoadCorrectionFile(std::string correction);

  void Start();
  void Stop();
  bool isFinished();

 private:
  void Init();
  void ProcessLidarPacket();
  void PushLiDARData(PandarPacket packet);
  int ParseXTData(HS_LIDAR_XT_Packet *packet, const uint8_t *recvbuf, const int len);

    void CalcXTPointXYZIT(HS_LIDAR_XT_Packet *pkt, int blockid, char chLaserNumber,
                          PPointCloud::Ptr cld);
  void FillPacket(const uint8_t *buf, const int len);

  void EmitBackMessege(char chLaserNumber, PPointCloud::Ptr cld);
  std::thread *lidar_process_thr_;
  bool enable_lidar_process_thr_;
  int start_angle_;

  std::function<void(PPointCloud::Ptr cld, double timestamp)>
      pcl_callback_;

  float sin_lookup_table_[ROTATION_MAX_UNITS];
  float cos_lookup_table_[ROTATION_MAX_UNITS];

  uint16_t last_azimuth_;
  double last_timestamp_;

  float General_elev_angle_map_[MAX_LASER_NUM];
  float General_horizatal_azimuth_offset_map_[MAX_LASER_NUM];

  float blockXTOffsetSingle_[HS_LIDAR_XT_BLOCK_NUMBER];
  float blockXTOffsetDual_[HS_LIDAR_XT_BLOCK_NUMBER];
  float blockXTOffsetTriple_[HS_LIDAR_XT_BLOCK_NUMBER];
  float laserXTOffset_[HS_LIDAR_XT_UNIT_NUM];

  int tz_second_;
  std::string frame_id_;
  int pcl_type_;
  PcapReader *pcap_reader_;
  std::string m_sLidarType;
  std::vector<float> m_sin_azimuth_map_;
  std::vector<float> m_cos_azimuth_map_;
  std::vector<float> m_sin_elevation_map_;
  std::vector<float> m_cos_elevation_map_;
  std::vector<float> m_sin_azimuth_map_h;
  std::vector<float> m_cos_azimuth_map_h;
  std::vector<float> m_sin_azimuth_map_b;
  std::vector<float> m_cos_azimuth_map_b;
  PacketsBuffer m_PacketsBuffer;
  bool m_bCoordinateCorrectionFlag;

};

#endif  // SRC_PANDARGENERAL_H_
