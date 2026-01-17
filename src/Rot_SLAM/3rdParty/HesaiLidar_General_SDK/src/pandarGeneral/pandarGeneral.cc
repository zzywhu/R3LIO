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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double degreeToRadian(double degree) { return degree * M_PI / 180; }

static const float pandarGeneral_elev_angle_map[] = {
    14.882f, 11.032f, 8.059f, 5.057f, 3.04f, 2.028f, 1.86f, 1.688f, \
    1.522f, 1.351f, 1.184f, 1.013f, 0.846f, 0.675f, 0.508f, 0.337f, \
    0.169f, 0.0f, -0.169f, -0.337f, -0.508f, -0.675f, -0.845f, -1.013f, \
    -1.184f, -1.351f, -1.522f, -1.688f, -1.86f, -2.028f, -2.198f, -2.365f, \
    -2.536f, -2.7f, -2.873f, -3.04f, -3.21f, -3.375f, -3.548f, -3.712f, \
    -3.884f, -4.05f, -4.221f, -4.385f, -4.558f, -4.72f, -4.892f, -5.057f, \
    -5.229f, -5.391f, -5.565f, -5.726f, -5.898f, -6.061f, -7.063f, -8.059f, \
    -9.06f, -9.885f, -11.032f, -12.006f, -12.974f, -13.93f, -18.889f, -24.897f
};

static std::vector<std::vector<PPoint> > PointCloudList(MAX_LASER_NUM);
static std::vector<PPoint> PointCloud(MAX_POINT_CLOUD_NUM);
static int iPointCloudIndex = 0;

PandarGeneral::PandarGeneral(std::string pcap_path, \
    std::function<void(PPointCloud::Ptr, double)> \
    pcl_callback, uint16_t start_angle, int tz, int pcl_type, \
    std::string lidar_type, std::string frame_id, \
    std::string timestampType, bool coordinate_correction_flag) {

  lidar_process_thr_ = NULL;

  enable_lidar_process_thr_ = false;

  pcap_reader_ = new PcapReader(pcap_path, lidar_type);

  start_angle_ = start_angle;
  pcl_callback_ = pcl_callback;
  last_azimuth_ = 0;
  last_timestamp_ = 0;
  m_sLidarType = lidar_type;
  frame_id_ = frame_id;
  tz_second_ = tz * 3600;
  pcl_type_ = pcl_type;
  m_bCoordinateCorrectionFlag = coordinate_correction_flag;

  Init();
}

PandarGeneral::~PandarGeneral() {
  Stop();

  if (pcap_reader_ != NULL) {
    delete pcap_reader_;
    pcap_reader_ = NULL;
  }
}

void PandarGeneral::Init() {
  for (uint16_t rotIndex = 0; rotIndex < ROTATION_MAX_UNITS; ++rotIndex) {
    float rotation = degreeToRadian(0.01 * static_cast<double>(rotIndex));
    cos_lookup_table_[rotIndex] = cosf(rotation);
    sin_lookup_table_[rotIndex] = sinf(rotation);
  }
  m_sin_azimuth_map_.resize(MAX_AZIMUTH_DEGREE_NUM);
  m_cos_azimuth_map_.resize(MAX_AZIMUTH_DEGREE_NUM);
  for(int i = 0; i < MAX_AZIMUTH_DEGREE_NUM; ++i) {
    m_sin_azimuth_map_[i] = sinf(i * M_PI / 18000);
    m_cos_azimuth_map_[i] = cosf(i * M_PI / 18000);
  }
  m_sin_azimuth_map_h.resize(MAX_AZIMUTH_DEGREE_NUM);
  m_cos_azimuth_map_h.resize(MAX_AZIMUTH_DEGREE_NUM);
  for(int i = 0; i < MAX_AZIMUTH_DEGREE_NUM; ++i) {
    m_sin_azimuth_map_h[i] = sinf(i * M_PI / 18000) * HS_LIDAR_XT_COORDINATE_CORRECTION_H;
    m_cos_azimuth_map_h[i] = cosf(i * M_PI / 18000) * HS_LIDAR_XT_COORDINATE_CORRECTION_H;
  }
  m_sin_azimuth_map_b.resize(MAX_AZIMUTH_DEGREE_NUM);
  m_cos_azimuth_map_b.resize(MAX_AZIMUTH_DEGREE_NUM);
  for(int i = 0; i < MAX_AZIMUTH_DEGREE_NUM; ++i) {
    m_sin_azimuth_map_b[i] = sinf(i * M_PI / 18000) * HS_LIDAR_XT_COORDINATE_CORRECTION_B;
    m_cos_azimuth_map_b[i] = cosf(i * M_PI / 18000) * HS_LIDAR_XT_COORDINATE_CORRECTION_B;
  }
  if (pcl_type_) {
    for (int i = 0; i < MAX_LASER_NUM; i++) {
      PointCloudList[i].reserve(MAX_POINT_CLOUD_NUM_PER_CHANNEL);
    }
  }

  if (m_sLidarType == "PandarXT-32") {
    m_sin_elevation_map_.resize(HS_LIDAR_XT_UNIT_NUM);
    m_cos_elevation_map_.resize(HS_LIDAR_XT_UNIT_NUM);
    for (int i = 0; i < HS_LIDAR_XT_UNIT_NUM; i++) {
      m_sin_elevation_map_[i] = sinf(degreeToRadian(pandarXT_elev_angle_map[i]));
      m_cos_elevation_map_[i] = cosf(degreeToRadian(pandarXT_elev_angle_map[i]));
      General_elev_angle_map_[i] = pandarXT_elev_angle_map[i];
      General_horizatal_azimuth_offset_map_[i] = pandarXT_horizatal_azimuth_offset_map[i];
      laserXTOffset_[i] = laserXTOffset[i];
    }
  }

  if (m_sLidarType == "PandarXT-16") {
    m_sin_elevation_map_.resize(HS_LIDAR_XT16_UNIT_NUM);
    m_cos_elevation_map_.resize(HS_LIDAR_XT16_UNIT_NUM);
    for (int i = 0; i < HS_LIDAR_XT16_UNIT_NUM; i++) {
      m_sin_elevation_map_[i] = sinf(degreeToRadian(pandarXT_elev_angle_map[i*2]));
      m_cos_elevation_map_[i] = cosf(degreeToRadian(pandarXT_elev_angle_map[i*2]));
      General_elev_angle_map_[i] = pandarXT_elev_angle_map[i*2];
      General_horizatal_azimuth_offset_map_[i] = pandarXT_horizatal_azimuth_offset_map[i*2];
      laserXTOffset_[i] = laserXTOffset[i*2];
    }
  }

  for (int i = 0; i < HS_LIDAR_XT_BLOCK_NUMBER; i++) {
    blockXTOffsetSingle_[i] = blockXTOffsetSingle[i];
    blockXTOffsetDual_[i] = blockXTOffsetDual[i];
  }

}

/**
 * @brief load the correction file
 * @param file The path of correction file
 */
int PandarGeneral::LoadCorrectionFile(std::string correct_file_path) {
  std::ifstream correct_fin(correct_file_path);
  if (!correct_fin.is_open()) {
    std::cerr << "Open correction file " << correct_file_path << " failed" << std::endl;
    return -1;
  }
  std::cout << "Open correction file " << correct_file_path << " succeed" << std::endl;
  
  std::string correction_content;
  correct_fin.seekg(0, std::ios::end);
  int length = correct_fin.tellg();
  correct_fin.seekg(0, std::ios::beg);
  char *buffer = new char[length];
  correct_fin.read(buffer, length);
  correct_fin.close();
  correction_content = buffer;
  std::istringstream ifs(correction_content);

  std::string line;
  if (std::getline(ifs, line)) {  // first line "Laser id,Elevation,Azimuth"
    std::cout << "Parse Lidar Correction..." << std::endl;
  }

  double azimuthOffset[HS_LIDAR_L64_UNIT_NUM];
  double elev_angle[HS_LIDAR_L64_UNIT_NUM];

  int lineCounter = 0;
  while (std::getline(ifs, line)) {
    // correction file has 3 columns, min length is len(0,0,0)
    if (line.length() < 5) {
      break;
    }
    lineCounter++;

    int lineId = 0;
    double elev, azimuth;

    std::stringstream ss(line);
    std::string subline;
    std::getline(ss, subline, ',');
    std::stringstream(subline) >> lineId;
    std::getline(ss, subline, ',');
    std::stringstream(subline) >> elev;
    std::getline(ss, subline, ',');
    std::stringstream(subline) >> azimuth;

    if (lineId != lineCounter) {
      break;
    }

    elev_angle[lineId - 1] = elev;
    azimuthOffset[lineId - 1] = azimuth;
  }
  m_sin_elevation_map_.resize(lineCounter);
  m_cos_elevation_map_.resize(lineCounter);
  for (int i = 0; i < lineCounter; ++i) {
    /* for all the laser offset */
    General_elev_angle_map_[i] = elev_angle[i];
    m_sin_elevation_map_[i] = sinf(degreeToRadian(General_elev_angle_map_[i]));
    m_cos_elevation_map_[i] = cosf(degreeToRadian(General_elev_angle_map_[i]));
    General_horizatal_azimuth_offset_map_[i] = azimuthOffset[i];
  }

  return 0;
}

void PandarGeneral::Start() {
  Stop();
  enable_lidar_process_thr_ = true;

  pcap_reader_->start(std::bind(&PandarGeneral::FillPacket, this, std::placeholders::_1, std::placeholders::_2));
}

void PandarGeneral::Stop() {
  enable_lidar_process_thr_ = false;

  if (lidar_process_thr_) {
    // lidar_process_thr_->interrupt(); // no interrupt before c++20
    lidar_process_thr_->join();
    delete lidar_process_thr_;
    lidar_process_thr_ = NULL;
  }

  if (pcap_reader_ != NULL) {
    pcap_reader_->stop();
  }

  return;
}

bool PandarGeneral::isFinished() {
  return pcap_reader_->isFinished();
}

void PandarGeneral::FillPacket(const uint8_t *buf, const int len) {
  if (len != GPS_PACKET_SIZE) {
    PandarPacket pkt;
    memcpy(pkt.data, buf, len);
    pkt.size = len;
    PushLiDARData(pkt);
  }
}

void PandarGeneral::ProcessLidarPacket() {

  if (!m_PacketsBuffer.hasEnoughPackets()) {
    return;
  }

  struct timespec ts;
  int ret = 0;

  PPointCloud::Ptr outMsg(new PPointCloud());

  PandarPacket packet = *(m_PacketsBuffer.getIterCalc());
  m_PacketsBuffer.moveIterCalc();

  if((packet.size == HS_LIDAR_XT_PACKET_SIZE && (m_sLidarType == "XT" || m_sLidarType == "PandarXT-32")) || \
      (packet.size == HS_LIDAR_XT16_PACKET_SIZE && (m_sLidarType == "PandarXT-16"))) {
    HS_LIDAR_XT_Packet pkt;
    ret = ParseXTData(&pkt, packet.data, packet.size);
    if (ret != 0) {
      return;
    }

    for (int i = 0; i < pkt.header.chBlockNumber; ++i) {
      int azimuthGap = 0; /* To do */
      double timestampGap = 0; /* To do */
      if(last_azimuth_ > pkt.blocks[i].azimuth) {
        azimuthGap = static_cast<int>(pkt.blocks[i].azimuth) + (36000 - static_cast<int>(last_azimuth_));
      } else {
        azimuthGap = static_cast<int>(pkt.blocks[i].azimuth) - static_cast<int>(last_azimuth_);
      }
      timestampGap = pkt.timestamp_point - last_timestamp_ + 0.001;
      if (last_azimuth_ != pkt.blocks[i].azimuth && \
          (azimuthGap / timestampGap) < MAX_AZIMUTH_DEGREE_NUM * 100 ) {
        /* for all the blocks */
        if ((last_azimuth_ > pkt.blocks[i].azimuth &&
              start_angle_ <= pkt.blocks[i].azimuth) ||
            (last_azimuth_ < start_angle_ &&
              start_angle_ <= pkt.blocks[i].azimuth)) {
          if (pcl_callback_ && (iPointCloudIndex > 0 || PointCloudList[0].size() > 0)) {
            EmitBackMessege(pkt.header.chLaserNumber, outMsg);
          }
        }
      } else {
        //printf("last_azimuth_:%d pkt.blocks[i].azimuth:%d  *******azimuthGap:%d\n", last_azimuth_, pkt.blocks[i].azimuth, azimuthGap);
      }
      CalcXTPointXYZIT(&pkt, i, pkt.header.chLaserNumber, outMsg);
      last_azimuth_ = pkt.blocks[i].azimuth;
      last_timestamp_ = pkt.timestamp_point;
    }
  } else {
    return;
  }

  outMsg->header.frame_id = frame_id_;
  outMsg->height = 1;
}

void PandarGeneral::PushLiDARData(PandarPacket packet) {
  m_PacketsBuffer.push_back(packet);
  ProcessLidarPacket();
}

int PandarGeneral::ParseXTData(HS_LIDAR_XT_Packet *packet,
                                const uint8_t *recvbuf, const int len) {
  if (len != HS_LIDAR_XT_PACKET_SIZE && len != HS_LIDAR_XT16_PACKET_SIZE) {
    std::cout << "packet size mismatch PandarXT " << len << "," << \
        len << std::endl;
    return -1;
  }
 

  int index = 0;
  int block = 0;
  //Parse 12 Bytes Header
  packet->header.sob = (recvbuf[index] & 0xff) << 8| ((recvbuf[index+1] & 0xff));
  packet->header.chProtocolMajor = recvbuf[index+2] & 0xff;
  packet->header.chProtocolMinor = recvbuf[index+3] & 0xff;
  packet->header.chLaserNumber = recvbuf[index+6] & 0xff;
  packet->header.chBlockNumber = recvbuf[index+7] & 0xff;
  packet->header.chReturnType = recvbuf[index+8] & 0xff;
  packet->header.chDisUnit = recvbuf[index+9] & 0xff;
  index += HS_LIDAR_XT_HEAD_SIZE;

  if (packet->header.sob != 0xEEFF) {
    printf("Error Start of Packet!\n");
    return -1;
  }

  for(block = 0; block < packet->header.chBlockNumber; block++) {
    packet->blocks[block].azimuth = (recvbuf[index] & 0xff) | \
        ((recvbuf[index + 1] & 0xff) << 8);
    index += HS_LIDAR_XT_BLOCK_HEADER_AZIMUTH;

    int unit;

    for(unit = 0; unit < packet->header.chLaserNumber; unit++) {
      unsigned int unRange = (recvbuf[index]& 0xff) | ((recvbuf[index + 1]& 0xff) << 8);

      packet->blocks[block].units[unit].distance = \
          (static_cast<double>(unRange * packet->header.chDisUnit)) / (double)1000;
      packet->blocks[block].units[unit].intensity = (recvbuf[index+2]& 0xff);
      packet->blocks[block].units[unit].confidence = (recvbuf[index+3]& 0xff);
      index += HS_LIDAR_XT_UNIT_SIZE;
    }
  }

  index += HS_LIDAR_XT_RESERVED_SIZE; // skip reserved bytes

  packet->echo = recvbuf[index]& 0xff;

  index += HS_LIDAR_XT_ECHO_SIZE;
  index += HS_LIDAR_XT_ENGINE_VELOCITY;

  packet->addtime[0] = recvbuf[index]& 0xff;
  packet->addtime[1] = recvbuf[index+1]& 0xff;
  packet->addtime[2] = recvbuf[index+2]& 0xff;
  packet->addtime[3] = recvbuf[index+3]& 0xff;
  packet->addtime[4] = recvbuf[index+4]& 0xff;
  packet->addtime[5] = recvbuf[index+5]& 0xff;

  index += HS_LIDAR_XT_UTC_SIZE;

  packet->timestamp = (recvbuf[index] & 0xff)| (recvbuf[index+1] & 0xff) << 8 | \
      ((recvbuf[index+2] & 0xff) << 16) | ((recvbuf[index+3] & 0xff) << 24);
    // printf("timestamp %u \n", packet->timestamp);
    struct tm tTm = {0};
  // UTC's year only include 0 - 99 year , which indicate 2000 to 2099.
  // and mktime's year start from 1900 which is 0. so we need add 100 year.
  tTm.tm_year = packet->addtime[0] + 100;

  // in case of time error
  if (tTm.tm_year >= 200) {
    tTm.tm_year -= 100;
  }

  // UTC's month start from 1, but mktime only accept month from 0.
  tTm.tm_mon = packet->addtime[1] - 1;
  tTm.tm_mday = packet->addtime[2];
  tTm.tm_hour = packet->addtime[3];
  tTm.tm_min = packet->addtime[4];
  tTm.tm_sec = packet->addtime[5];
  tTm.tm_isdst = 0;
  packet->timestamp_point = mktime(&tTm) + static_cast<double>(packet->timestamp) / 1000000.0;
  return 0;
}

void PandarGeneral::CalcXTPointXYZIT(HS_LIDAR_XT_Packet *pkt, int blockid, \
    char chLaserNumber, PPointCloud::Ptr cld) {
  HS_LIDAR_XT_Block *block = &pkt->blocks[blockid];

  for (int i = 0; i < chLaserNumber; ++i) {
    /* for all the units in a block */
    HS_LIDAR_XT_Unit &unit = block->units[i];
    PPoint point;

    /* skip wrong points */
    if (unit.distance <= 0.1 || unit.distance > 200.0) {
      continue;
    }

    int azimuth = static_cast<int>(General_horizatal_azimuth_offset_map_[i] * 100 + block->azimuth);
    if(azimuth < 0)
      azimuth += 36000;
    if(azimuth >= 36000)
      azimuth -= 36000;
    if(m_bCoordinateCorrectionFlag){
      float distance = unit.distance - (m_cos_azimuth_map_h[abs(int(General_horizatal_azimuth_offset_map_[i] * 100))] * m_cos_elevation_map_[i] -
                      m_sin_azimuth_map_b[abs(int(General_horizatal_azimuth_offset_map_[i] * 100))] * m_cos_elevation_map_[i]);
      float xyDistance = distance * m_cos_elevation_map_[i];
      point.x = xyDistance * m_sin_azimuth_map_[azimuth] - m_cos_azimuth_map_b[azimuth] + m_sin_azimuth_map_h[azimuth];
      point.y = xyDistance * m_cos_azimuth_map_[azimuth] + m_sin_azimuth_map_b[azimuth] + m_cos_azimuth_map_h[azimuth];
      point.z = distance * m_sin_elevation_map_[i];

      if (COORDINATE_CORRECTION_CHECK){
        float xyDistance = unit.distance * m_cos_elevation_map_[i];
        float point_x = static_cast<float>(xyDistance * m_sin_azimuth_map_[azimuth]);
        float point_y = static_cast<float>(xyDistance * m_cos_azimuth_map_[azimuth]);
        float point_z = static_cast<float>(unit.distance * m_sin_elevation_map_[i]);
        printf("distance = %f; elevation = %f; azimuth = %f; delta X = %f; delta Y = %f; delta Z = %f; \n", 
              unit.distance, pandarGeneral_elev_angle_map[i], float(azimuth / 100), point.x - point_x, point.y - point_y, point.z - point_z);
      }
    }
    else{
      float xyDistance = unit.distance * m_cos_elevation_map_[i];
      point.x = static_cast<float>(xyDistance * m_sin_azimuth_map_[azimuth]);
      point.y = static_cast<float>(xyDistance * m_cos_azimuth_map_[azimuth]);
      point.z = static_cast<float>(unit.distance * m_sin_elevation_map_[i]);
    } 
    point.intensity = unit.intensity;

    {
      point.timestamp = pkt->timestamp_point + tz_second_;

      if (pkt->echo == 0x3d){
        point.timestamp =
            point.timestamp + (static_cast<double>(blockXTOffsetTriple_[blockid] +
                                                  laserXTOffset_[i]) /
                              1000000.0f);
      }
      else if (pkt->echo == 0x39 || pkt->echo == 0x3b || pkt->echo == 0x3c) {
        point.timestamp =
            point.timestamp + (static_cast<double>(blockXTOffsetDual_[blockid] +
                                                  laserXTOffset_[i]) /
                              1000000.0f);
      } else {
        point.timestamp = point.timestamp + \
            (static_cast<double>(blockXTOffsetSingle_[blockid] + laserXTOffset_[i]) / \
            1000000.0f);
      }
    }

    point.ring = i;
    if (pcl_type_) {
      PointCloudList[i].push_back(point);
    } else {
      PointCloud[iPointCloudIndex] = point;
      iPointCloudIndex++;
    }
  }
}

void PandarGeneral::EmitBackMessege(char chLaserNumber, PPointCloud::Ptr cld) {
  if (pcl_type_) {
    for (int i=0; i<chLaserNumber; i++) {
      for (int j=0; j<PointCloudList[i].size(); j++) {
        cld->push_back(PointCloudList[i][j]);
      }
    }
  }
  else{
    cld->points.assign(PointCloud.begin(),PointCloud.begin() + iPointCloudIndex);
    cld->width = (uint32_t)cld->points.size();
    cld->height = 1;
    iPointCloudIndex = 0;
  }
  pcl_callback_(cld, cld->points[0].timestamp);
  if (pcl_type_) {
    for (int i=0; i<chLaserNumber; i++) {
      PointCloudList[i].clear();
      PointCloudList[i].reserve(MAX_POINT_CLOUD_NUM_PER_CHANNEL);
      cld->points.clear();
      cld->width = (uint32_t)cld->points.size();
      cld->height = 1;
    }
  }
}
