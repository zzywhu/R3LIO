//
// Created by w on 2023/6/2.
//
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Vector3.h>
#include <glog/logging.h>
#include <malloc.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <scout_gazebo/LidarAngle.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include <sys/resource.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/opencv.hpp>

#include "ReadWriter.h"
#include "RigelSLAMRawIOTools.h"
#include "System.h"
#include "plog/Log.h"
#include "rigelslam_rot/motor.h"

#define OFFLINE_TEST 0
#define OUTPUT_LOG 0

namespace Hesai_ZG_ros {
// ZG-Hesai-16 Lidar
struct PointXYZIRT16 {
  float x;
  float y;
  float z;
  float intensity;
};

struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace Hesai_ZG_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(
    Hesai_ZG_ros::PointXYZIRT16,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity))

POINT_CLOUD_REGISTER_POINT_STRUCT(
    Hesai_ZG_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity))

bool _exitFlag = false;

std::string _mapTopic = "/rigelslam_rot/lidar/mapping/cloud_registered";
// std::string _odomTopic = "/rigelslam_rot/lidar/mapping/odometry";

std::string _preMapTopic = "/rigelslam_rot/lidar/mapping/cloud_premap";

std::string _odom2mapTopic = "/rigelslam_rot/lidar/mapping/odometry2map";

std::string _bevmatchTopic = "/rigelslam_rot/lidar/mapping/bevmatch";

System *_sys = nullptr;

nav_msgs::Path _path;
PointCloudXYZI::Ptr _featsFromMap(new PointCloudXYZI());
nav_msgs::Odometry _odomAftMapped;

Eigen::Vector3d _pntOnImu(-112.32 / 1000, 33.19 / 1000, -156.774 / 1000);

pcl::PointCloud<pcl::PointXYZI>::Ptr _controlMarkers;
std::vector<ControlPointInfoStru> _ctrlPointsInfos;
ros::Publisher _pubControlMarkers;
ros::Publisher _pubControlMarkerCaptured;
ros::Publisher _pubStopedNode;
ros::Publisher _pubLoopNode;

double localization_threshold = 0.2;  // 配准残差阈值

bool _bStopNode = false;
bool _enableCaptureControlMarker = false;
std::vector<pcl::PointXYZI> _ctrlMarkersForAverage;
std::vector<LoopPairInfoStru> _loopPairInfos;
double _ctrlMarkerBeginStamp = 0;
int _controlMarkerIndex = 1;
bool _bReadedPlayBackFile = false;
bool _isPlayBackTask = false;
bool _isOutputFeature = false;
std::string _playBackFilePath;
std::string _saveRawPath;
std::string _scanScene;
int _loopCount = 0;
bool _isContinueTask = false;
std::string _lastTaskPath;
std::string _saveDirectory;

std::deque<double>
    _ctrlPtTimestampFromPlayBack;  // 从任务文件读取到的控制点对应KF的ID

V3D _meanPos = V3D(0, 0, 0);
V3D _meanRot = V3D(0, 0, 0);
int _nTotal = 0;
int _lastMotorSeqID = -1;
int _lastIMUSeqID = -1;
int _lastLidSeqID = -1;
int _msgCount = 0;

double _premapdownsampleLeaf = 0.2;

std::thread *_releaseThread = nullptr;

double _coolCount = 200;
double _memUsage = 0;
double _meanFrameTime = 0;
bool _isMaxOptimized = false;

bool _islidarget = false;

PointCloudXYZI::Ptr _addedMapCloud(new PointCloudXYZI);

std::string _workspace;

std::vector<SensorMsgs::ImuData::Ptr> _imuDataList;

void sigHandle(int sig) {
  _exitFlag = true;
  std::printf("catch sig %d", sig);
  _sys->_sigBuffer.notify_all();
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(_sys->_stateIkfom.offset_R_L_I * p_body_lidar +
                 _sys->_stateIkfom.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void publishPreMap(const ros::Publisher &pubLaserCloudPreMap) {
  sensor_msgs::PointCloud2 laserCloudmsg;

  // downsmaple
  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(_sys->_premapPtr);
  sor.setLeafSize(0.2f, 0.2f, 0.2f);
  sor.filter(*downsampled);
  pcl::toROSMsg(*downsampled, laserCloudmsg);

  laserCloudmsg.header.stamp = ros::Time().fromSec(_sys->_lidarBegTime);
  laserCloudmsg.header.frame_id = "map";
  pubLaserCloudPreMap.publish(laserCloudmsg);
  // std::cout << "publish premap size: " << downsampled->points.size() <<
  // std::endl;
}

void mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  _sys->_mtxBuffer.lock();
  pcl::PointCloud<Hesai_ZG_ros::PointXYZIRT16>::Ptr oriCloud(
      new pcl::PointCloud<Hesai_ZG_ros::PointXYZIRT16>);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::fromROSMsg(*msg, *oriCloud);
  for (auto &p : oriCloud->points) {
    pcl::PointXYZINormal pp;
    pp.intensity = p.intensity;
    pp.x = p.x;
    pp.y = p.y;
    pp.z = p.z;
    cloud->push_back(pp);
  }
  _sys->_lidarBegTime = msg->header.stamp.toSec();
  _sys->_cloudBuffer.push_back(cloud);
  if (_sys->_cloudBuffer.size() > 100) {
    _sys->_cloudBuffer.pop_front();
  }
  _sys->_mtxBuffer.unlock();
  _sys->_sigBuffer.notify_all();
  //_islidarget = true;
}

void stopNodeHandler(const std_msgs::Int32ConstPtr &msgIn) {
  _exitFlag = true;  // 关闭节点
}

void setParams() {
  _sys->_config._imuFilePath = "";
  _sys->_config._rasterFilePath = "";
  _sys->_config._pcapFilePath = "";
  _sys->_config._lidarCorrectFilePath = "";
  _sys->_config._isEnable3DViewer = false;
  _sys->_config._isTimeSyncEn = false;
  _sys->_config._isMotorInitialized = true;
  _sys->_config._isImuInitialized = true;
  _sys->_config._nSkipFrames = 20;
  _sys->_config._enableGravityAlign = false;

  _sys->_config._minFramePoint = 1000;
  _sys->_config._isFeatExtractEn = false;

  LidarProcess::mutableConfig()._pointFilterNum = 1;
  LidarProcess::mutableConfig()._lidarType = 4;
  LidarProcess::mutableConfig()._nScans = 16;
  LidarProcess::mutableConfig()._timeUnit = 0;
  LidarProcess::mutableConfig()._scanRate = 10;

  _sys->_config._isCutFrame = false;
  _sys->_config._meanAccNorm = 9.805;
  _sys->_config._imuMaxInitCount = 1000;
  _sys->_config._gnssInitSize = 3;

  _sys->_config._udpateMethod = 0;
  _sys->_config._matchMethod = 1;
  _sys->_config._nMaxInterations = 5;
  _sys->_config._detRange = 120;
  _sys->_config._cubeLen = 2000;
  _sys->_config._filterSizeSurf = 0.1;
  _sys->_config._filterSizeMap = 0.2;
  _sys->_config._gyrCov = 0.001;
  _sys->_config._accCov = 0.1;
  _sys->_config._bAccCov = 0.0001;
  _sys->_config._bGyrCov = 0.0001;
  _sys->_config._timeLagIMUWtrLidar = 0;
  _sys->_config._isEstiExtrinsic = false;
  _sys->_config._isUseIntensity = false;
  _sys->_config._gnssMaxError = 5;

  double tilVec[] = {0, 0, 0};
  double RilVec[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  _sys->_config._tilVec = std::vector<double>(tilVec, tilVec + 3);
  _sys->_config._RilVec = std::vector<double>(RilVec, RilVec + 9);
  double tolVec[] = {0, 0, 0};
  double RolVec[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  _sys->_config._tolVec = std::vector<double>(tolVec, tolVec + 3);
  _sys->_config._RolVec = std::vector<double>(RolVec, RolVec + 9);
  double tigVec[] = {-0.11232, 0.03319, -0.156774};
  double RigVec[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  _sys->_config._tigVec = std::vector<double>(tigVec, tigVec + 3);
  _sys->_config._RigVec = std::vector<double>(RigVec, RigVec + 9);

  _sys->_config._maxPointsSize = 400;
  _sys->_config._maxCovPointsSize = 400;
  for (int i = 0; i < 4; i++) _sys->_config._layerPointSizeList.emplace_back(5);

  _sys->_config._maxLayers = 2;
  _sys->_config._voxelLength = 0.5;
  _sys->_config._minSurfEigenValue = 0.005;
  _sys->_config._rangingCov = 0.08;
  _sys->_config._angleCov = 0.2;
  _sys->_config._covType = 1;

  _sys->_config._isEnableBA = false;

  _sys->_config._isLoopEn = true;
  LoopCloser::mutableConfig()._poseGraphOptimizationMethod = "ceres";
  LoopCloser::mutableConfig()._coolingSubmapNum = 2;
  LoopCloser::mutableConfig()._numFrameLargeDrift = 6000;
  LoopCloser::mutableConfig()._minSubmapIdDiff = 30;
  LoopCloser::mutableConfig()._maxSubMapAccuTran = 30;
  LoopCloser::mutableConfig()._maxSubMapAccuRot = 90;
  LoopCloser::mutableConfig()._maxSubMapAccuFrames = 25;
  LoopCloser::mutableConfig()._normalRadiusRatio = 8;

  LoopCloser::mutableConfig()._voxelDownSize = 0.3;
  LoopCloser::mutableConfig()._vgicpVoxRes = 0.5;
  LoopCloser::mutableConfig()._isFramewisePGO = true;
  LoopCloser::mutableConfig()._isPgoIncremental = false;
  LoopCloser::mutableConfig()._isStoreFrameBlock = true;
  LoopCloser::mutableConfig()._frameStoreInterval = 1;
  LoopCloser::mutableConfig()._isCorrectRealTime = true;
  LoopCloser::mutableConfig()._isLoopDetectEn = true;
  LoopCloser::mutableConfig()._isOutputPCD = false;
  LoopCloser::mutableConfig()._frontendWinSize = 100;

  LoopCloser::mutableConfig()._voxelLength = _sys->config()._voxelLength;
  LoopCloser::mutableConfig()._maxLayers = _sys->config()._maxLayers;
  LoopCloser::mutableConfig()._layerPointSizeList =
      _sys->config()._layerPointSizeList;
  LoopCloser::mutableConfig()._maxPointsSize = _sys->config()._maxPointsSize;
  LoopCloser::mutableConfig()._maxCovPointsSize =
      _sys->config()._maxCovPointsSize;
  LoopCloser::mutableConfig()._minEigenValue =
      _sys->config()._minSurfEigenValue;
  LoopCloser::mutableConfig()._gnssMaxError = _sys->config()._gnssMaxError;

  _sys->_config._isSaveMap = false;
}

void loadRosParams(ros::NodeHandle &nh) {
  nh.param<double>("rigelslam_rot/minRange",
                   LidarProcess::mutableConfig()._blindMin, 0);
  nh.param<double>("rigelslam_rot/ridus_k", _sys->_config._radius_k, 3);
  nh.param<double>("rigelslam_rot/maxRange",
                   LidarProcess::mutableConfig()._blindMax, 0);
  nh.param<int>("rigelslam_rot/scanLines",
                LidarProcess::mutableConfig()._nScans, 16);
  nh.param<int>("rigelslam_rot/cov_type", _sys->_config._covType, 1);
  nh.param<std::string>("rigelslam_rot/map_path", _sys->_map_path, "");
  nh.param<int>("rigelslam_rot/updatemethod", _sys->_config._udpateMethod, 0);
  nh.param<std::string>("rigelslam_rot/detector_config_path",
                        LoopCloser::mutableConfig()._detectorConfigPath, "");
  nh.param<bool>("rigelslam_rot/isPlayBackTask", _isPlayBackTask, false);
  nh.param<bool>("rigelslam_rot/GravityAlign",
                 _sys->_config._enableGravityAlign, false);
  nh.param<bool>("rigelslam_rot/isContinueTask", _isContinueTask, false);
  nh.param<bool>("rigelslam_rot/isOutputFeature", _isOutputFeature, false);
  nh.param<bool>("rigelslam_rot/isOutputIMU",
                 _sys->_imuProcessor->_bOutpoutImuInfo, false);
  nh.param<std::string>("rigelslam_rot/playBackFileName", _playBackFilePath,
                        "/tmp/rigelslam_s_playBack-01.bin");
  nh.param<std::string>("rigelslam_rot/saveDirectory", _saveDirectory, "");

  nh.param<double>("bev/gicpthreshold", localization_threshold, 0.2);

  std::string extrinsicFilePath;
  nh.param<std::string>(
      "rigelslam_rot/equipmentParamsFile", extrinsicFilePath,
      "/home/w/code/fast_lio_win/src/config/zg_equipment_param.txt");
  nh.param<std::string>("rigelslam_rot/saveRawPath", _saveRawPath, "/tmp/");
  nh.param<std::string>("rigelslam_rot/scanSceneName", _scanScene, "");
  nh.param<bool>("rigelslam_rot/usemutiview", _sys->_config._isUseMultiview,
                 false);
  nh.param<bool>("rigelslam_rot/useintensity", _sys->_config._isUseIntensity,
                 false);
  nh.param<bool>("rigelslam_rot/loopClosureEnableFlag", _sys->_config._isLoopEn,
                 false);
  nh.param<bool>("rigelslam_rot/saveMap", _sys->_config._issavemap, false);
  LoopCloser::mutableConfig()._issavemap = _sys->_config._issavemap;
  nh.param<bool>("rigelslam_rot/loopcorrect",
                 LoopCloser::mutableConfig()._isCorrectRealTime, false);
  _lastTaskPath = _saveDirectory + "/lastTaskInfo.bin";

  _sys->_cloudAxisTransfer = new ZGAxisTransfer();
  _sys->_cloudAxisTransfer->ReadEquipmentParams(extrinsicFilePath.c_str());
  _sys->_cloudAxisTransfer->readZGExtrinsic(extrinsicFilePath.c_str());

  if (_scanScene == "Street" || _scanScene == "OpenCountry" ||
      _scanScene == "Terrian" || _scanScene == "Building" ||
      _scanScene == "forest")
    _sys->mutableConfig()._matchMethod = 0;
  else
    _sys->mutableConfig()._matchMethod = 1;
  if (_scanScene == "Indoor" || _scanScene == "UndergroundPark" ||
      _scanScene == "Stair" || _scanScene == "Tunnel" ||
      _scanScene == "Windrow" || _scanScene == "SteelStructure" ||
      _scanScene == "Factory") {
    LoopCloser::mutableConfig()._voxelDownSize = 0.2;
    LoopCloser::mutableConfig()._detectLoopMethod = 0;

    if (_scanScene == "Windrow" || _scanScene == "Stair")
      LoopCloser::mutableConfig()._neighborSearchDist = 3;
    if (_scanScene == "Indoor" || _scanScene == "UndergroundPark" ||
        _scanScene == "SteelStructure" || _scanScene == "Factory" ||
        _scanScene == "Tunnel")
      LoopCloser::mutableConfig()._neighborSearchDist = 5;
  } else
    LoopCloser::mutableConfig()._detectLoopMethod = 1;
}

int main(int argc, char **argv) {
  google::InitGoogleLogging("XXX");
  google::SetCommandLineOption("GLOG_minloglevel", "2");

  // plog::init(plog::debug,
  // "/home/zzy/RigelSLAM_Rot/devel/logs/mapOptimization.txt", 64 * 1024 * 1024,
  // 2);

  ros::init(argc, argv, "ZGSlamRosTest");
  ros::NodeHandle _nh;

  _sys = new System();
  setParams();
  loadRosParams(_nh);

  _sys->projectormap = std::make_shared<BEVProjector>(_nh);
  _sys->projectorcurrent = std::make_shared<BEVProjector>(_nh);

  _sys->bev_manager =
      std::make_shared<BEVFeatureManager>(_nh, _sys->projectormap);

  if (!_sys->initSystem()) {
    std::cerr << "System initializing failed!" << std::endl;
    return EXIT_FAILURE;
  }

  ros::Subscriber subMap = _nh.subscribe(_mapTopic, 2000, mapCallback);
  ros::Publisher pubPreMap =
      _nh.advertise<sensor_msgs::PointCloud2>(_preMapTopic, 10);
  ros::Publisher pubOdom2Map =
      _nh.advertise<nav_msgs::Odometry>(_odom2mapTopic, 10);
  ros::Publisher pubbevmatch =
      _nh.advertise<sensor_msgs::Image>(_bevmatchTopic, 1, true);

  std::cout << "Subscribed to: " << _mapTopic << std::endl;

  std::cout << "Publishing to: " << _odom2mapTopic << std::endl;

  //预先加载地图
  pcl::io::loadPCDFile(_sys->_map_path, *_sys->_premapPtr);
  std::cout << "PreMap Loaded, point size: " << _sys->_premapPtr->points.size()
            << std::endl;
  //发布预先加载的地图
  sensor_msgs::PointCloud2 laserCloudmsg;

  // downsmaple
  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(_sys->_premapPtr);
  sor.setLeafSize(0.5f, 0.5f, 0.5f);
  sor.filter(*downsampled);
  pcl::toROSMsg(*downsampled, laserCloudmsg);

  laserCloudmsg.header.stamp = ros::Time().fromSec(_sys->_lidarBegTime);
  laserCloudmsg.header.frame_id = "map";
  pubPreMap.publish(laserCloudmsg);
  std::cout << "Publishing to: " << _preMapTopic << std::endl;

  ///////////////online mode///////////////////
  signal(SIGINT, sigHandle);
  bool status = ros::ok();

  // 在main函数中定义局部的map_frame变量，替代_sys->map_frame
  BEVFrame local_map_frame;

  //制作premap的BEV
  std_msgs::Header header;
  header.stamp.fromSec(_sys->_lidarBegTime);
  header.frame_id = "map";
  local_map_frame = BEVFrame();
  local_map_frame.points = _sys->_premapPtr;
  local_map_frame.header = header;
  _sys->projectormap->getMapBEV(local_map_frame);

  std::cout << "PreMap BEV Generated." << std::endl;
  cv::imwrite(std::string(ROOT_DIR) + "BEV/database/Map_BEV.png",
              local_map_frame.img_dense);

  // 对premap进行降采样 - 只在初始化时执行一次
  pcl::PointCloud<pcl::PointXYZI>::Ptr premap_downsampled(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> premap_voxel_grid;
  premap_voxel_grid.setInputCloud(_sys->_premapPtr);
  premap_voxel_grid.setLeafSize(_premapdownsampleLeaf, _premapdownsampleLeaf,
                                _premapdownsampleLeaf);
  premap_voxel_grid.filter(*premap_downsampled);
  std::cout << "PreMap downsampled: " << _sys->_premapPtr->points.size()
            << " -> " << premap_downsampled->points.size() << " points"
            << std::endl;

  pcl::PointCloud<pcl::PointXYZI>::Ptr bev_downsampled(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::VoxelGrid<pcl::PointXYZI> bev_voxel_grid;

  // 定义配准相关变量
  bool is_localized = false;
  bool is_fully_localized = false;  // 新增：彻底定位成功标志
  Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();
  nav_msgs::Odometry final_odom;

  // 延迟特征提取 - 不在程序开始时提取，避免段错误
  // 在需要匹配时再提取地图特征

  // tf broadcaster
  static tf::TransformBroadcaster tf_broadcaster;

  while (status) {
    if (_exitFlag) break;
    ros::spinOnce();

    if (!_sys->_cloudBuffer.empty()) {
      // std::cout << "1: " << std::endl;

      _sys->_mapBuffer.push_back(_sys->_cloudBuffer.front());
      _sys->_cloudBuffer.pop_front();

      _sys->_frameId++;
      std::cout << "Frame id: " << _sys->_frameId << std::endl;

      // 还未“彻底定位”时仍继续计算；但一旦 hasConverged 过，就开始发布定位结果
      if (!is_fully_localized) {
        if (!_sys->_mapBuffer.empty() && _sys->_frameId % 50 == 0) {
          std::cout << "Processing point clouds for localization..."
                    << std::endl;

          // 1. 合并所有点云
          pcl::PointCloud<pcl::PointXYZINormal>::Ptr merged_cloud(
              new pcl::PointCloud<pcl::PointXYZINormal>);
          for (auto &cloud : _sys->_mapBuffer) {
            *merged_cloud += *cloud;
          }
          std::cout << "Merged cloud size: " << merged_cloud->points.size()
                    << std::endl;

          if (merged_cloud->points.size() >= 1000) {
            // 2. 转换为BEV格式的点云
            pcl::PointCloud<pcl::PointXYZI>::Ptr bev_cloud(
                new pcl::PointCloud<pcl::PointXYZI>);
            for (const auto &p : merged_cloud->points) {
              pcl::PointXYZI pt;
              pt.x = p.x;
              pt.y = p.y;
              pt.z = p.z;
              pt.intensity = p.intensity;
              bev_cloud->push_back(pt);
            }

            // 定义角度搜索参数
            const double angle_step_deg = 30.0;  // 每次旋转30度
            const double max_angle_deg = 180.0;

            // 遍历不同旋转角度进行匹配
            for (double angle_deg = -max_angle_deg; angle_deg <= max_angle_deg;
                 angle_deg += angle_step_deg) {
              std::cout << "\n--- Trying localization with rotation: "
                        << angle_deg << " degrees ---" << std::endl;

              // a. 对当前帧点云进行旋转
              double angle_rad = angle_deg * M_PI / 180.0;
              Eigen::Matrix4f rotation_transform = Eigen::Matrix4f::Identity();
              rotation_transform(0, 0) = std::cos(angle_rad);
              rotation_transform(0, 1) = -std::sin(angle_rad);
              rotation_transform(1, 0) = std::sin(angle_rad);
              rotation_transform(1, 1) = std::cos(angle_rad);

              pcl::PointCloud<pcl::PointXYZI>::Ptr rotated_bev_cloud(
                  new pcl::PointCloud<pcl::PointXYZI>);
              pcl::transformPointCloud(*bev_cloud, *rotated_bev_cloud,
                                       rotation_transform);

              // b. 构建当前帧的BEV
              std_msgs::Header current_header;
              current_header.stamp = ros::Time().fromSec(_sys->_lidarBegTime);
              current_header.frame_id = "odom";

              BEVFrame current_frame;
              current_frame.points = rotated_bev_cloud;
              current_frame.header = current_header;
              _sys->projectorcurrent->getMapBEV(current_frame);

              // ... (后续的XFeat匹配, RANSAC, GICP等逻辑保持不变) ...
              // 4. 使用系统中的 XFeat 进行特征匹配
              std::cout << "Starting XFeat matching with system detector..."
                        << std::endl;
              cv::Mat mkpts_0, mkpts_1;
              _sys->_XFDetector.match_xfeat(current_frame.img_dense,
                                            local_map_frame.img_dense, mkpts_0,
                                            mkpts_1);
              std::cout << "XFeat found " << mkpts_0.rows << " initial matches"
                        << std::endl;

              // 5. 如果有足够的匹配点，进行RANSAC粗差剔除
              if (mkpts_0.rows > 10) {
                std::cout << "Processing " << mkpts_0.rows
                          << " XFeat matches with RANSAC..." << std::endl;

                // 转换为 vector<Point2f> 格式，按照示例的方式
                std::vector<cv::Point2f> pts_current, pts_map;
                for (int i = 0; i < mkpts_0.rows; ++i) {
                  pts_current.push_back(mkpts_0.at<cv::Point2f>(i, 0));
                  pts_map.push_back(mkpts_1.at<cv::Point2f>(i, 0));
                }

                // 使用 RANSAC 估计 Homography（按照示例方式）
                cv::Mat mask;
                cv::Mat H = cv::findHomography(pts_current, pts_map, cv::RANSAC,
                                               20.0, mask, 100, 0.99);

                if (!H.empty()) {
                  // 计算内点数量
                  mask = mask.reshape(1);
                  int num_inliers = cv::countNonZero(mask);
                  std::cout << "RANSAC result: " << num_inliers << "/"
                            << pts_current.size() << " inliers" << std::endl;

                  // 6. 创建经过RANSAC筛选的匹配可视化
                  cv::Mat img_matches;
                  if (num_inliers > 8) {  // 需要足够的内点才进行可视化
                    // 按照示例代码的方式准备keypoints和matches，只包含内点
                    std::vector<cv::KeyPoint> keypoints_current, keypoints_map;
                    std::vector<cv::DMatch> good_matches;

                    int match_idx = 0;
                    for (int i = 0; i < mkpts_0.rows; ++i) {
                      if (mask.at<uchar>(i, 0)) {  // 只处理内点
                        cv::Point2f pt_current = mkpts_0.at<cv::Point2f>(i, 0);
                        cv::Point2f pt_map = mkpts_1.at<cv::Point2f>(i, 0);

                        keypoints_current.emplace_back(pt_current, 5);
                        keypoints_map.emplace_back(pt_map, 5);
                        good_matches.emplace_back(match_idx, match_idx, 0);
                        match_idx++;
                      }
                    }

                    // 使用drawMatches函数，按照示例的方式绘制内点匹配
                    if (!keypoints_current.empty() && !keypoints_map.empty() &&
                        !good_matches.empty()) {
                      cv::drawMatches(
                          current_frame.img_dense, keypoints_current,
                          local_map_frame.img_dense, keypoints_map,
                          good_matches, img_matches,
                          cv::Scalar(0, 255, 0),  // 绿色表示good matches
                          cv::Scalar::all(-1), std::vector<char>(),
                          cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                      // 在图像上添加文本信息
                      std::string info_text =
                          "Inliers: " + std::to_string(num_inliers) + "/" +
                          std::to_string(mkpts_0.rows);
                      cv::putText(img_matches, info_text, cv::Point(10, 30),
                                  cv::FONT_HERSHEY_SIMPLEX, 1,
                                  cv::Scalar(0, 255, 0), 2);

                      std::cout << "Created visualization with "
                                << good_matches.size() << " inlier matches"
                                << std::endl;
                    }
                  } else {
                    std::cout << "Not enough inliers for visualization: "
                              << num_inliers << std::endl;

                    // 如果内点太少，显示所有匹配但用不同颜色标识
                    std::vector<cv::KeyPoint> keypoints_current, keypoints_map;
                    std::vector<cv::DMatch> all_matches;

                    for (int i = 0; i < mkpts_0.rows; ++i) {
                      cv::Point2f pt_current = mkpts_0.at<cv::Point2f>(i, 0);
                      cv::Point2f pt_map = mkpts_1.at<cv::Point2f>(i, 0);

                      keypoints_current.emplace_back(pt_current, 5);
                      keypoints_map.emplace_back(pt_map, 5);
                      all_matches.emplace_back(i, i, 0);
                    }

                    cv::drawMatches(
                        current_frame.img_dense, keypoints_current,
                        local_map_frame.img_dense, keypoints_map, all_matches,
                        img_matches,
                        cv::Scalar(0, 0, 255),  // 红色表示质量较差的匹配
                        cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                    std::string warning_text =
                        "Poor matches: " + std::to_string(num_inliers) + "/" +
                        std::to_string(mkpts_0.rows);
                    cv::putText(img_matches, warning_text, cv::Point(10, 30),
                                cv::FONT_HERSHEY_SIMPLEX, 1,
                                cv::Scalar(0, 0, 255), 2);
                  }

                  // 发布经过RANSAC处理的匹配图像
                  if (!img_matches.empty()) {
                    sensor_msgs::ImagePtr match_img_msg =
                        cv_bridge::CvImage(current_header, "bgr8", img_matches)
                            .toImageMsg();
                    pubbevmatch.publish(match_img_msg);
                    std::cout << "Published RANSAC-filtered match image with "
                              << num_inliers << " inliers" << std::endl;
                  }

                  // //打印2d点对
                  // for (int i = 0; i < pts_current.size(); i++) {
                  //     std::cout << "Match " << i << ": Current(" <<
                  //     pts_current[i].x << ", " << pts_current[i].y
                  //               << ") <-> Map(" << pts_map[i].x << ", " <<
                  //               pts_map[i].y << ")" << std::endl;
                  // }

                  // 7. 继续后续的定位处理（只有在有足够内点时）
                  if (num_inliers > 8) {
                    // 使用2D刚性变换(基于RANSAC的部分仿射)估计初值
                    // cv::Mat inliers_rigid;
                    // cv::Mat A = cv::estimateAffinePartial2D(
                    //     pts_current, pts_map, inliers_rigid,
                    //     cv::RANSAC, // 鲁棒估计
                    //     1.0,        // 像素阈值
                    //     2000,       // 最大迭代次数
                    //     0.99        // 置信度
                    // );
                    // 仅使用上一阶段 Homography RANSAC 的内点再次估计 2D
                    // 刚性初值
                    std::vector<cv::Point2f> inlier_curr, inlier_map;
                    inlier_curr.reserve(num_inliers);
                    inlier_map.reserve(num_inliers);
                    for (int i = 0; i < mkpts_0.rows; ++i) {
                      if (mask.at<uchar>(i, 0)) {
                        inlier_curr.push_back(
                            _sys->projectorcurrent->backProjectBEVPixelToXY(
                                mkpts_0.at<cv::Point2f>(i, 0)));
                        inlier_map.push_back(
                            _sys->projectormap->backProjectBEVPixelToXY(
                                mkpts_1.at<cv::Point2f>(i, 0)));
                      }
                    }
                    //打印所有内点点对
                    for (int i = 0; i < inlier_curr.size(); i++) {
                      std::cout << "Inlier " << i << ": Current("
                                << inlier_curr[i].x << ", " << inlier_curr[i].y
                                << ") <-> Map(" << inlier_map[i].x << ", "
                                << inlier_map[i].y << ")" << std::endl;
                    }
                    if (inlier_curr.size() < 3) {
                      std::cout << "Inlier pairs < 3, skip rigid init."
                                << std::endl;
                      continue;
                    }
                    cv::Mat inliers_rigid;
                    cv::Mat A = cv::estimateAffinePartial2D(
                        inlier_curr, inlier_map, inliers_rigid, cv::RANSAC,
                        1.0,  // 像素阈值(可调 1~3)
                        2000, 0.99);
                    if (A.empty()) {
                      std::cout
                          << "Failed to estimate 2D rigid transform with RANSAC"
                          << std::endl;
                      continue;
                      // 可直接 return/continue，或让后续逻辑自行判断
                    }

                    // 从2×3矩阵中提取旋转和平移，并去掉尺度成为刚性
                    double a00 = A.at<double>(0, 0), a01 = A.at<double>(0, 1),
                           a02 = A.at<double>(0, 2);
                    double a10 = A.at<double>(1, 0), a11 = A.at<double>(1, 1),
                           a12 = A.at<double>(1, 2);

                    // 估计公共尺度并归一化为纯旋转
                    double s = std::sqrt(a00 * a00 + a10 * a10);
                    std::cout << "Estimated scale from 2D rigid: " << s
                              << std::endl;
                    float yaw = 0.f;
                    if (s > 1e-8) {
                      double r00 = a00 / s;
                      double r10 = a10 / s;
                      yaw = static_cast<float>(std::atan2(r10, r00));
                    }

                    // 像素到米：乘以 BEV 分辨率
                    float dx = static_cast<float>(a02);
                    float dy = static_cast<float>(a12);

                    std::cout << "Initial estimate from rigid 2D (inliers "
                                 "only) - dx: "
                              << dx << ", dy: " << dy << ", yaw: " << yaw
                              << std::endl;

                    // 8. 构建初始变换矩阵 (这是从旋转后的帧到地图的变换)
                    Eigen::Matrix4f initial_transform_from_rotated =
                        Eigen::Matrix4f::Identity();
                    initial_transform_from_rotated(0, 0) = std::cos(yaw);
                    initial_transform_from_rotated(0, 1) = -std::sin(yaw);
                    initial_transform_from_rotated(1, 0) = std::sin(yaw);
                    initial_transform_from_rotated(1, 1) = std::cos(yaw);
                    initial_transform_from_rotated(0, 3) = dx;
                    initial_transform_from_rotated(1, 3) = dy;

                    // 关键修正：计算从原始帧到地图的真正初始变换
                    Eigen::Matrix4f initial_transform =
                        initial_transform_from_rotated * rotation_transform;

                    // 9. 使用GICP进行精细配准（后续代码保持不变）
                    std::cout << "Starting GICP refinement..." << std::endl;

                    // downsmaple bev_cloud

                    bev_voxel_grid.setInputCloud(bev_cloud);
                    bev_voxel_grid.setLeafSize(_premapdownsampleLeaf,
                                               _premapdownsampleLeaf,
                                               _premapdownsampleLeaf);
                    bev_voxel_grid.filter(*bev_downsampled);
                    // bev_cloud = bev_downsampled;
                    std::cout
                        << "BEV cloud downsampled: " << bev_cloud->points.size()
                        << " points" << std::endl;
                    // ... GICP配准代码保持不变 ...
                    // pcl::PointCloud<pcl::PointXYZI>::Ptr
                    // bev_cloud_aligned(new pcl::PointCloud<pcl::PointXYZI>);
                    // pcl::transformPointCloud(*bev_downsampled,
                    // *bev_cloud_aligned, initial_transform);
                    // pcl::io::savePCDFileBinary(std::string(ROOT_DIR) +
                    // "BEV/query/current_bev.pcd", *bev_cloud_aligned);

                    // 2. GICP输入变为变换后的点云，初始变换用单位阵
                    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI,
                                                          pcl::PointXYZI>
                        gicp;
                    gicp.setInputSource(bev_downsampled);
                    gicp.setInputTarget(premap_downsampled);

                    gicp.setMaximumIterations(50);
                    gicp.setTransformationEpsilon(1e-6);
                    gicp.setEuclideanFitnessEpsilon(1e-6);
                    gicp.setMaxCorrespondenceDistance(2.0);
                    gicp.setRANSACOutlierRejectionThreshold(0.5);
                    // gicp.setRANSACOutlierRejectionThreshold(0.5);

                    // 执行配准
                    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(
                        new pcl::PointCloud<pcl::PointXYZI>);
                    auto start_time = std::chrono::high_resolution_clock::now();
                    gicp.align(*aligned_cloud,
                               initial_transform);  // 使用修正后的初值
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time);

                    std::cout << "GICP alignment completed in "
                              << duration.count() << " ms" << std::endl;

                    if (gicp.hasConverged()) {
                      double fitness_score = gicp.getFitnessScore();
                      std::cout << "GICP converged with fitness score: "
                                << fitness_score << std::endl;

                      // GICP的结果已经是最终的 map <- odom 变换，无需再乘
                      // rotation_transform
                      final_transform = gicp.getFinalTransformation();
                      is_localized = true;

                      // 构建里程计消息（每次收敛都更新一次）
                      final_odom.header.stamp = ros::Time(_sys->_lidarBegTime);
                      final_odom.header.frame_id = "map";
                      final_odom.child_frame_id = "odom";

                      final_odom.pose.pose.position.x = final_transform(0, 3);
                      final_odom.pose.pose.position.y = final_transform(1, 3);
                      final_odom.pose.pose.position.z = final_transform(2, 3);

                      Eigen::Matrix3f rotation =
                          final_transform.block<3, 3>(0, 0);
                      Eigen::Quaternionf quat(rotation);
                      quat.normalize();

                      final_odom.pose.pose.orientation.x = quat.x();
                      final_odom.pose.pose.orientation.y = quat.y();
                      final_odom.pose.pose.orientation.z = quat.z();
                      final_odom.pose.pose.orientation.w = quat.w();

                      // 当fitness低于阈值，认为彻底定位成功，不再计算，只发布
                      if (fitness_score < localization_threshold) {
                        is_fully_localized = true;
                        std::cout
                            << "Fully localized. Stop further GICP computation."
                            << std::endl;
                      } else {
                        std::cout << "Partially localized. Continue refining..."
                                  << std::endl;
                      }
                    }

                  } else {
                    std::cout << "Not enough RANSAC inliers for localization: "
                              << num_inliers << std::endl;
                  }
                } else {
                  std::cout
                      << "Failed to estimate Homography with XFeat matches"
                      << std::endl;
                }
              } else {
                std::cout << "Not enough XFeat matches for RANSAC: "
                          << mkpts_0.rows << std::endl;
              }
              // 如果已经完全定位，则跳出角度搜索循环
              if (is_fully_localized) {
                break;
              }
            }  // 角度搜索循环结束

          } else {
            std::cout << "Not enough points for localization, continuing..."
                      << std::endl;
          }
        }
        std::cout << "Localization status: "
                  << (is_fully_localized ? "Fully localized"
                                         : (is_localized ? "Partially localized"
                                                         : "Not localized"))
                  << std::endl;

        // 未彻底定位时：若已定位则发布定位结果；否则发布单位阵
        if (is_localized) {
          final_odom.header.stamp = ros::Time().fromSec(_sys->_lidarBegTime);
          pubOdom2Map.publish(final_odom);
          std::cout << "Published odometry message." << std::endl;

          tf::Transform transform;
          transform.setOrigin(tf::Vector3(final_odom.pose.pose.position.x,
                                          final_odom.pose.pose.position.y,
                                          final_odom.pose.pose.position.z));
          tf::Quaternion q(final_odom.pose.pose.orientation.x,
                           final_odom.pose.pose.orientation.y,
                           final_odom.pose.pose.orientation.z,
                           final_odom.pose.pose.orientation.w);
          transform.setRotation(q);
          tf_broadcaster.sendTransform(tf::StampedTransform(
              transform, ros::Time().fromSec(_sys->_lidarBegTime), "map",
              "odom"));
        } else {
          tf::Transform identity_transform;
          identity_transform.setIdentity();
          tf_broadcaster.sendTransform(tf::StampedTransform(
              identity_transform, ros::Time().fromSec(_sys->_lidarBegTime),
              "map", "odom"));
        }
      } else {
        // 已彻底定位：不再计算，仅发布最终结果
        std::cout << "Already fully localized. Publishing final odometry only."
                  << std::endl;
        final_odom.header.stamp = ros::Time().fromSec(_sys->_lidarBegTime);
        pubOdom2Map.publish(final_odom);

        tf::Transform transform;
        transform.setOrigin(tf::Vector3(final_odom.pose.pose.position.x,
                                        final_odom.pose.pose.position.y,
                                        final_odom.pose.pose.position.z));
        tf::Quaternion q(final_odom.pose.pose.orientation.x,
                         final_odom.pose.pose.orientation.y,
                         final_odom.pose.pose.orientation.z,
                         final_odom.pose.pose.orientation.w);
        transform.setRotation(q);
        tf_broadcaster.sendTransform(tf::StampedTransform(
            transform, ros::Time().fromSec(_sys->_lidarBegTime), "map",
            "odom"));
      }
    }
  }

  ros::shutdown();
  return EXIT_SUCCESS;
}
