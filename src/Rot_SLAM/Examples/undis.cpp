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
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
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
  double time;
  float x;
  float y;
  float z;
  uint16_t intensity;
  uint16_t ring;
};

struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace Hesai_ZG_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(Hesai_ZG_ros::PointXYZIRT16,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      uint16_t, intensity,
                                      intensity)(double, time,
                                                 time)(uint16_t, ring, ring))

POINT_CLOUD_REGISTER_POINT_STRUCT(Hesai_ZG_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      float, intensity,
                                      intensity)(float, time, time)(uint16_t,
                                                                    ring, ring))

bool _exitFlag = false;
std::string _lidTopic = "/ZG/lidar";
std::string _lineTopic = "/rigelslam_rot/lidar/mapping/cloudline";
std::string _imuTopic = "/ZG/IMU_data";
std::string _motorTopic = "/ZG/Motor";
std::string _imgTopic = "/rigelslam_rot/intensity_img";
std::string _imgTopic_left = "/rigelslam_rot/intensity_img_left";
std::string _imgTopic_right = "/rigelslam_rot/intensity_img_right";
std::string _stopTopic = "/ZG/StopSLAMNode";
std::string _endTopic = "/ZG/SLAMNodeStoped";
std::string _errorTopic = "/errorCase";
std::string _ctrlMarkerDispTopic =
    "/rigelslam_rot/lidar/feature/control_makers";
std::string _ctrlMarkerGetTopic = "/CaptureControlMarker";
std::string _ctrlMarkerPubTopic = "/ControlMarkerCaptured";
std::string _loopTopic =
    "/rigelslam_rot/lidar/mapping/loop_closure_constraints";

std::string _mapTopic = "/rigelslam_rot/lidar/mapping/cloud_registered";
std::string _pathTopic = "/rigelslam_rot/lidar/mapping/path";
std::string _frameTopic = "/rigelslam_rot/lidar/deskew/cloud_deskewed";
std::string _odomTopic = "/rigelslam_rot/lidar/mapping/odometry";

System* _sys = nullptr;

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

std::thread* _releaseThread = nullptr;

double _coolCount = 200;
double _memUsage = 0;
double _meanFrameTime = 0;
bool _isMaxOptimized = false;

PointCloudXYZI::Ptr _addedMapCloud(new PointCloudXYZI);

std::string _workspace;

std::vector<SensorMsgs::ImuData::Ptr> _imuDataList;

void sigHandle(int sig) {
  _exitFlag = true;
  std::printf("catch sig %d", sig);
  _sys->_sigBuffer.notify_all();
}

void RGBpointBodyLidarToIMU(PointType const* const pi, PointType* const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(_sys->_stateIkfom.offset_R_L_I * p_body_lidar +
                 _sys->_stateIkfom.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void publishImage(const ros::Publisher& pubImage) {
  // 确保图像有效
  if (_sys->_matchImg.empty()) {
    ROS_ERROR("Input image is empty!");
    return;
  }
  // 根据图像通道数选择编码格式
  std::string encoding;
  if (_sys->_matchImg.type() == CV_8UC1) {
    encoding = "mono8";  // 单通道灰度图
  } else if (_sys->_matchImg.type() == CV_8UC3) {
    encoding = "bgr8";  // 三通道彩色图
  } else {
    ROS_ERROR("Unsupported image type: %d", _sys->_matchImg.type());
    return;
  }
  // 创建 sensor_msgs::Image 消息
  sensor_msgs::ImagePtr img_msg =
      cv_bridge::CvImage(std_msgs::Header(), encoding, _sys->_matchImg)
          .toImageMsg();

  // 创建 Image 消息
  // sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),
  // "mono8", _sys->_intensityImg).toImageMsg();

  // 设置时间戳
  img_msg->header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);

  // 设置坐标系 ID
  img_msg->header.frame_id = "odom";

  // 发布图像
  pubImage.publish(img_msg);
}
void publishImage_left(const ros::Publisher& pubImage) {
  // 确保图像有效
  if (_sys->_matchImg.empty()) {
    ROS_ERROR("Input image is empty!");
    return;
  }
  // 根据图像通道数选择编码格式
  std::string encoding;
  if (_sys->_matchImg.type() == CV_8UC1) {
    encoding = "mono8";  // 单通道灰度图
  } else if (_sys->_matchImg.type() == CV_8UC3) {
    encoding = "bgr8";  // 三通道彩色图
  } else {
    ROS_ERROR("Unsupported image type: %d", _sys->_matchImg.type());
    return;
  }
  // 创建 sensor_msgs::Image 消息
  sensor_msgs::ImagePtr img_msg =
      cv_bridge::CvImage(std_msgs::Header(), encoding, _sys->_matchImg_left)
          .toImageMsg();

  // 创建 Image 消息
  // sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),
  // "mono8", _sys->_intensityImg).toImageMsg();

  // 设置时间戳
  img_msg->header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);

  // 设置坐标系 ID
  img_msg->header.frame_id = "odom";

  // 发布图像
  pubImage.publish(img_msg);
}
void publishImage_right(const ros::Publisher& pubImage) {
  // 确保图像有效
  if (_sys->_matchImg_right.empty()) {
    ROS_ERROR("Input image is empty!");
    return;
  }
  // 根据图像通道数选择编码格式
  std::string encoding;
  if (_sys->_matchImg_right.type() == CV_8UC1) {
    encoding = "mono8";  // 单通道灰度图
  } else if (_sys->_matchImg_right.type() == CV_8UC3) {
    encoding = "bgr8";  // 三通道彩色图
  } else {
    ROS_ERROR("Unsupported image type: %d", _sys->_matchImg_right.type());
    return;
  }
  // 创建 sensor_msgs::Image 消息
  sensor_msgs::ImagePtr img_msg =
      cv_bridge::CvImage(std_msgs::Header(), encoding, _sys->_matchImg_right)
          .toImageMsg();

  // 创建 Image 消息
  // sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(),
  // "mono8", _sys->_intensityImg).toImageMsg();

  // 设置时间戳
  img_msg->header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);

  // 设置坐标系 ID
  img_msg->header.frame_id = "odom";

  // 发布图像
  pubImage.publish(img_msg);
}

void publishFrameBody(const ros::Publisher& pubLaserCloudBody) {
  if (_sys->_frameId % 2 != 0) return;
  int size = _sys->_localCloudPtr->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
    RGBpointBodyLidarToIMU(&_sys->_localCloudPtr->points[i],
                           &laserCloudIMUBody->points[i]);

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
  laserCloudmsg.header.frame_id = "odom_mapping";
  pubLaserCloudBody.publish(laserCloudmsg);
}

void publishFrameBody_undist(const ros::Publisher& pubLaserCloudBody) {
  // if(_sys->_frameId%2!=0)
  //     return;
  int size = _sys->_localCloudPtr->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
  for (int i = 0; i < size; i++)
    RGBpointBodyLidarToIMU(&_sys->_localCloudPtr->points[i],
                           &laserCloudIMUBody->points[i]);

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
  laserCloudmsg.header.frame_id = "body";

  sensor_msgs::PointCloud2 cloud_msg;
  sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
  modifier.setPointCloud2Fields(6, "x", 1, sensor_msgs::PointField::FLOAT32,
                                "y", 1, sensor_msgs::PointField::FLOAT32, "z",
                                1, sensor_msgs::PointField::FLOAT32,
                                "intensity", 1,
                                sensor_msgs::PointField::FLOAT32, "ring", 1,
                                sensor_msgs::PointField::UINT16, "time", 1,
                                sensor_msgs::PointField::FLOAT32);

  // 设置点云大小
  modifier.resize(size);

  // 创建迭代器以访问数据
  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud_msg,
                                                         "intensity");
  sensor_msgs::PointCloud2Iterator<uint16_t> iter_ring(cloud_msg, "ring");
  sensor_msgs::PointCloud2Iterator<float> iter_time(cloud_msg, "time");

  // 填充数据
  for (size_t i = 0; i < size; ++i) {
    *iter_x = _sys->_localCloudPtr->points[i].x;
    *iter_y = _sys->_localCloudPtr->points[i].y;
    *iter_z = _sys->_localCloudPtr->points[i].z;
    *iter_intensity = _sys->_localCloudPtr->points[i].intensity;
    *iter_ring = _sys->_localCloudPtr->points[i].normal_x;  // 设置 ring 值
    *iter_time =
        _sys->_localCloudPtr->points[i].curvature / 1000;  // 设置 time 值

    // 移动到下一个点
    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_intensity;
    ++iter_ring;
    ++iter_time;
  }
  cloud_msg.header.stamp = ros::Time().fromSec(_sys->_lidarBegTime);
  cloud_msg.header.frame_id = "body";

  pubLaserCloudBody.publish(cloud_msg);
}

void publishFrameGlobal(const ros::Publisher& pubLaserCloudFull) {
  sensor_msgs::PointCloud2 laserCloudmsg;
  PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
  _sys->transCloud(_sys->_globalCloudDownPtr, curWorldCloudPtr, _sys->_rotAlign,
                   V3D::Zero());
  pcl::toROSMsg(*curWorldCloudPtr, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
  laserCloudmsg.header.frame_id = "odom";
  pubLaserCloudFull.publish(laserCloudmsg);
}

void publishMapline(const ros::Publisher& pubLaserCloudFull) {
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*_sys->_matchworldlinecloudrgb, laserCloudmsg);
  //_addedMapCloud.reset(new PointCloudXYZI());
  laserCloudmsg.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
  laserCloudmsg.header.frame_id = "odom_mapping";
  pubLaserCloudFull.publish(laserCloudmsg);
}

void publishMapIncremental(const ros::Publisher& pubLaserCloudFull) {
  if (!_sys->config()._isEnable3DViewer &&
      _sys->config()._matchMethod != 0)  // not kdtree match, no visualization
  {
    if (!_sys->_ikdtree.Root_Node) {
      PointCloudXYZI::Ptr globalSurfCloudPtr(new PointCloudXYZI());
      _sys->transCloud(_sys->_localSurfCloudPtr, globalSurfCloudPtr, _sys->_Rwl,
                       _sys->_twl);
      _sys->initKdTreeMap(globalSurfCloudPtr->points);
    } else
      _sys->_addedPoints =
          _sys->_ikdtree.Add_Points(_sys->_globalCloudDownPtr->points, true);
  }
  _addedMapCloud->points.insert(_addedMapCloud->points.end(),
                                _sys->_addedPoints.begin(),
                                _sys->_addedPoints.end());

  if (_sys->_frameId % 20 != 0) return;

  _sys->transCloud(_addedMapCloud, _addedMapCloud, _sys->_rotAlign,
                   Eigen::Vector3d::Zero());
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*_addedMapCloud, laserCloudmsg);
  _addedMapCloud.reset(new PointCloudXYZI());
  laserCloudmsg.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
  laserCloudmsg.header.frame_id = "odom";
  pubLaserCloudFull.publish(laserCloudmsg);
}

void publishMap(const ros::Publisher& pubLaserCloudMap, bool isOnce = false) {
  if (!_sys->config()._isEnable3DViewer &&
      _sys->config()._matchMethod != 0)  // not kdtree match, no visualization
  {
    if (!_sys->_ikdtree.Root_Node) {
      PointCloudXYZI::Ptr globalSurfCloudPtr(new PointCloudXYZI());
      _sys->transCloud(_sys->_localSurfCloudPtr, globalSurfCloudPtr, _sys->_Rwl,
                       _sys->_twl);
      _sys->initKdTreeMap(globalSurfCloudPtr->points);
    } else
      _sys->_addedPoints =
          _sys->_ikdtree.Add_Points(_sys->_globalCloudDownPtr->points, true);
  }

  //    TicToc time;
  //    _featsFromMap->points.insert(_featsFromMap->points.end(),_sys->_addedPoints.begin(),_sys->_addedPoints.end());
  //    printf("ikdtree update time %f ms\n", time.toc());

  if (_sys->_frameId % 20 != 0 && !isOnce) return;

  TicToc time;
  PointVector().swap(_sys->_ikdtree.PCL_Storage);
  _sys->_ikdtree.flatten(_sys->_ikdtree.Root_Node, _sys->_ikdtree.PCL_Storage,
                         NOT_RECORD);
  _featsFromMap->points = _sys->_ikdtree.PCL_Storage;
  printf("ikdtree points get time %f ms\n", time.toc());

  _sys->transCloud(_featsFromMap, _featsFromMap, _sys->_rotAlign,
                   Eigen::Vector3d::Zero());
  sensor_msgs::PointCloud2 laserCloudMap;
  pcl::toROSMsg(*_featsFromMap, laserCloudMap);
  // std::cout<<"published _featsFromMap:"<<laserCloudMap.width<<std::endl;
  laserCloudMap.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
  laserCloudMap.header.frame_id = "odom";
  pubLaserCloudMap.publish(laserCloudMap);
}

void publishPath(const ros::Publisher& pubPath) {
  /*** if path is too large, the rvis will crash ***/
  static int i = 0;
  i++;
  if (i % 2 == 0) {
    V3D posAlign = _sys->_rotAlign * _sys->_stateIkfom.pos;
    Eigen::Quaterniond quatAlign =
        Eigen::Quaterniond(_sys->_rotAlign * _sys->_stateIkfom.rot);

    geometry_msgs::PoseStamped msg_body_pose;
    msg_body_pose.pose.position.x = posAlign(0);
    msg_body_pose.pose.position.y = posAlign(1);
    msg_body_pose.pose.position.z = posAlign(2);
    msg_body_pose.pose.orientation.x = quatAlign.coeffs()[0];
    msg_body_pose.pose.orientation.y = quatAlign.coeffs()[1];
    msg_body_pose.pose.orientation.z = quatAlign.coeffs()[2];
    msg_body_pose.pose.orientation.w = quatAlign.coeffs()[3];

    msg_body_pose.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
    msg_body_pose.header.frame_id = "odom";
    _path.poses.push_back(msg_body_pose);
    pubPath.publish(_path);
  }
}

void publishOdometry(const ros::Publisher& pubOdomAftMapped) {
  V3D posAlign = _sys->_rotAlign * _sys->_stateIkfom.pos;
  Eigen::Quaterniond quatAlign =
      Eigen::Quaterniond(_sys->_rotAlign * _sys->_stateIkfom.rot);

  _odomAftMapped.header.frame_id = "odom";
  _odomAftMapped.child_frame_id = "odom_mapping";
  _odomAftMapped.header.stamp = ros::Time().fromSec(
      _sys->_lidarEndTime);  // ros::Time().fromSec(lidar_end_time);
  _odomAftMapped.pose.pose.position.x = posAlign(0);
  _odomAftMapped.pose.pose.position.y = posAlign(1);
  _odomAftMapped.pose.pose.position.z = posAlign(2);
  _odomAftMapped.pose.pose.orientation.x = quatAlign.coeffs()[0];
  _odomAftMapped.pose.pose.orientation.y = quatAlign.coeffs()[1];
  _odomAftMapped.pose.pose.orientation.z = quatAlign.coeffs()[2];
  _odomAftMapped.pose.pose.orientation.w = quatAlign.coeffs()[3];
  pubOdomAftMapped.publish(_odomAftMapped);
  auto P = _sys->_kf.get_P();
  for (int i = 0; i < 6; i++) {
    int k = i < 3 ? i + 3 : i - 3;
    _odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    _odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    _odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    _odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    _odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    _odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_odomAftMapped.pose.pose.position.x,
                                  _odomAftMapped.pose.pose.position.y,
                                  _odomAftMapped.pose.pose.position.z));
  q.setW(_odomAftMapped.pose.pose.orientation.w);
  q.setX(_odomAftMapped.pose.pose.orientation.x);
  q.setY(_odomAftMapped.pose.pose.orientation.y);
  q.setZ(_odomAftMapped.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, _odomAftMapped.header.stamp,
                                        "odom", "odom_mapping"));
}

template <typename T>
sensor_msgs::PointCloud2 publishCloud(ros::Publisher* thisPub, T thisCloud,
                                      ros::Time thisStamp,
                                      std::string thisFrame) {
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0) thisPub->publish(tempCloud);
  return tempCloud;
}

void publishLoopInfos() {
  if (!_sys->loopCloser()) return;
  if (_sys->loopCloser()->loopCount() != _loopCount) {
    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "odom";
    markerNode.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "odom";
    markerEdge.header.stamp = ros::Time().fromSec(_sys->_lidarEndTime);
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.0;
    markerEdge.color.g = 1.0;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    auto& pgoEdges = _sys->_loopCloser->getPgoEdges();
    const auto& _subMapBlocks = _sys->_loopCloser->getSubMapBlocks();
    Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
    gravAlignMat.block<3, 3>(0, 0) = _sys->_rotAlign;
    for (auto& edge : pgoEdges) {
      if (edge.con_type != REGISTRATION) continue;

      Eigen::Matrix4d poseLo1 = gravAlignMat * edge.block1->_poseLo;
      Eigen::Matrix4d poseLo2 = gravAlignMat * edge.block2->_poseLo;

      float x, y, z, roll, pitch, yaw;
      pcl::getTranslationAndEulerAngles(Eigen::Affine3d(poseLo2).cast<float>(),
                                        x, y, z, roll, pitch, yaw);
      geometry_msgs::Point p;
      p.x = x;
      p.y = y;
      p.z = z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      pcl::getTranslationAndEulerAngles(Eigen::Affine3d(poseLo1).cast<float>(),
                                        x, y, z, roll, pitch, yaw);
      p.x = x;
      p.y = y;
      p.z = z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    _pubLoopNode.publish(markerArray);
    _loopCount = _sys->loopCloser()->loopCount();
  }
}

void captureControlMarkerHandler(const std_msgs::Int32ConstPtr& msgIn) {
  _ctrlMarkersForAverage.clear();
  _controlMarkerIndex = msgIn->data;
  _enableCaptureControlMarker = true;
  _ctrlMarkerBeginStamp = _sys->_lidarEndTime;
  printf("Begin capture controlMarker:%d\n", msgIn->data);
}

void saveControlPointInfoFile() {
  if (_ctrlPointsInfos.empty()) return;  // 无控制点就不写出.bin文件

  if (_isPlayBackTask)  // 回放任务也不会写出控制点bin文件，防止将原始bin文件篡改
    return;
  std::cout << "output control point data...\n";
  FILE* fout = fopen(_playBackFilePath.c_str(), "wb");
  if (fout) {
    double addKFAngle = 0;
    double addKFDist = 0;
    // 输出关键帧选择的角度和距离阈值
    fwrite(&addKFAngle, sizeof(float), 1, fout);
    fwrite(&addKFDist, sizeof(float), 1, fout);

    // 输出逆向标定读取到的lidar在IMU坐标系下的坐标
    fwrite(&_pntOnImu[0], sizeof(double), 1, fout);
    fwrite(&_pntOnImu[1], sizeof(double), 1, fout);
    fwrite(&_pntOnImu[2], sizeof(double), 1, fout);

    // 输出控制点的个数
    unsigned long ctrlPtNum = _ctrlPointsInfos.size();
    fwrite(&ctrlPtNum, sizeof(unsigned long), 1, fout);

    // 输出控制点对应KF的时间戳
    for (int idx = 0; idx < _ctrlPointsInfos.size(); idx++)
      fwrite(&_ctrlPointsInfos[idx].collectTimestamp, sizeof(double), 1, fout);

    fclose(fout);
  }
}

// 读取回放控制点的文件
bool readControlPointInfoFile() {
  std::cout << "Control points file:" << _playBackFilePath << std::endl;
  FILE* fp = fopen(_playBackFilePath.c_str(), "rb");
  if (fp) {
    // 读取关键帧选择的角度和距离阈值
    double addKFAngle, addKFDist;
    size_t cnt = fread(&addKFAngle, 1, sizeof(float), fp);
    if (cnt != sizeof(float)) return false;
    cnt = fread(&addKFDist, 1, sizeof(float), fp);
    if (cnt != sizeof(float)) return false;

    // 读取bin文件中记录的逆向标定得到的lidar在IMU坐标系下的坐标
    cnt = fread(&_pntOnImu[0], 1, sizeof(double), fp);
    if (cnt != sizeof(double)) return false;
    cnt = fread(&_pntOnImu[1], 1, sizeof(double), fp);
    if (cnt != sizeof(double)) return false;
    cnt = fread(&_pntOnImu[2], 1, sizeof(double), fp);
    if (cnt != sizeof(double)) return false;

    printf("控制点回放, 读取到idar在IMU坐标系下的坐标: %lf, %lf, %lf\n",
           _pntOnImu.x(), _pntOnImu.y(), _pntOnImu.z());
    // printf("控制点回放, 读取到lidar在IMU坐标系下的坐标: %lf, %lf, %lf\n",
    // _pntOnImu.x, _pntOnImu.y, _pntOnImu.z);

    unsigned long ctrlPtNum = 0;
    cnt = fread(&ctrlPtNum, 1, sizeof(unsigned long), fp);
    if (cnt != sizeof(unsigned long)) return false;

    for (int i = 0; i < ctrlPtNum; i++) {
      double ctrlPtKFTimestamp = -1;
      cnt = fread(&ctrlPtKFTimestamp, 1, sizeof(double), fp);
      if (cnt != sizeof(double)) {
        std::cerr << "fread error, cnt=" << cnt << std::endl;
        return false;
      }

      _ctrlPtTimestampFromPlayBack.push_back(ctrlPtKFTimestamp);

      printf("读取到控制点对应KF timestamp: %lf\n", ctrlPtKFTimestamp);
      // printf("读取到控制点对应KF timestamp: %lf\n",  ctrlPtKFTimestamp);
    }
    return true;
  }
  return false;
}

void addControlPointByFile() {
  if (!_sys->loopCloser()) return;
  _sys->loopCloser()->getBufLock().lock();
  CloudBlockPtrs frameBlocks = _sys->loopCloser()->getFrameBlocks();
  _sys->loopCloser()->getBufLock().unlock();
  if (frameBlocks.size() < 10) return;

  if (_ctrlPtTimestampFromPlayBack.empty()) return;

  static int playBackCtrlPtNum = 0;

  /**
   * 【注意】这里的控制点回放有一个假设：采集控制点时LiDAR
   *pose的Z值在6s局部是最小值，如果在实际应用中该假设不能满足，
   * 那么采集的控制点无法正确回放 zqy 0321
   **/

  double curKFTime = frameBlocks.back()->_timestamp;
  double controlPointTime = _ctrlPtTimestampFromPlayBack.front();
  //    if(controlPointTime<=0)
  //        _ctrlPtTimestampFromPlayBack.pop_front();
  // 在过去15 + 15 s找一个z值最小的KF为控制点采集所属的KF
  double timeSpan = 15;
  if (curKFTime - controlPointTime < timeSpan) return;
  std::cout << "Add control point" << std::endl;

  std::vector<std::vector<size_t>> windowKFAndTimestamp;

  std::vector<size_t> aWindowKF;

  // 采用时间跨度为6s，前进量为2s的滑动窗口，在滑动窗口内找Z变化最大的窗口中，z最小值对应的KF为控制点所属KF
  double baseTime = frameBlocks.back()->_timestamp;

  double windowTimeSpan = 6;  // 6s
  double step = 2;
  while (1) {
    if (fabs(baseTime - controlPointTime) > timeSpan + 4) {
      windowKFAndTimestamp.push_back(aWindowKF);  // 保存最后一个窗口中的KFs的id
      break;
    }

    for (size_t i = frameBlocks.size() - 1; i > 0; i--) {
      if (baseTime < frameBlocks[i]->_timestamp) continue;
      if (baseTime - frameBlocks[i]->_timestamp > windowTimeSpan) break;

      if (baseTime - frameBlocks[i]->_timestamp <= windowTimeSpan)
        aWindowKF.push_back(i);
    }

    baseTime -= step;  // 窗口向前滑动
    windowKFAndTimestamp.push_back(aWindowKF);
    aWindowKF.clear();
  }
  Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
  gravAlignMat.block<3, 3>(0, 0) = _sys->_rotAlign;
  std::vector<std::pair<float, size_t>> dZAndTargetKFs;
  for (size_t i = 0; i < windowKFAndTimestamp.size(); i++) {
    if (windowKFAndTimestamp[i].empty()) continue;

    float minZ = FLT_MAX;
    float maxZ = FLT_MIN;
    size_t minZKfID = 0;
    for (size_t j = 0; j < windowKFAndTimestamp[i].size(); j++) {
      auto kfID = windowKFAndTimestamp[i][j];

      if (kfID < 0 || kfID >= frameBlocks.size()) continue;
      Eigen::Matrix4d pose = gravAlignMat * frameBlocks[kfID]->_poseLo;
      double z = pose(2, 3);
      if (z < minZ) {
        minZ = z;
        minZKfID = kfID;
      }
      if (z > maxZ) maxZ = z;
    }

    if (minZKfID > 0)
      dZAndTargetKFs.push_back(std::make_pair(maxZ - minZ, minZKfID));
  }

  if (dZAndTargetKFs.empty()) {
    _ctrlPtTimestampFromPlayBack.pop_front();  // 查找失败
    std::cerr << "#1:Searching failed" << std::endl;
    return;
  }

  // 找具有局部Z变化最大的窗口内的minZ对应的kf
  float dZMax = dZAndTargetKFs[0].first;
  size_t selectedKFID = dZAndTargetKFs[0].second;
  for (size_t id = 1; id < dZAndTargetKFs.size(); id++) {
    if (dZAndTargetKFs[id].first > dZMax) {
      dZMax = dZAndTargetKFs[id].first;
      selectedKFID = dZAndTargetKFs[id].second;
    }
  }

  if (dZMax < 0.6)  // 普通身高的人，胸前只脚底一般> 0.6m
  {
    std::cout << "dZ max过小" << dZMax << std::endl;
    _ctrlPtTimestampFromPlayBack.pop_front();  // 查找失败
    std::cerr << "#2:Searching failed" << std::endl;
    return;
  }

  // printf("selectID, 文件记录和探测到的时间戳: %d, %lf, %lf\n", selectedKFID,
  // controlPointTime, cloudKeyPoses6D->points[selectedKFID].time);

  // 防止因为局部抖动，在localMinZ最小附近选择运动量最小的一帧，(采集控制点时设备处于静止状态)
  int bestKFID = selectedKFID;
  float minMoveDist = 0.6f;
  for (size_t id = selectedKFID - 3; id < selectedKFID + 3; id++) {
    if (id - 1 < 0 || (id + 1 >= frameBlocks.size())) break;
    auto prevKFPose = gravAlignMat * frameBlocks[id - 1]->_poseLo;
    auto curKFPose = gravAlignMat * frameBlocks[id]->_poseLo;
    auto lastKFPose = gravAlignMat * frameBlocks[id + 1]->_poseLo;

    auto prevKFPos = prevKFPose.block<3, 1>(0, 3);
    auto curKFPos = curKFPose.block<3, 1>(0, 3);
    auto lastKFPos = lastKFPose.block<3, 1>(0, 3);

    float dx =
        fabs(curKFPos.x() - prevKFPos.x()) + fabs(curKFPos.x() - lastKFPos.x());
    float dy =
        fabs(curKFPos.y() - prevKFPos.y()) + fabs(curKFPos.y() - lastKFPos.y());
    float dz =
        fabs(curKFPos.z() - prevKFPos.z()) + fabs(curKFPos.z() - lastKFPos.z());
    float droll = 0;  // 角元素也会影响控制点的精度所以也考虑
    float dpitch = 0;
    float dyaw = 0;
    //        float droll = fabs(curKFPose.roll - prevKFPose.roll) +
    //        fabs(curKFPose.roll - lastKFPose.roll);        //
    //        角元素也会影响控制点的精度所以也考虑 float dpitch =
    //        fabs(curKFPose.pitch - prevKFPose.pitch) + fabs(curKFPose.pitch -
    //        lastKFPose.pitch); float dyaw = fabs(curKFPose.yaw -
    //        prevKFPose.yaw) + fabs(curKFPose.yaw - lastKFPose.yaw);

    float dist = sqrt(dx * dx + dy * dy + dz * dz + droll * droll +
                      dpitch * dpitch + dyaw * dyaw);
    if (dist < minMoveDist) {
      bestKFID = id;
      minMoveDist = dist;
    }
  }

  if (bestKFID < 0 || bestKFID >= frameBlocks.size()) {
    _ctrlPtTimestampFromPlayBack.pop_front();  // 查找失败
    std::cerr << "#3:Searching failed" << std::endl;
    return;
  }

  printf("回放到控制点, 文件记录时间戳和探测到的时间戳: %lf, %lf\n",
         controlPointTime, frameBlocks[bestKFID]->_timestamp);
  auto bestKFPose = gravAlignMat * frameBlocks[bestKFID]->_poseLo;
  auto bestKFPos = bestKFPose.block<3, 1>(0, 3);
  double x = bestKFPos.x();
  double y = bestKFPos.y();
  double z = bestKFPos.z();

  pcl::PointXYZI srcPnt, transfPnt;
  srcPnt.x = x;
  srcPnt.y = y;
  srcPnt.z = z;
  srcPnt.intensity = 255;
  transfPnt.x = srcPnt.x + _pntOnImu.x();
  transfPnt.y = srcPnt.y + _pntOnImu.y();
  transfPnt.z = srcPnt.z + _pntOnImu.z();
  transfPnt.intensity = srcPnt.intensity;
  printf("src->transferd:%f,%f,%f->%f,%f,%f\n", srcPnt.x, srcPnt.y, srcPnt.z,
         transfPnt.x, transfPnt.y, transfPnt.z);
  _controlMarkers->points.push_back(transfPnt);
  printf("Add control marker %d:%f,%f,%f\n", _controlMarkers->points.size() - 1,
         transfPnt.x, transfPnt.y, transfPnt.z);

  // publish control markers
  publishCloud(&_pubControlMarkers, _controlMarkers,
               ros::Time().fromSec(_sys->_lidarEndTime), "odom");
  _enableCaptureControlMarker = false;
  std_msgs::Int32 msg;
  msg.data = (int32_t)_controlMarkers->points.size();
  _pubControlMarkerCaptured.publish(msg);

  ControlPointInfoStru ctrlPtInfoForPlayback(frameBlocks[bestKFID]->_timestamp,
                                             bestKFID, _pntOnImu.x(),
                                             _pntOnImu.y(), _pntOnImu.z());
  _ctrlPointsInfos.push_back(ctrlPtInfoForPlayback);

  playBackCtrlPtNum++;

  _ctrlPtTimestampFromPlayBack.pop_front();
}

void addControlPoint() {
  if (!_sys->loopCloser()) return;
  if (_bReadedPlayBackFile) {
    addControlPointByFile();
    return;
  }

  CloudBlockPtrs frameBlocks = _sys->loopCloser()->getFrameBlocks();
  if (frameBlocks.empty()) return;
  if (!_enableCaptureControlMarker)  // 未收到控制点采集的信号
    return;

  Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
  gravAlignMat.block<3, 3>(0, 0) = _sys->_rotAlign;
  if (_sys->_lidarEndTime - _ctrlMarkerBeginStamp <
      5)  // 按下按钮静止超过2s才会采集
  {
    auto lastPose = gravAlignMat * frameBlocks.back()->_poseLo;
    auto lastPos = lastPose.block<3, 1>(0, 3);
    _meanPos += lastPos;
    _nTotal++;
    return;
  }
  _meanPos /= _nTotal;
  double x = _meanPos.x();
  double y = _meanPos.y();
  double z = _meanPos.z();
  printf("Mean Pose x,y,z:%lf,%lf,%lf,%lf,%lf,%lf\n", x, y, z);
  _meanPos = V3D(0, 0, 0);
  _nTotal = 0;

  pcl::PointXYZI srcPnt, transfPnt;
  srcPnt.x = x;
  srcPnt.y = y;
  srcPnt.z = z;
  srcPnt.intensity = 255;
  transfPnt.x = srcPnt.x + _pntOnImu.x();
  transfPnt.y = srcPnt.y + _pntOnImu.y();
  transfPnt.z = srcPnt.z + _pntOnImu.z();
  transfPnt.intensity = srcPnt.intensity;
  printf("src->transferd:%f,%f,%f->%f,%f,%f\n", srcPnt.x, srcPnt.y, srcPnt.z,
         transfPnt.x, transfPnt.y, transfPnt.z);
  _controlMarkers->points.push_back(transfPnt);
  printf("Add control marker %d:%f,%f,%f\n", _controlMarkers->points.size() - 1,
         transfPnt.x, transfPnt.y, transfPnt.z);

  // publish control markers
  publishCloud(&_pubControlMarkers, _controlMarkers,
               ros::Time().fromSec(_sys->_lidarEndTime), "odom");
  _enableCaptureControlMarker = false;
  std_msgs::Int32 msg;
  msg.data = _controlMarkerIndex;
  _pubControlMarkerCaptured.publish(msg);

  ControlPointInfoStru ctrlPtInfoForPlayback(
      _sys->_lidarEndTime, static_cast<int>(frameBlocks.size() - 1),
      _pntOnImu.x(), _pntOnImu.y(), _pntOnImu.z());
  _ctrlPointsInfos.push_back(ctrlPtInfoForPlayback);
}

void lidCallback(PPointCloud::Ptr cloudIn, double lidTime) {
  //    if(!_sys->mergeNConsecutiveScans(_sys->_config._nMergedFrames, cloudIn,
  //    lidTime))
  //        return;
  if (lidTime < _sys->_lidarEndTime) {
    std::cout << "Lidar end time:" << _sys->_lidarEndTime << std::endl;
    std::cout << "lidTime:" << _sys->_lidarEndTime << std::endl;
    std::cerr << "lidar loop back, clear buffer" << std::endl;
    _sys->_lidarBuffer.clear();
    _sys->_timeBuffer.clear();
  }
  _sys->_lidarBegTime = lidTime;
  // cout << std::fixed << std::setprecision(6) << "Lidar beg time: " << lidTime
  // << endl;

  if (!_sys->_config._isCutFrame) {
    PointCloudXYZI::Ptr cloudOut(new PointCloudXYZI());
    _sys->_lidarProcessor->process(cloudIn, lidTime, cloudOut);

    _sys->_lidarBuffer.push_back(cloudOut);
    _sys->_timeBuffer.push_back(lidTime);
    // std::cout<<"Lidar begin:"<<cloudOut->front().curvature<<std::endl;
    // std::cout<<"Lidar end:"<<cloudOut->back().curvature<<std::endl;
  } else {
    static int scanCount = 0;
    deque<PointCloudXYZI::Ptr> ptrQueue;
    deque<double> timeQueue;

    _sys->_lidarProcessor->processCutFrameCloud(
        cloudIn, lidTime, ptrQueue, timeQueue, _sys->_config._cutFrameNum,
        scanCount++);
    while (!ptrQueue.empty() && !timeQueue.empty()) {
      _sys->_lidarBuffer.push_back(ptrQueue.front());
      ptrQueue.pop_front();
      _sys->_timeBuffer.push_back(timeQueue.front() / double(1000));  // unit:s
      timeQueue.pop_front();
    }
  }
}

void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  _sys->_mtxBuffer.lock();
#if OUTPUT_LOG
  static std::ofstream lidSeqOfs =
      std::ofstream(_workspace + "/lidSeq", std::ios::trunc | std::ios::in);
  // lidSeqOfs<<std::setprecision(9)<<std::fixed<<" "<<msg->header.seq<<"
  // "<<msg->header.stamp.toSec()<<std::endl;
  lidSeqOfs << std::setprecision(9) << std::fixed << _msgCount++ << " "
            << msg->header.seq << " " << msg->header.stamp.toSec() << std::endl;
#endif
  //
  //    if(_lastLidSeqID!=-1&&msg->header.seq>_lastLidSeqID+5)
  //    {
  //        //std::cerr<<"IMU Seq jump!"<<std::endl;
  //        _sys->_mtxBuffer.unlock();
  //        return;
  //    }
  //    else
  //        _lastLidSeqID=msg->header.seq;

  pcl::PointCloud<Hesai_ZG_ros::PointXYZIRT16>::Ptr oriCloud(
      new pcl::PointCloud<Hesai_ZG_ros::PointXYZIRT16>);
  pcl::PointCloud<PPoint>::Ptr cloud(new pcl::PointCloud<PPoint>);
  pcl::fromROSMsg(*msg, *oriCloud);
  for (auto& p : oriCloud->points) {
    PPoint pp;
    pp.ring = p.ring;
    pp.timestamp = msg->header.stamp.toSec() + p.time;
    pp.intensity = p.intensity;
    pp.x = p.x;
    pp.y = p.y;
    pp.z = p.z;
    cloud->push_back(pp);
  }
  lidCallback(cloud, msg->header.stamp.toSec());

  _sys->_mtxBuffer.unlock();
  _sys->_sigBuffer.notify_all();
}

void imuCallback(const sensor_msgs::Imu::ConstPtr& msgIn) {
  _sys->_mtxBuffer.lock();
#if OUTPUT_LOG
  static std::ofstream imuSeqOfs =
      std::ofstream(_workspace + "/imuSeq", std::ios::trunc | std::ios::in);
  // imuSeqOfs<<std::setprecision(9)<<std::fixed<<" "<<msgIn->header.seq<<"
  // "<<msgIn->header.stamp.toSec()<<std::endl;
  imuSeqOfs << std::setprecision(9) << std::fixed << _msgCount++ << " "
            << msgIn->header.seq << " " << msgIn->header.stamp.toSec()
            << std::endl;
#endif
  //
  //    if(_lastIMUSeqID!=-1&&msgIn->header.seq>_lastIMUSeqID+10)
  //    {
  //        //std::cerr<<"IMU Seq jump!"<<std::endl;
  //        _sys->_mtxBuffer.unlock();
  //        return;
  //    }
  //    else
  //        _lastIMUSeqID=msgIn->header.seq;
  SensorMsgs::ImuData::Ptr msg(new SensorMsgs::ImuData);
  msg->angular_velocity =
      Eigen::Vector3d(msgIn->angular_velocity.x, msgIn->angular_velocity.y,
                      msgIn->angular_velocity.z);
  msg->linear_acceleration = Eigen::Vector3d(msgIn->linear_acceleration.x,
                                             msgIn->linear_acceleration.y,
                                             msgIn->linear_acceleration.z);
  // IMU Time Compensation
  msg->header = msgIn->header.stamp.toSec() - _sys->_timediffImuWrtLidar -
                _sys->config()._timeLagIMUWtrLidar;

  if (msg->header < _sys->_lastImuTimestamp) {
    std::cerr << "IMU loop back, clear IMU buffer." << std::endl;
    _sys->_imuBuffer.clear();
  }

  _sys->_lastImuTimestamp = msg->header;
  _sys->_imuBuffer.push_back(msg);

  _imuDataList.emplace_back(msg);

  // push all IMU meas into Init_LI
  if (!_sys->mutableConfig()._isImuInitialized && !_sys->_isDataAccumFinished)
    _sys->_initiatorLI->push_ALL_IMU_CalibState(
        msg, _sys->mutableConfig()._meanAccNorm);

  _sys->_mtxBuffer.unlock();
  _sys->_sigBuffer.notify_all();
}

void motorCallback(const rigelslam_rot::motor::ConstPtr& msgIn) {
  // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
  _sys->_mtxBuffer.lock();

  const double timestamp = msgIn->header.stamp.toSec();
#if OUTPUT_LOG
  static std::ofstream motorSeqOfs =
      std::ofstream(_workspace + "/motorSeq", std::ios::trunc | std::ios::in);
  // motorSeqOfs<<std::setprecision(9)<<std::fixed<<" "<<msgIn->header.seq<<"
  // "<<timestamp<<std::endl;
  motorSeqOfs << std::setprecision(9) << std::fixed << _msgCount++ << " "
              << msgIn->header.seq << " " << timestamp << std::endl;
#endif
  // std::cout<<msgIn->header.seq<<std::endl;
  //    if(_lastMotorSeqID!=-1&&msgIn->header.seq>_lastMotorSeqID+10)
  //    {
  //        //std::cerr<<"Motor Seq jump!"<<std::endl;
  //        _sys->_mtxBuffer.unlock();
  //        return;
  //    }
  //    else
  //        _lastMotorSeqID=msgIn->header.seq;
  if (timestamp >= _sys->_lastMotorTimestamp) {
    _sys->_lastMotorTimestamp = timestamp;
    _sys->_motorAngleBuf.emplace_back(timestamp, deg2rad(msgIn->motor_angle));
  } else {
    std::cout << "Last motor time:" << _sys->_lastMotorTimestamp << std::endl;
    std::cout << "cur motor time:" << timestamp << std::endl;
    std::printf("motor loop back\n");
    // PAUSE;
    // _sys->_motorAngleBuf.clear();
  }
  _sys->_mtxBuffer.unlock();
  _sys->_sigBuffer.notify_all();
}

void saveRawData() {
  TicToc time;
  std::cout << "Output raw data...\n";

  CRawFileData rawFileData;

  auto& fileHead = rawFileData.getFileHead();
  auto& lidarFrameInfos = rawFileData.getLidarFramesInfos();
  auto& ctrlPointsInfos = rawFileData.getControlPointInfos();
  auto& loopPairInfos = rawFileData.getLoopPairInfos();
  auto& odomInfos = rawFileData.getOdometryInfos();
  auto& imuInfos = rawFileData.getIMUInfos();
  // auto& TilList=rawFileData.getExtrinsicInfos();

  fileHead.version = RAWFileVersionEnum::VERSION_3_0;
  char chs[128] = {
      "RigelSLAM-Pano\0"};  // 后边显示的赋\0空字符，避免后续拷贝中出现意外的失败
  memcpy(fileHead.deviceType, chs, 128 * sizeof(char));

  char chs2[128] = {"Linux\0"};
  memcpy(fileHead.writeFileSystem, chs2, 128 * sizeof(char));
  for (int i = 0; i < _scanScene.size() && i < 127; i++)
    fileHead.scanSceneName[i] = _scanScene
        [i];  // ubunu中采用strcpy拷贝没有显示\0结束的字符串会造成访问越界

  // TODO 修改 rawFileData.fileHead.scanTime
  CloudBlockPtrs frameBlocks = _sys->loopCloser()->getFrameBlocks();

  fileHead.lidarFrameNum = frameBlocks.size();
  fileHead.scanDuration =
      static_cast<float>(frameBlocks.back()->_timestamp -
                         frameBlocks.front()->_timestamp);  // 扫描时长，s
  double trajectoryLength = 0;
  for (int i = 1; i < frameBlocks.size(); i++) {
    auto curPos = frameBlocks[i]->_poseLo.block<3, 1>(0, 3);
    auto prevPos = frameBlocks[i - 1]->_poseLo.block<3, 1>(0, 3);
    float dx = curPos.x() - prevPos.x();
    float dy = curPos.y() - prevPos.y();
    float dz = curPos.z() - prevPos.z();
    trajectoryLength += sqrt(dx * dx + dy * dy + dz * dz);  // 计算累积轨迹长度
  }
  fileHead.scanPathLength = trajectoryLength;

  fileHead.isContinueTask = _isContinueTask;
  fileHead.isPlaybackTask = _isPlayBackTask;
  fileHead.isEmergedRawFile = false;

  fileHead.controlPointsNum = _ctrlPointsInfos.size();
  std::cout << "Control points size:" << fileHead.controlPointsNum << std::endl;
  ctrlPointsInfos = _ctrlPointsInfos;
  //    PointCloudXYZI::Ptr rawCloudPtr(new PointCloudXYZI());
  //    PointCloudXYZI::Ptr globalCloudPtr(new PointCloudXYZI());
  //    PointCloudXYZI::Ptr rawSurfPtr(new PointCloudXYZI());
  //    PointCloudXYZI::Ptr rawCornPtr(new PointCloudXYZI());
  //    PointCloudXYZI::Ptr trajCloud(new PointCloudXYZI());
  pcl::VoxelGrid<PointType> downSizeFilter;
  downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
  Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
  if (_sys->_gnssTrans.isIdentity())
    gravAlignMat.block<3, 3>(0, 0) = _sys->_rotAlign;
  else
    gravAlignMat = _sys->_gnssTrans.inverse();
  // std::cout<<"Gravity Align Matrix:"<<gravAlignMat<<std::endl;

  lidarFrameInfos.resize(frameBlocks.size());
#pragma omp parallel for num_threads(MP_PROC_NUM)
  for (int i = 0; i < frameBlocks.size(); i++) {
    LidarFrameInfoStru lidarFrameInfo;
    lidarFrameInfo.kfID = i;
    lidarFrameInfo.timestamp = frameBlocks[i]->_timestamp;
    lidarFrameInfo.Til = _sys->_TilList[i];

    Eigen::Matrix4d pose = gravAlignMat * frameBlocks[i]->_poseLo;
    auto curPos = pose.block<3, 1>(0, 3);
    lidarFrameInfo.x = curPos.x();
    lidarFrameInfo.y = curPos.y();
    lidarFrameInfo.z = curPos.z();
    auto curQ = Eigen::Quaterniond(pose.block<3, 3>(0, 0));
    lidarFrameInfo.q_orientation.w() = curQ.w();
    lidarFrameInfo.q_orientation.x() = curQ.x();
    lidarFrameInfo.q_orientation.y() = curQ.y();
    lidarFrameInfo.q_orientation.z() = curQ.z();
    pcl::io::loadPCDFile(frameBlocks[i]->_pcdFilePath, *frameBlocks[i]->_pcRaw);
    lidarFrameInfo.fullCloud = frameBlocks[i]->_pcRaw;
    //        for(auto& p:frameBlocks[i]->_pcRaw->points)
    //        {
    //            pcl::PointXYZI pt;
    //            pt.x=p.x;
    //            pt.y=p.y;
    //            pt.z=p.z;
    //            pt.intensity=p.intensity;
    //            lidarFrameInfo.fullCloud->push_back(pt);
    //        }
    if (_isOutputFeature) {
      _sys->_lidarProcessor->extractFeatures(frameBlocks[i]->_pcRaw,
                                             *_sys->_localSurfCloudPtr,
                                             *_sys->_localCornerCloudPtr);

      downSizeFilter.setInputCloud(_sys->_localSurfCloudPtr->makeShared());
      downSizeFilter.filter(*_sys->_localSurfCloudPtr);
      downSizeFilter.setInputCloud(_sys->_localCornerCloudPtr->makeShared());
      downSizeFilter.filter(*_sys->_localCornerCloudPtr);
      lidarFrameInfo.surfCloud = _sys->_localSurfCloudPtr;
      lidarFrameInfo.cornCloud = _sys->_localCornerCloudPtr;
      //            for(auto& p:_sys->_localSurfCloudPtr->points)
      //            {
      //                pcl::PointXYZI pt;
      //                pt.x=p.x;
      //                pt.y=p.y;
      //                pt.z=p.z;
      //                pt.intensity=p.intensity;
      //                lidarFrameInfo.surfCloud->push_back(pt);
      //            }
      //            for(auto& p:_sys->_localCornerCloudPtr->points)
      //            {
      //                pcl::PointXYZI pt;
      //                pt.x=p.x;
      //                pt.y=p.y;
      //                pt.z=p.z;
      //                pt.intensity=p.intensity;
      //                lidarFrameInfo.cornCloud->push_back(pt);
      //            }
    }

    // pcl::transformPointCloud(*frameBlocks[i]->_pcRaw, *globalCloudPtr,
    // pose);//pw=Tw1*p1
    //*rawCloudPtr+=*globalCloudPtr;
    //        pcl::transformPointCloud(*_sys->_localSurfCloudPtr,
    //        *globalCloudPtr, pose);//pw=Tw1*p1 *rawSurfPtr+=*globalCloudPtr;
    //        pcl::transformPointCloud(*_sys->_localCornerCloudPtr,
    //        *globalCloudPtr, pose);//pw=Tw1*p1 *rawCornPtr+=*globalCloudPtr;
    //
    //        PointType trajPt;
    //        trajPt.x = frameBlocks[i]->_poseLo(0, 3);
    //        trajPt.y = frameBlocks[i]->_poseLo(1, 3);
    //        trajPt.z = frameBlocks[i]->_poseLo(2, 3);
    //        trajCloud->push_back(trajPt) ;
    //////////////////debug///////////////////////
    // static pcl::PCDWriter pcdWriter;
    // string pcdPath(string(ROOT_DIR) + "PCD/frame" + to_string(i) +
    // string(".pcd")); pcdWriter.writeBinary(pcdPath, *frameBlocks[i]->_pcRaw);
    ////////////////////////////////////////////////

    frameBlocks[i]->_pcRaw.reset(new PointCloudXYZI);
    lidarFrameInfos[i] = lidarFrameInfo;
  }
  //    pcl::PCDWriter pcdWriter;
  //    pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/rawFrameCloud.pcd",
  //    *rawCloudPtr); pcdWriter.writeBinary(string(ROOT_DIR) +
  //    "PCD/rawSurfCloud.pcd", *rawSurfPtr);
  //    pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/rawCornCloud.pcd",
  //    *rawCornPtr); pcdWriter.writeBinary(string(ROOT_DIR) +
  //    "PCD/rawTrajCloud.pcd", *trajCloud);
  std::cout << "Raw frames size:" << frameBlocks.size() << std::endl;

  //////////////////debug///////////////////////
  //    std::ofstream fPose = std::ofstream(string(ROOT_DIR) + "PCD/pose.json",
  //    std::ios::trunc | std::ios::in); for(int i = 0; i < frameBlocks.size();
  //    i++)
  //    {
  //        Eigen::Matrix4d pose=gravAlignMat*frameBlocks[i]->_poseLo;
  //        auto pos =  pose.block<3,1>(0,3);
  //        auto rot =  pose.block<3,3>(0,0);
  //        Eigen::Quaterniond quat=Eigen::Quaterniond (rot);
  //        fPose<<pos.x()<<" "<<pos.y()<<" "<<pos.z()<<" "<<quat.w()<<"
  //        "<<quat.x()<<" "<<quat.y()<<" "<<quat.z()<<std::endl; fPose <<
  //        setprecision(6) << std::fixed;
  //    }
  ////////////////////////////////////////////////

  fileHead.loopPairNum = _sys->_loopCloser->loopCount();
  std::vector<LoopPairInfoStru> loopPairsInfo;
  auto& pgoEdges = _sys->_loopCloser->getPgoEdges();
  const auto& _subMapBlocks = _sys->_loopCloser->getSubMapBlocks();
  for (auto& edge : pgoEdges) {
    if (edge.con_type != REGISTRATION) continue;
    LoopPairInfoStru loopPairInfo;  // loopPairsInfo;
    loopPairInfo.prevKFID = edge.block1->_uniqueId;
    loopPairInfo.curKFID = edge.block2->_uniqueId;

    std::cout << "Loop info:" << loopPairInfo.prevKFID << "-"
              << loopPairInfo.curKFID << std::endl;
    Eigen::Matrix4d poseLo1 = gravAlignMat * edge.block1->_poseLo;
    Eigen::Matrix4d poseLo2 = gravAlignMat * edge.block2->_poseLo;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(Eigen::Affine3d(poseLo2).cast<float>(), x,
                                      y, z, roll, pitch, yaw);
    loopPairInfo.poseFrom[0] = roll;
    loopPairInfo.poseFrom[1] = pitch;
    loopPairInfo.poseFrom[2] = yaw;
    loopPairInfo.poseFrom[3] = x;
    loopPairInfo.poseFrom[4] = y;
    loopPairInfo.poseFrom[5] = z;

    pcl::getTranslationAndEulerAngles(Eigen::Affine3d(poseLo1).cast<float>(), x,
                                      y, z, roll, pitch, yaw);
    loopPairInfo.poseTo[0] = roll;
    loopPairInfo.poseTo[1] = pitch;
    loopPairInfo.poseTo[2] = yaw;
    loopPairInfo.poseTo[3] = x;
    loopPairInfo.poseTo[4] = y;
    loopPairInfo.poseTo[5] = z;
    loopPairsInfo.push_back(loopPairInfo);
  }
  loopPairInfos = loopPairsInfo;

  fileHead.imuNum = _imuDataList.size();
  for (auto& imuData : _imuDataList)
    imuInfos.emplace_back(imuData->header, imuData->angular_velocity,
                          imuData->linear_acceleration);

  // PointCloudXYZI::Ptr trajIMUCloud(new PointCloudXYZI());
  for (auto frameBlock : frameBlocks) {
    for (auto& imuData : frameBlock->_imuDataList) {
      OdometryInfoStru odomInfo;
      odomInfo.second_ = imuData._timestamp;
      odomInfo.position_ = _sys->_rotAlign * imuData._position;
      odomInfo.orientation_ = _sys->_rotAlign * imuData._orientation;
      odomInfo.velocity_ = imuData._velocity;
      odomInfo.accelerate_ = imuData._accelerate;
      odomInfo.angular_velocity_ = imuData._angular_velocity;
      odomInfo.ba_ = imuData._ba;
      odomInfo.bg_ = imuData._bg;
      odomInfos.emplace_back(odomInfo);
      //            PointType trajPt;
      //            trajPt.x = odomInfo.position_(0);
      //            trajPt.y = odomInfo.position_(1);
      //            trajPt.z = odomInfo.position_(2);
      // trajIMUCloud->push_back(trajPt) ;
    }
  }
  // pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/rawIMUTrajCloud.pcd",
  // *trajIMUCloud);

  // std::string outputPath=std::string(ROOT_DIR) + "PCD/output.raw";
  if (CRigelSLAMRawIOTools::WriteRawFile(_saveRawPath, rawFileData) !=
      WRITE_RAW_FILE_RETURN::WRITE_RAW_SUCCESSED)
    printf("写出raw文件输出失败!\n");
  else
    std::cout << "Raw frame clouds saved!" << std::endl;
  printf("Raw output time: %f ms\n", time.toc());
}

void stopNodeHandler(const std_msgs::Int32ConstPtr& msgIn) {
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

void loadRosParams(ros::NodeHandle& nh) {
  nh.param<double>("rigelslam_rot/minRange",
                   LidarProcess::mutableConfig()._blindMin, 0);
  nh.param<double>("rigelslam_rot/ridus_k", _sys->_config._radius_k, 3);
  nh.param<double>("rigelslam_rot/maxRange",
                   LidarProcess::mutableConfig()._blindMax, 0);
  nh.param<int>("rigelslam_rot/scanLines",
                LidarProcess::mutableConfig()._nScans, 16);
  nh.param<int>("rigelslam_rot/cov_type", _sys->_config._covType, 1);
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

void releaseVoxelMap() {
  if (_sys->_voxelSurfMap.empty()) return;
  // int delCount=0;
  for (auto locOctItr : _sys->_voxelSurfMap) {
    delete locOctItr.second;
    locOctItr.second = nullptr;
    // delCount++;
    // if(delCount==1)
    //{
    //  delCount=0;
    //  usleep(1);
    // }
  }
  std::unordered_map<VOXEL_LOC, OctoTree*>().swap(_sys->_voxelSurfMap);
  malloc_trim(0);
}

double getAvailableMemory() {
  long long pages = sysconf(_SC_AVPHYS_PAGES);
  double page_size =
      sysconf(_SC_PAGE_SIZE) / (1073741824.);  // 1024*1024*1024(以KB为单位)
  return pages * page_size;
}

void checkMatchMethod() {
  if (_isMaxOptimized) return;
  if (_coolCount != 0) _coolCount--;

  if (_sys->mutableConfig()._matchMethod == 0) {
    if (_meanFrameTime > 100.) {
      if (_coolCount == 0) {
        if (LidarProcess::mutableConfig()._pointFilterNum == 1) {
          LidarProcess::mutableConfig()._pointFilterNum = 2;
          std::cout << "Filter points to 1/2" << std::endl;
          _coolCount = 200;
        } else {
          _isMaxOptimized = true;
          _sys->_lidarProcessor->setDownsampleLeafSize(0.3, 0.3, 0.3);
          std::cout << "Downsample points size to 0.3" << std::endl;
          _coolCount = 200;
        }
      }
    }
  } else {
    if ((_memUsage > 30 && _meanFrameTime > 100) || _memUsage > 50 ||
        _meanFrameTime > 110.) {
      if (_sys->_ikdtree.Root_Node) {
        _sys->mutableConfig()._matchMethod = 0;
        _sys->_kf.get_h_dyn_share() =
            std::bind(&System::hShareModelKdTree, _sys, std::placeholders::_1,
                      std::placeholders::_2);

        _releaseThread = new std::thread(releaseVoxelMap);
        // releaseVoxelMap();
        // std::cout<<"Switch to kdtree matching"<<std::endl;
      }
    }
  }
}

void checkFilterPointNum() {
  if (_meanFrameTime > 100.) {
    if (LidarProcess::mutableConfig()._pointFilterNum == 1) {
      LidarProcess::mutableConfig()._pointFilterNum = 2;
      std::cout << "Filter points to 1/2" << std::endl;
    }
  } else {
    if (_sys->mutableConfig()._matchMethod == 1) {
      if (LidarProcess::mutableConfig()._pointFilterNum == 2) {
        LidarProcess::mutableConfig()._pointFilterNum = 1;
        std::cout << "Filter points to 1" << std::endl;
      }
    }
  }
}

void createWorkspace() {
  int nPos = _saveRawPath.rfind('/');
  std::string rawName =
      _saveRawPath.substr(nPos + 1, _saveRawPath.length() - nPos);
  nPos = rawName.rfind('.');
  rawName = rawName.substr(0, nPos);

  _workspace = string(ROOT_DIR) + "result/" + rawName;
  mkdir(_workspace.c_str(), 0777);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging("XXX");
  google::SetCommandLineOption("GLOG_minloglevel", "2");

  plog::init(plog::debug,
             "/home/zzy/RigelSLAM_Rot/devel/logs/mapOptimization.txt",
             64 * 1024 * 1024, 2);

  ros::init(argc, argv, "ZGSlamRosTest");
  ros::NodeHandle _nh;

  _sys = new System();
  setParams();
  loadRosParams(_nh);

  if (!_sys->initSystem()) {
    std::cerr << "System initializing failed!" << std::endl;
    return EXIT_FAILURE;
  }
#if OUTPUT_LOG
  createWorkspace();
  std::ofstream fStatOfs =
      std::ofstream(_workspace + "/stats.txt", std::ios::trunc | std::ios::in);
  _sys->_fTimeOfs = std::ofstream(_workspace + "/timeCost.txt",
                                  std::ios::trunc | std::ios::in);
#endif
  // std::ofstream fout(string(ROOT_DIR)+"time/timecost.txt", std::ios::out);

  _controlMarkers.reset(new pcl::PointCloud<pcl::PointXYZI>());
  _path.header.stamp = ros::Time::now();
  _path.header.frame_id = "odom";

  ros::Subscriber subPcl = _nh.subscribe(_lidTopic, 2000, lidarCallback);
  ros::Subscriber subImu = _nh.subscribe(_imuTopic, 20000, imuCallback);
  ros::Subscriber subMotor = _nh.subscribe(_motorTopic, 200000, motorCallback);
  ros::Subscriber subStopNode =
      _nh.subscribe<std_msgs::Int32>(_stopTopic, 2, &stopNodeHandler, nullptr,
                                     ros::TransportHints().tcpNoDelay());
  ros::Subscriber subCapControlMarker = _nh.subscribe<std_msgs::Int32>(
      _ctrlMarkerGetTopic, 5, &captureControlMarkerHandler, nullptr,
      ros::TransportHints().tcpNoDelay());
  ros::Publisher pubImg = _nh.advertise<sensor_msgs::Image>(_imgTopic, 1);
  ros::Publisher pubImg_left =
      _nh.advertise<sensor_msgs::Image>(_imgTopic_left, 1);
  ros::Publisher pubImg_right =
      _nh.advertise<sensor_msgs::Image>(_imgTopic_right, 1);
  ros::Publisher publine =
      _nh.advertise<sensor_msgs::PointCloud2>(_lineTopic, 1);
  ros::Publisher pubMap = _nh.advertise<sensor_msgs::PointCloud2>(_mapTopic, 1);
  ros::Publisher pubPath = _nh.advertise<nav_msgs::Path>(_pathTopic, 1);
  ros::Publisher pubOdomAftMapped =
      _nh.advertise<nav_msgs::Odometry>(_odomTopic, 10);
  ros::Publisher pubLaserCloudFullBody =
      _nh.advertise<sensor_msgs::PointCloud2>(_frameTopic, 1);
  // ros::Publisher pubLaserCloudFull =
  // _nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
  _pubControlMarkers =
      _nh.advertise<sensor_msgs::PointCloud2>(_ctrlMarkerDispTopic, 1);
  _pubControlMarkerCaptured =
      _nh.advertise<std_msgs::Int32>(_ctrlMarkerPubTopic, 5);
  _pubStopedNode = _nh.advertise<std_msgs::Int32>(_endTopic, 2);
  _pubLoopNode = _nh.advertise<visualization_msgs::MarkerArray>(_loopTopic, 2);

  std::cout << "Subscribed lidar topic:" << _lidTopic << std::endl;
  std::cout << "Subscribed imu topic:" << _imuTopic << std::endl;
  std::cout << "Subscribed motor topic:" << _motorTopic << std::endl;

  // 读取控制点回放的文件， 会修改KF选择的角度和距离阈值
  if (_isPlayBackTask) {
    if (readControlPointInfoFile()) {
      _bReadedPlayBackFile = true;
      printf("控制点信息文件读取完成....\n");
    }
  }

  // rusage rs1,rs2;
  // getrusage(RUSAGE_SELF, &rs1);
  double availableMemOri = getAvailableMemory();

#if OFFLINE_TEST
  ///////////////Offline mode//////////////////
  if (argc < 1) {
    std::cout << "Not enough params!" << std::endl;
    return EXIT_FAILURE;
  }
  std::string bagPath = argv[1];
  rosbag::Bag bag;
  bag.open(bagPath, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back("/ZG/Motor");
  topics.push_back("/ZG/IMU_data");
  topics.push_back("/ZG/lidar");
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  std::cout << "Collecting imu and motor messages..." << std::endl;
  for (auto const& m : view) {
    rigelslam_rot::motor::ConstPtr motorMsg =
        m.instantiate<rigelslam_rot::motor>();
    if (motorMsg != nullptr) motorCallback(motorMsg);

    sensor_msgs::ImuConstPtr imuMsg = m.instantiate<sensor_msgs::Imu>();
    if (imuMsg != nullptr) imuCallback(imuMsg);

    sensor_msgs::PointCloud2ConstPtr lidMsg =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (lidMsg) {
      TicToc frameTime;
      lidarCallback(lidMsg);
      if (_sys->syncPackages(_sys->_measures)) {
        // checkFilterPointNum();
        // checkMatchMethod();

        if (!_sys->config()._isMotorInitialized)
          _sys->motorInitialize();
        else
          _sys->mapping();

        // for display
        publishPath(pubPath);
        publishOdometry(pubOdomAftMapped);
        publishFrameBody(pubLaserCloudFullBody);
        // publishMap(pubMap);
        publishMapIncremental(pubMap);
        publishLoopInfos();

        addControlPoint();

        if (_sys->_frameId % 20 == 0) malloc_trim(0);

        _memUsage = availableMemOri - getAvailableMemory();
        _meanFrameTime += (frameTime.toc() - _meanFrameTime) / 100;
        printf("Frame procecssed time %f ms\n", _meanFrameTime);
        printf("Memory usage %lf Gb\n", std::max(0., _memUsage));
#if OUTPUT_LOG
        fStatOfs << std::fixed << std::setprecision(8) << _meanFrameTime << ","
                 << _memUsage << std::endl;
#endif
        std::cout << "Frame id: " << _sys->_frameId << std::endl << std::endl;
        _sys->_frameId++;
        // PAUSE;
      }
    }
  }

  bag.close();
  _sys->saveMap(true);  // save frames cloud left

  if (!_sys->loopCloser()->config()._isCorrectRealTime)
    _sys->correctLoop();  // loop closing after all loop info detected

  // for display
  // publishMap(pubMap, true);
  publishPath(pubPath);
  publishOdometry(pubOdomAftMapped);
  publishFrameBody(pubLaserCloudFullBody);
  // publish_frame_world(pubLaserCloudFull);
  publishLoopInfos();

  if (!_sys->_loopCloser->isFinished()) {
    _sys->_loopCloser->requestFinish();
    std::cout << "Loopcloser request finish" << std::endl;
    while (!_sys->_loopCloser->isFinished()) usleep(5000);
  }
  if (_isContinueTask) _sys->processContinualTask(_lastTaskPath);
  _sys->saveLastTaskInfo(_lastTaskPath);

  saveRawData();
  saveControlPointInfoFile();

  return EXIT_SUCCESS;
#endif  // OFFLINE_TEST

  ///////////////online mode///////////////////
  signal(SIGINT, sigHandle);
  bool status = ros::ok();
  while (status) {
    if (_exitFlag) break;
    ros::spinOnce();

    if (_sys->syncPackages(_sys->_measures)) {
      // checkFilterPointNum();
      // checkMatchMethod();

      TicToc frameTime;
      if (!_sys->config()._isMotorInitialized)
        _sys->motorInitialize();
      else
        _sys->mapping_undist();

      // for display
      // imagecreatortest();
      // if(_sys->_frameId==41)
      // {
      //     cv::imwrite("/home/zzy/SLAM/my_slam_work/src/Rot_intensity_augm_LIVO/image/sparse.png",_sys->_intensityImg);
      // }
      // publishImage(pubImg);
      // publishImage_left(pubImg_left);
      // publishImage_right(pubImg_right);
      // publishPath(pubPath);
      // publishMapline(publine);
      // publishOdometry(pubOdomAftMapped);
      publishFrameBody_undist(pubLaserCloudFullBody);
      // publishMap(pubMap);
      // publishMapIncremental(pubMap);
      // publishLoopInfos();

      // addControlPoint();
      if (_sys->_frameId % 20 == 0) malloc_trim(0);
      _memUsage = availableMemOri - getAvailableMemory();
      _meanFrameTime += (frameTime.toc() - _meanFrameTime) / 100;
      printf("Frame procecssed time %f ms\n", _meanFrameTime);
      printf("Memory usage %lf Gb\n", std::max(0., _memUsage));
#if OUTPUT_LOG
      fStatOfs << std::fixed << std::setprecision(8) << _meanFrameTime << ","
               << _memUsage << std::endl;
#endif
      // fout<<_sys->_frameId<<" "<<_meanFrameTime<<" "<<_memUsage<<std::endl;
      std::cout << "Frame id: " << _sys->_frameId << std::endl << std::endl;

      _sys->_frameId++;
    }
  }

  ros::shutdown();

  return EXIT_SUCCESS;
}
