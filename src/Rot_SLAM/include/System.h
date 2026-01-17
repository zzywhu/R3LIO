//
// Created by w on 2022/9/26.
//

#ifndef SRC_SYSTEM_H
#define SRC_SYSTEM_H

#ifdef _WIN32
#ifndef _UNISTD_H
#define _UNISTD_H
#include <io.h>
#include <process.h>
#endif /* _UNISTD_H */
#endif

#include <ceres/rotation.h>
#include <math.h>
#include <omp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <boost/circular_buffer.hpp>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <tuple>

#include "list"
#include "set"
//#include "pandarGeneral/pandarGeneral.h"
//#include "pandarGeneral/point_types.h"
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

#include "BEVProcess/BEVprojector.h"
#include "BEVProcess/bev_feature.h"
#include "DataIO/ReadWriter.h"
#include "Feature/FeatureExtractor.h"
#include "Feature/PCA.hpp"
#include "ImageProcess/imageprocess.h"
#include "Initializer/LI_Initiator.h"
#include "Initializer/LidMotorCalib.h"
#include "LoopCloser/LoopCloser.h"
#include "MLSD/mlsd.h"
#include "Mapper/CloudBlock.h"
#include "Mapper/IkdTree.h"
#include "Mapper/VoxelMapper.h"
#include "Mapper/VoxelVGICP.hpp"
#include "Misc/Common.h"
#include "Misc/So3Math.h"
#include "Misc/TicToc.h"
#include "Misc/calculate_pose.h"
#include "Misc/use-ikfom.h"
#include "Optimizer/Optimizer.h"
#include "Registrator/AbsoluteOrientation.h"
#include "Registrator/BaseRegistration.h"
#include "Registrator/LsqRegPCL_impl.hpp"
#include "Registrator/RegistrationICP.h"
#include "Registrator/RegistrationLoam.h"
#include "Registrator/RegistrationVGICP.h"
#include "Registrator/fast_gicp/gicp/fast_gicp.hpp"
#include "Registrator/fast_gicp/gicp/fast_vgicp.hpp"
#include "Registrator/fast_gicp/gicp/impl/fast_gicp_impl.hpp"
#include "Registrator/fast_gicp/gicp/impl/fast_vgicp_impl.hpp"
#include "Registrator/fast_gicp/gicp/impl/lsq_registration_impl.hpp"
#include "RigelSLAMRawIOTools.h"
#include "SensorProcess/IMUPreintegration.h"
#include "SensorProcess/IMUProcess.h"
#include "SensorProcess/LidarProcess.h"
#include "Utility.hpp"
#include "Viewer/Viewer.h"
#include "patchwork/patchworkpp.h"
#include "xfeat/XFeat.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define SYSTEM_API __declspec(dllexport)
#else
#define SYSTEM_API __declspec(dllimport)
#endif
#else
#define SYSTEM_API
#endif

using pcl::visualization::PointCloudColorHandlerCustom;

#define INIT_TIME (0.1)

const float MOV_THRESHOLD = 1.5f;

class SYSTEM_API System {
  struct SYSTEM_API Config {
    ////////////System common setting////////////
    int _udpateMethod;
    int _matchMethod;
    std::string _imuFilePath;
    std::string _rasterFilePath;
    std::string _pcapFilePath;
    std::string _lidarCorrectFilePath;
    bool _isTimeSyncEn;
    bool _isEnable3DViewer;
    bool _isMotorInitialized;
    bool _isImuInitialized;
    bool _isFeatExtractEn;
    bool _isEstiExtrinsic;
    bool _isUseIntensity;
    bool _isUseMultiview;
    int _featureExtractSegNum;
    int _minFramePoint;
    int _nSkipFrames;
    int _nMergedFrames;

    bool _isCutFrame;
    int _cutFrameNum;
    int _origOdomFreq;
    double _onlineRefineTime;
    double _meanAccNorm;
    double _dataAccumLength;
    std::vector<double> _rotLICov;
    std::vector<double> _transLICov;
    bool _enableGravityAlign;
    double _radius_k;
    double _maxInierError;

    int _nMaxInterations;
    float _detRange;
    double _cubeLen;
    double _filterSizeSurf;
    double _filterSizeMap;
    double _gyrCov;
    double _accCov;
    double _bGyrCov;
    double _bAccCov;
    double _timeLagIMUWtrLidar;
    std::vector<double> _tolVec;
    std::vector<double> _RolVec;
    std::vector<double> _tilVec;
    std::vector<double> _RilVec;
    std::vector<double> _tigVec;
    std::vector<double> _RigVec;
    int _imuMaxInitCount;
    int _gnssInitSize;
    double _gnssMaxError;

    int _maxPointsSize;
    int _maxCovPointsSize;
    std::vector<int> _layerPointSizeList;
    int _maxLayers;
    double _voxelLength;
    double _minSurfEigenValue;
    double _rangingCov;
    double _angleCov;
    int _covType;

    // BundleAdjustment:
    bool _isEnableBA;
    bool _isVerbose;
    int _filterNum;
    int _thdNum;
    int _windowSize;
    int _marginSize;

    bool _isLoopEn;
    bool _issavemap;
    bool _isSaveMap;
    int _pcdSaveInterval;

    Config() {
      _imuFilePath = "";
      _rasterFilePath = "";
      _pcapFilePath = "";
      _lidarCorrectFilePath = "";
      _udpateMethod =
          1;  // 0: EKF with IMU, 1: LSQ with IMU, other: LSQ without IMU
      _matchMethod = 1;  //# 0: kdtree,  1: voxelmap
      _isTimeSyncEn = false;
      _isEnable3DViewer = false;
      _isMotorInitialized = true;
      _isImuInitialized = true;
      _isFeatExtractEn = true;
      _isEstiExtrinsic = false;
      _featureExtractSegNum = 6;
      _minFramePoint = 1000;
      _nSkipFrames = 0;
      _nMergedFrames = 0;

      _isCutFrame = false;
      _cutFrameNum = 5;
      _origOdomFreq = 10;
      _onlineRefineTime = 20.;
      _meanAccNorm = 9.81;
      _dataAccumLength = 300;
      _rotLICov = std::vector<double>(3, 0.00005);
      _transLICov = std::vector<double>(3, 0.0005);
      _radius_k = 3;
      _enableGravityAlign = true;
      _maxInierError = 1;

      _nMaxInterations = 5;
      _detRange = 300.0f;
      _cubeLen = 2000;
      _filterSizeSurf = 0.5;
      _filterSizeMap = 0.5;
      _gyrCov = 0.1;
      _accCov = 0.1;
      _bGyrCov = 0.0001;
      _bAccCov = 0.0001;
      _timeLagIMUWtrLidar = 0.;
      _tolVec = std::vector<double>(3, 0.0);
      _RolVec = std::vector<double>(9, 0.0);
      _tilVec = std::vector<double>(3, 0.0);
      _RilVec = std::vector<double>(9, 0.0);
      _tigVec = std::vector<double>(3, 0.0);
      _RigVec = std::vector<double>(9, 0.0);
      _imuMaxInitCount = 1000;
      _gnssInitSize = 20;
      _gnssMaxError = 0.3;

      _maxPointsSize = 100;
      _maxCovPointsSize = 100;
      _layerPointSizeList = std::vector<int>();
      _maxLayers = 2;
      _voxelLength = 1.0;
      _minSurfEigenValue = 0.01;
      _rangingCov = 0.02;
      _angleCov = 0.05;
      _covType = 0;  // 0: observation,  1: vgicp

      _isEnableBA = false;
      _isVerbose = false;
      _filterNum = 1;
      _thdNum = 4;
      _windowSize = 10;
      _marginSize = 5;

      _isLoopEn = false;

      _isSaveMap = true;
      _pcdSaveInterval = -1;
    }
  };

 public:
  System()
      : _sysID(0),
        _localCloudPtr(new PointCloudXYZI()),
        _localSurfCloudPtr(new PointCloudXYZI()),
        _localCornerCloudPtr(new PointCloudXYZI()),
        _localCloudDownPtr(new PointCloudXYZI()),
        _localSurfCloudDownPtr(new PointCloudXYZI()),
        _localCornerCloudDownPtr(new PointCloudXYZI()),
        _globalCloudDownPtr(new PointCloudXYZI()),
        _globalSurfCloudDownPtr(new PointCloudXYZI()),
        _globalCornerCloudDownPtr(new PointCloudXYZI()),
        _downCloudMap(new PointCloudXYZI()),
        _denseCloudMap(new PointCloudXYZI()),
        _trajCloud(new PointCloudXYZI()),
        _lidarProcessor(new LidarProcess()),
        fout_traj(string(ROOT_DIR) + "MapResult/traj.txt", std::ios::out),
        _imuProcessor(new ImuProcess()),
        _initiatorLI(nullptr),
        _jacoRot(MatrixXd(30000, 3)),
        _trajPts(new PointCloudXYZI()),
        _trajPropagatPts(new PointCloudXYZI()),
        _trajGNSSPts(new PointCloudXYZI()),
        _regVGICP(RegistrationVGICP(1)),
        _isFinished(false),
        _curMotorAngle(-1.),
        _initMotorAngle(-1.),
        _frameNum(0),
        _moveStartTime(0.0),
        _onlineCalibStartTime(0.0),
        _Rwg(Eigen::Matrix3d::Identity()),
        _twg(Eigen::Vector3d::Zero()),
        _Rwl(Eigen::Matrix3d::Identity()),
        _twl(Eigen::Vector3d::Zero()),
        _prevTwl(Eigen::Matrix4d::Identity()),
        _Rol(Eigen::Matrix3d::Identity()),
        _Ril(Eigen::Matrix3d::Identity()),
        _isResetShow(false),
        _isEKFInited(false),
        _isInitMap(false),
        _isOnlineCalibFinish(false),
        _isDataAccumFinished(false),
        _isDataAccumStart(false),
        _frameId(0),
        _pcdIndex(0),
        _cloudDownSize(0),
        _cloudSurfDownSize(0),
        _cloudCornerDownSize(0),
        _lidarBegTime(0),
        _lastImuTimestamp(-1.0),
        _lastMotorTimestamp(-1.0),
        _Patchworkpp(patchwork_parameters),
        _timediffImuWrtLidar(0.0),
        _isTimediffSetFlg(false),
        _lidarEndTime(0),
        _isLidarPushed(false),
        _firstLidarTime(0),
        _scanNum(0),
        //_pandarReader(nullptr),
        _dispThread(nullptr),
        _p(nullptr),
        _isDispStop(false),
        _minDispZ(0),
        _maxDispZ(0),
        _isLocalMapInitialized(false),
        _isFirstFrame(true),
        _timeLastScan(0),
        _dt(0.0),
        _mapCloudQueue(200),
        _XFDetector(4096, 0.05, true),
        _loopCloser(nullptr),
        _isLoopCorrected(false),
        _isFirstLidarFrame(true),
        _sensorTimeDiff(0),
        _globalMapPtr(new pcl::PointCloud<pcl::PointXYZ>),
        _globalCloudXYZPtr(new pcl::PointCloud<pcl::PointXYZ>),
        _localCloudXYZPtr(new pcl::PointCloud<pcl::PointXYZ>),
        _sparselinecloud(new pcl::PointCloud<pcl::PointXYZI>),
        _premapPtr(new pcl::PointCloud<pcl::PointXYZI>),
        _curlinecloud(new pcl::PointCloud<pcl::PointXYZI>),
        _sparseworldlinecloud(new pcl::PointCloud<pcl::PointXYZI>),
        _matchlinecloud(new pcl::PointCloud<pcl::PointXYZI>),
        _matchworldlinecloud(new pcl::PointCloud<pcl::PointXYZI>),
        _matchworldlinecloudrgb(new pcl::PointCloud<pcl::PointXYZRGB>),
        intensityMapdense(new pcl::PointCloud<pcl::PointXYZI>),
        intensityMapdense_left(new pcl::PointCloud<pcl::PointXYZI>),
        intensityMapdense_right(new pcl::PointCloud<pcl::PointXYZI>),
        _Twl(Eigen::Matrix4d::Identity()),
        _rotAlign(Eigen::Matrix3d::Identity()),
        _rotAlign_traj(Eigen::Matrix3d::Identity()),
        _normvec(new PointCloudXYZI(100000, 1)),
        _cloudAxisTransfer(nullptr),
        _frameIdDisp(0),
        _isGnssInited(false),
        _gnssTrans(Eigen::Matrix4d::Identity()),
        _gnssDisp(nullptr) {
    _resLast = new float[100000];
    for (int i = 0; i < 100000; i++) _resLast[i] = float(0.0);

    _isPointSelectedSurf = new bool[100000];
    for (int i = 0; i < 100000; i++) _isPointSelectedSurf[i] = bool(0);

    _fTimeOfs = std::ofstream(string(ROOT_DIR) + "result/timeCost.txt",
                              std::ios::trunc | std::ios::in);
  }

  ~System() {
    delete _resLast;
    delete _isPointSelectedSurf;
  }

  void shutdown() {
    if (_loopCloser) {
      if (!_loopCloser->isFinished()) {
        _loopCloser->requestFinish();
        std::cout << "Loopcloser request finish" << std::endl;
        while (!_loopCloser->isFinished()) usleep(5000);
      }
      delete _loopCloser;
      _loopCloser = nullptr;
    }

    if (_dispThread) {
      std::cout << "Shutdown pcl visualization ...";
      _isDispStop = true;
      _dispThread->join();
      _p->close();
      delete _dispThread;
      _dispThread = nullptr;
    }

    // while (!_p->wasStopped())
    // usleep(5000);
  }

  static Config &mutableConfig() { return _config; }
  static const Config &config() { return _config; }

  StatesGroup &getCurState() { return _stateCur; }

  Eigen::Matrix3d &Rol() { return _Rol; }
  Eigen::Vector3d &tol() { return _tol; }

  Eigen::Matrix3d &Ril() { return _Ril; }
  Eigen::Vector3d &til() { return _til; }

  LoopCloser *loopCloser() { return _loopCloser; }

  deque<SensorMsgs::ImuData::Ptr> &imuBufferAll() { return _imuBufferAll; }

  std::list<pair<double, double>> &motorAngleBufferAll() {
    return _motorAngleBufAll;
  }

  // PandarGeneral* pandarReader(){return _pandarReader;}

  long int getCurFrameID() { return _frameId; }

  void setID(const int id) { _sysID = id; }

  bool start();

  void stop();

  bool isFinished();

  void loadParams(const std::string &filePath);

  bool initSystem();

  void resetSystem();

  void lidCallback(PPointCloud::Ptr cloudIn, double lidTime);

  void correctLoop();

  void fileoutCalibResult(ofstream &foutResult, const StatesGroup &state,
                          const double &timeLagIMUWrtLidar);

  // private:
 public:
  float calcDist(const PointType &p1, const PointType &p2);

  static bool varContrast(PointWithCov &x, PointWithCov &y) {
    return (x.obsCov.diagonal().norm() < y.obsCov.diagonal().norm());
  };

  void updateDenseMap(PointCloudXYZI::Ptr cloudPtr);

  void pointBodyToWorld(state_ikfom &stateCur, PointType const *const pi,
                        PointType *const po);

  void pointBodyToWorld(StatesGroup &stateCur, PointType const *const pi,
                        PointType *const po);

  void transCloud(const PointCloudXYZI::Ptr &cloudIn,
                  PointCloudXYZI::Ptr &cloudOut, const Eigen::Matrix3d &Rol,
                  const Eigen::Vector3d &tol);
  void transCloud2(const PointCloudXYZI::Ptr &cloudIn,
                   pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudOut,
                   const Eigen::Matrix3d &Rol, const Eigen::Vector3d &tol);
  void transCloud3(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudIn,
                   pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudOut,
                   const Eigen::Matrix3d &Rol, const Eigen::Vector3d &tol);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertPointCloudWithLines(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud);

  void transCloudInMotorAxis(const PointCloudXYZI::Ptr &cloudIn,
                             PointCloudXYZI::Ptr &cloudOut, const double &angle,
                             const Eigen::Matrix3d &Rol,
                             const Eigen::Vector3d &tol);

  void convertToRGBCloud(PointCloudXYZI &cloudIn,
                         pcl::PointCloud<pcl::PointXYZRGB> &cloudRGB);

  /// viewer
  void voxelMap2RGBClouds(
      const std::unordered_map<VOXEL_LOC, OctoTree *> &voxelMap,
      PointCloudXYZRGB::Ptr octClouds);

  void initPCLViewer();

  void resetViewer();

  void updateViewer(bool bForceUpdateMap = false);

  void updateTrajectory();

  void showKdTree(const int &vp);

  void showLocalDownCloud(const int &vp);

  void showLocalFeatCloud(const int &vp);

  void showDenseMapCloud(const int &vp);

  void showTrajectory(const int &vp);

  void showLoopClosure(const int &vp);

  void showVoxelMap(const int &vp);

  void showMatchedPoints(const int &vp);

  void show();

  /// viewer

  void printState(const state_ikfom &state,
                  const std::string &prefix = "Final Result");

  void printState(const std::string &prefix = "Final Result");

  void printProgress(double percentage);

  void laserMapFovSegment();

  bool collectRasterAngles(const double &begTime, const double &endTime);

  bool getRasterAngle(const double &curTime);

  int matchKdTree(const PointCloudXYZI::Ptr &localCloudPtr,
                  const PointCloudXYZI::Ptr &globalCloudPtr, KD_TREE &kdTree,
                  MatchedInfoList &matches);

  void saveMap(bool isForced = false);
  void Savemap();
  void Savetraj();
  void motorMotionCompensation();

  void motorMotionCompensationZG();

  void motorMotionCompensationZGforSIM();

  bool lidarMotionCompensationWithTimestamp(
      typename pcl::PointCloud<PointType>::Ptr &cloudInOut,
      const Eigen::Matrix4d &T12,  // T12 (from k+1 to k frame)
      float minScanDurationMs = 50);

  void predictAndUndistort(const MeasureGroup &meas,
                           PointCloudXYZI::Ptr pclOut);

  void initKdTreeMap(PointVector pointCloud = PointVector());

  void updateKdTreeMap(PointVector &pointsIn);

  void initVoxelMapWithKdtree(PointVector pointCloud = PointVector());

  void updateVoxelMapWithKdtree(PointVector &pointsIn);

  void initVoxelMap();

  void updateVoxelMap();

  bool motionCheck();

  void calibAndRefineLI();

  void motorInitialize();

  void processCloudOnlyMotor();

  void processCloudCeres();

  void processCloudLSQ(Registration &reg);

  int processCloudLSQ(RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> &reg);

  void processCloudVGICP();

  void processCloudLOAM();

  void processCloudESIKF();

  cv::Mat projectToXZ(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                      bool isinter);

  cv::Mat overlayRedOnGrayscale(const cv::Mat &gray1, const cv::Mat &gray2);

  cv::Mat stackImagesVertical(const cv::Mat &gray1, const cv::Mat &gray2);

  cv::Mat interpolateBlackRegions(const cv::Mat &inputImage);

  void removeLines(cv::Mat &img);

  void filterBrightness(cv::Mat &img);

  void imagecreator_calib();
  void imagecreatortest();
  void imagecreatoropt();
  void buildsurfmap(pcl::PointCloud<pcl::PointXYZI>::Ptr &densecloud);
  void processCloudIKFoM();
  void processCloudIKFoM_calib();

  void hShareModelKdTree(state_ikfom &s,
                         esekfom::dyn_share_datastruct<double> &ekfom_data);

  void hShareModelVoxelMap(state_ikfom &s,
                           esekfom::dyn_share_datastruct<double> &ekfom_data);

  bool updateKFVoxelMap();

  bool updateKFKdTree();

  bool assertDegeneracy();

  void updateFrontendMap();

  void updateLoopStatus(state_ikfom &state);

  void updateLoopStatus(StatesGroup &state);

  void loopClosing(state_ikfom &state);

  void loopClosing(StatesGroup &state);

  void processGNSS();

  void mapping();
  void mapping_undist();
  void mapping_calib();
  void mapping_sim();
  bool syncPackages(MeasureGroup &meas);

  void collectMotorIMU();

  double getSensorTimeDiff(double lidTime, double imuTime);

  void deleteUnstablePoints();

  void voxelFilter(pcl::PointCloud<PointType>::Ptr &cloudIn,
                   pcl::PointCloud<PointType>::Ptr &cloudOut, float gridSize);

  void saveLastTaskInfo(const std::string &lastTaskPath);

  bool readLastTaskInfo(const std::string &lastTaskPath, float *refXYZ,
                        pcl::PointCloud<PointType>::Ptr &lastSubmap);

  bool processContinualTask(const std::string &lastTaskPath);

  struct surfmaplist {
    std::vector<cv::Mat> surfmap;
  };

 public:
  static Config _config;
  int _sysID;

  pcl::visualization::PCLVisualizer *_p;
  int _vp1;

  bool _isResetShow;
  bool _isEKFInited;
  bool _isInitMap;
  bool _isOnlineCalibFinish;
  bool _isDataAccumFinished;
  bool _isDataAccumStart;

  ofstream _foutResult;

  long int _frameId;
  int _pcdIndex;

  int _cloudDownSize;
  int _cloudSurfDownSize;
  int _cloudCornerDownSize;
  PointCloudXYZI::Ptr _localCloudPtr;
  PointCloudXYZI::Ptr _localSurfCloudPtr;
  PointCloudXYZI::Ptr _localCornerCloudPtr;
  boost::circular_buffer<PointCloudXYZI::Ptr> _mapCloudQueue;
  cv::Mat _intensityImg;
  cv::Mat _intensityImg_left;
  cv::Mat _intensityImg_right;
  pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense;
  pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense_left;
  pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense_right;
  cv::Mat _matchImg;
  cv::Mat _matchImg_left;
  cv::Mat _matchImg_right;
  PointCloudXYZI::Ptr _localCloudDownPtr;
  PointCloudXYZI::Ptr _localSurfCloudDownPtr;
  PointCloudXYZI::Ptr _localCornerCloudDownPtr;

  PointCloudXYZI::Ptr _globalCloudDownPtr;
  PointCloudXYZI::Ptr _globalSurfCloudDownPtr;
  PointCloudXYZI::Ptr _globalCornerCloudDownPtr;

  PointCloudXYZI::Ptr _downCloudMap;
  PointCloudXYZI::Ptr _denseCloudMap;
  PointCloudXYZI::Ptr _trajCloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr _sparselinecloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr _curlinecloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr _sparseworldlinecloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr _matchlinecloud;
  pcl::PointCloud<pcl::PointXYZI>::Ptr _matchworldlinecloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr _matchworldlinecloudrgb;
  cv::Mat _cannyimg;

  vector<PointVector> _nearestPoints;
  PointVector _addedPoints;
  KD_TREE _ikdtree;
  std::unordered_map<VOXEL_LOC, OctoTree *> _voxelSurfMap;
  std::unordered_map<VOXEL_LOC, OctoTree *> _voxelCornMap;

  shared_ptr<LidarProcess> _lidarProcessor;
  shared_ptr<ImuProcess> _imuProcessor;
  shared_ptr<ImuPreintegration> _imuPreintegration;
  XFeat::XFDetector _XFDetector;
  imgProcesser _imgProcesser;
  cv::Mat _mkpts_0;
  cv::Mat _mkpts_1;
  cv::Mat _scanlineIdMap;
  surfmaplist _surfmaplist;
  std::unordered_map<VOXEL_LOC, Voxel *> _sparsevoxelmap;
  // raster angle
  double _curMotorAngle;
  double _initMotorAngle;

  // Extrinsic
  //  motor <- lidar
  Eigen::Matrix3d _Rol;
  Eigen::Vector3d _tol;
  // imu <- motor
  Eigen::Matrix3d _Ril;
  Eigen::Vector3d _til;
  // imu <- RTK
  Eigen::Matrix3d _Rig;
  Eigen::Vector3d _tig;
  // GNSS pose
  Eigen::Matrix3d _Rwg;
  Eigen::Vector3d _twg;
  // Lidar pose
  Eigen::Matrix3d _Rwl;
  Eigen::Vector3d _twl;
  Eigen::Matrix3d _Rwlprop;
  Eigen::Vector3d _twlprop;
  Eigen::Matrix3d _Rl2l;
  Eigen::Vector3d _tl2l;
  Eigen::Matrix4d _prevTwl;
  std::ofstream fout_traj;

  std::vector<Eigen::Matrix4d> _relToList;
  std::vector<Eigen::Matrix4d> _relTlList;

  // // LI Calib Parameters

  int _frameNum;
  double _moveStartTime;
  double _onlineCalibStartTime;

  shared_ptr<LI_Init> _initiatorLI;

  MatrixXd _jacoRot;
  MD(DIM_STATE, DIM_STATE)
  _G;
  MD(DIM_STATE, DIM_STATE)
  _H_T_H;
  MD(DIM_STATE, DIM_STATE)
  _I_STATE;

  MeasureGroup _measures;
  StatesGroup _stateCur;
  StatesGroup _statePropagat;

  esekfom::esekf<state_ikfom, 12, input_ikfom> _kf;
  state_ikfom _stateIkfom;
  state_ikfom _statePropIkfom;

  mutex _mtxBuffer;
  condition_variable _sigBuffer;
  double _lidarBegTime;
  double _lastImuTimestamp;
  double _lastMotorTimestamp;
  double _lastGPSTimestamp;
  double _timediffImuWrtLidar;
  bool _isTimediffSetFlg;
  double _lidarEndTime;
  bool _isLidarPushed;
  double _firstLidarTime;

  int _scanNum;

  std::deque<double> _timeBuffer;
  std::deque<PointCloudXYZI::Ptr> _lidarBuffer;
  std::deque<PointCloudXYZI::Ptr> _mapBuffer;
  std::deque<PointCloudXYZI::Ptr> _cloudBuffer;
  std::deque<SensorMsgs::ImuData::Ptr> _imuBuffer;
  std::list<pair<double, double>> _motorAngleBuf;
  std::deque<SensorMsgs::GNSSData::Ptr> _gpsBuffer;
  std::deque<SensorMsgs::ImuData::Ptr> _imuBufferAll;
  std::list<pair<double, double>> _motorAngleBufAll;
  std::deque<SensorMsgs::GNSSData::Ptr> _gpsBufferAll;
  std::deque<std::pair<SensorMsgs::GNSSData::Ptr, int>> _gpsConstraintBuf;
  // PandarGeneral * _pandarReader;

  MatchedInfoList _matchInfos;

  queue<PointCloudXYZI::Ptr> _localCloudQueue;
  queue<PointCloudXYZI::Ptr> _localCloudDownQueue;
  queue<PointCloudXYZI::Ptr> _localPlaneDownQueue;
  queue<PointCloudXYZI::Ptr> _localLineDownQueue;
  queue<PointCloudXYZI::Ptr> _downCloudMapQueue;
  queue<state_ikfom> _stateQueue;
  queue<state_ikfom> _statePropagatQueue;
  queue<Eigen::Matrix3d> _RQueue;
  queue<Eigen::Vector3d> _tQueue;
  queue<double> _motorAngleQueue;
  queue<PointCloudXYZRGB::Ptr> _voxelCloudQueue;
  queue<std::vector<MatchOctoTreeInfo>> _matchedSurfListQueue;
  queue<std::vector<std::pair<int, int>>> _lcIdxPairsQueue;
  queue<SensorMsgs::GNSSData::Ptr> _gnssQueue;

  long int _frameIdDisp;
  PointCloudXYZRGB::Ptr _voxelMapCloudDisp;
  PointCloudXYZI::Ptr _localCloudDisp;
  PointCloudXYZI::Ptr _localCloudDownDisp;
  PointCloudXYZI::Ptr _localPlaneDownDisp;
  PointCloudXYZI::Ptr _localLineDownDisp;
  PointCloudXYZI::Ptr _downCloudMapDisp;
  state_ikfom _stateDisp;
  state_ikfom _statePropagatDisp;
  Eigen::Matrix3d _RDisp;
  Eigen::Vector3d _tDisp;
  double _motorAngleDisp;
  std::vector<MatchOctoTreeInfo> _matchedSurfListDisp;
  std::vector<std::pair<int, int>> _lcIdxPairsDisp;
  SensorMsgs::GNSSData::Ptr _gnssDisp;

  PointCloudXYZI::Ptr _trajPts;
  PointCloudXYZI::Ptr _trajPropagatPts;
  PointCloudXYZI::Ptr _trajGNSSPts;
  std::vector<Colour> _voxelColors;

  std::mutex _dispMutex;
  std::thread *_dispThread;
  bool _isDispStop;

  double _minDispZ;
  double _maxDispZ;

  RegistrationICP _regICP;
  RegistrationVGICP _regVGICP;
  RegistrationPCL_VGICP<pcl::PointXYZ, pcl::PointXYZ> _regPCL_VGICP;

  std::unique_ptr<GaussianVoxelMap<PointType>> _voxelMapGaussian;

  vector<BoxPointType> _cubNeedrm;
  bool _isLocalMapInitialized;
  BoxPointType _localMapPoints;

  bool _isFirstFrame;
  double _timeLastScan;
  double _dt;

  std::vector<double> _rotDeltas;
  std::vector<double> _posDeltas;

  std::vector<MatchOctoTreeInfo> _matchedSurfList;
  std::vector<MatchOctoTreeInfo> _matchedCornerList;
  std::vector<int> _matchSurfSizeList;
  std::vector<int> _matchCornerSizeList;
  std::vector<PointWithCov> _curSurfPvList;
  std::vector<PointWithCov> _curCornerPvList;
  std::vector<BoxPointType> _boxToDel;
  std::vector<std::vector<int>> _linelist;
  std::vector<std::vector<int>> _linelist_left;
  std::vector<std::vector<int>> _linelist_right;
  std::vector<Matchlinelist> _matchlinelist;
  std::vector<Matchlinelist> _matchlinelist_left;
  std::vector<Matchlinelist> _matchlinelist_right;

  std::shared_ptr<BEVProjector> projectormap;
  std::shared_ptr<BEVProjector> projectorcurrent;
  std::shared_ptr<BEVFeatureManager> bev_manager;
  BEVFrame map_frame;
  patchwork::Params patchwork_parameters;
  patchwork::PatchWorkpp _Patchworkpp;

  std::string _map_path;

  LoopCloser *_loopCloser;
  bool _isLoopCorrected;

  bool _isFirstLidarFrame;
  double _sensorTimeDiff;

  bool _isFinished;

  pcl::PointCloud<pcl::PointXYZI>::Ptr _premapPtr;

  pcl::PointCloud<pcl::PointXYZ>::Ptr _globalMapPtr;
  pcl::PointCloud<pcl::PointXYZ>::Ptr _globalCloudXYZPtr;
  pcl::PointCloud<pcl::PointXYZ>::Ptr _localCloudXYZPtr;
  Eigen::Matrix4d _Twl;
  fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> _fastVGICPReg;

  LoamTracker _loamTracker;

  Eigen::Vector3d _globalGrav;
  Eigen::Matrix3d _rotAlign;
  Eigen::Matrix3d _rotAlign_traj;
  PointCloudXYZI::Ptr _normvec;
  bool *_isPointSelectedSurf;
  float *_resLast;

  std::set<OctoTree *> _voxelCovUpdated;

  ZGAxisTransfer *_cloudAxisTransfer;

  std::ofstream _fTimeOfs;
  bool _isGnssInited;
  Eigen::Matrix4d _gnssTrans;
  std::vector<Eigen::Matrix4d> _TilList;

  SensorMsgs::GNSSData::Ptr _lastGNSSData;
};

#endif  // SRC_SYSTEM_H
