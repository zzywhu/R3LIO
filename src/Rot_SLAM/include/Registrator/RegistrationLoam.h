//
// Created by w on 2022/11/23.
//

#ifndef FAST_LIO_WIN_REGISTRATIONLOAM_H
#define FAST_LIO_WIN_REGISTRATIONLOAM_H
#include "Common.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>



#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define LOAM_TRACKER_API __declspec(dllexport)
#else
#define LOAM_TRACKER_API __declspec(dllimport)
#endif
#else
#define LOAM_TRACKER_API
#endif

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::G; // GNSS pose
using namespace std;

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                           (float, z, z) (float, intensity, intensity)
                                           (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                           (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class LOAM_TRACKER_API LoamTracker
{
public:
    struct Config
    {
        int      _nScans;
        int      _nHorizonScan;
        double   _mappingCornerLeafSize;
        double   _mappingSurfLeafSize;
        double   _surroundingKeyframeDensity;
        double   _mappingProcessInterval;
        double   _surroundingKeyframeSearchRadius;
        int      _edgeFeatureMinValidNum;
        int      _surfFeatureMinValidNum;
        double   _surroundingkeyframeAddingAngleThreshold;
        double   _surroundingkeyframeAddingDistThreshold;
        double   _zTollerance;
        double   _rotationTollerance;
        Config()
        {
            _nScans=16;
            _nHorizonScan=2000;
            _mappingCornerLeafSize=0.2;
            _mappingSurfLeafSize=0.4;
            _surroundingKeyframeDensity=0.1;
            _mappingProcessInterval=0.05;
            _surroundingKeyframeSearchRadius=50.;
            _edgeFeatureMinValidNum=10;
            _surfFeatureMinValidNum=100;
            _surroundingkeyframeAddingAngleThreshold=0;
            _surroundingkeyframeAddingDistThreshold=0;
            _zTollerance = 1000;
            _rotationTollerance=1000;
        }
    };


    LoamTracker();

    static Config& mutableConfig(){return _config;}

    static const Config& config(){return _config;}

    void allocateMemory();

    void setInitGuess(Eigen::Matrix4d& guess);

    bool isDegenerate(){return _isDegenerate;}

    Eigen::Affine3f track(const double& timestamp,
                          const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
                          const pcl::PointCloud<PointType>::Ptr& laserCloudSurf);

    void pointAssociateToMap(PointType const * const pi, PointType * const po);

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn);

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }


    PointTypePose trans2PointTypePose(float transformIn[]);


    void updateInitialGuess();

    void extractNearby();

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract);

    void extractSurroundingKeyFrames();

    void downsampleCurrentScan();

    void updatePointAssociateToMap();

    void cornerOptimization();

    void surfOptimization();

    void combineOptimizationCoeffs();

    bool LMOptimization(int iterCount);

    void scan2MapOptimization();

    void transformUpdate();

    float constraintTransformation(float value, float limit);

    bool saveFrame();

    void addOdomFactor();

    void saveKeyFramesAndFactor();

    Eigen::Affine3f getLastPose(){return trans2Affine3f(transformTobeMapped);}

private:
    // gtsam
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::ISAM2 *isam;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

//    CloudInfo cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> fullCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudFullLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    double timeLaserCloudInfoLast;

    float transformTobeMapped[6];

    std::mutex mtx;

    double timeLastProcessing = -1;

    bool _isDegenerate;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int  laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    int imuPreintegrationResetId = 0;

    Eigen::Affine3f transPointAssociateToMap;


    static Config  _config;
};


#endif //FAST_LIO_WIN_REGISTRATIONLOAM_H
