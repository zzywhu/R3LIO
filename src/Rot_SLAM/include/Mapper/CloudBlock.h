//
// Created by w on 2022/9/14.
//

#ifndef SRC_CLOUDBLOCK_H
#define SRC_CLOUDBLOCK_H
#include <boost/make_shared.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/point_representation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include "Common.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define CLOUD_BLOCK_API __declspec(dllexport)
#else
#define CLOUD_BLOCK_API __declspec(dllimport)
#endif
#else
#define CLOUD_BLOCK_API
#endif

typedef pcl::search::KdTree<PointType>::Ptr PcTreePtr;
typedef pcl::search::KdTree<PointType> PcTree;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

//the edge (constraint)'s type
enum ConstraintType
{
    REGISTRATION,
    ADJACENT,
    HISTORY,
    SMOOTH,
    NONE
};

//regular bounding box whose edges are parallel to x,y,z axises
struct CLOUD_BLOCK_API Bounds
{
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;
    int type;

    Bounds()
    {
        min_x = min_y = min_z = max_x = max_y = max_z = 0.0;
    }
    void inf_x()
    {
        min_x = -DBL_MAX;
        max_x = DBL_MAX;
    }
    void inf_y()
    {
        min_y = -DBL_MAX;
        max_y = DBL_MAX;
    }
    void inf_z()
    {
        min_z = -DBL_MAX;
        max_z = DBL_MAX;
    }
    void inf_xyz()
    {
        inf_x();
        inf_y();
        inf_z();
    }
};

struct CLOUD_BLOCK_API CenterPoint
{
    double x;
    double y;
    double z;
    CenterPoint(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
};


class CLOUD_BLOCK_API CloudBlock
{
public:
    CloudBlock()
    {
        init();
    }

    ~CloudBlock()
    {
        freeAll();
    }

    CloudBlock(const CloudBlock &in_block, bool clone_raw = false)
    {
        init();
        cloneMetadata(in_block);
    }

    void init();

    void freeRawCloud();

    void freeAll();

    void cloneMetadata(const CloudBlock &inBlock);

    void cloneCloud(PointCloudXYZI::Ptr &pcOut, bool getPcDone);

    void appendFeature(const CloudBlock &inBlock);

    void transformFeature(const Eigen::Matrix4d &transMat);

public:
    double                        _timestamp;
    int                           _uniqueId;		  //Unique ID
    int                           _idInStrip;	  //ID in the strip
    int                           _lastFrameId; //_lastFrameId is the frame index (not _uniqueId) of the last frame of the submap
    //ID means the number may not be continous and begining from 0 (like 3, 7, 11, ...),
    //but index should begin from 0 and its step (interval) should be 1 (like 0, 1, 2, 3, ...)

    Bounds                        _bound;				  //Bounding Box in geo-coordinate system
    CenterPoint                   _center;			  //Center Point in geo-coordinate system
    CenterPoint                   _station;		  //Station position in geo-coordinate system
    Eigen::Matrix4d               _stationPose; //Station pose in geo-coordinate system

    Bounds                        _localBound;				//Bounding Box in local coordinate system
    CenterPoint                   _localCenter;				//Center Point in local coordinate system
    CenterPoint                   _localStation;		//Station position in local coordinate system
    Eigen::Matrix4d               _localStationPose; //Station pose in local coordinate system
    bool                          _stationPositionAvailable; //If the approximate position of the _station is provided

    bool                          _poseFixed = false;  //the pose is fixed or not
    bool                          _poseStable = false; //the pose is stable or not after the optimization
    Eigen::Matrix4d               _poseLo;		//used for lidar odometry
    Eigen::Matrix4d               _poseOptimized; //optimized pose
    Eigen::Matrix4d               _poseInit;		//used for the init guess for pgo
    Eigen::Matrix4d               _poseOri;		//used for the init guess for pgo
    Eigen::Vector3d               _gravity;

    Matrix6d                      _informationMatrixToNext;

    PointCloudXYZI::Ptr           _pcRaw; //Raw point cloud
    //PointCloudXYZI::Ptr           _pcRawWorld; //in world coordinate system (for lidar odometry)
    PointCloudXYZI::Ptr           _pcDown;   //Downsampled point cloud
    //PointCloudXYZI::Ptr           _pcDownWorld; //in world coordinate system (for lidar odometry)
    PointCloudXYZI::Ptr           _pcDownAligned;   //Downsampled point cloud
    std::vector<PointWithCov>     _pvList;
    std::vector<IMUData>          _imuDataList;

    std::string                   _pcdFilePath;


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::vector<CloudBlock, Eigen::aligned_allocator<CloudBlock>> strip;
typedef std::vector<strip> strips;
typedef boost::shared_ptr<CloudBlock> CloudBlockPtr;
typedef std::deque<CloudBlockPtr> CloudBlockPtrs;



#endif //SRC_CLOUDBLOCK_H
