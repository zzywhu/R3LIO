#ifndef LIDARPROCESS_H
#define LIDARPROCESS_H

#include <pcl/filters/voxel_grid.h>
#include "pandarGeneral/point_types.h"

#include <iomanip>

#include "CloudInfo.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define LIDAR_PROCESS_API __declspec(dllexport)
#else
#define LIDAR_PROCESS_API __declspec(dllimport)
#endif
#else
#define LIDAR_PROCESS_API
#endif

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE
{
    AVIA = 1, VELO, OUSTER, HESAI, TIMOO16
}; //{1, 2, 3, 4, 5}
enum TIME_UNIT
{
    Sec = 0, Ms = 1, Us = 2, Ns = 3
};

const bool time_list_cut_frame(PointType &x, PointType &y);

namespace VelodynePoint
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        uint16_t ring;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace VelodynePoint
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePoint::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, time, time)
                                  (uint16_t, ring, ring)
)

namespace OusterPoint
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}  // namespace OusterPoint
// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPoint::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                  (std::uint32_t, t, t)
                                  (std::uint16_t, reflectivity, reflectivity)
                                  (std::uint8_t, ring, ring)
                                  (std::uint16_t, ambient, ambient)
                                  (std::uint32_t, range, range)
)

namespace HesaiPoint
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D

        float intensity;
        double timestamp;
        uint16_t ring;                   //laser ring number
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace HesaiPoint

POINT_CLOUD_REGISTER_POINT_STRUCT(HesaiPoint::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (double, timestamp, timestamp)
                                  (uint16_t, ring, ring))

namespace TimooPoint
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D

        float intensity;
        float time;
        uint16_t ring;                   //laser ring number
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace TimooPoint

POINT_CLOUD_REGISTER_POINT_STRUCT(TimooPoint::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, time, time)
                                  (uint16_t, ring, ring))

struct smoothness_t
{
    float value;
    size_t ind;
};

struct by_value
{
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
        return left.value < right.value;
    }
};

class LIDAR_PROCESS_API LidarProcess
{
public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct Config
    {

        int      _pointFilterNum;
		double   _blindMin;
		double   _blindMax;
        int      _lidarType;
        int      _nScans;
        int      _timeUnit;
        int      _scanRate;
        int      _nHorizonScan;
        double   _edgeThreshold;
        double   _surfThreshold;
        double   _surfLeafSize;
        Config()
        {
            _pointFilterNum=5;
            _blindMin=0.2;
            _blindMax=200;
            _lidarType=TIMOO16;
            _nScans=16;
            _timeUnit=Us;
            _scanRate=10;//hz
            _nHorizonScan=2000;
            _edgeThreshold=1.0;//0.1;
            _surfThreshold=0.1;
            _surfLeafSize=0.1;
        }
    };
    LidarProcess():
            _isGivenOffsetTime(false)
    {
        _cloudSmoothness.resize(_config._nScans * _config._nHorizonScan);

        _cloudCurvature = new float[_config._nScans * _config._nHorizonScan];
        _cloudNeighborPicked = new int[_config._nScans * _config._nHorizonScan];
        _cloudLabel = new int[_config._nScans * _config._nHorizonScan];
    }

    ~LidarProcess();

    static Config& mutableConfig(){return _config;}

    static const Config& config(){return _config;}

    void process(PPointCloud::Ptr pl_orig, const double& lidTime, PointCloudXYZI::Ptr &pcl_out);


    void processCutFrameCloud(PPointCloud::Ptr pl_orig, const double& lidTime, std::deque<PointCloudXYZI::Ptr>& pcl_out,
                               std::deque<double>& time_lidar, const int required_frame_num, int scan_count);


    void setDownsampleLeafSize(const float& lx, const float& ly, const float& lz){_downSizeFilter.setLeafSize(lx, ly, lz);};

    int downSampleLocalCloud(PointCloudXYZI::Ptr oriCloud,PointCloudXYZI::Ptr downCloud);

    void filterCloud(PointCloudXYZI::Ptr oriCloud, PointCloudXYZI::Ptr filteredCloud, int filterNum);

    void projectPointCloud(PointCloudXYZI::Ptr oriCloud, CloudInfo& cloudInfo);

    void calculateSmoothness(CloudInfo& cloudInfo);

    void markOccludedPoints(CloudInfo& cloudInfo);

    void extractFeatures(PointCloudXYZI::Ptr oriCloud, PointCloudXYZI& surfaceCloud, PointCloudXYZI& cornerCloud);

private:
    PointCloudXYZI                          _plFull;
    float                                   _timeUnitScale;
    bool                                    _isGivenOffsetTime;
    pcl::VoxelGrid<pcl::PCLPointCloud2>     _downSizeFilter; // fix bug for cloud ptr releasing(meet with pcl11 and eigen 3.x)
    std::vector<smoothness_t>               _cloudSmoothness;
    float *                                 _cloudCurvature;
    int *                                   _cloudNeighborPicked;
    int *                                   _cloudLabel;

    static Config                           _config;
};



#endif //LIDARPROCESS_H