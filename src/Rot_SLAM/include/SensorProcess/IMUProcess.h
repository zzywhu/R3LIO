#ifndef FAST_LIO_IMU_PROCESS_H
#define FAST_LIO_IMU_PROCESS_H

#include <cmath>
#include <math.h>
#include <deque>
#include <iomanip>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>

#include <Eigen/Eigen>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "Pose6D.h"
#include <Misc/So3Math.h>
#include <Misc/use-ikfom.h>
#include <Misc/Common.h>

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define IMU_PROCESS_API __declspec(dllexport)
#else
#define IMU_PROCESS_API __declspec(dllimport)
#endif // __DLL_EXPORTS__
#else
#define IMU_PROCESS_API
#endif // __SHARED_LIBS__

/// *************Preconfiguration

const bool IMU_PROCESS_API timeComprator(PointType &x, PointType &y);


/// *************IMU Process and undistortion
class IMU_PROCESS_API ImuProcess
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();

    ~ImuProcess();

    void Reset();

    void Reset(double start_timestamp, const SensorMsgs::ImuConstPtr &lastimu);

    void clear();

    bool& imu_need_init(){return imu_need_init_;}

    int max_init_count(){return maxInitCount;}
    void set_imu_max_init_count(const int count){maxInitCount=count;}

    void set_gyr_cov(const V3D &scaler)
    { cov_gyr_scale = scaler; }

    void set_R_LI_cov(const V3D &R_LI_cov)
    { cov_R_LI = R_LI_cov; }

    void set_T_LI_cov(const V3D &T_LI_cov)
    { cov_T_LI = T_LI_cov; }

    void set_acc_cov(const V3D &scaler)
    { cov_acc_scale = scaler; }

    void set_gyr_bias_cov(const V3D &b_g)
    { cov_bias_gyr = b_g; }

    void set_acc_bias_cov(const V3D &b_a)
    { cov_bias_acc = b_a; }

    void set_mean_acc_norm(const double &mean_acc_norm)
    { IMU_mean_acc_norm = mean_acc_norm; }

    void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

    void Process(const MeasureGroup &meas, StatesGroup &state, PointCloudXYZI::Ptr pcl_un_);

    void imuMotionCompensation(PointCloudXYZI &cloud, const StatesGroup& stateEnd);

    std::vector<Pose6D>& getIMUPoses(){return IMUpose;}
    void pushIMUPose(const Pose6D& pose6D){IMUpose.push_back(pose6D);}
    void clearIMUPose(){IMUpose.clear();}


private:
    void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);

    void IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N, bool enable_bias_init = false);

    void undistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out);

    void propagation_and_undist(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_in_out);

    void forward_propagation_without_imu(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);

public:
    V3D cov_acc;
    V3D cov_gyr;
    V3D cov_acc_scale;
    V3D cov_gyr_scale;
    V3D cov_bias_gyr;
    V3D cov_bias_acc;
    V3D cov_R_LI;
    V3D cov_T_LI;
    Eigen::Matrix<double, 12, 12> Q;
    bool imu_en;
    bool LI_init_done = false;
    double IMU_mean_acc_norm;
    bool undistort_en;
    bool _bOutpoutImuInfo = false;    // zqy Add 0828
    std::vector<IMUData> _imuDatas;

private:
    PointCloudXYZI::Ptr cur_pcl_un_;
    SensorMsgs::ImuConstPtr last_imu_;
    std::vector<Pose6D> IMUpose;
    V3D mean_acc;
    V3D mean_gyr;
    V3D angvel_last;
    V3D acc_s_last;
    double start_timestamp_;
    double last_lidar_end_time_;
    int init_iter_num = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
    double time_last_scan;
    int  maxInitCount;
};

#endif//FAST_LIO_IMU_PROCESS_H



