#ifndef IMU_PREINTERGRATION_H
#define IMU_PREINTERGRATION_H

//#include "ImuData.h"
#include "Utility.hpp"
#include "Common.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "ImuData.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define IMU_PREINTERGRATION_API __declspec(dllexport)
#else
#define IMU_PREINTERGRATION_API __declspec(dllimport)
#endif
#else
#define IMU_PREINTERGRATION_API
#endif

using gtsam::symbol_shorthand::X; // pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class IMU_PREINTERGRATION_API ImuPreintegration
{
public:
    struct Config
    {
        double _imuAccNoise;
        double _imuGyrNoise;
        double _imuAccBiasNoise;
        double _imuGyrBiasNoise;
        double _correctRotNoise; // rad,rad,rad,m, m, m
        double _correctPosNoise; // rad,rad,rad,m, m, m
        double _correctRotNoiseDegenerate; // rad,rad,rad,m, m, m
        double _correctPosNoiseDegenerate; // rad,rad,rad,m, m, m
        double _meanAccNorm;
        Eigen::Matrix3d _Ril;
        Eigen::Vector3d _til;
        Config()
        {
            _imuAccNoise=0.01;
            _imuGyrNoise=0.001;
            _imuAccBiasNoise=0.0002;
            _imuGyrBiasNoise=0.00003;
            _correctRotNoise = 0.05;
            _correctPosNoise = 0.1;// rad,rad,rad,m, m, m
            _correctRotNoiseDegenerate = 1.;
            _correctPosNoiseDegenerate = 1.;// rad,rad,rad,m, m, m;
            _meanAccNorm=9.81;
            _Ril=Eigen::Matrix3d::Identity();
            _til=Eigen::Vector3d::Identity();
        }
    };


    ImuPreintegration() :_isDoneFirstOpt(false),
                         _lastImuTime(-1),
                         _lastImuTimeOpt(-1),
                         _key(1),
                         _nMaxIterCount(100),
                         _isFirstData(true),
                         _initIterNum(0),
                         _isImuNeedInit(true),
                         _imuIntegratorOpt(nullptr),
                         _imuIntegratorImu(nullptr),
                         _initialized(false) {}


    void init();

    static Config &mutableConfig() { return _config; }
    static const Config &config() { return _config; }

    gtsam::PreintegratedImuMeasurements* getPreintegratedImu(){return _imuIntegratorImu;}
    gtsam::PreintegratedImuMeasurements* getPreintegratedOpt(){return _imuIntegratorOpt;}

    gtsam::NavState& getCurState(){return _curState;}

    void resetOptimization();

    void resetParams()
    {
        _lastImuTime = -1;
        _isDoneFirstOpt = false;
        _initialized = false;
    }

    void optimizeOdometry(StatesGroup &odomState, const double& curOdomTime,bool isDegenerate=false);

    bool predictByImu(SensorMsgs::Imu::Ptr &imuData, StatesGroup& stateOut);

    bool imuInit(SensorMsgs::Imu::Ptr &imuData, StatesGroup& stateCur);

    bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            std::cerr<<"Large velocity, reset IMU-preintegration!"<<std::endl;
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            std::cerr<<"Large bias, reset IMU-preintegration!"<<std::endl;
            return true;
        }

        return false;
    }

private:
    std::mutex                              _mtx;
    bool                                    _initialized;

    gtsam::noiseModel::Diagonal::shared_ptr _priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr _priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr _priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr _correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr _correctionNoise2;
    gtsam::Vector                           _noiseModelBetweenBias;

    gtsam::PreintegratedImuMeasurements*    _imuIntegratorOpt;
    gtsam::PreintegratedImuMeasurements*    _imuIntegratorImu;

    std::deque<SensorMsgs::Imu>             _imuQueOpt;
    std::deque<SensorMsgs::Imu>             _imuQueImu;

    gtsam::NavState                         _curState;
    gtsam::Pose3                            _prevPose;
    gtsam::Vector3                          _prevVel;
    gtsam::NavState                         _prevState;
    gtsam::imuBias::ConstantBias            _prevBias;

    gtsam::NavState                         _prevStateOdom;
    gtsam::imuBias::ConstantBias            _prevBiasOdom;

    bool                                    _isDoneFirstOpt;
    double                                  _lastImuTime;
    double                                  _lastImuTimeOpt;

    gtsam::ISAM2                            _optimizer;
    gtsam::NonlinearFactorGraph             _graphFactors;
    gtsam::Values _graphValues;

    int                                     _key;

    int                                     _nMaxIterCount;
    V3D                                     _meanAcc;
    V3D                                     _meanGyr;
    bool                                    _isFirstData;
    V3D                                     _covAcc;
    V3D                                     _covGyr;
    int                                     _initIterNum;
    bool                                    _isImuNeedInit;

    static Config                           _config;
};




#endif 