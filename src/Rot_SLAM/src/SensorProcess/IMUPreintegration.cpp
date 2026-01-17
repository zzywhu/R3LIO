#include "IMUPreintegration.h"
using namespace std;


ImuPreintegration::Config ImuPreintegration::_config;


void ImuPreintegration::init()
{
    boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU();
    p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(_config._imuAccNoise, 2); // acc white noise in continuous
    p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(_config._imuGyrNoise, 2); // gyro white noise in continuous
    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2); // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

    _priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
    _priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-4); // m/s
    // _priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);
    _priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
    //_correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
    _correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2); // meter
    _correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
    _noiseModelBetweenBias = (gtsam::Vector(6) <<_config._imuAccBiasNoise,_config._imuAccBiasNoise,_config._imuAccBiasNoise,
                                                 _config._imuGyrBiasNoise,_config._imuGyrBiasNoise,_config._imuGyrBiasNoise).finished();

    _imuIntegratorImu = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
    _imuIntegratorOpt = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization

}


void ImuPreintegration::resetOptimization()
{
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    _optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    _graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    _graphValues = NewGraphValues;
}


void ImuPreintegration::optimizeOdometry(StatesGroup &odomState, const double& curOdomTime, bool isDegenerate)
{
    std::lock_guard<std::mutex> lock(_mtx);

    // make sure we have imu data to integrate
    if (_imuQueOpt.empty())
        return;

    gtsam::Pose3 curImuPose = gtsam::Pose3(gtsam::Rot3(odomState.rot_end), odomState.pos_end);

    // 0. initialize system
    if (_initialized == false)
    {
        resetOptimization();

        // pop old IMU message
        while (!_imuQueOpt.empty())
        {
            if (_imuQueOpt.front().header < curOdomTime)
            {
                _lastImuTimeOpt = _imuQueOpt.front().header;
                _imuQueOpt.pop_front();
            }
            else
                break;
        }
        // initial pose
        _prevPose = curImuPose;
        printf("\033[1m\033[34m"  "[%s] " "\033[0m", "Init lidar frame");
        cout << "imu Pos  = " << _prevPose.translation().transpose() << endl;

        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), _prevPose, _priorPoseNoise);
        _graphFactors.add(priorPose);
        // initial velocity
        _prevVel = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), _prevVel, _priorVelNoise);
        _graphFactors.add(priorVel);
        // initial bias
        _prevBias = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), _prevBias, _priorBiasNoise);
        _graphFactors.add(priorBias);
        // add values
        _graphValues.insert(X(0), _prevPose);
        _graphValues.insert(V(0), _prevVel);
        _graphValues.insert(B(0), _prevBias);
        // optimize once
        _optimizer.update(_graphFactors, _graphValues);
        _graphFactors.resize(0);
        _graphValues.clear();

        _imuIntegratorImu->resetIntegrationAndSetBias(_prevBias);
        _imuIntegratorOpt->resetIntegrationAndSetBias(_prevBias);

        _key = 1;
        _initialized = true;
        return;
    }


    // reset graph for speed                                                                                                                                                                                                                                                                                                             
    if (_key == 100)
    {
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(_optimizer.marginalCovariance(X(_key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(_optimizer.marginalCovariance(V(_key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(_optimizer.marginalCovariance(B(_key - 1)));
        // reset graph
        resetOptimization();
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), _prevPose, updatedPoseNoise);
        _graphFactors.add(priorPose);
        // add velocity
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), _prevVel, updatedVelNoise);
        _graphFactors.add(priorVel);
        // add bias
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), _prevBias, updatedBiasNoise);
        _graphFactors.add(priorBias);
        // add values
        _graphValues.insert(X(0), _prevPose);
        _graphValues.insert(V(0), _prevVel);
        _graphValues.insert(B(0), _prevBias);
        // optimize once
        _optimizer.update(_graphFactors, _graphValues);
        _graphFactors.resize(0);
        _graphValues.clear();

        _key = 1;
    }


    // 1. integrate imu data and optimize
    while (!_imuQueOpt.empty())
    {
        // pop and integrate imu data that is between two optimizations
        const SensorMsgs::Imu& thisImu = _imuQueOpt.front();
        if (thisImu.header < curOdomTime)
        {
            double dt = (_lastImuTimeOpt < 0) ? (1.0 / 500.0) : (thisImu.header - _lastImuTimeOpt);
            _imuIntegratorOpt->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x(), thisImu.linear_acceleration.y(), thisImu.linear_acceleration.z()),
                                                    gtsam::Vector3(thisImu.angular_velocity.x(), thisImu.angular_velocity.y(), thisImu.angular_velocity.z()), dt);
            //std::cout<<"Intergrate IMU opt time:"<<thisImu.header<<std::endl;
            //printf("\033[1m\033[34m" "[%s] " "\033[0m", "Opt Imu intergrate");
           // std::cout<<std::setprecision(6)<<std::fixed<<"imu time:"<< thisImu.header<<std::endl;
            _lastImuTimeOpt = thisImu.header;
            _imuQueOpt.pop_front();
        }
        else
            break;
    }
    // add imu factor to graph
    const gtsam::PreintegratedImuMeasurements &preintImu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*_imuIntegratorOpt);
    gtsam::ImuFactor imuFactor(X(_key - 1), V(_key - 1), X(_key), V(_key), B(_key - 1), preintImu);
    _graphFactors.add(imuFactor);
    // add imu bias between factor
    _graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(_key - 1), B(_key), gtsam::imuBias::ConstantBias(),
                                                                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(_imuIntegratorOpt->deltaTij()) * _noiseModelBetweenBias)));

    printf("\033[1m\033[34m"  "[%s] " "\033[0m", "before gtsam fustion");
    //cout << "lidar Pos  = " << lidarPose.translation().transpose() << endl;
    cout << "Registrated imu Pos  = " << curImuPose.translation().transpose() << endl;
    // add pose factor
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(_key), curImuPose, isDegenerate ? _correctionNoise2 : _correctionNoise);
    _graphFactors.add(pose_factor);
    // insert predicted values
    gtsam::NavState propState_ = _imuIntegratorOpt->predict(_prevState, _prevBias);
    std::cout<<"imu prop state: "<<propState_.pose().translation().transpose()<<std::endl;
    _graphValues.insert(X(_key), propState_.pose());
    _graphValues.insert(V(_key), propState_.v());
    _graphValues.insert(B(_key), _prevBias);
    // optimize
    _optimizer.update(_graphFactors, _graphValues);
    _optimizer.update();
    _graphFactors.resize(0);
    _graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = _optimizer.calculateEstimate();
    _prevPose = result.at<gtsam::Pose3>(X(_key));
    _prevVel = result.at<gtsam::Vector3>(V(_key));
    _prevBias = result.at<gtsam::imuBias::ConstantBias>(B(_key));

    // Reset the optimization preintegration object.
    _imuIntegratorOpt->resetIntegrationAndSetBias(_prevBias);
    // check optimization
    if (failureDetection(_prevVel, _prevBias))
    {
        resetParams();
        std::cerr<<"Detected failure motion"<<std::endl;
        return;
    }

    odomState.pos_end=_prevPose.translation();
    odomState.rot_end=_prevPose.rotation().matrix();
    odomState.vel_end=_prevVel;
    odomState.bias_g=_prevBias.gyroscope();
    odomState.bias_a=_prevBias.accelerometer();
    odomState.gravity=_imuIntegratorOpt->p().getGravity();

    // 2. after optimization, repropagate imu odometry preintegration
    _prevState = gtsam::NavState(_prevPose, _prevVel);
    _prevStateOdom = _prevState;
    _prevBiasOdom = _prevBias;
    // first pop imu data earlier than current correction data
    double lastImuQT = -1;
    while (!_imuQueImu.empty() && _imuQueImu.front().header < curOdomTime)
    {
        lastImuQT = _imuQueImu.front().header;
        _imuQueImu.pop_front();
    }
    // repropogate
    _imuIntegratorImu->resetIntegrationAndSetBias(_prevBiasOdom); // reset bias use the newly optimized bias
    if (!_imuQueImu.empty())
    {
        // integrate imu message from the beginning of this optimization
        for (int i = 0; i < (int) _imuQueImu.size(); ++i)
        {
            double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (_imuQueImu[i].header - lastImuQT);
            std::cout<<"Integrate measurement:"<<_imuQueImu[i].header<<std::endl;
            _imuIntegratorImu->integrateMeasurement(gtsam::Vector3(_imuQueImu[i].linear_acceleration.x(), _imuQueImu[i].linear_acceleration.y(), _imuQueImu[i].linear_acceleration.z()),
                                                    gtsam::Vector3(_imuQueImu[i].angular_velocity.x(), _imuQueImu[i].angular_velocity.y(), _imuQueImu[i].angular_velocity.z()), dt);
            lastImuQT = _imuQueImu[i].header;
        }
    }

    ++_key;
    _isDoneFirstOpt = true;
}


bool ImuPreintegration::predictByImu(SensorMsgs::Imu::Ptr &imuData, StatesGroup& stateOut)
{
    std::lock_guard<std::mutex> lock(_mtx);

    _imuQueOpt.push_back(*imuData);
    _imuQueImu.push_back(*imuData);

    if(_isImuNeedInit)
    {
        if(imuInit(imuData, stateOut))
            _isImuNeedInit=false;
        return false;
    }
    else if (!_isDoneFirstOpt)
        return false;

    double dt = (_lastImuTime < 0) ? (1.0 / 1000.0) : (imuData->header - _lastImuTime);
    _lastImuTime = imuData->header;

    // integrate this single imu message
    //std::cout<<"Integrate measurement:"<<imuData->header<<std::endl;
    _imuIntegratorImu->integrateMeasurement(gtsam::Vector3(imuData->linear_acceleration.x(), imuData->linear_acceleration.y(), imuData->linear_acceleration.z()),
                                            gtsam::Vector3(imuData->angular_velocity.x(), imuData->angular_velocity.y(), imuData->angular_velocity.z()), dt);

    // predict odometry
    _curState = _imuIntegratorImu->predict(_prevStateOdom, _prevBiasOdom);
    //std::cout<<"_prevStateOdom:"<<_prevStateOdom.position().transpose()<<std::endl;
    // transform imu pose to ldiar

    stateOut.pos_end=_curState.position();
    stateOut.rot_end=_curState.R();
    stateOut.vel_end=_curState.velocity();
    stateOut.bias_g=_prevBiasOdom.gyroscope();
    stateOut.bias_a=_prevBiasOdom.accelerometer();
    stateOut.gravity=_imuIntegratorImu->p().getGravity();

    return true;
}


bool ImuPreintegration::imuInit(SensorMsgs::Imu::Ptr &imuData, StatesGroup& stateCur)
{
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurements to unit gravity **/
    //printf("IMU Initializing: %.1f %%", double(_initIterNum) / _nMaxIterCount * 100);
    V3D curAcc, curGyr;

    if (_isFirstData)
    {
        _initIterNum = 1;
        _isFirstData = false;
        const auto &imuAcc = imuData->linear_acceleration;
        const auto &gyrAcc = imuData->angular_velocity;
        _meanAcc << imuAcc.x(), imuAcc.y(), imuAcc.z();
        _meanGyr << gyrAcc.x(), gyrAcc.y(), gyrAcc.z();
    }
    else
    {
        _initIterNum++;
        const auto &imuAcc = imuData->linear_acceleration;
        const auto &gyrAcc = imuData->angular_velocity;
        curAcc << imuAcc.x(), imuAcc.y(), imuAcc.z();
        curGyr << gyrAcc.x(), gyrAcc.y(), gyrAcc.z();

        _meanAcc += (curAcc - _meanAcc) / _initIterNum;
        _meanGyr += (curGyr - _meanGyr) / _initIterNum;

        _covAcc = _covAcc * (_initIterNum - 1.0) / _initIterNum + (curAcc - _meanAcc).cwiseProduct(curAcc - _meanAcc) * (_initIterNum - 1.0) / (_initIterNum * _initIterNum);
        _covGyr = _covGyr * (_initIterNum - 1.0) / _initIterNum + (curGyr - _meanGyr).cwiseProduct(curGyr - _meanGyr) * (_initIterNum - 1.0) / (_initIterNum * _initIterNum);
//        std::cout<<"covAcc: "<<_covAcc.norm()<<std::endl;
        if(_covAcc.norm()>1)
            std::cerr<<"Stop moving, IMU needs to be initialized!"<<std::endl;
    }
    if(_initIterNum > _nMaxIterCount)
    {
        _isImuNeedInit = false;

        stateCur.gravity = -_meanAcc / _meanAcc.norm() * G_m_s2;
        stateCur.rot_end = Eye3d;
        stateCur.bias_g.setZero();

        _covAcc *= pow(G_m_s2 / _meanAcc.norm(), 2);

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU();
//        p->accelerometerCovariance = Eigen::Matrix3d::Identity(); // acc white noise in continuous
//        p->accelerometerCovariance(0,0)=_covAcc(0);
//        p->accelerometerCovariance(1,1)=_covAcc(1);
//        p->accelerometerCovariance(2,2)=_covAcc(2);
//        p->gyroscopeCovariance = Eigen::Matrix3d::Identity(); // acc white noise in continuous
//        p->gyroscopeCovariance(0,0)=_covGyr(0);
//        p->gyroscopeCovariance(1,1)=_covGyr(1);
//        p->gyroscopeCovariance(2,2)=_covGyr(2);
        p->n_gravity=stateCur.gravity;

        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(_config._imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(_config._imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias

        if(_imuIntegratorImu)
            delete _imuIntegratorImu;
        if(_imuIntegratorOpt)
            delete _imuIntegratorOpt;

        _imuIntegratorImu = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        _imuIntegratorOpt = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
//        printf("IMU Initialization Done: Gravity: %.4f %.4f %.4f, Acc norm: %.4f\n",
//               stateCur.gravity[0], stateCur.gravity[1], stateCur.gravity[2], _meanAcc.norm());
        std::cout<<"****IMU initialized result****"<<std::endl;
        std::cout<<"GyroscopeCovariance:"<<std::endl<<p->gyroscopeCovariance<<std::endl;
        std::cout<<"AccelerometerCovariance:"<<std::endl<<p->accelerometerCovariance<<std::endl;
        std::cout<<"Gravity:"<<p->getGravity().transpose()<<std::endl;
        std::cout<<"********************************"<<std::endl;
        return true;
    }
    return false;
}

