//
// Created by w on 2022/10/19.
//

#ifndef FAST_LIO_IMU_H
#define FAST_LIO_IMU_H

#include <boost/make_shared.hpp>

namespace SensorMsgs
{
    struct ImuData
    {
        typedef ImuData Type;

        ImuData()
                : header()
                , orientation()
                , angular_velocity()
                , angular_velocity_covariance()
                , linear_acceleration()
                , linear_acceleration_covariance()
        {
            angular_velocity_covariance.setZero();
            linear_acceleration_covariance.setZero();
        }
        ImuData(const ImuData& imu)
                : header(imu.header),
                  orientation(imu.orientation),
                  angular_velocity(imu.angular_velocity),
                  angular_velocity_covariance(imu.angular_velocity_covariance),
                  linear_acceleration(imu.linear_acceleration),
                  linear_acceleration_covariance(imu.linear_acceleration_covariance)
       {}


        double header;

        typedef  Eigen::Quaterniond  _orientation_type;
        _orientation_type orientation;

        typedef Eigen::Matrix<double,3,3>  _orientation_covariance_type;
        _orientation_covariance_type orientation_covariance;

        typedef  Eigen::Matrix<double,3,1> _angular_velocity_type;
        _angular_velocity_type angular_velocity;

        typedef Eigen::Matrix<double,3,3>  _angular_velocity_covariance_type;
        _angular_velocity_covariance_type angular_velocity_covariance;

        typedef  Eigen::Matrix<double,3,1>  _linear_acceleration_type;
        _linear_acceleration_type linear_acceleration;

        typedef Eigen::Matrix<double,3,3>  _linear_acceleration_covariance_type;
        _linear_acceleration_covariance_type linear_acceleration_covariance;


        typedef boost::shared_ptr< ::SensorMsgs::ImuData> Ptr;
        typedef boost::shared_ptr< ::SensorMsgs::ImuData> ConstPtr;

    }; // struct Imu_

    typedef ::SensorMsgs::ImuData Imu;

    typedef boost::shared_ptr< ::SensorMsgs::ImuData > ImuPtr;
    typedef boost::shared_ptr< ::SensorMsgs::ImuData const> ImuConstPtr;

// constants requiring out of line definition


} // namespace SensorMsgs
#endif //FAST_LIO_IMU_H
