//
// Created by w on 2022/8/18.
//

#ifndef SRC_BASEREGISTRATION_H
#define SRC_BASEREGISTRATION_H

#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include "Misc/So3.h"
#include "Common.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define BASE_REGISTRATION_API __declspec(dllexport)
#else
#define BASE_REGISTRATION_API __declspec(dllimport)
#endif // __DLL_EXPORTS__
#else
#define BASE_REGISTRATION_API
#endif // __SHARED_LIBS__

class BASE_REGISTRATION_API Registration
{
public:
    Registration(int maxInteration = 20) :
            _maxInteration(maxInteration)
    {}

    void setInterationNum(const int &num)
    { _maxInteration = num; }

    virtual void
    evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1> &g,
                  double &residual) = 0;

    virtual void evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual) = 0;

    virtual void clear() = 0;

    void damping_iter();

public:
    int _maxInteration;
    SO3 _so3Pose, _so3Temp;
    Eigen::Vector3d _tPose, _tTemp;
};


#endif //SRC_BASEREGISTRATION_H
