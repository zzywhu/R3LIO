//
// Created by w on 2022/9/15.
//

#ifndef SRC_LIDMOTORCALIB_H
#define SRC_LIDMOTORCALIB_H

#include "vector"
#include "iostream"

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include "Misc/Common.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define LIDAR_MOTOR_CALIB_API __declspec(dllexport)
#else
#define LIDAR_MOTOR_CALIB_API __declspec(dllimport)
#endif
#else
#define LIDAR_MOTOR_CALIB_API
#endif

bool LIDAR_MOTOR_CALIB_API calcEXRotation(std::vector<Eigen::Matrix4d> relToList,
                    std::vector<Eigen::Matrix4d> relTlList,
                    Eigen::Matrix3d& Rol,  Eigen::Vector3d& tol,
                    double maxIter = 100,
                    double maxAngleErr = 1.);

#endif //SRC_LIDMOTORCALIB_H
