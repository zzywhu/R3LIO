//
// Created by w on 2022/9/15.
//

#ifndef SRC_REGISTRATIONVGICP_H
#define SRC_REGISTRATIONVGICP_H

#include "Registrator/BaseRegistration.h"
#include "Mapper/VoxelVGICP.hpp"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define REGISTRATION_VGICP_API __declspec(dllexport)
#else
#define REGISTRATION_VGICP_API __declspec(dllimport)
#endif
#else
#define REGISTRATION_VGICP_API
#endif

class REGISTRATION_VGICP_API RegistrationVGICP : virtual public Registration
{
public:
    RegistrationVGICP(int maxInteration = 20) :
            Registration(maxInteration)
    {}

    void setMatchedList(std::vector<GuassinVoxelMatchInfo<PointWithCov>> &matchList)
    { matchedList = matchList; }

    void
    evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1> &g,
                  double &residual);

    void evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual);

    void clear();

public:
    std::vector<GuassinVoxelMatchInfo<PointWithCov>> matchedList;
    SO3 so3_pose, so3_temp;
    Eigen::Vector3d t_pose, t_temp;

};

#endif //SRC_REGISTRATIONVGICP_H
