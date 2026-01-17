//
// Created by w on 2022/9/15.
//

#ifndef SRC_REGISTRATIONICP_H
#define SRC_REGISTRATIONICP_H

#include "Registrator/BaseRegistration.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define REGISTRATION_ICP_API __declspec(dllexport)
#else
#define REGISTRATION_ICP_API __declspec(dllimport)
#endif
#else
#define REGISTRATION_ICP_API
#endif

class REGISTRATION_ICP_API RegistrationICP : virtual public Registration
{

public:

    void push_surf(const Eigen::Vector3d &orip, const Eigen::Matrix<double, 3, 1> &centor, Eigen::Vector3d &direct,
                   double coeff);

    void push_line(const Eigen::Vector3d &orip, const Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff);

    void
    evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1> &g,
                  double &residual);

    void evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual);

    void clear();

public:
    SO3 so3_pose, so3_temp;
    Eigen::Vector3d t_pose, t_temp;
    std::vector<Eigen::Vector3d> surf_centor, surf_direct;
    std::vector<Eigen::Vector3d> corn_centor, corn_direct;
    std::vector<Eigen::Vector3d> surf_gather, corn_gather;
    std::vector<double> surf_coeffs, corn_coeffs;
};

#endif //SRC_REGISTRATIONICP_H
