//
// Created by w on 2022/9/15.
//
#include "Registrator/RegistrationICP.h"

void RegistrationICP::clear()
{
    surf_centor.clear(),
            surf_direct.clear();
    corn_centor.clear(),
            corn_direct.clear();
    surf_gather.clear(),
            corn_gather.clear();
    surf_coeffs.clear();
    corn_coeffs.clear();
}

void RegistrationICP::push_surf(const Eigen::Vector3d &orip, const Eigen::Matrix<double, 3, 1> &centor,
                                Eigen::Vector3d &direct, double coeff)
{
    direct.normalize();
    surf_direct.push_back(direct);
    surf_centor.push_back(centor);
    surf_gather.push_back(orip);
    surf_coeffs.push_back(coeff);
}

void RegistrationICP::push_line(const Eigen::Vector3d &orip, const Eigen::Vector3d &centor, Eigen::Vector3d &direct,
                                double coeff)
{
    direct.normalize();
    corn_direct.push_back(direct);
    corn_centor.push_back(centor);
    corn_gather.push_back(orip);
    corn_coeffs.push_back(coeff);
}

void RegistrationICP::evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess,
                                    Eigen::Matrix<double, 6, 1> &g, double &residual)
{
    Hess.setZero();
    g.setZero();
    residual = 0;
    uint a_size = surf_gather.size();
    for (uint i = 0; i < a_size; i++)
    {
        Eigen::Matrix3d _jac = surf_direct[i] * surf_direct[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather[i];
        Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - surf_centor[i];
        Eigen::Vector3d d_vec = _jac * v_ac;
        Eigen::Matrix<double, 3, 6> jacob;
        jacob.block<3, 3>(0, 0) = _jac * point_xi;
        jacob.block<3, 3>(0, 3) = _jac;

        residual += surf_coeffs[i] * d_vec.dot(d_vec);
        Hess += surf_coeffs[i] * jacob.transpose() * jacob;
        g += surf_coeffs[i] * jacob.transpose() * d_vec;
    }

    a_size = corn_gather.size();
    for (uint i = 0; i < a_size; i++)
    {
        Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct[i] * corn_direct[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather[i];
        Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - corn_centor[i];
        Eigen::Vector3d d_vec = _jac * v_ac;
        Eigen::Matrix<double, 3, 6> jacob;
        jacob.block<3, 3>(0, 0) = _jac * point_xi;
        jacob.block<3, 3>(0, 3) = _jac;

        residual += corn_coeffs[i] * d_vec.dot(d_vec);
        Hess += corn_coeffs[i] * jacob.transpose() * jacob;
        g += corn_coeffs[i] * jacob.transpose() * d_vec;
    }
}

void RegistrationICP::evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual)
{
    residual = 0;
    uint a_size = surf_gather.size();
    for (uint i = 0; i < a_size; i++)
    {
        Eigen::Matrix3d _jac = surf_direct[i] * surf_direct[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather[i];
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - surf_centor[i];
        double d_vec = surf_direct[i].transpose() * v_ac;

        residual += surf_coeffs[i] * d_vec * d_vec;
    }

    a_size = corn_gather.size();
    for (uint i = 0; i < a_size; i++)
    {
        Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct[i] * corn_direct[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather[i];
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - corn_centor[i];
        Eigen::Vector3d d_vec = _jac * v_ac;

        residual += corn_coeffs[i] * d_vec.dot(d_vec);
    }

}