//
// Created by w on 2022/9/15.
//
#include "Registrator/RegistrationVGICP.h"

void RegistrationVGICP::clear()
{
    matchedList.clear();
}


void RegistrationVGICP::evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess,
                                      Eigen::Matrix<double, 6, 1> &g, double &residual)
{
    Hess.setZero();
    g.setZero();
    double toatalResidual = 0;
    std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(MP_PROC_NUM);
    std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(MP_PROC_NUM);
    for (int i = 0; i < MP_PROC_NUM; i++)
    {
        Hs[i].setZero();
        bs[i].setZero();
    }

#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : toatalResidual) schedule(guided, 8)
    for (int i = 0; i < matchedList.size(); i++)
    {
        const Eigen::Vector4d mean_A = Eigen::Vector4d(matchedList[i]._pv.pl.x(), matchedList[i]._pv.pl.y(),
                                                       matchedList[i]._pv.pl.z(), 1.0);
        const Eigen::Vector4d mean_B = matchedList[i]._matchedVoxel->_mean;
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans.block<3, 3>(0, 0) = so3_p.matrix();
        trans.block<3, 1>(0, 3) = t_p;

        const Eigen::Vector4d transed_mean_A = trans * mean_A;
        const Eigen::Vector4d error = mean_B - transed_mean_A;

        double w = std::sqrt(matchedList[i]._matchedVoxel->_numPoints);
        //double w=1;
        Eigen::Matrix4d RCR = matchedList[i]._matchedVoxel->_cov +
                              trans.matrix() * matchedList[i]._pv.neighCov * trans.matrix().transpose(); // CB + T*CA*T'
        RCR(3, 3) = 1.0;
        Eigen::Matrix4d voxelMahalanobis;
        voxelMahalanobis = RCR.inverse();
        voxelMahalanobis(3, 3) = 0.0;

        toatalResidual += w * error.transpose() * voxelMahalanobis * error;

        Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
        dtdx0.block<3, 3>(0, 0) = SO3::hat(transed_mean_A.head<3>());
        dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

        Eigen::Matrix<double, 6, 6> Hi = w * jlossexp.transpose() * voxelMahalanobis * jlossexp;
        Eigen::Matrix<double, 6, 1> bi = w * jlossexp.transpose() * voxelMahalanobis * error;

        int thread_num = omp_get_thread_num();
        Hs[thread_num] += Hi;
        bs[thread_num] += bi;
    }


    Hess.setZero();
    g.setZero();
    for (int i = 0; i < MP_PROC_NUM; i++)
    {
        Hess += Hs[i];
        g += bs[i];
    }
    residual = toatalResidual;
}

void RegistrationVGICP::evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual)
{
    double totalResidual = 0.0;
#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : totalResidual)
    for (int i = 0; i < matchedList.size(); i++)
    {
        auto target_voxel = matchedList[i]._matchedVoxel;

        const Eigen::Vector4d mean_A = Eigen::Vector4d(matchedList[i]._pv.pl.x(), matchedList[i]._pv.pl.y(),
                                                       matchedList[i]._pv.pl.z(), 1.0);
        const Eigen::Vector4d mean_B = matchedList[i]._matchedVoxel->_mean;
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans.block<3, 3>(0, 0) = so3_p.matrix();
        trans.block<3, 1>(0, 3) = t_p;

        const Eigen::Vector4d transed_mean_A = trans * mean_A;
        const Eigen::Vector4d error = mean_B - transed_mean_A;

        double w = std::sqrt(matchedList[i]._matchedVoxel->_numPoints);
        //double w=1;
        Eigen::Matrix4d RCR = matchedList[i]._matchedVoxel->_cov +
                              trans.matrix() * matchedList[i]._pv.neighCov * trans.matrix().transpose(); // CB + T*CA*T'
        RCR(3, 3) = 1.0;
        Eigen::Matrix4d voxelMahalanobis;
        voxelMahalanobis = RCR.inverse();
        voxelMahalanobis(3, 3) = 0.0;
//        std::cout<<w<<" / "<<mean_A.transpose()<<" / "<<mean_B.transpose()<<"/ "<<error.transpose()<<std::endl;
//        std::cout<<voxelMahalanobis<<std::endl;
        totalResidual += w * error.transpose() * voxelMahalanobis * error;

    }
    residual = totalResidual;
}