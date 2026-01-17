//
// Created by w on 2022/8/30.
//
#ifndef SRC_LSQREGPCL_IMPL_HPP
#define SRC_LSQREGPCL_IMPL_HPP

#include <pcl/search/impl/search.hpp>
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_types.h>
#include "boost/format.hpp"

#include "LsqRegPCL.h"
#include "Mapper/VoxelVGICP.hpp"
#include "Misc/So3.h"
#include "Mapper/IkdTree.h"

PCL_INSTANTIATE(Search, PCL_POINT_TYPES)

template<typename PointTarget, typename PointSource>
Eigen::Quaterniond RegistrationPCL<PointTarget, PointSource>::so3_exp(const Eigen::Vector3d &omega)
{
    double theta_sq = omega.dot(omega);

    double theta;
    double imag_factor;
    double real_factor;
    if (theta_sq < 1e-10)
    {
        theta = 0;
        double theta_quad = theta_sq * theta_sq;
        imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
        real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
    }
    else
    {
        theta = std::sqrt(theta_sq);
        double half_theta = 0.5 * theta;
        imag_factor = std::sin(half_theta) / theta;
        real_factor = std::cos(half_theta);
    }

    return Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
                              imag_factor * omega.z());
}

template<typename PointTarget, typename PointSource>
RegistrationPCL<PointTarget, PointSource>::RegistrationPCL()
{
    this->reg_name_ = "RegistrationPCL";
    max_iterations_ = 64;
    rotation_epsilon_ = 2e-3;
    transformation_epsilon_ = 5e-4;

    lsq_optimizer_type_ = LSQ_OPTIMIZER_TYPE::LevenbergMarquardt;
    lm_debug_print_ = false;
    lm_max_iterations_ = 10;
    lm_init_lambda_factor_ = 1e-9;
    lm_lambda_ = -1.0;

    final_hessian_.setIdentity();
}

template<typename PointTarget, typename PointSource>
RegistrationPCL<PointTarget, PointSource>::~RegistrationPCL()
{}

template<typename PointTarget, typename PointSource>
void RegistrationPCL<PointTarget, PointSource>::setLMLambda(double val)
{
    lm_lambda_ = val;
}

template<typename PointTarget, typename PointSource>
void RegistrationPCL<PointTarget, PointSource>::setRotationEpsilon(double eps)
{
    rotation_epsilon_ = eps;
}


template<typename PointTarget, typename PointSource>
void RegistrationPCL<PointTarget, PointSource>::setInitialLambdaFactor(double init_lambda_factor)
{
    lm_init_lambda_factor_ = init_lambda_factor;
}

template<typename PointTarget, typename PointSource>
void RegistrationPCL<PointTarget, PointSource>::setDebugPrint(bool lm_debug_print)
{
    lm_debug_print_ = lm_debug_print;
}

template<typename PointTarget, typename PointSource>
const Eigen::Matrix<double, 6, 6> &RegistrationPCL<PointTarget, PointSource>::getFinalHessian() const
{
    return final_hessian_;
}

template<typename PointTarget, typename PointSource>
double RegistrationPCL<PointTarget, PointSource>::evaluateCost(const Eigen::Matrix4f &relative_pose,
                                                               Eigen::Matrix<double, 6, 6> *H,
                                                               Eigen::Matrix<double, 6, 1> *b)
{
    return this->linearize(Eigen::Isometry3f(relative_pose).cast<double>(), H, b);
}

template<typename PointTarget, typename PointSource>
void RegistrationPCL<PointTarget, PointSource>::computeTransformation(PointCloudSource &output, const Matrix4 &guess)
{
    Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

    lm_lambda_ = -1.0;
    converged_ = false;

    if (lm_debug_print_)
    {
        std::cout << "********************************************" << std::endl;
        std::cout << "***************** optimize *****************" << std::endl;
        std::cout << "********************************************" << std::endl;
    }

    for (int i = 0; i < max_iterations_ && !converged_; i++)
    {
        nr_iterations_ = i;

        Eigen::Isometry3d delta;
        if (!step_optimize(x0, delta))
        {
            std::cerr << "lm not converged!!" << std::endl;
            break;
        }

        converged_ = is_converged(delta);
    }

    final_transformation_ = x0.cast<float>().matrix();
    pcl::transformPointCloud(*input_, output, final_transformation_);
}

template<typename PointTarget, typename PointSource>
bool RegistrationPCL<PointTarget, PointSource>::is_converged(const Eigen::Isometry3d &delta) const
{
    double accum = 0.0;
    Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = delta.translation();

    Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
    Eigen::Vector3d t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

    return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template<typename PointTarget, typename PointSource>
bool RegistrationPCL<PointTarget, PointSource>::step_optimize(Eigen::Isometry3d &x0, Eigen::Isometry3d &delta)
{
    switch (lsq_optimizer_type_)
    {
        case LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
            return step_lm(x0, delta);
        case LSQ_OPTIMIZER_TYPE::GaussNewton:
            return step_gn(x0, delta);
    }

    return step_lm(x0, delta);
}

template<typename PointTarget, typename PointSource>
bool RegistrationPCL<PointTarget, PointSource>::step_gn(Eigen::Isometry3d &x0, Eigen::Isometry3d &delta)
{
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    double y0 = linearize(x0, &H, &b);

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
    Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

    delta.setIdentity();
    delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    x0 = delta * x0;
    final_hessian_ = H;

    return true;
}

template<typename PointTarget, typename PointSource>
bool RegistrationPCL<PointTarget, PointSource>::step_lm(Eigen::Isometry3d &x0, Eigen::Isometry3d &delta)
{
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;
    double y0 = linearize(x0, &H, &b);

    if (lm_lambda_ < 0.0)
    {
        lm_lambda_ = lm_init_lambda_factor_ * H.diagonal().array().abs().maxCoeff();
    }

    double nu = 2.0;
    for (int i = 0; i < lm_max_iterations_; i++)
    {
        Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H + lm_lambda_ * Eigen::Matrix<double, 6, 6>::Identity());
        Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

        delta.setIdentity();
        delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
        delta.translation() = d.tail<3>();

        Eigen::Isometry3d xi = delta * x0;
        double yi = compute_error(xi);
        double rho = (y0 - yi) / (d.dot(lm_lambda_ * d - b));

        if (lm_debug_print_)
        {
            if (i == 0)
            {
                std::cout << boost::format("--- LM optimization ---\n%5s %15s %15s %15s %15s %15s %5s\n") % "i" %
                             "y0" % "yi" % "rho" % "lambda" % "|delta|" % "dec";
            }
            char dec = rho > 0.0 ? 'x' : ' ';
            std::cout << boost::format("%5d %15g %15g %15g %15g %15g %5c") % i % y0 % yi % rho % lm_lambda_ %
                         d.norm() % dec << std::endl;
        }

        if (rho < 0)
        {
            if (is_converged(delta))
            {
                return true;
            }

            lm_lambda_ = nu * lm_lambda_;
            nu = 2 * nu;
            continue;
        }

        x0 = xi;
        lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
        final_hessian_ = H;
        return true;
    }

    return false;
}

template <typename PointSource, typename PointTarget>
double RegistrationPCL_VGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b)
{
    H->setZero();
    b->setZero();
    double totalResidual = 0;
    std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(MP_PROC_NUM);
    std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(MP_PROC_NUM);
    for (int i = 0; i < MP_PROC_NUM; i++)
    {
        Hs[i].setZero();
        bs[i].setZero();
    }
#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : totalResidual) schedule(guided, 8)
    for (int i = 0; i < _matchedGuassinVoxelList.size(); i++)
    {
        const Eigen::Vector4d mean_A = Eigen::Vector4d(_matchedGuassinVoxelList[i]._pv.pl.x(), _matchedGuassinVoxelList[i]._pv.pl.y(), _matchedGuassinVoxelList[i]._pv.pl.z(), 1.0);
        const Eigen::Vector4d mean_B = _matchedGuassinVoxelList[i]._matchedVoxel->_mean;

        const Eigen::Vector4d transed_mean_A = trans * mean_A;
        const Eigen::Vector4d error = mean_B - transed_mean_A;

        double w = std::sqrt(_matchedGuassinVoxelList[i]._matchedVoxel->_numPoints);
        //double w=1;
        Eigen::Matrix4d RCR = _matchedGuassinVoxelList[i]._matchedVoxel->_cov +
                              trans.matrix() * _matchedGuassinVoxelList[i]._pv.neighCov * trans.matrix().transpose(); // CB + T*CA*T'
        RCR(3, 3) = 1.0;
        Eigen::Matrix4d voxelMahalanobis;
        voxelMahalanobis = RCR.inverse();
        voxelMahalanobis(3, 3) = 0.0;
        totalResidual += w * error.transpose() * voxelMahalanobis * error;

//        std::cout<<"id "<<i<<std::endl;
//        std::cout<<mean_A.transpose()<<std::endl;
//        std::cout<<transed_mean_A.transpose()<<std::endl;
//        std::cout<<_matchedGuassinVoxelList[i]._pv.neighCov<<std::endl;
//        std::cout<<mean_B.transpose()<<std::endl;
//        std::cout<<_matchedGuassinVoxelList[i]._matchedVoxel->_numPoints<<std::endl;
//        std::cout<<_matchedGuassinVoxelList[i]._matchedVoxel->_cov<<std::endl;
//        std::cout<<w<<std::endl;

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

#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : totalResidual) schedule(guided, 8)
    for (int i = 0; i < _matchedOctreeList.size(); i++)
    {
        const Eigen::Vector4d mean_A = Eigen::Vector4d(_matchedOctreeList[i].pv.pl.x(), _matchedOctreeList[i].pv.pl.y(), _matchedOctreeList[i].pv.pl.z(), 1.0);
        const Eigen::Vector4d mean_B = _matchedOctreeList[i].voxel_correspondence->_mean;

        const Eigen::Vector4d transed_mean_A = trans * mean_A;
        const Eigen::Vector4d error = mean_B - transed_mean_A;

        double w = std::sqrt(_matchedOctreeList[i].voxel_correspondence->_allPointsNum);
        //double w=1;
        Eigen::Matrix4d RCR = _matchedOctreeList[i].voxel_correspondence->_cloudCov +
                              trans.matrix() * _matchedOctreeList[i].pv.neighCov * trans.matrix().transpose(); // CB + T*CA*T'
        RCR(3, 3) = 1.0;
        Eigen::Matrix4d voxelMahalanobis;
        voxelMahalanobis = RCR.inverse();
        voxelMahalanobis(3, 3) = 0.0;
        totalResidual += w * error.transpose() * voxelMahalanobis * error;

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


    H->setZero();
    b->setZero();
    for (int i = 0; i < MP_PROC_NUM; i++)
    {
        *H += Hs[i];
        *b += bs[i];
    }
    return totalResidual;
}

template <typename PointSource, typename PointTarget>
double RegistrationPCL_VGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans)
{
    double residual = 0.0;
#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : residual)
    for (int i = 0; i < _matchedGuassinVoxelList.size(); i++)
    {
        const Eigen::Vector4d mean_A = Eigen::Vector4d(_matchedGuassinVoxelList[i]._pv.pl.x(), _matchedGuassinVoxelList[i]._pv.pl.y(), _matchedGuassinVoxelList[i]._pv.pl.z(), 1.0);
        const Eigen::Vector4d mean_B = _matchedGuassinVoxelList[i]._matchedVoxel->_mean;

        const Eigen::Vector4d transed_mean_A = trans * mean_A;
        const Eigen::Vector4d error = mean_B - transed_mean_A;

        double w = std::sqrt(_matchedGuassinVoxelList[i]._matchedVoxel->_numPoints);
        //double w=1;
        Eigen::Matrix4d RCR = _matchedGuassinVoxelList[i]._matchedVoxel->_cov + trans.matrix() * _matchedGuassinVoxelList[i]._pv.neighCov * trans.matrix().transpose(); // CB + T*CA*T'
        RCR(3, 3) = 1.0;
        Eigen::Matrix4d voxelMahalanobis;
        voxelMahalanobis = RCR.inverse();
        voxelMahalanobis(3, 3) = 0.0;
//        std::cout<<w<<" / "<<mean_A.transpose()<<" / "<<mean_B.transpose()<<"/ "<<error.transpose()<<std::endl;
//        std::cout<<voxelMahalanobis<<std::endl;
        residual += w * error.transpose() * voxelMahalanobis * error;
    }
#pragma omp parallel for num_threads(MP_PROC_NUM) reduction(+ : residual)
    for (int i = 0; i < _matchedOctreeList.size(); i++)
    {
        const Eigen::Vector4d mean_A = Eigen::Vector4d(_matchedOctreeList[i].pv.pl.x(), _matchedOctreeList[i].pv.pl.y(), _matchedOctreeList[i].pv.pl.z(), 1.0);
        const Eigen::Vector4d mean_B = _matchedOctreeList[i].voxel_correspondence->_mean;

        const Eigen::Vector4d transed_mean_A = trans * mean_A;
        const Eigen::Vector4d error = mean_B - transed_mean_A;

        double w = std::sqrt(_matchedOctreeList[i].voxel_correspondence->_allPointsNum);
        //double w=1;
        Eigen::Matrix4d RCR = _matchedOctreeList[i].voxel_correspondence->_cloudCov + trans.matrix() * _matchedOctreeList[i].pv.neighCov * trans.matrix().transpose(); // CB + T*CA*T'
        RCR(3, 3) = 1.0;
        Eigen::Matrix4d voxelMahalanobis;
        voxelMahalanobis = RCR.inverse();
        voxelMahalanobis(3, 3) = 0.0;
//        std::cout<<w<<" / "<<mean_A.transpose()<<" / "<<mean_B.transpose()<<"/ "<<error.transpose()<<std::endl;
//        std::cout<<voxelMahalanobis<<std::endl;
        residual += w * error.transpose() * voxelMahalanobis * error;
    }

    return residual;
}

template <typename PointSource, typename PointTarget>
bool RegistrationPCL_VGICP<PointSource, PointTarget>::reg(typename pcl::PointCloud<PointSource>::Ptr srcCloud,
                                                          typename pcl::PointCloud<PointTarget>::Ptr dstCloud,
                                                          Eigen::Matrix4d& InitTrans,
                                                          Eigen::Matrix4d& trans,
                                                          double voxelSize)
{
    std::unique_ptr<GaussianVoxelMap<PointSource>> voxelMapVGICP;
    voxelMapVGICP.reset(new GaussianVoxelMap<PointSource>(voxelSize, VoxelAccumulationMode::ADDITIVE));
    voxelMapVGICP->updateVoxelMap(dstCloud->points);

    std::vector<PointWithCov> pvList;
    for (size_t i = 0; i < srcCloud->size(); i++)
    {
        PointWithCov pv;
        pv.pl << srcCloud->points[i].x, srcCloud->points[i].y, srcCloud->points[i].z;
        pvList.push_back(pv);
    }
    std::cout<<"pv size:"<<pvList.size()<<std::endl;
    calcPointNeighCov(pvList, 20);

    Eigen::Isometry3d guess = Eigen::Isometry3d(InitTrans);
    this->setLMLambda(-1.0);
    std::set<GaussianVoxel::Ptr> updatedVoxelSet;
    bool isCoverged=false;
    for (int iterCount = 0; iterCount < 64; iterCount++)
    {
        std::cout<<"iter:"<<iterCount<<std::endl;
        typename pcl::PointCloud<PointSource>::Ptr transCloud(new pcl::PointCloud<PointSource>());
        transCloud->resize(srcCloud->size());
        pcl::transformPointCloud(*srcCloud, *transCloud, guess.matrix().template cast<float>());

        for (size_t i = 0; i < srcCloud->size(); i++)
            pvList[i].pw << transCloud->points[i].x, transCloud->points[i].y, transCloud->points[i].z;

        voxelMapVGICP->getCorrespondences(pvList, _matchedGuassinVoxelList, NeighborSearchMethod::DIRECT1);
        std::cout<<"_matchedGuassinVoxelList size:"<<_matchedGuassinVoxelList.size()<<std::endl;
        /*** Calculate voxel map covariances ***/
        for (auto &match: _matchedGuassinVoxelList)
        {
            //std::cout<<match._pv.pl.transpose()<<std::endl;
            if(!updatedVoxelSet.count(match._matchedVoxel))
            {
                updatedVoxelSet.insert(match._matchedVoxel);
                match._matchedVoxel->calcCovariances(dstCloud, 20);
            }
        }

        Eigen::Isometry3d delta;
        if (!this->step_optimize(guess, delta))
        {
            std::cerr << "lm not converged!!" << std::endl;
            break;
        }

        if (this->is_converged(delta))
        {
            isCoverged=true;
            break;
        }
    }
    if(isCoverged)
    {
        trans=guess.matrix();
        return true;
    }
    return false;
}

#endif //SRC_LSQREGPCL_IMPL_HPP
