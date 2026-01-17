//
// Created by w on 2022/5/26.
//
#include "Optimizer/Optimizer.h"

void eigToDouble(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double *pose) {
    Eigen::Vector3d Rvec = RotationToAngleAxis(R);
    pose[0] = Rvec.x();
    pose[1] = Rvec.y();
    pose[2] = Rvec.z();
    pose[3] = t(0);
    pose[4] = t(1);
    pose[5] = t(2);
}

void doubleToEig(const double *pose, Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    R = AngleAxisToRotationMatrix(Eigen::Vector3d(pose[0], pose[1], pose[2]));
    t = Eigen::Vector3d(pose[3], pose[4], pose[5]);
}

void poseOptimize(MatchedInfoList &matchedInfoList, const double &motorRotAngle,
                  Eigen::Matrix3d &exR, Eigen::Vector3d &ext,
                  double maxOulierErr, bool isVerbose) {
    ceres::Problem problem;
    ceres::LossFunction *lossFunction;
    lossFunction = new ceres::HuberLoss(2.0);
    //loss_function = new ceres::CauchyLoss(10.0);
    double exPose6d[6];
    eigToDouble(exR, ext, exPose6d);
    problem.AddParameterBlock(exPose6d, 6);

    for (auto &match : matchedInfoList) {
        const double *pabcd = match._pabcd.data();
        ceres::CostFunction *costfunc = RegErrorOnlyExPose::Create(pabcd, match._pl, motorRotAngle);
        problem.AddResidualBlock(costfunc, lossFunction, exPose6d);
    }

    //    double totalResidual=0;
    //    for(int i=0; i < frameSize; i++)
    //    {
    //        auto& matchedInfos = matchedInfoList[i];
    //        const double& theta = rotationAngleList[i];
    //        for(auto mItr =  matchedInfos.begin();mItr!=matchedInfos.end();mItr++)
    //        {
    //            double point3d[3]={mItr->first.x,mItr->first.y,mItr->first.z};
    //            double* pabcd = mItr->second.data();
    //            double res=computeResidual(exPose6d, pabcd, point3d, theta);
    //            //std::cout<<"Residual:"<<res<<std::endl;
    //            totalResidual+=res;
    //        }
    //    }
    //    std::cout<<"Total residual before BA:"<<totalResidual<<std::endl;

    ceres::Solver::Options options;
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.max_num_iterations = 1;
    options.max_solver_time_in_seconds = 1000;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1;
    options.eta = 1e-2;
    options.use_nonmonotonic_steps = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //std::cout << summary.FullReport() << "\n";
    //std::cout << summary.BriefReport() << "\n";

    //Discard ouliers
    if (isVerbose || maxOulierErr) {
        double totalResidual = 0;
        int nDiscard = 0;
        for (auto mItr = matchedInfoList.begin(); mItr != matchedInfoList.end();) {
            double point3d[3] = {mItr->_pl.x(), mItr->_pl.y(), mItr->_pl.z()};
            double *pabcd = mItr->_pabcd.data();
            double res = computeResidual(exPose6d, pabcd, point3d, motorRotAngle);
            //std::cout<<"Residual:"<<res<<std::endl;

            if (maxOulierErr > 0 && res > maxOulierErr) {
                mItr = matchedInfoList.erase(mItr);
                nDiscard++;
            } else {
                mItr++;
                totalResidual += res;
            }
        }
        double meanRes = totalResidual / matchedInfoList.size();
        if ((!isVerbose && nDiscard > 0) || isVerbose)
            std::cout << "matches/discard/mean residual:" << matchedInfoList.size() << "/" << nDiscard << "/" << meanRes << std::endl;
    }

    doubleToEig(exPose6d, exR, ext);
}

void initOptimize(const MatchedInfoList &matchedInfoList, const double &rotationAngle,
                  Eigen::Matrix3d &exR, Eigen::Vector3d &ext) {
    ceres::Problem problem;
    ceres::LossFunction *lossFunction;
    lossFunction = new ceres::HuberLoss(2.0);
    //loss_function = new ceres::CauchyLoss(10.0);
    double exPose6d[6];
    eigToDouble(exR, ext, exPose6d);
    problem.AddParameterBlock(exPose6d, 6);

    for (auto &match : matchedInfoList) {
        Eigen::Vector3d point3d;
        point3d << match._pl.x(), match._pl.y(), match._pl.z();
        const double *_pabcd = match._pabcd.data();
        ceres::CostFunction *costfunc = InitRegErrorExPose::Create(_pabcd, point3d, rotationAngle);
        problem.AddResidualBlock(costfunc, lossFunction, exPose6d);
    }

    ceres::Solver::Options options;
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.max_num_iterations = 1;
    options.max_solver_time_in_seconds = 1000;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 1;
    options.eta = 1e-2;
    options.use_nonmonotonic_steps = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //std::cout << summary.FullReport() << "\n";
    std::cout << summary.BriefReport() << "\n";

    doubleToEig(exPose6d, exR, ext);
}

void calcHandeyeExtrinsic(const std::vector<Eigen::Matrix4d> &relativeTos, const std::vector<Eigen::Matrix4d> &relativeTls,
                          Eigen::Matrix3d &exR, Eigen::Vector3d &ext) {
    ceres::Problem problem;
    ceres::LossFunction *lossFunction;
    lossFunction = new ceres::HuberLoss(0.1);
    //loss_function = new ceres::CauchyLoss(10.0);

    Eigen::Quaterniond exQ(exR);
    ceres::LocalParameterization *quaternionLocalParameterization = new ceres::EigenQuaternionParameterization;
    int frameSize = relativeTls.size();
    for (int i = 0; i < frameSize; i++) {
        Eigen::Quaterniond QoRel(relativeTos[i].block<3, 3>(0, 0));
        Eigen::Quaterniond QlRel(relativeTls[i].block<3, 3>(0, 0));
        Eigen::Vector3d toRel(relativeTos[i].block<3, 1>(0, 3));
        Eigen::Vector3d tlRel(relativeTls[i].block<3, 1>(0, 3));

        ceres::CostFunction *costfunc = RelativeRTError::Create(toRel.data(), QoRel.coeffs().data(), tlRel.data(), QlRel.coeffs().data(), 0.1, 0.01);
        problem.AddResidualBlock(costfunc, lossFunction, ext.data(), exQ.coeffs().data());
        problem.SetParameterization(exQ.coeffs().data(), quaternionLocalParameterization);
    }
    ceres::Solver::Options options;
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 100;
    options.max_solver_time_in_seconds = 1000;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 1;
    options.eta = 1e-2;
    options.use_nonmonotonic_steps = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    //std::cout << summary.FullReport() << "\n";
    std::cout << summary.BriefReport() << "\n";

    exR = exQ.toRotationMatrix();
}
