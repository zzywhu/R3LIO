//
// Created by w on 2022/8/18.
//

#include "Registrator/BaseRegistration.h"

void Registration::damping_iter()
{
    double u = 0.01, v = 2;
    Eigen::Matrix<double, 6, 6> D;
    D.setIdentity();
    Eigen::Matrix<double, 6, 6> Hess, Hess2;
    Eigen::Matrix<double, 6, 1> g;
    Eigen::Matrix<double, 6, 1> dxi;
    double residual1, residual2;

    Eigen::MatrixXd matA(6, 6);
    Eigen::VectorXd matB(6);
    Eigen::VectorXd matX(6);

//    cv::Mat matA(6, 6, CV_64F, cv::Scalar::all(0));
//    cv::Mat matB(6, 1, CV_64F, cv::Scalar::all(0));
//    cv::Mat matX(6, 1, CV_64F, cv::Scalar::all(0));

    for (int i = 0; i < _maxInteration; i++)
    {
        evaluate_para(_so3Pose, _tPose, Hess, g, residual1);
        D = Hess.diagonal().asDiagonal();

        // dxi = (Hess + u*D).bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-g);

        Hess2 = Hess + u * D;
        for (int j = 0; j < 6; j++)
        {
            matB(j, 0) = -g(j, 0);
            for (int f = 0; f < 6; f++)
                matA(j, f) = Hess2(j, f);
        }
        dxi = matA.colPivHouseholderQr().solve(matB);

        _so3Temp = SO3::exp(dxi.block<3, 1>(0, 0)) * _so3Pose;
        _tTemp = _tPose + dxi.block<3, 1>(3, 0);
        evaluate_only_residual(_so3Temp, _tTemp, residual2);
        double q1 = dxi.dot(u * D * dxi - g);
        double q = residual1 - residual2;
        //printf("residual: %lf u: %lf v: %lf q: %lf %lf %lf\n", residual1, u, v, q/q1, q1, q);
        if (q > 0)
        {
            _so3Pose = _so3Temp;
            _tPose = _tTemp;
            q = q / q1;
            v = 2;
            q = 1 - pow(2 * q - 1, 3);
            u *= (q < 1. / 3. ? 1. / 3. : q);
        }
        else
        {
            u = u * v;
            v = 2 * v;
        }

        if (fabs(residual1 - residual2) < 1e-9)
            break;
    }
}



