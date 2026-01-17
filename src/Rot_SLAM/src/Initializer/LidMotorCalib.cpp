//
// Created by w on 2022/9/15.
//

#include "Initializer/LidMotorCalib.h"

bool calcEXRotation(std::vector<Eigen::Matrix4d> relToList,
                    std::vector<Eigen::Matrix4d> relTlList,
                    Eigen::Matrix3d& Rol,  Eigen::Vector3d& tol,
                    double maxIter, double maxAngleErr)
{
    if (relToList.size() != relTlList.size())
    {
        std::cerr << "Wrong size with relative lidar and motor motions!" << std::endl;
        return false;
    }
    int relMotionSize = relTlList.size();
    std::cout << "Relative motions size: " << relMotionSize << std::endl;

    Eigen::Quaterniond QloEstimate;
    Eigen::Vector3d RolCov;
    bool isConvReached = false;
    for (int i = 0; i < maxIter; i++)
    {
        relMotionSize = relTlList.size();
        if (relMotionSize < 3)
        {
            std::cerr << "Not enough relative motions!" << std::endl;
            return false;
        }
        //std::cout<<"Iter time:"<<i<<std::endl;
        //std::cout<<"Relative motions size: "<<relMotionSize<<std::endl;
        //Qlo*Qo1o2=Ql1l2*Qlo => ([Ql1l2]L-[Qo1o2]R)*Qlo=0 => A*Qlo=0
        Eigen::MatrixXd A(relMotionSize * 4, 4);
        A.setZero();
        for (int i = 0; i < relMotionSize; i++)
        {
			// for Feima
            // Eigen::AngleAxis<double> angleaxis;
            // angleaxis.fromRotationMatrix(relToList[i].block<3, 3>(0, 0));
            // if(angleaxis.angle() * 180. / M_PI>=140&&angleaxis.angle() * 180. / M_PI<=180)
            //    continue;
            Eigen::Quaterniond r1(relTlList[i].block<3, 3>(0, 0));//Ql1l2
            Eigen::Quaterniond r2(relToList[i].block<3, 3>(0, 0));//Qo1o2

            double angularDist = 180 / M_PI * r1.angularDistance(r2);
            std::printf("%d/%f ", i, angularDist);

            //double huber = angular_distance > 1.0 ? 1.0 / angular_distance : 1.0;
            double huber = 1.0;
            Eigen::Matrix4d L, R;
            //[Ql1l2]L
            double w = r1.w();
            Eigen::Vector3d q = r1.vec();
            L.block<3, 3>(0, 0) = w *   Eigen::Matrix3d::Identity() +   skewSymmetric(q);
            L.block<3, 1>(0, 3) = q;
            L.block<1, 3>(3, 0) = -q.transpose();
            L(3, 3) = w;
            //[Qo1o2]R
            w = r2.w();
            q = r2.vec();
            R.block<3, 3>(0, 0) = w *   Eigen::Matrix3d::Identity() -   skewSymmetric(q);
            R.block<3, 1>(0, 3) = q;
            R.block<1, 3>(3, 0) = -q.transpose();
            R(3, 3) = w;

            A.block<4, 4>(i * 4, 0) = huber * (L - R);
        }

        Eigen::JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
        Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
        QloEstimate = Quaterniond(x); //Qlo

        //Quaterniond r1=(Quaterniond(_relTlList.back().block<3,3>(0,0)*QloEstimate));
        //Quaterniond r2=(Quaterniond(QloEstimate*relToList.back().block<3,3>(0,0)));
        //double angularDist = 180 / M_PI * r1.angularDistance(r2);

       // cout << "Rol guess:" << R2ypr(QloEstimate.toRotationMatrix().inverse()).transpose() << endl;
        std::cout << "Rol guess:" << QloEstimate.toRotationMatrix() << std::endl;
//        for (int i = 0; i < relMotionSize; i++)
//        {
//            Quaterniond r1=(Quaterniond(relTlList[i].block<3,3>(0,0))*QloEstimate);//Qo0li=Qol*Ql0li
//            Quaterniond r2=(Quaterniond(QloEstimate*relToList[i].block<3,3>(0,0)));//Qo0li'=Qo0oi*Qol
//            double angularDist = 180 / M_PI * r1.angularDistance(r2);
//            std::printf("%d/%f ", i, angularDist);
//        }

        int nDiscard = 0;
        i = 0;
        for (auto itrTl = relTlList.begin(), itrTo = relToList.begin(); itrTl != relTlList.end();)
        {
            Quaterniond r1((*itrTl).block<3, 3>(0, 0));//Ql1l2
            Quaterniond r2(QloEstimate * (*itrTo).block<3, 3>(0, 0) * QloEstimate.inverse());//Qlo*Qo1o2*Qol

            double angularDist = 180 / M_PI * r1.angularDistance(r2);
            std::printf("%d/%f ", i++, angularDist);
            if (angularDist > maxAngleErr)
            {
                itrTl = relTlList.erase(itrTl);
                itrTo = relToList.erase(itrTo);
                nDiscard++;
            }
            else
            {
                itrTl++;
                itrTo++;
            }
        }
        RolCov = svd.singularValues().tail<3>();
        std::cout << "Rol Cov:" << RolCov.transpose() << std::endl;
        if (RolCov(1) > 0.25)
            isConvReached = true;
        if (nDiscard == 0)
        {
            std::cout << "No angle error exceeded: exit iteration" << std::endl;
            break;
        }

        std::cout << "Discard large angle error size:" << nDiscard << std::endl;
    }

    if (isConvReached)
    {
        Rol = QloEstimate.toRotationMatrix().inverse();//Qol
        tol = Eigen::Vector3d();
        std::cout << "Rol guess:" << Rol << std::endl;
        Eigen::AngleAxis<double> angleaxis;
        angleaxis.fromRotationMatrix(relToList.back().block<3, 3>(0, 0));
        std::cout << "Current rotation angle from init frame:" << angleaxis.angle() * 180. / M_PI << std::endl;

        return true;
    }

    return false;
}