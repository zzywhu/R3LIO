//
// Created by w on 2022/8/25.
//
#include <pcl/search/kdtree.h>

#include "Misc/Common.h"

#ifdef _WIN32
void usleep(unsigned long usec)
{
	HANDLE timer;
	LARGE_INTEGER interval;
	interval.QuadPart = -(10 * usec);

	timer = CreateWaitableTimer(NULL, TRUE, NULL);
	SetWaitableTimer(timer, &interval, 0, NULL, NULL, 0);
	WaitForSingleObject(timer, INFINITE);
	CloseHandle(timer);
}
#endif // _WIN32

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

void calcBodyCov(PointWithCov &pv, const float range_inc, const float degree_inc)
{
    float range = sqrt(pv.pl[0] * pv.pl[0] + pv.pl[1] * pv.pl[1] + pv.pl[2] * pv.pl[2]);
    float range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
    Eigen::Vector3d direction(pv.pl);
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1),
            direction(2), 0, -direction(0),
            -direction(1), direction(0), 0;

    Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    pv.bodyCov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void calcPointNeighCov(PvList &pvList, int kCorrespondences)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (auto &pv:pvList)
        pclCloud->push_back(pcl::PointXYZ(pv.pl.x(), pv.pl.y(), pv.pl.z()));

    static pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    kdtree.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    kdtree->setInputCloud(pclCloud);

    const int cloudSize = pclCloud->size();
    const int nearSize = std::min(cloudSize, kCorrespondences);
#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
    for (int i = 0; i < cloudSize; i++)
    {
        std::vector<int> kIndices;
        std::vector<float> kSqDistances;
        kdtree->nearestKSearch(pclCloud->at(i), nearSize, kIndices, kSqDistances);

        Eigen::Matrix<double, 4, -1> neighbors(4, nearSize);
        for (int j = 0; j < kIndices.size(); j++)
            neighbors.col(j) = pclCloud->at(kIndices[j]).getVector4fMap().template cast<double>();

        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        Eigen::Matrix4d cov = neighbors * neighbors.transpose() / nearSize;

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        pvList[i].neighCov = Eigen::Matrix4d::Zero();
        pvList[i].neighCov.template block<3, 3>(0, 0) = svd.matrixU() * Eigen::Vector3d(1, 1, 1e-3).asDiagonal() * svd.matrixV().transpose();
    }
}


Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

double calVelocity(Matrix4ds &Trans, int mean_frame_num, int frame_per_second)
{
    int mean_frame_num_used;

    mean_frame_num_used = min_(mean_frame_num, Trans.size());

    double accumulate_tx = 0, accumulate_ty = 0, accumulate_tz = 0;

    int count_sum = 0;

    for (auto iter = Trans.end() - 1; iter >= Trans.end() - mean_frame_num_used; iter--)
    {
        //LOG(INFO) << *iter;
        Eigen::Matrix4d tempMat = *iter;
        //simple implement
        accumulate_tx += tempMat(0, 3);
        accumulate_ty += tempMat(1, 3);
        accumulate_tz += tempMat(2, 3);
        count_sum++;
    }

    double vx = accumulate_tx / mean_frame_num_used * frame_per_second;
    double vy = accumulate_ty / mean_frame_num_used * frame_per_second;
    double vz = accumulate_tz / mean_frame_num_used * frame_per_second;

    double mean_linear_velocity = std::sqrt(vx * vx + vy * vy + vz * vz);

    //std::cout << "current approximate velocity: " << mean_linear_velocity * 3.6 << " (km/h)\n";

    return mean_linear_velocity;
}

double calTranslationFromTranmat(Eigen::Matrix4d &tran_mat)
{
    Eigen::Vector3d translation_vec;
    translation_vec = tran_mat.block<3, 1>(0, 3);
    return (translation_vec.norm());
}

double calHeadingDegFromTranmat(Eigen::Matrix4d &tran_mat)
{
    Eigen::Vector3d euler_angle = (tran_mat.block<3, 3>(0, 0)).eulerAngles(0, 1, 2); //rotation axis : z,y',x''

    double roll_deg = std::abs(euler_angle(0) / M_PI * 180.0);
    double pitch_deg = std::abs(euler_angle(1) / M_PI * 180.0);
    double yaw_deg = std::abs(euler_angle(2) / M_PI * 180.0);

    if (roll_deg > 90)
        roll_deg = 180 - roll_deg;
    if (pitch_deg > 90)
        pitch_deg = 180 - pitch_deg;
    if (yaw_deg > 90)
        yaw_deg = 180 - yaw_deg;

    return yaw_deg;
}

double calRotationDegFromTranmat(Eigen::Matrix4d &tran_mat)
{
    Eigen::AngleAxisd rs(tran_mat.block<3, 3>(0, 0));

    double rotation_deg = std::abs(rs.angle()) * 180.0 / M_PI;
    return rotation_deg;
}

