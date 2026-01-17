#ifndef ABSOLUTE_ORIENTATION_HPP
#define ABSOLUTE_ORIENTATION_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <climits>
#include <locale>
#include <vector>
#include <Eigen/Dense>

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define ABSOULUTE_ORIENT_API __declspec(dllexport)
#else
#define ABSOULUTE_ORIENT_API __declspec(dllimport)
#endif
#else
#define ABSOULUTE_ORIENT_API
#endif


class ABSOULUTE_ORIENT_API AbsouteOrientation
{
public:
    struct Point3d
    {
        Point3d() {}

        Point3d(const double &_x, const double &_y, const double &_z) : x(_x), y(_y), z(_z)
        {
        }

        double x, y, z;

        bool Normalize()
        {
            double norm = sqrt(x * x + y * y + z * z);
            if (norm < 0.000001f)
                return false;
            x /= norm;
            y /= norm;
            z /= norm;
            return true;
        }

        inline double Dot(const Point3d &pt) const
        {
            return x * pt.x + y * pt.y + z * pt.z;
        }

        inline Point3d Cross(const Point3d &pt)
        {
            return Point3d(y * pt.z - z * pt.y, -x * pt.z + z * pt.x, x * pt.y - y * pt.x);
        }

        Point3d operator-(const Point3d &pt) const
        {
            return Point3d(x - pt.x, y - pt.y, z - pt.z);
        }
    };

    enum OrientationMethod
    {
        LOCAL_COOR_P3 = 1,
        USE_QUATERNION = 2,
        USE_ORTHO_MAT = 3
    };
public:
    AbsouteOrientation();

    ~AbsouteOrientation();

    // int SetCorresPts(const Point3d* pts1, const Point3d* pts2, const int& pt_num);
    static bool runCalSevenParams(std::vector<double> &targetFramePts, std::vector<double> &sourceFramePts, const size_t ptsNum, double *R, double *t, double &Scale);

    static bool orientation(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num, double *R, double *T, double &Scale, const OrientationMethod &omethod = USE_QUATERNION);

private:
    static inline double dis(const Point3d &p1, const Point3d &p2)
    {
        return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y * p2.y) + (p1.z - p2.z) * (p1.z * p2.z));
    }

    static bool computeRotP3(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num, double *R);

    static bool computeRotQuaternion(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num, double *R);

    static bool computeRotOrthoMat(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num, double *R);

    static void quaternion2Rotation(const double *q, double *rot)
    {
        rot[0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
        rot[1] = (q[1] * q[2] - q[0] * q[3]) * 2;
        rot[2] = (q[1] * q[3] + q[0] * q[2]) * 2;

        rot[3] = (q[2] * q[1] + q[0] * q[3]) * 2;
        rot[4] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
        rot[5] = (q[2] * q[3] - q[0] * q[1]) * 2;

        rot[6] = (q[3] * q[1] - q[0] * q[2]) * 2;
        rot[7] = (q[3] * q[2] + q[0] * q[1]) * 2;
        rot[8] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
    }

    static void rotation2Quaternion(const double *rot, double *q)
    {
        double q0_2 = (1.0 + rot[0] + rot[4] + rot[8]) / 4;
        double qx_2 = (1.0 + rot[0] - rot[4] - rot[8]) / 4;
        double qy_2 = (1.0 - rot[0] + rot[4] - rot[8]) / 4;
        double qz_2 = (1.0 - rot[0] - rot[4] + rot[8]) / 4;
        if (q0_2 > qx_2 && q0_2 > qy_2 && q0_2 > qz_2)
        {
            q[0] = sqrt(q0_2);
            q[1] = (rot[7] - rot[5]) / 4 / q[0];
            q[2] = (rot[2] - rot[6]) / 4 / q[0];
            q[3] = (rot[3] - rot[1]) / 4 / q[0];
        } else if (qx_2 > qy_2 && qx_2 > qz_2)
        {
            q[1] = sqrt(qx_2);
            q[0] = (rot[7] - rot[5]) / 4 / q[1];
            q[2] = (rot[3] + rot[1]) / 4 / q[1];
            q[3] = (rot[2] + rot[6]) / 4 / q[1];
        } else if (qy_2 > qz_2)
        {
            q[2] = sqrt(qy_2);
            q[0] = (rot[2] - rot[6]) / 4 / q[2];
            q[1] = (rot[3] + rot[1]) / 4 / q[2];
            q[3] = (rot[7] + rot[5]) / 4 / q[2];
        } else
        {
            q[3] = sqrt(qz_2);
            q[0] = (rot[3] - rot[1]) / 4 / q[3];
            q[1] = (rot[2] + rot[6]) / 4 / q[3];
            q[2] = (rot[7] + rot[5]) / 4 / q[3];
        }
    }
};



#endif
