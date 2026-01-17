
#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#ifdef _WIN32
#include <windows.h>
#define PAUSE system("pause")
#else

#include <unistd.h>

#define PAUSE system("read -p 'Press Enter to continue...' var")
#endif // _WIN32
#include "deque"

#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv2/opencv.hpp>
#include <std_msgs/Header.h>

#include <Misc/Color.h>
#include <Misc/So3Math.h>
#include "ImuData.h"
#include "Pose6D.h"
#include "GNSSData.h"
#include <torch/torch.h>
#include <torch/script.h>

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define COMMON_API __declspec(dllexport)
#else
#define COMMON_API __declspec(dllimport)
#endif // __DLL_EXPORTS__
#else
#define COMMON_API
#endif // __SHARED_LIBS__

using namespace Eigen;

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 30
#define PI_M (3.14159265358)
#define G_m_s2 (9.81)  // Gravity const in GuangDong/China
#define DIM_STATE (24) // Dimension of states (Let Dim(SO(3)) = 3)

#define LIDAR_SP_LEN (2)
#define INIT_COV (1)
//#define INIT_COV (0.0000001)
#define NUM_MATCH_POINTS (5)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]

#define DEBUG_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "Log/" + name))
#define RESULT_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "result/" + name))

//#define uint unsigned int

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZRGB PointTypeRGB;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Vector3d V3D;
typedef Vector2d V2D;
typedef Vector4d V4D;
typedef Matrix3d M3D;
typedef Matrix4d M4D;
typedef Vector3f V3F;
typedef Matrix3f M3F;
typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Matrix4ds;

//typedef std::pair<PointType, Eigen::Vector4d> MatchedInfo;
struct MatchedInfo {
    MatchedInfo(const V3D &pl, const Eigen::Vector4d &pabcd, const V3D &center, const V3D &norm, const double &coeffs) {
        _pl = pl;
        _center = center;
        _norm = norm;
        _pabcd = pabcd;
        _coeffs = coeffs;
    }

    V3D _pl;
    V3D _center;
    V3D _norm;
    Eigen::Vector4d _pabcd;
    double _coeffs;
};

const M3D Eye3d(M3D::Identity());
const V3D Zero3d(0, 0, 0);
const M4D Eye4d(M4D::Identity());

struct BEVFrame {
    std_msgs::Header header;
    cv::Mat img_photo_u8;
    cv::Mat img_matches;
    cv::Mat img_with_keypoints;
    cv::Mat img_dense;
    M4D T_current_last;
    pcl::PointCloud<pcl::PointXYZI>::Ptr points;

    // std::vector<std::vector<std::vector<float>>> local_feats; // 128
    // std::vector<float> global_desc; // 8192
    torch::Tensor local_feats;
    std::vector<float> global_desc;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat query_descriptors;

    BEVFrame clone() const {
        BEVFrame f;
        f.header = header;
        f.img_photo_u8 = img_photo_u8.clone();
        f.img_matches = img_matches.clone();
        f.img_with_keypoints = img_with_keypoints.clone();
        f.img_dense = img_dense.clone();
        f.T_current_last = T_current_last;
        f.points = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*points);
        f.local_feats = local_feats.clone();
        f.global_desc = global_desc;
        f.keypoints = keypoints;
        f.query_descriptors = query_descriptors.clone();
        return f;
    }
};

typedef std::vector<MatchedInfo> MatchedInfoList;

#define MD(a, b) Eigen::Matrix<double, (a), (b)>
#define VD(a) Eigen::Matrix<double, (a), 1>
#define MF(a, b) Eigen::Matrix<float, (a), (b)>
#define VF(a) Eigen::Matrix<float, (a), 1>

#define max_(a, b) (((a) > (b)) ? (a) : (b))
#define min_(a, b) (((a) < (b)) ? (a) : (b))

enum FeatType {
    Plane,
    Line
};

// 3D point with covariance
typedef struct PointWithCov {
    Eigen::Vector3d pi; // cood in imu
    Eigen::Vector3d pl; // coord in lidar
    Eigen::Vector3d pw; // coord in world
    Eigen::Matrix3d bodyCov = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d obsCov = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d neighCov = Eigen::Matrix4d::Identity();
    FeatType featType;
};

typedef std::vector<PointWithCov> PvList;

struct MeasureGroup // Lidar data and imu dates for the curent process
{
    MeasureGroup() {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
    std::deque<SensorMsgs::ImuConstPtr> imu;
    std::vector<std::pair<double, double>> tMotorAngles;
    SensorMsgs::GNSSData::Ptr gnss;
};

struct StatesGroup {
    StatesGroup() {
        rot_end = M3D::Identity();
        pos_end = Zero3d;
        offset_R_L_I = M3D::Identity();
        offset_T_L_I = Zero3d;
        vel_end = Zero3d;
        bias_g = Zero3d;
        bias_a = Zero3d;
        gravity = Zero3d;
        cov = MD(DIM_STATE, DIM_STATE)::Identity() * INIT_COV;
        cov.block<9, 9>(15, 15) = MD(9, 9)::Identity() * 0.00001;
    };

    StatesGroup(const StatesGroup &b) {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->offset_R_L_I = b.offset_R_L_I;
        this->offset_T_L_I = b.offset_T_L_I;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->gravity = b.gravity;
        this->cov = b.cov;
    };

    StatesGroup &operator=(const StatesGroup &b) {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->offset_R_L_I = b.offset_R_L_I;
        this->offset_T_L_I = b.offset_T_L_I;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->gravity = b.gravity;
        this->cov = b.cov;
        return *this;
    };

    StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add) {
        StatesGroup a;
        a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
        a.offset_R_L_I = this->offset_R_L_I * Exp(state_add(6, 0), state_add(7, 0), state_add(8, 0));
        a.offset_T_L_I = this->offset_T_L_I + state_add.block<3, 1>(9, 0);
        a.vel_end = this->vel_end + state_add.block<3, 1>(12, 0);
        a.bias_g = this->bias_g + state_add.block<3, 1>(15, 0);
        a.bias_a = this->bias_a + state_add.block<3, 1>(18, 0);
        a.gravity = this->gravity + state_add.block<3, 1>(21, 0);
        a.cov = this->cov;
        return a;
    };

    StatesGroup &operator+=(const Matrix<double, DIM_STATE, 1> &state_add) {
        this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        this->pos_end += state_add.block<3, 1>(3, 0);
        this->offset_R_L_I = this->offset_R_L_I * Exp(state_add(6, 0), state_add(7, 0), state_add(8, 0));
        this->offset_T_L_I += state_add.block<3, 1>(9, 0);
        this->vel_end += state_add.block<3, 1>(12, 0);
        this->bias_g += state_add.block<3, 1>(15, 0);
        this->bias_a += state_add.block<3, 1>(18, 0);
        this->gravity += state_add.block<3, 1>(21, 0);
        return *this;
    };

    Matrix<double, DIM_STATE, 1> operator-(const StatesGroup &b) {
        Matrix<double, DIM_STATE, 1> a;
        M3D rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3, 1>(0, 0) = Log(rotd);
        a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
        M3D offsetd(b.offset_R_L_I.transpose() * this->offset_R_L_I);
        a.block<3, 1>(6, 0) = Log(offsetd);
        a.block<3, 1>(9, 0) = this->offset_T_L_I - b.offset_T_L_I;
        a.block<3, 1>(12, 0) = this->vel_end - b.vel_end;
        a.block<3, 1>(15, 0) = this->bias_g - b.bias_g;
        a.block<3, 1>(18, 0) = this->bias_a - b.bias_a;
        a.block<3, 1>(21, 0) = this->gravity - b.gravity;
        return a;
    };

    void resetpose() {
        this->rot_end = M3D::Identity();
        this->pos_end = Zero3d;
        this->vel_end = Zero3d;
    }

    M3D rot_end;                              // the estimated attitude (rotation matrix) at the end lidar point
    V3D pos_end;                              // the estimated position at the end lidar point (world frame)
    M3D offset_R_L_I;                         // Rotation from Lidar frame L to IMU frame I
    V3D offset_T_L_I;                         // Translation from Lidar frame L to IMU frame I
    V3D vel_end;                              // the estimated velocity at the end lidar point (world frame)
    V3D bias_g;                               // gyroscope bias
    V3D bias_a;                               // accelerator bias
    V3D gravity;                              // the estimated gravity acceleration
    Matrix<double, DIM_STATE, DIM_STATE> cov; // states covariance
};

typedef struct IMUData {
    double _timestamp;
    Eigen::Vector3d _position;
    Eigen::Quaterniond _orientation;
    Eigen::Vector3d _velocity;
    Eigen::Vector3d _accelerate;
    Eigen::Vector3d _angular_velocity;
    Eigen::Vector3d _ba;
    Eigen::Vector3d _bg;
} IMUData;

#ifdef _WIN32
void COMMON_API usleep(unsigned long usec);
#endif // _WIN32

template <typename T>
T rad2deg(T radians) {
    return radians * 180.0 / PI_M;
}

template <typename T>
T deg2rad(T degrees) {
    return degrees * PI_M / 180.0;
}

float COMMON_API pointDistance(PointType p);

float COMMON_API pointDistance(PointType p1, PointType p2);

template <typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g,
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R) {
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++) {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
    }
    // Map<M3D>(rot_kp.rot, 3,3) = R;
    return move(rot_kp);
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
        q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

void COMMON_API calcBodyCov(PointWithCov &pv, const float range_inc, const float degree_inc);

void COMMON_API calcPointNeighCov(std::vector<PointWithCov> &pvList, int kCorrespondences = 20);

template <typename T>
Eigen::Matrix<T, 3, 3> AngleAxisToRotationMatrix(const Eigen::Matrix<T, 3, 1> &rvec) {
    T angle = rvec.norm();
    if (angle == T(0))
        return Eigen::Matrix<T, 3, 3>::Identity();

    Eigen::Matrix<T, 3, 1> axis;
    axis = rvec.normalized();

    Eigen::Matrix<T, 3, 3> rmat;
    rmat = Eigen::AngleAxis<T>(angle, axis);

    return rmat;
}

template <typename T>
Eigen::Matrix<T, 3, 1> RotationToAngleAxis(const Eigen::Matrix<T, 3, 3> &rmat) {
    Eigen::AngleAxis<T> angleaxis;
    angleaxis.fromRotationMatrix(rmat);
    return angleaxis.angle() * angleaxis.axis();
}

template <class T>
void calc(T matrix[4][5], Eigen::Vector3d &solution) //计算行列式
{
    T base_D = matrix[1][1] * matrix[2][2] * matrix[3][3] + matrix[2][1] * matrix[3][2] * matrix[1][3] + matrix[3][1] * matrix[1][2] * matrix[2][3];
    base_D = base_D - (matrix[1][3] * matrix[2][2] * matrix[3][1] + matrix[1][1] * matrix[2][3] * matrix[3][2] + matrix[1][2] * matrix[2][1] * matrix[3][3]);

    if (base_D != 0) {
        T x_D = matrix[1][4] * matrix[2][2] * matrix[3][3] + matrix[2][4] * matrix[3][2] * matrix[1][3] + matrix[3][4] * matrix[1][2] * matrix[2][3];
        x_D = x_D - (matrix[1][3] * matrix[2][2] * matrix[3][4] + matrix[1][4] * matrix[2][3] * matrix[3][2] + matrix[1][2] * matrix[2][4] * matrix[3][3]);
        T y_D = matrix[1][1] * matrix[2][4] * matrix[3][3] + matrix[2][1] * matrix[3][4] * matrix[1][3] + matrix[3][1] * matrix[1][4] * matrix[2][3];
        y_D = y_D - (matrix[1][3] * matrix[2][4] * matrix[3][1] + matrix[1][1] * matrix[2][3] * matrix[3][4] + matrix[1][4] * matrix[2][1] * matrix[3][3]);
        T z_D = matrix[1][1] * matrix[2][2] * matrix[3][4] + matrix[2][1] * matrix[3][2] * matrix[1][4] + matrix[3][1] * matrix[1][2] * matrix[2][4];
        z_D = z_D - (matrix[1][4] * matrix[2][2] * matrix[3][1] + matrix[1][1] * matrix[2][4] * matrix[3][2] + matrix[1][2] * matrix[2][1] * matrix[3][4]);

        T x = x_D / base_D;
        T y = y_D / base_D;
        T z = z_D / base_D;
        // cout << "[ x:" << x << "; y:" << y << "; z:" << z << " ]" << endl;
        solution[0] = x;
        solution[1] = y;
        solution[2] = z;
    } else {
        std::cout << "【无解】";
        solution[0] = 0;
        solution[1] = 0;
        solution[2] = 0;
        //return DBL_MIN;
    }
}

Eigen::Vector3d COMMON_API R2ypr(const Eigen::Matrix3d &R);

double COMMON_API calVelocity(Matrix4ds &Trans, int mean_frame_num = 20, int frame_per_second = 10);

double COMMON_API calTranslationFromTranmat(Eigen::Matrix4d &tran_mat);

double COMMON_API calHeadingDegFromTranmat(Eigen::Matrix4d &tran_mat);

double COMMON_API calRotationDegFromTranmat(Eigen::Matrix4d &tran_mat);

#endif