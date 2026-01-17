#ifndef INCLUDE_ZG_TRANSFER_H
#define INCLUDE_ZG_TRANSFER_H
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <string>

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define ZG_TRANSFER_API __declspec(dllexport)
#else
#define ZG_TRANSFER_API __declspec(dllimport)
#endif
#else
#define ZG_TRANSFER_API
#endif
using namespace std;
//This class is used to calculate the position and attitude of the lidar to the IMU
class ZG_TRANSFER_API ZGAxisTransfer {
public:
    ZGAxisTransfer() {
        m_shift_angle_Z = 20.0;
        //--------------------NO.1---------------------//
        m_rotation_center(0) = -0.12591;
        m_rotation_center(1) = 0.03319;
        m_rotation_center(2) = 0.12548 - 0.0094 / sin(GetRad(90 - m_shift_angle_Z));
        m_rotation_line(0) = -sin(GetRad(90 - m_shift_angle_Z));
        m_rotation_line(1) = 0.0;
        m_rotation_line(2) = cos(GetRad(90 - m_shift_angle_Z));

        //--------------------NO.2---------------------//
        m_benchmark_angle = 303;
        m_lidar_coord = m_rotation_center;
        m_lidar_coord(2) = 0.12548;

        m_lidar_X = Eigen::Vector3d(0, -1, 0);

        m_lidar_Z(0) = sin(GetRad(m_shift_angle_Z));
        m_lidar_Z(1) = 0.0;
        m_lidar_Z(2) = cos(GetRad(m_shift_angle_Z));

        m_lidar_Y = -m_rotation_line;

        m_lidar_2_rotation_center.setZero();
        m_lidar_2_rotation_center(2) = 0.0084 / sin(GetRad(90 - m_shift_angle_Z));

        m_imu_X = Eigen::Vector3d(1, 0, 0);
        m_imu_Y = Eigen::Vector3d(0, 1, 0);
        m_imu_Z = Eigen::Vector3d(0, 0, 1);

        m_is_use_rpyxyz_error_param = false;
    }

    ~ZGAxisTransfer() {
    }

public:
    //Read device parameters
    bool ReadEquipmentParams(const char *path) {
        FILE *fp = fopen(path, "r+");
        if (!fp) {
            printf("error cannot params file: %s\n", path);
            return false;
        }
        double benchmarkA, shift_Z_A, D, X, Y, Z; //
        fscanf(fp, "%lf %lf %lf %lf %lf %lf", &benchmarkA, &shift_Z_A, &D, &X, &Y, &Z);
        m_shift_angle_Z = shift_Z_A;
        m_benchmark_angle = benchmarkA;

        //--------------------NO.1---------------------//
        m_rotation_center(0) = X;
        m_rotation_center(1) = Y;
        m_rotation_center(2) = Z - D / sin(GetRad(90 - m_shift_angle_Z));

        m_rotation_line(0) = -sin(GetRad(90 - m_shift_angle_Z));
        m_rotation_line(1) = 0.0;
        m_rotation_line(2) = cos(GetRad(90 - m_shift_angle_Z));

        //--------------------NO.2---------------------//
        m_lidar_coord = m_rotation_center;
        m_lidar_coord(2) = Z;

        m_lidar_X = Eigen::Vector3d(0, -1, 0);

        m_lidar_Z(0) = sin(GetRad(m_shift_angle_Z));
        m_lidar_Z(1) = 0.0;
        m_lidar_Z(2) = cos(GetRad(m_shift_angle_Z));

        m_lidar_Y = -m_rotation_line;

        m_lidar_2_rotation_center.setZero();
        m_lidar_2_rotation_center(2) = D / sin(GetRad(90 - m_shift_angle_Z));

        fclose(fp);
        return true;
    }

    bool readZGExtrinsic(const std::string &paramPath) {
        FILE *fp = fopen(paramPath.c_str(), "r+");
        if (!fp) {
            printf("Failed opening param file: %s\n", paramPath.c_str());
            return false;
        }
        fscanf(fp, "%lf %lf %lf %lf %lf %lf",
               &motoraxis(0), &motoraxis(1), &motoraxis(2),
               &trans(0), &trans(1), &trans(2));

        _RLI_calib.matrix() << 0, 0.939693, 0.34202, 0.122378,
            -1, 0, 0, 0.033,
            0, -0.34202, 0.939693, 0.12356,
            0, 0, 0, 1;
        Eigen::Matrix3d rotMatrix;
        Eigen::Vector3d vectorBefore(0, 1, 0);
        rotMatrix = Eigen::Quaterniond::FromTwoVectors(vectorBefore, motoraxis).toRotationMatrix();
        _fixPart_new = Eigen::Affine3d::Identity();
        _fixPart_new.rotate(rotMatrix.inverse());
        _fixPart_new.translation() << trans(0, 0), 0, trans(2, 0);
        fclose(fp);
        return true;
    }

    //Read error curve parameters
    bool ReadRPYXYZErrorParams(const char *path) {
        FILE *fp = fopen(path, "r+");
        if (!fp) {
            return false;
        }
        int param_num, param_size;
        fscanf(fp, "%d %d", &param_num, &param_size);
        m_rpyxyz_error_params.resize(param_num);
        double A;
        for (int i = 0; i < param_num; i++) {
            m_rpyxyz_error_params[i].resize(param_size);
            for (int j = 0; j < param_size; j++) {
                fscanf(fp, " %lf", &A);
                m_rpyxyz_error_params[i](j) = A;
            }
        }
        m_is_use_rpyxyz_error_param = true;
        fclose(fp);
        return true;
    }

    //Set reference angle
    void SetBenchmarkAngle(const double angle) {
        m_benchmark_angle = angle;
    }

    //Set error parameters
    void SetErrorParams(const vector<Eigen::VectorXd> &params) {
        m_rpyxyz_error_params = params;
        m_is_use_rpyxyz_error_param = true;
    }

    //Calculate posture
    void CalculatePose(const double &angle, Eigen::Matrix3d &R, Eigen::Vector3d &T) {
        //1 calculate error
        double error[6] = {0};
        if (m_is_use_rpyxyz_error_param) {
            int offset = angle >= 180 ? 1 : 0;
            for (int i = 0; i < 6; i++) {
                error[i] = CalcError(m_rpyxyz_error_params[2 * i + offset], angle);
                if (i < 3) {
                    error[i] = GetRad(error[i]);
                }
            }
        }
        double angle_gap = angle - m_benchmark_angle;
        if (angle_gap < 0) {
            angle_gap += 360.0;
        }
        double angle_gap_rad = GetRad(angle_gap);

        Eigen::Vector3d lidar_coord = RodriguesRotation(angle_gap_rad, m_rotation_line, m_lidar_2_rotation_center) + m_rotation_center;
        Eigen::Vector3d lidar_X = RodriguesRotation(angle_gap_rad, m_rotation_line, m_lidar_X);
        Eigen::Vector3d lidar_Y = m_lidar_Y;
        Eigen::Vector3d lidar_Z = RodriguesRotation(angle_gap_rad, m_rotation_line, m_lidar_Z);
        lidar_X.normalize();
        lidar_Y.normalize();
        lidar_Z.normalize();

        T = lidar_coord + Eigen::Vector3d(error[3], error[4], error[5]);
        R.block<3, 1>(0, 0) = lidar_X;
        R.block<3, 1>(0, 1) = lidar_Y;
        R.block<3, 1>(0, 2) = lidar_Z;

        Eigen::AngleAxisd roll, pitch, yaw;
        roll = Eigen::AngleAxisd(error[0], m_imu_X);
        pitch = Eigen::AngleAxisd(error[1], m_imu_Y);
        yaw = Eigen::AngleAxisd(error[2], m_imu_Z);
        Eigen::Matrix3d errorR = (yaw * pitch * roll).toRotationMatrix();
        R = errorR * R;
    }

    void getLid2IMUTrans(const double &motorAngle, Eigen::Matrix3d &Ril, Eigen::Vector3d &til) {
        Eigen::Affine3d Til = _fixPart * Eigen::AngleAxisd(-motorAngle, Vector3d::UnitY()) * Eigen::Translation3d(0, 0, -_dzLid2MotAxis);
        Ril = Til.linear();
        til = Til.translation();
    }

    void getLid2IMUTransRefine(const double &motorAngle, Eigen::Matrix3d &Ril, Eigen::Vector3d &til) {
        Eigen::Affine3d Til = _RLI_calib * _fixPart_new.inverse() * Eigen::AngleAxisd(-motorAngle, Eigen::Vector3d::UnitY()) * _fixPart_new;

        Ril = Til.linear();
        til = Til.translation();
    }

    void getLid2IMUTransRefineforSIM(const double &motorAngle, Eigen::Matrix3d &Ril, Eigen::Vector3d &til) {
        _RLI_calib.matrix() << 0.8658698, 0.0000000, -0.5002694, -0.089,
            0.0000000, 1.0000000, 0.0000000, -0.000,
            0.5002694, 0.0000000, 0.8658698, 0.288,
            0, 0, 0, 1;
        Eigen::Affine3d Til = _RLI_calib * _fixPart_new.inverse() * Eigen::AngleAxisd(motorAngle, Eigen::Vector3d::UnitX()) * _fixPart_new;

        Ril = Til.linear();
        til = Til.translation();

        // std::cout << "Ril: \n"
        //           << Ril << std::endl;
        // std::cout << "til: \n"
        //           << til.transpose() << std::endl;
    }

private:
    //Rodri's rotation formula
    Eigen::Vector3d RodriguesRotation(double angle, Eigen::Vector3d rotation_vec, Eigen::Vector3d &vec) {
        return cos(angle) * vec + (1 - cos(angle)) * (vec.dot(rotation_vec)) * rotation_vec + sin(angle) * rotation_vec.cross(vec);
    }

    //Angle to radian
    double GetRad(double angle) {
        return angle * 3.14159265358 / 180;
    }

    //Calculate error
    double CalcError(Eigen::VectorXd param, double A) {
        double E = param(0) * pow(A, 7) + param(1) * pow(A, 6) + param(2) * pow(A, 5) + param(3) * pow(A, 4) + param(4) * pow(A, 3) + param(5) * pow(A, 2) + param(6) * pow(A, 1) + param(7) + param(8) * sin(param(9) * A + param(10));
        return E;
    }

private:
    double m_shift_angle_Z; //Offset angle between Lidar Z axis and IMU Z axis

    Eigen::Vector3d m_rotation_center; //rotation center
    Eigen::Vector3d m_rotation_line;   //rotation axis

    double m_benchmark_angle;                  //benchmark angle
    Eigen::Vector3d m_lidar_coord;             //Lidar origin
    Eigen::Vector3d m_lidar_X;                 //LiDAR X axis
    Eigen::Vector3d m_lidar_Y;                 //LiDAR Y axis
    Eigen::Vector3d m_lidar_Z;                 //LiDAR Z axis
    Eigen::Vector3d m_lidar_2_rotation_center; //Lidar origin to center of rotation vector

    Eigen::Vector3d m_imu_X;
    Eigen::Vector3d m_imu_Y;
    Eigen::Vector3d m_imu_Z;
    double _dzLid2MotAxis;
    double _angleLid2IMU;
    double _dxLid2IMU;
    double _dyLid2IMU;
    Eigen::Vector3d motoraxis;
    Eigen::Vector3d trans;
    double _dzLid2IMU;
    double _dzOffset;
    double _dxOffset;
    int _deviceType;
    Eigen::Affine3d _fixPart;
    Eigen::Affine3d _fixPart_new;
    Eigen::Affine3d _RLI_calib;
    bool m_is_use_rpyxyz_error_param;              //Whether to use error parameters
    vector<Eigen::VectorXd> m_rpyxyz_error_params; //error parameters
};

class ZG_TRANSFER_API Interpolation {
public:
    enum InterpolationMethon {
        USE_INITIAL_BENCHMARK_ZERO_DEGREE = 0,
        USE_OPTIMISE_TWELVE_POSTURE_RT,
        USE_OPTIMISE_TWELVE_POSTURE_Q
    };

public:
    Interpolation(InterpolationMethon methon) {
        methon_ = methon;
        spacing_angle_ = 0.0;
        m_is_use_rpyxyz_error_param = false;
    }

    ~Interpolation() {
    }

public:
    bool ReadEquipmentParams(const char *file_path) {
        ifstream read((string)file_path, ios::in);
        if (!read.is_open()) {
            printf("could not open file！\n");
            return false;
        }

        if (methon_ == USE_OPTIMISE_TWELVE_POSTURE_RT) {
            int num = 0;
            read >> num;
            if (num == 0) return false;
            spacing_angle_ = 360 / num;

            Rs_.resize(num);
            ts_.resize(num);

            for (int i = 0; i < num; i++) {
                for (int j = 0; j < 9; j++) {
                    read >> Rs_[i](j);
                }
                for (int j = 0; j < 3; j++) {
                    read >> ts_[i](j);
                }
            }
            printf("2\n");
        }
        if (methon_ == USE_INITIAL_BENCHMARK_ZERO_DEGREE) {
            int row = 0;
            while (!read.eof()) {
                if (row == 4) break;

                double a, b, c;
                read >> a >> b >> c;
                if (row == 0) {
                    T_(0) = a;
                    T_(1) = b;
                    T_(2) = c;
                }
                if (row != 0) {
                    R_(row - 1, 0) = a;
                    R_(row - 1, 1) = b;
                    R_(row - 1, 2) = c;
                    //R((row - 1) * 3 + 0) = a;
                    //R((row - 1) * 3 + 1) = b;
                    //R((row - 1) * 3 + 2) = c;
                }
                row++;
            }
        }
        if (methon_ == USE_OPTIMISE_TWELVE_POSTURE_Q) {
            for (int i = 0; i < 12; i++) {
                Eigen::Quaterniond q;
                Eigen::Vector3d t;
                read >> q.w() >> q.x() >> q.y() >> q.z() >> t.x() >> t.y() >> t.z();
                qs_.push_back(q);
                ts_.push_back(t);
            }
        }
        read.close();
        printf("read file success!\n");
        return true;
    }

    bool ReadRPYXYZErrorParams(const char *path) {
        FILE *fp = fopen(path, "r+");
        if (!fp) {
            return false;
        }
        int param_num, param_size;
        fscanf(fp, "%d %d", &param_num, &param_size);
        m_rpyxyz_error_params.resize(param_num);
        double A;
        for (int i = 0; i < param_num; i++) {
            m_rpyxyz_error_params[i].resize(param_size);
            for (int j = 0; j < param_size; j++) {
                fscanf(fp, " %lf", &A);
                m_rpyxyz_error_params[i](j) = A;
            }
        }
        m_is_use_rpyxyz_error_param = true;
        printf("1\n");
        fclose(fp);
        return true;
    }

    inline void SetErrorParams(const vector<Eigen::VectorXd> &params) {
        m_rpyxyz_error_params = params;
        m_is_use_rpyxyz_error_param = true;
    }

    //2. angle->[0,360]
    bool CalculatePose(const double angle, Eigen::Matrix3d &R, Eigen::Vector3d &T) {
        if (angle > 360 || angle < 0) return false;

        if (methon_ == USE_OPTIMISE_TWELVE_POSTURE_RT) {
            T = ts_[0];
            int start = (int)angle / 30;
            int end = start + 1;
            if (start == 11) {
                end = 0;
            }
            double t = (angle - spacing_angle_ * start) / spacing_angle_;

            Eigen::Quaterniond qstrat(Rs_[start]);
            qstrat.normalize();
            Eigen::Quaterniond qend(Rs_[end]);
            qend.normalize();
            R = qstrat.slerp(t, qend).toRotationMatrix();

            /*
      double error[6] = { 0 };
      if (m_is_use_rpyxyz_error_param)
      {
        int offset = angle >= 180 ? 1 : 0;
        for (int i = 0; i < 6; i++)
        {
          error[i] = CalcError(m_rpyxyz_error_params[2 * i + offset], angle);
          if (i < 3)
          {
            error[i] = GetRad(error[i]);
          }
        }
      }

      //T = T + Eigen::Vector3d(error[3], error[4], error[5]);

      Eigen::AngleAxisd roll, pitch, yaw;
      roll = Eigen::AngleAxisd(error[0], Eigen::Vector3d::UnitX());
      pitch = Eigen::AngleAxisd(error[1], Eigen::Vector3d::UnitY());
      yaw = Eigen::AngleAxisd(error[2], Eigen::Vector3d::UnitZ());
      Eigen::Matrix3d errorR = (yaw * pitch * roll).toRotationMatrix();
      R = errorR * R;
      */
        }
        if (methon_ == USE_INITIAL_BENCHMARK_ZERO_DEGREE) {
            T = T_;
            Eigen::Vector3d vec(0, 1, 0);
            double t = angle * 3.14159265358 / 180;

            R = R_ * Eigen::AngleAxisd(t, vec).toRotationMatrix().inverse();
            double error[6] = {0};
            if (m_is_use_rpyxyz_error_param) {
                int offset = angle >= 180 ? 1 : 0;
                for (int i = 0; i < 6; i++) {
                    error[i] = CalcError(m_rpyxyz_error_params[2 * i + offset], angle);
                    if (i < 3) {
                        error[i] = GetRad(error[i]);
                    }
                }
            }

            T = T + Eigen::Vector3d(error[3], error[4], error[5]);

            Eigen::AngleAxisd roll, pitch, yaw;
            roll = Eigen::AngleAxisd(error[0], Eigen::Vector3d::UnitX());
            pitch = Eigen::AngleAxisd(error[1], Eigen::Vector3d::UnitY());
            yaw = Eigen::AngleAxisd(error[2], Eigen::Vector3d::UnitZ());
            Eigen::Matrix3d errorR = (yaw * pitch * roll).toRotationMatrix();
            R = errorR * R;
        }
        if (methon_ == USE_OPTIMISE_TWELVE_POSTURE_Q) {
            T = ts_[0];

            int start = angle / spacing_angle_;
            int end = start + 1;
            if (start == 11) {
                end = 0;
            }

            double t = (angle - spacing_angle_ * start) / spacing_angle_;
            if (t > 1) t = 1;

            R = qs_[start].slerp(t, qs_[end]).toRotationMatrix();
            double error[6] = {0};
            if (m_is_use_rpyxyz_error_param) {
                int offset = angle >= 180 ? 1 : 0;
                for (int i = 0; i < 6; i++) {
                    error[i] = CalcError(m_rpyxyz_error_params[2 * i + offset], angle);
                    if (i < 3) {
                        error[i] = GetRad(error[i]);
                    }
                }
            }

            T = T + Eigen::Vector3d(error[3], error[4], error[5]);

            Eigen::AngleAxisd roll, pitch, yaw;
            roll = Eigen::AngleAxisd(error[0], Eigen::Vector3d::UnitX());
            pitch = Eigen::AngleAxisd(error[1], Eigen::Vector3d::UnitY());
            yaw = Eigen::AngleAxisd(error[2], Eigen::Vector3d::UnitZ());
            Eigen::Matrix3d errorR = (yaw * pitch * roll).toRotationMatrix();
            R = errorR * R;
        }
        return true;
    }

private:
    //Rodri's rotation formula
    inline Eigen::Vector3d RodriguesRotation(double angle, Eigen::Vector3d rotation_vec, Eigen::Vector3d &vec) {
        return cos(angle) * vec + (1 - cos(angle)) * (vec.dot(rotation_vec)) * rotation_vec + sin(angle) * rotation_vec.cross(vec);
    }

    //Angle to radian
    inline double GetRad(double angle) {
        return angle * 3.14159265358 / 180;
    }

    //Calculate error
    inline double CalcError(Eigen::VectorXd param, double A) {
        double E = param(0) * pow(A, 7) + param(1) * pow(A, 6) + param(2) * pow(A, 5) + param(3) * pow(A, 4) + param(4) * pow(A, 3) + param(5) * pow(A, 2) + param(6) * pow(A, 1) + param(7) + param(8) * sin(param(9) * A + param(10));
        return E;
    }

private:
    InterpolationMethon methon_;
    double spacing_angle_;

    vector<Eigen::Matrix3d> Rs_;
    vector<Eigen::Vector3d> ts_;

    Eigen::Matrix3d R_;
    Eigen::Vector3d T_;

    vector<Eigen::Quaterniond> qs_;

    bool m_is_use_rpyxyz_error_param;              //Whether to use error parameters
    vector<Eigen::VectorXd> m_rpyxyz_error_params; //error parameters
};

#endif