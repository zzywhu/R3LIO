#include "SensorProcess/IMUProcess.h"

const bool timeComprator(PointType &x, PointType &y){ return (x.curvature < y.curvature); }

ImuProcess::ImuProcess() : b_first_frame_(true),
                           imu_need_init_(true),
                           start_timestamp_(-1),
                           undistort_en(true),
                           maxInitCount(1000)
{
    init_iter_num = 1;
    cov_acc = V3D(0.1, 0.1, 0.1);
    cov_gyr = V3D(0.1, 0.1, 0.1);
    cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    Q = Eigen::Matrix<double, 12, 12>::Zero();
    last_imu_.reset(new SensorMsgs::ImuData());

}

ImuProcess::~ImuProcess()
{}

void ImuProcess::Reset()
{
    // ROS_WARN("Reset ImuProcess");
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    imu_need_init_ = true;
    start_timestamp_ = -1;
    init_iter_num = 1;
    IMUpose.clear();
    last_imu_.reset(new SensorMsgs::ImuData());
    cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::clear()
{
    // ROS_WARN("Reset ImuProcess");
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    start_timestamp_ = -1;
    init_iter_num = 1;
    IMUpose.clear();
    last_imu_.reset(new SensorMsgs::ImuData());
    cur_pcl_un_.reset(new PointCloudXYZI());
    last_lidar_end_time_= -1;
    b_first_frame_ = true;
    time_last_scan=-1;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity **/

    V3D cur_acc, cur_gyr;

    if (b_first_frame_)
    {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
        mean_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();
    }

    for (const auto &imu : meas.imu)
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
        cur_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();

        mean_acc      += (cur_acc - mean_acc) / N;
        mean_gyr      += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

        // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

        N ++;
    }
    state_ikfom init_state = kf_state.get_x();
    init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);

    init_state.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
    init_state.pos = Zero3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
    init_state.bg  = mean_gyr;
    kf_state.change_x(init_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
    init_P.setIdentity();
    init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
    init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
    init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
    init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
    init_P(21,21) = init_P(22,22) = 0.00001;
    kf_state.change_P(init_P);
    last_imu_ = meas.imu.back();

}

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout, int &N, bool enable_bias_init)
{
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurements to unit gravity **/
    //printf("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
    V3D cur_acc, cur_gyr;

    if (b_first_frame_)
    {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
        mean_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();
    }

    for (const auto &imu : meas.imu)
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
        cur_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();

        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

        N++;
    }

    state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;
    state_inout.rot_end = Eye3d;
    if (enable_bias_init)
    {
        state_inout.bias_g = mean_gyr;
        // this bias is better than 0 but not accurate still
        state_inout.bias_a = (-mean_acc) - state_inout.gravity;
    }
    last_imu_ = meas.imu.back();
}

void ImuProcess::Process(const MeasureGroup &meas,
                         esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloudXYZI::Ptr cur_pcl_un_)
{
    if(meas.imu.empty()) {return;}
    assert(meas.lidar != nullptr);
    cur_pcl_un_->clear();

    if (imu_need_init_)
    {
        /// The very first lidar frame
        IMU_init(meas, kf_state, init_iter_num);

        last_imu_   = meas.imu.back();

        state_ikfom imu_state = kf_state.get_x();
        if (init_iter_num > maxInitCount)
        {
            cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
            imu_need_init_ = false;

            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
            printf("IMU Initialization Done: \n"
                   "Gravity: %.4f %.4f %.4f, Acc norm: %.4f Count: %d\n",
                   imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), init_iter_num);
            std::cout<<"Gyroscope noise:"<<cov_gyr.transpose()<<std::endl;
            std::cout<<"Accelerometer noise:"<<cov_acc.transpose()<<std::endl;
            std::cout << "Gyroscope bias:" << imu_state.bg.transpose() << std::endl;
            std::cout << "Accelerometer bias:" << imu_state.ba.transpose() << std::endl;

        }

        return;
    }

    undistortPcl(meas, kf_state, *cur_pcl_un_);
}

void ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_)
{
    if (imu_en)
    {
        if (meas.imu.empty()) return;
        assert(meas.lidar != nullptr);

        if (imu_need_init_)
        {
            if (!LI_init_done)
            {
                /// The very first lidar frame
                IMU_init(meas, stat, init_iter_num);
                imu_need_init_ = true;
                last_imu_ = meas.imu.back();
                if (init_iter_num > maxInitCount)
                {
                    cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
                    imu_need_init_ = false;

                    cov_acc = cov_acc_scale;
                    cov_gyr = cov_gyr_scale;

                    printf("IMU Initialization Done: Gravity: %.4f %.4f %.4f, Acc norm: %.4f\n", stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm());
                    std::cout<<"Gyroscope noise:"<<cov_gyr.transpose()<<std::endl;
                    std::cout<<"Accelerometer noise:"<<cov_acc.transpose()<<std::endl;
                    IMU_mean_acc_norm = mean_acc.norm();
                }
            }
            else
            {
                std::cout << std::endl;
//                printf(BOLDMAGENTA "[Refinement] Switch to LIO mode, online refinement begins.\n\n" RESET);
                last_imu_ = meas.imu.back();
                imu_need_init_ = false;
                cov_acc = cov_acc_scale;
                cov_gyr = cov_gyr_scale;
            }
            return;
        }
//        std::cout<<"propagation_and_undist"<<std::endl;
        propagation_and_undist(meas, stat, *cur_pcl_un_);
    }
    else
    {
//        std::cout<<"Forward_propagation_without_imu"<<std::endl;
        forward_propagation_without_imu(meas, stat, *cur_pcl_un_);
    }
}

void ImuProcess::undistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu;
    v_imu.push_front(last_imu_);
    const double &imu_beg_time = v_imu.front()->header;
    const double &imu_end_time = v_imu.back()->header;
    const double &pcl_beg_time = meas.lidar_beg_time;
    const double &pcl_end_time = meas.lidar_end_time;

    /*** sort point clouds by offset time ***/
    pcl_out = *(meas.lidar);
    sort(pcl_out.points.begin(), pcl_out.points.end(), timeComprator);
    // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

    /*** Initialize IMU pose ***/
    state_ikfom imu_state = kf_state.get_x();
    IMUpose.clear();
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

    /*** forward propagation at each imu point ***/
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;

    double dt = 0;

    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->header < last_lidar_end_time_)
            continue;

        angvel_avr<<0.5 * (head->angular_velocity.x() + tail->angular_velocity.x()),
                0.5 * (head->angular_velocity.y() + tail->angular_velocity.y()),
                0.5 * (head->angular_velocity.z() + tail->angular_velocity.z());
        acc_avr   <<0.5 * (head->linear_acceleration.x() + tail->linear_acceleration.x()),
                0.5 * (head->linear_acceleration.y() + tail->linear_acceleration.y()),
                0.5 * (head->linear_acceleration.z() + tail->linear_acceleration.z());

        // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

        acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

        if(head->header < last_lidar_end_time_)
            dt = tail->header - last_lidar_end_time_;
            // dt = tail->header.stamp.toSec() - pcl_beg_time;
        else
            dt = tail->header - head->header;

        in.acc = acc_avr;
        in.gyro = angvel_avr;
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
        kf_state.predict(dt, Q, in);
       // std::cout<<std::setprecision(32)<<std::fixed<<"dt:"<<dt<<"/acc_avr:"<<acc_avr.transpose()<<angvel_avr<<angvel_avr.transpose()<<std::endl;
        /* save the poses at each IMU measurements */
        imu_state = kf_state.get_x();

        if(_bOutpoutImuInfo)
		{
            IMUData imuData;
            imuData._timestamp=tail->header;
            imuData._position=imu_state.pos;
            imuData._orientation=imu_state.rot.matrix();
            imuData._velocity=imu_state.vel;
            imuData._accelerate=tail->linear_acceleration;
            imuData._angular_velocity=tail->angular_velocity;
            imuData._ba=imu_state.ba;
            imuData._bg=imu_state.bg;
            _imuDatas.emplace_back(imuData);
        }

        angvel_last = angvel_avr - imu_state.bg;//w-bg
        acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);//Rwi(a-ba)
        for(int i=0; i<3; i++)
        {
            acc_s_last[i] += imu_state.grav[i];//Rwi(a-ba)+g
        }
        double &&offs_t = tail->header - pcl_beg_time;
        IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);

    imu_state = kf_state.get_x();
    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    if(!undistort_en)
        return;
    if (pcl_out.points.begin() == pcl_out.points.end()) return;
    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu<<MAT_FROM_ARRAY(head->rot);
        // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
        vel_imu<<VEC_FROM_ARRAY(head->vel);//V
        pos_imu<<VEC_FROM_ARRAY(head->pos);//P
        acc_imu<<VEC_FROM_ARRAY(tail->acc);//Rwi(a-ba)+g
        angvel_avr<<VEC_FROM_ARRAY(tail->gyr);//w-bg

        for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
        {
            dt = it_pcl->curvature / double(1000) - head->offset_time;

            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
            M3D R_i(R_imu * Exp(angvel_avr, dt));//Rwi*Exp[(Rwi(a-ba)+g)*dt]

            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);//(P+v*dt+0.5(Rwi(a-ba)+g)*dt^2)-P'
            //pl=Tli*Tii'(Til*pl')  (Tii'=Riw*[Rwi'|(twi'-twi)]<=>Tiw*Twi'=[Riw|tiw]*[Rwi'|twi']=[Riw*Rwi'|Riw*twi'+tiw]=[Riw*Rwi'|Riw*twi'+Riw*twi]
            V3D P_compensate = imu_state.offset_R_L_I.conjugate() *
                               (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!

            // save Undistorted points and their rotation
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin())
                break;
        }
    }
}

void ImuProcess::forward_propagation_without_imu(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
    pcl_out = *(meas.lidar);
    /*** sort point clouds by offset time ***/
    const double &pcl_beg_time = meas.lidar_beg_time;
    sort(pcl_out.points.begin(), pcl_out.points.end(), timeComprator);
    const double &pcl_end_offset_time = pcl_out.points.back().curvature / double(1000);

    MD(DIM_STATE, DIM_STATE) F_x, cov_w;
    double dt = 0.0;

    if (b_first_frame_)
    {
        dt = 0.1;
        b_first_frame_ = false;
        time_last_scan = pcl_beg_time;
    }
    else
    {
        dt = pcl_beg_time - time_last_scan;
        time_last_scan = pcl_beg_time;
    }

    /* covariance propagation */
    F_x.setIdentity();
    cov_w.setZero();
    /** In CV model, bias_g represents angular velocity **/
    /** In CV model，bias_a represents linear acceleration **/
    M3D Exp_f = Exp(state_inout.bias_g, dt);
    F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);
    F_x.block<3, 3>(0, 15) = Eye3d * dt;
    F_x.block<3, 3>(3, 12) = Eye3d * dt;
    cov_w.block<3, 3>(15, 15).diagonal() = cov_gyr_scale * dt * dt;
    cov_w.block<3, 3>(12, 12).diagonal() = cov_acc_scale * dt * dt;

    /** Forward propagation of covariance**/
//    std::cout<<"F_x:"<<F_x<<std::endl;
//    std::cout<<"cov:"<<state_inout.cov<<std::endl;
//    std::cout<<"cov_w"<<cov_w<<std::endl;
    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
//    std::cout<<"cov:"<<state_inout.cov<<std::endl;
    /** Forward propagation of attitude **/
    state_inout.rot_end = state_inout.rot_end * Exp_f;
    /** Position Propagation **/
    state_inout.pos_end += state_inout.vel_end * dt;

    /**CV model： un-distort pcl using linear interpolation **/
    auto it_pcl = pcl_out.points.end() - 1;
    double dt_j = 0.0;
    for (; it_pcl != pcl_out.points.begin(); it_pcl--)
    {
        dt_j = pcl_end_offset_time - it_pcl->curvature / double(1000);
        M3D R_jk(Exp(state_inout.bias_g, -dt_j));
        V3D P_j(it_pcl->x, it_pcl->y, it_pcl->z);
        // Using rotation and translation to un-distort points
        V3D p_jk;
        p_jk = -state_inout.rot_end.transpose() * state_inout.vel_end * dt_j;

        V3D P_compensate = R_jk * P_j + p_jk;

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);
    }

}

void ImuProcess::propagation_and_undist(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out)
{
    /*** add the imu of the last frame-tail to the current frame-head ***/
    pcl_out = *(meas.lidar);
    auto v_imu = meas.imu;
    v_imu.push_front(last_imu_);
    double imu_end_time = v_imu.back()->header;
    double pcl_beg_time, pcl_end_time;

    pcl_beg_time = meas.lidar_beg_time;
    /*** sort point clouds by offset time ***/
    sort(pcl_out.points.begin(), pcl_out.points.end(), timeComprator);
    pcl_end_time = pcl_beg_time + pcl_out.points.back().curvature / double(1000);

    /*** Initialize IMU pose ***/
    IMUpose.clear();
    IMUpose.push_back(set_pose6d(v_imu.front()->header - pcl_beg_time, acc_s_last, angvel_last, state_inout.vel_end, state_inout.pos_end, state_inout.rot_end));

    /*** forward propagation at each imu point ***/
    V3D acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
    M3D R_imu(state_inout.rot_end);
    MD(DIM_STATE, DIM_STATE) F_x, cov_w;

    double dt = 0;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->header < last_lidar_end_time_) continue;

        angvel_avr << 0.5 * (head->angular_velocity.x() + tail->angular_velocity.x()),
                0.5 * (head->angular_velocity.y() + tail->angular_velocity.y()),
                0.5 * (head->angular_velocity.z() + tail->angular_velocity.z());


        acc_avr << 0.5 * (head->linear_acceleration.x() + tail->linear_acceleration.x()),
                   0.5 * (head->linear_acceleration.y() + tail->linear_acceleration.y()),
                   0.5 * (head->linear_acceleration.z() + tail->linear_acceleration.z());

        angvel_avr -= state_inout.bias_g;
        acc_avr = acc_avr / IMU_mean_acc_norm * G_m_s2 - state_inout.bias_a;

        if (head->header < last_lidar_end_time_)
            dt = tail->header - last_lidar_end_time_;
        else
            dt = tail->header - head->header;

        /* covariance propagation */
        M3D acc_avr_skew;
        M3D Exp_f = Exp(angvel_avr, dt);
        acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

        F_x.setIdentity();
        cov_w.setZero();

        F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
        F_x.block<3, 3>(0, 15) = -Eye3d * dt;
        F_x.block<3, 3>(3, 12) = Eye3d * dt;
        F_x.block<3, 3>(12, 0) = -R_imu * acc_avr_skew * dt;
        F_x.block<3, 3>(12, 18) = -R_imu * dt;
        F_x.block<3, 3>(12, 21) = Eye3d * dt;

        cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
        cov_w.block<3, 3>(6, 6).diagonal() = cov_R_LI * dt * dt;
        cov_w.block<3, 3>(9, 9).diagonal() = cov_T_LI * dt * dt;
        cov_w.block<3, 3>(12, 12) = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
        cov_w.block<3, 3>(15, 15).diagonal() = cov_bias_gyr * dt * dt; // bias gyro covariance
        cov_w.block<3, 3>(18, 18).diagonal() = cov_bias_acc * dt * dt; // bias acc covariance

        state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

        /* propagation of IMU attitude (global frame)*/
        R_imu = R_imu * Exp_f;

        /* Specific acceleration (global frame) of IMU */
        acc_imu = R_imu * acc_avr + state_inout.gravity;

        /* propagation of IMU position (global frame)*/
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

        /* velocity of IMU (global frame)*/
        vel_imu = vel_imu + acc_imu * dt;

        /* save the poses at each IMU measurements (global frame)*/
        angvel_last = angvel_avr;
        acc_s_last = acc_imu;
        double &&offs_t = tail->header - pcl_beg_time;
        IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;


    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;


    auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * state_inout.offset_T_L_I;
    // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

#ifdef DEBUG_PRINT
    cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<endl;
      cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<endl;
#endif

    /*** un-distort each lidar point (backward propagation) ***/
    auto it_pcl = pcl_out.points.end() - 1; //a single point in k-th frame
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        R_imu << MAT_FROM_ARRAY(head->rot);
        acc_imu << VEC_FROM_ARRAY(head->acc);
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        angvel_avr << VEC_FROM_ARRAY(head->gyr);
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
        {
            dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = t_j - t_i > 0

            /* Transform to the 'scan-end' IMU frame（I_k frame）using only rotation
            * Note: Compensation direction is INVERSE of Frame's moving direction
            * So if we want to compensate a point at timestamp-i to the frame-e
            * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */

            M3D R_i(R_imu * Exp(angvel_avr, dt));
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * state_inout.offset_T_L_I - pos_liD_e);

            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

            /// save Undistorted points and their rotation
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin())
                break;
        }
    }
}

void ImuProcess::imuMotionCompensation(PointCloudXYZI &cloud, const StatesGroup& stateEnd)
{
    auto tLidarEnd = stateEnd.pos_end + stateEnd.rot_end * stateEnd.offset_T_L_I;
    V3D acc_imu, angvel_avr, acc_avr, vel_imu, pos_imu;
    M3D R_imu;
    /*** un-distort each lidar point (backward propagation) ***/
    auto it_pcl = cloud.points.end() - 1; //a single point in k-th frame
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        R_imu << MAT_FROM_ARRAY(head->rot);
        acc_imu << VEC_FROM_ARRAY(head->acc);
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        angvel_avr << VEC_FROM_ARRAY(head->gyr);
        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
        {
            double dt = it_pcl->curvature / double(1000) - head->offset_time; //dt = t_j - t_i > 0
            /* Transform to the 'scan-end' IMU frame（I_k frame）using only rotation
            * Note: Compensation direction is INVERSE of Frame's moving direction
            * So if we want to compensate a point at timestamp-i to the frame-e
            * pCompensate = R_imu_e ^ T * (Ri * pi + tei) where tei is represented in global frame */

            M3D Ri(R_imu * Exp(angvel_avr, dt));
            V3D tei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + Ri * stateEnd.offset_T_L_I - tLidarEnd);

            V3D pi(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D pCompensate = stateEnd.rot_end.transpose() * (Ri * pi + tei);

            /// save Undistorted points and their rotation
            it_pcl->x = pCompensate(0);
            it_pcl->y = pCompensate(1);
            it_pcl->z = pCompensate(2);

            if (it_pcl == cloud.points.begin())
                break;
        }
        if (it_pcl == cloud.points.begin())
            break;
    }
}

