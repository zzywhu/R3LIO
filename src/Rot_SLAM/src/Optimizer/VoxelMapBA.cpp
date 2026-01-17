//
// Created by w on 2022/9/15.
//

#include "Optimizer/VoxelMapBA.h"

LM_SLWD_VOXEL::LM_SLWD_VOXEL(int ss,
                             int fn,
                             int thnum) : slwd_size(ss),
                                          filternum(fn),
                                          thd_num(thnum)
{
    so3_poses.resize(ss);
    t_poses.resize(ss);
    so3_poses_temp.resize(ss);
    t_poses_temp.resize(ss);
    jac_leng = 6 * ss;
    corn_less = 0.1;
    map_refine_flag = 0;
    is_verbose = false;
}

// Used by "push_voxel"
void LM_SLWD_VOXEL::downsample(PvList &plvec,
                               int cur_frame,
                               std::vector<Eigen::Vector3d> &plvec_voxel,
                               std::vector<int> &slwd_num,
                               int filternum2use)
{
    uint plsize = plvec.size();
    if (plsize <= (uint) filternum2use)
    {
        for (uint i = 0; i < plsize; i++)
        {
            plvec_voxel.push_back(plvec[i].pl);
            slwd_num.push_back(cur_frame);
        }
        return;
    }

    Eigen::Vector3d center;
    double part = 1.0 * plsize / filternum2use;

    for (int i = 0; i < filternum2use; i++)
    {
        uint np = part * i;
        uint nn = part * (i + 1);
        center.setZero();
        for (uint j = np; j < nn; j++)
            center += plvec[j].pl;
        center = center / (nn - np);
        plvec_voxel.push_back(center);
        slwd_num.push_back(cur_frame);
    }
}

// Push voxel into _optimizer
void LM_SLWD_VOXEL::push_voxel(std::vector<PvList *> &plvec, SigVecClass &sig_vec, int lam_type)
{
    int process_points_size = 0;
    for (int i = 0; i < slwd_size; i++)
        if (!plvec[i]->empty())
            process_points_size++;

    // Only one scan
    if (process_points_size <= 1)
        return;

    int filternum2use = filternum;
    if (filternum * process_points_size < MIN_PS)
        filternum2use = MIN_PS / process_points_size + 1;

    std::vector<Eigen::Vector3d> *plvec_voxel = new std::vector<Eigen::Vector3d>();
    // Frame num in sliding window for each point in "plvec_voxel"
    std::vector<int> *slwd_num = new std::vector<int>();
    plvec_voxel->reserve(filternum2use * slwd_size);
    slwd_num->reserve(filternum2use * slwd_size);

    // retain one point for one scan (you can modify)
    for (int i = 0; i < slwd_size; i++)
        if (!plvec[i]->empty())
            downsample(*plvec[i], i, *plvec_voxel, *slwd_num, filternum2use);

//    for(int i=0; i<slwd_size; i++)
//    {
//        for(uint j=0; j<plvec[i]->size(); j++)
//        {
//            plvec_voxel->push_back(plvec[i]->at(j).point);
//            slwd_num->push_back(i);
////            Eigen::Vector3d vec_tran = so3_poses[i].matrix() * plvec[i]->at(j).point+t_poses[i];
//        }
//    }

    plvec_voxels.push_back(plvec_voxel); // Push a voxel into _optimizer
    slwd_nums.push_back(slwd_num);
    lam_types.push_back(lam_type);
    sig_vecs.push_back(sig_vec); // history points out of sliding window
}

// Calculate Hessian, Jacobian, residual
void LM_SLWD_VOXEL::acc_t_evaluate(std::vector<SO3> &so3_ps, std::vector<Eigen::Vector3d> &t_ps,
                                   int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
{
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    Eigen::MatrixXd _hess(Hess);
    Eigen::MatrixXd _jact(JacT);

    // In program, lambda_0 < lambda_1 < lambda_2
    // For plane, the residual is lambda_0
    // For line, the residual is lambda_0+lambda_1
    // We only calculate lambda_1 here
    for (int a = head; a < end; a++)
    {
        uint k = lam_types[a]; // 0 is surf, 1 is line
        SigVecClass &sig_vec = sig_vecs[a];
        std::vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels[a];
        // Position in slidingwindow for each point in "plvec_voxel"
        std::vector<int> &slwd_num = *slwd_nums[a];
        uint backnum = plvec_voxel.size();

        Eigen::Vector3d vec_tran;
        std::vector<Eigen::Vector3d> plvec_back(backnum);
        // derivative point to T (R, t)
        std::vector<Eigen::Matrix3d> point_xis(backnum);
        Eigen::Vector3d centor(Eigen::Vector3d::Zero());
        Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

        for (uint i = 0; i < backnum; i++)
        {
            vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i];
            // left multiplication instead of right muliplication in paper
            point_xis[i] = -SO3::hat(vec_tran);
            plvec_back[i] = vec_tran + t_ps[slwd_num[i]]; // after trans

            centor += plvec_back[i];
            covMat += plvec_back[i] * plvec_back[i].transpose();
        }

        double N_points = backnum + sig_vec._sigmaSize;
        centor += sig_vec._sigmaCenter;
        covMat += sig_vec._sigmaCov;

        covMat = covMat - centor * centor.transpose() / N_points;
        covMat = covMat / N_points;
        centor = centor / N_points;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        Eigen::Vector3d eigen_value = saes.eigenvalues();

        Eigen::Matrix3d U = saes.eigenvectors();
        Eigen::Vector3d u[3]; // eigenvectors
        for (int j = 0; j < 3; j++)
            u[j] = U.block<3, 1>(0, j);

        // Jacobian matrix
        Eigen::Matrix3d ukukT = u[k] * u[k].transpose();
        Eigen::Vector3d vec_Jt;
        for (uint i = 0; i < backnum; i++)
        {
            plvec_back[i] = plvec_back[i] - centor;
            vec_Jt = 2.0 / N_points * ukukT * plvec_back[i];
            _jact.block<3, 1>(6 * slwd_num[i] + 3, 0) += vec_Jt;
            _jact.block<3, 1>(6 * slwd_num[i], 0) -= point_xis[i] * vec_Jt;
        }

        // Hessian matrix
        Eigen::Matrix3d Hessian33;
        Eigen::Matrix3d C_k;
        std::vector<Eigen::Matrix3d> C_k_np(3);
        for (uint i = 0; i < 3; i++)
        {
            if (i == k)
            {
                C_k_np[i].setZero();
                continue;
            }
            Hessian33 = u[i] * u[k].transpose();
            // part of F matrix in paper
            C_k_np[i] = -1.0 / N_points / (eigen_value[i] - eigen_value[k]) * (Hessian33 + Hessian33.transpose());
        }

        Eigen::Matrix3d h33;
        uint rownum, colnum;
        for (uint j = 0; j < backnum; j++)
        {
            for (int f = 0; f < 3; f++)
            {
                C_k.block<1, 3>(f, 0) = plvec_back[j].transpose() * C_k_np[f];
            }
            C_k = U * C_k;
            colnum = 6 * slwd_num[j];
            // block matrix operation, half Hessian matrix
            for (uint i = j; i < backnum; i++)
            {
                Hessian33 = u[k] * (plvec_back[i]).transpose() * C_k + u[k].dot(plvec_back[i]) * C_k;

                rownum = 6 * slwd_num[i];
                if (i == j)
                    Hessian33 += (N_points - 1) / N_points * ukukT;
                else
                    Hessian33 -= 1.0 / N_points * ukukT;
                Hessian33 = 2.0 / N_points * Hessian33; // Hessian matrix of lambda and point

                // Hessian matrix of lambda and pose
                if (rownum == colnum && i != j)
                {
                    _hess.block<3, 3>(rownum + 3, colnum + 3) += Hessian33 + Hessian33.transpose();

                    h33 = -point_xis[i] * Hessian33;
                    _hess.block<3, 3>(rownum, colnum + 3) += h33;
                    _hess.block<3, 3>(rownum + 3, colnum) += h33.transpose();
                    h33 = Hessian33 * point_xis[j];
                    _hess.block<3, 3>(rownum + 3, colnum) += h33;
                    _hess.block<3, 3>(rownum, colnum + 3) += h33.transpose();
                    h33 = -point_xis[i] * h33;
                    _hess.block<3, 3>(rownum, colnum) += h33 + h33.transpose();
                }
                else
                {
                    _hess.block<3, 3>(rownum + 3, colnum + 3) += Hessian33;
                    h33 = Hessian33 * point_xis[j];
                    _hess.block<3, 3>(rownum + 3, colnum) += h33;
                    _hess.block<3, 3>(rownum, colnum + 3) -= point_xis[i] * Hessian33;
                    _hess.block<3, 3>(rownum, colnum) -= point_xis[i] * h33;
                }
            }
        }

        if (k == 1)
        {
            // add weight for line feature
            residual += corn_less * eigen_value[k];
            Hess += corn_less * _hess;
            JacT += corn_less * _jact;
        }
        else
        {
            residual += eigen_value[k];
            Hess += _hess;
            JacT += _jact;
        }
        _hess.setZero();
        _jact.setZero();
    }

    // Hessian is symmetric, copy to save time
    for (int j = 0; j < jac_leng; j += 6)
        for (int i = j + 6; i < jac_leng; i += 6)
            Hess.block<6, 6>(j, i) = Hess.block<6, 6>(i, j).transpose();
}

// Multithread for "acc_t_evaluate"
void LM_SLWD_VOXEL::divide_thread(std::vector<SO3> &so3_ps, std::vector<Eigen::Vector3d> &t_ps, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
{
    Hess.setZero();
    JacT.setZero();
    residual = 0;

    std::vector<Eigen::MatrixXd> hessians(thd_num, Hess);
    std::vector<Eigen::VectorXd> jacobians(thd_num, JacT);
    std::vector<double> resis(thd_num, 0);

    uint gps_size = plvec_voxels.size();
    if (gps_size < (uint) thd_num)
    {
        acc_t_evaluate(so3_ps, t_ps, 0, gps_size, Hess, JacT, residual);
        Hess = hessians[0];
        JacT = jacobians[0];
        residual = resis[0];
        return;
    }

    std::vector<std::thread *> mthreads(thd_num);

    double part = 1.0 * (gps_size) / thd_num;
    for (int i = 0; i < thd_num; i++)
    {
        int np = part * i;
        int nn = part * (i + 1);

        mthreads[i] = new std::thread(&LM_SLWD_VOXEL::acc_t_evaluate, this, std::ref(so3_ps), std::ref(t_ps), np, nn, std::ref(hessians[i]), std::ref(jacobians[i]), std::ref(resis[i]));
    }

    for (int i = 0; i < thd_num; i++)
    {
        mthreads[i]->join();
        Hess += hessians[i];
        JacT += jacobians[i];
        residual += resis[i];
        delete mthreads[i];
    }
}

// Calculate residual
void LM_SLWD_VOXEL::evaluate_only_residual(std::vector<SO3> &so3_ps, std::vector<Eigen::Vector3d> &t_ps, double &residual)
{
    residual = 0;
    uint gps_size = plvec_voxels.size();
    Eigen::Vector3d vec_tran;

    for (uint a = 0; a < gps_size; a++)
    {
        uint k = lam_types[a];
        SigVecClass &sig_vec = sig_vecs[a];
        std::vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels[a];
        std::vector<int> &slwd_num = *slwd_nums[a];
        uint backnum = plvec_voxel.size();

        Eigen::Vector3d centor(Eigen::Vector3d::Zero());
        Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

        for (uint i = 0; i < backnum; i++)
        {
            vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i] + t_ps[slwd_num[i]];
            centor += vec_tran;
            covMat += vec_tran * vec_tran.transpose();
        }

        double N_points = backnum + sig_vec._sigmaSize;
        centor += sig_vec._sigmaCenter;
        covMat += sig_vec._sigmaCov;

        covMat = covMat - centor * centor.transpose() / N_points;
        covMat = covMat / N_points;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        Eigen::Vector3d eigen_value = saes.eigenvalues();

        if (k == 1)
            residual += corn_less * eigen_value[k];
        else
            residual += eigen_value[k];
    }
}

// LM process
void LM_SLWD_VOXEL::dampingIter()
{
    my_mutex.lock();
    map_refine_flag = 1;
    my_mutex.unlock();

    if (plvec_voxels.size() != slwd_nums.size() || plvec_voxels.size() != lam_types.size() || plvec_voxels.size() != sig_vecs.size())
    {
        printf("size is not equal\n");
        exit(0);
    }

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);

    Eigen::MatrixXd Hess2(jac_leng, jac_leng);
    Eigen::VectorXd JacT2(jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;

    Eigen::MatrixXd matA(jac_leng, jac_leng);
    Eigen::VectorXd matB(jac_leng);
    Eigen::VectorXd matX(jac_leng);
    matA.setZero();
    matB.setZero();
    for (int i = 0; i < iter_max; i++)
    {
        if (is_calc_hess)
        {// calculate Hessian, Jacobian, residual
            divide_thread(so3_poses, t_poses, Hess, JacT, residual1);
        }

        D = Hess.diagonal().asDiagonal();
        Hess2 = Hess + u * D;

        for (int j = 0; j < jac_leng; j++)
        {
            matB(j, 0) = -JacT(j, 0);
            for (int f = 0; f < jac_leng; f++)
                matA(j, f) = Hess2(j, f);
        }
        dxi = matA.colPivHouseholderQr().solve(matB);

//    cv::Mat matA(jac_leng, jac_leng, CV_64F, cv::Scalar::all(0));
//    cv::Mat matB(jac_leng, 1, CV_64F, cv::Scalar::all(0));
//    cv::Mat matX(jac_leng, 1, CV_64F, cv::Scalar::all(0));
//
//    for(int i=0; i<iter_max; i++)
//    {
//        if(is_calc_hess)
//        {
//            // calculate Hessian, Jacobian, residual
//            divide_thread(so3_poses, t_poses, Hess, JacT, residual1);
//        }
//
//        D = Hess.diagonal().asDiagonal();
//        Hess2 = Hess + u*D;
//
//        for(int j=0; j<jac_leng; j++)
//        {
//            matB.at<double>(j, 0) = -JacT(j, 0);
//            for(int f=0; f<jac_leng; f++)
//            {
//                matA.at<double>(j, f) = Hess2(j, f);
//            }
//        }
//        cv::solve(matA, matB, matX, cv::DECOMP_QR);
//        for(int j=0; j<jac_leng; j++)
//            dxi(j, 0) = matX.at<double>(j, 0);

        for (int j = 0; j < slwd_size; j++)
        {
            // left multiplication
            so3_poses_temp[j] = SO3::exp(dxi.block<3, 1>(6 * (j), 0)) * so3_poses[j];
            t_poses_temp[j] = t_poses[j] + dxi.block<3, 1>(6 * (j) + 3, 0);
        }

        // LM
        double q1 = 0.5 * (dxi.transpose() * (u * D * dxi - JacT))[0];
        // double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
        evaluate_only_residual(so3_poses_temp, t_poses_temp, residual2);

        q = (residual1 - residual2);
        if (is_verbose)
            printf("residual%d: %lf/%lf u: %lf v: %lf q: %lf %lf %lf\n", i, residual1, residual2, u, v, q / q1, q1, q);

        if (q > 0)
        {
            so3_poses = so3_poses_temp;
            t_poses = t_poses_temp;
            q = q / q1;
            v = 2;
            q = 1 - pow(2 * q - 1, 3);
            u *= (q < 1. / 3. ? 1. / 3. : q);
            is_calc_hess = true;
        }
        else
        {
            u = u * v;
            v = 2 * v;
            is_calc_hess = false;
        }

        if (fabs(residual1 - residual2) < 1e-9)
            break;
    }

    my_mutex.lock();
    map_refine_flag = 2;
    my_mutex.unlock();
}

int LM_SLWD_VOXEL::read_refine_state()
{
    int tem_flag;
    my_mutex.lock();
    tem_flag = map_refine_flag;
    my_mutex.unlock();
    return tem_flag;
}

void LM_SLWD_VOXEL::set_refine_state(int tem)
{
    my_mutex.lock();
    map_refine_flag = tem;
    my_mutex.unlock();
}

void LM_SLWD_VOXEL::free_voxel()
{
    uint a_size = plvec_voxels.size();
    for (uint i = 0; i < a_size; i++)
    {
        delete (plvec_voxels[i]);
        delete (slwd_nums[i]);
    }

    plvec_voxels.clear();
    slwd_nums.clear();
    sig_vecs.clear();
    lam_types.clear();
}