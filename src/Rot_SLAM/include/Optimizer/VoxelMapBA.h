//
// Created by w on 2022/9/15.
//
#ifndef SRC_VOXELMAPBA_H
#define SRC_VOXELMAPBA_H


#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "list"
#include "mutex"
#include "thread"

#include "Misc/So3.h"
#include "Mapper/VoxelMapper.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define VOXEL_MAP_BA_API __declspec(dllexport)
#else
#define VOXEL_MAP_BA_API __declspec(dllimport)
#endif
#else
#define VOXEL_MAP_BA_API
#endif

#define MIN_PS 5

// LM optimizer for map-refine
class VOXEL_MAP_BA_API LM_SLWD_VOXEL
{
public:
    LM_SLWD_VOXEL(int ss, int fn, int thnum);

    // Used by "push_voxel"
    void downsample(PvList &plvec_orig,
                    int cur_frame,
                    std::vector<Eigen::Vector3d> &plvec_voxel,
                    std::vector<int> &slwd_num,
                    int filternum2use);

    // Push voxel into optimizer
    void push_voxel(std::vector<PvList *> &plvec_orig, SigVecClass &sig_vec, int lam_type);

    // Calculate Hessian, Jacobian, residual
    void acc_t_evaluate(std::vector<SO3> &so3_ps, std::vector<Eigen::Vector3d> &t_ps,
                        int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual);

    // Multithread for "acc_t_evaluate"
    void divide_thread(std::vector<SO3> &so3_ps, std::vector<Eigen::Vector3d> &t_ps, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual);

    // Calculate residual
    void evaluate_only_residual(std::vector<SO3> &so3_ps, std::vector<Eigen::Vector3d> &t_ps, double &residual);

    // LM process
    void dampingIter();

    int read_refine_state();

    void set_refine_state(int tem);

    void free_voxel();

public:
    bool is_verbose;
    int slwd_size, filternum, thd_num, jac_leng;
    int iter_max = 20;

    double corn_less;

    std::vector<SO3> so3_poses, so3_poses_temp;
    std::vector<Eigen::Vector3d> t_poses, t_poses_temp;

    std::vector<int> lam_types; // 0 surf, 1 line
    std::vector<SigVecClass> sig_vecs;
    std::vector<std::vector<Eigen::Vector3d> *> plvec_voxels;
    std::vector<std::vector<int> *> slwd_nums;
    int map_refine_flag;
    std::mutex my_mutex;

};

#endif //SRC_VOXELMAPBA_H
