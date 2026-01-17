//
// Created by w on 2022/9/15.
//

#ifndef SRC_CONSTRAINTFINDER_H
#define SRC_CONSTRAINTFINDER_H
#include <vector>
#include <queue>

#include "Misc/Utility.hpp"
#include "Feature/CFilter.hpp"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define CONSTRAINT_FINDER_API __declspec(dllexport)
#else
#define CONSTRAINT_FINDER_API __declspec(dllimport)
#endif
#else
#define CONSTRAINT_FINDER_API
#endif

//the edge of pose(factor) graph
struct CONSTRAINT_FINDER_API Constraint
{
    Constraint()
    {
        block1 = CloudBlockPtr(new CloudBlock);
        block2 = CloudBlockPtr(new CloudBlock);
        Trans1_2.setIdentity();
        information_matrix.setIdentity();
        sigma = FLT_MAX;
        cov_updated = false;
    }

    void free_cloud()
    {
        block1->freeAll();
        block2->freeAll();
    }

    int unique_id;				   //Unique ID
    CloudBlockPtr block1, block2; //Two block  //Target: block1,  Source: block2
    ConstraintType con_type;	   //ConstraintType
    Eigen::Matrix4d Trans1_2;	  //transformation from 2 to 1 (in global shifted map coordinate system)
    Matrix6d information_matrix;
    float overlapping_ratio; //overlapping ratio (not bbx IOU) of two cloud blocks
    float confidence;
    float sigma;			  //standard deviation of the edge
    bool cov_updated = false; //has the information_matrix already updated

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::vector<Constraint, Eigen::aligned_allocator<Constraint>> constraints;

struct ConstraintParam
{
    std::string constraint_output_file;
    std::string registration_output_file;

    int find_constraint_knn;
    double find_constraint_overlap_ratio;
    int cloud_block_capacity;

    double vf_downsample_resolution_als;
    double vf_downsample_resolution_tls;
    double vf_downsample_resolution_mls;
    double vf_downsample_resolution_bpls;

    double gf_grid_resolution;
    double gf_max_grid_height_diff;
    double gf_neighbor_height_diff;
    int ground_downsample_rate;
    int nonground_downsample_rate;

    bool normal_downsampling_on;
    int normal_down_ratio;

    double pca_neigh_r;
    int pca_neigh_k;
    double pca_linearity_thre;
    double pca_planarity_thre;
    double pca_stablity_thre;

    double reg_corr_dis_thre;
    int reg_max_iteration_num;
    double converge_tran;
    double converge_rot_d;
};

class CONSTRAINT_FINDER_API ConstraintFinder
{
public:
    ConstraintFinder()
    {}

    ConstraintFinder(ConstraintParam &param)
    {
        param_ = param;
    }

    // Constraint Type 1
    bool find_strip_adjacent_constraint(strips &blocks_all, constraints &innerstrip_cons_all);

    bool find_adjacent_constraint_in_strip(strip &blocks_strip, constraints &innerstrip_cons);

    bool add_adjacent_constraint(CloudBlockPtrs &blocks, constraints &cons, int node_count); //add to cons

    bool add_adjacent_constraint(CloudBlockPtrs &blocks, constraints &cons, Eigen::Matrix4d &tran1_2, int node_index);

    // Constraint Type 2
    int find_overlap_registration_constraint(CloudBlockPtrs &blocks, constraints &cons,
                                             float neighbor_radius = 50.0, float min_iou_thre = 0.25, int adjacent_id_thre = 3,
                                             bool search_neighbor_2d = true, int max_neighbor = 10); //add to cons [search_neighbor_2d: true 2d, false 3d]

    int find_overlap_registration_constraint(CloudBlockPtrs &blocks, const CloudBlockPtr block, constraints &cons,
                                             float neighbor_radius = 50.0, float min_iou_thre = 0.25, int adjacent_id_thre = 3,
                                             bool search_neighbor_2d = true, int max_neighbor = 10);

    bool clear_registration_constraint(constraints &cons);

    bool cancel_registration_constraint(constraints &cons, double confidenceThre, double sigmaThre);

    bool batch_add_registration_edge_bfs(strip &all_blocks, constraints &all_cons, int visualization_level);

    bool assign_block2constraint(strip &all_blocks, constraints &all_cons);

    bool double_check_tran(Eigen::Matrix4d &global_reg_tran, Eigen::Matrix4d &lo_predicted_tran, Eigen::Matrix4d &trusted_tran,
                           double translation_thre = 8.0, double rotation_deg_thre = 45.0);

    double calculate_iou(Bounds &bound1, Bounds &bound2);

    double calculate_iou(Bounds &bound1, Bounds &bound2, double& iou1, double& iou2);

    std::pair<float, float> getFitnessScoreDist(pcl::PointCloud<PointType>::Ptr prevSubmap,
                                                pcl::PointCloud<PointType>::Ptr curSubmap);

private:
    void find_neighbor_k_cps(CenterPoint &cp_search, std::vector<CenterPoint> &cp_points, int k);
    void find_neighbor_r_cps(CenterPoint &cp_search, std::vector<CenterPoint> &cp_points, float r);


    bool judge_adjacent_by_id(int id_1, int id_2, int submap_id_diff = 3);

private:
    ConstraintParam param_;

    int total_con_count_;
    int registration_con_count_;
    int adjacent_con_count_;
    int ALS_adjacent_con_count_;
    int MLS_adjacent_con_count_;
    int BPLS_adjacent_con_count_;
};

#endif //SRC_CONSTRAINTFINDER_H
