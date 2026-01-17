#ifndef _INCLUDE_MAP_MANAGER_H
#define _INCLUDE_MAP_MANAGER_H

#include <vector>
#include <queue>

#include "Feature/CFilter.hpp"
#include "Misc/Utility.hpp"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define BLOCK_MANAGER_API __declspec(dllexport)
#else
#define BLOCK_MANAGER_API __declspec(dllimport)
#endif
#else
#define BLOCK_MANAGER_API
#endif


class BLOCK_MANAGER_API BlockManager
{
public:

    bool updateLocalMap(CloudBlockPtr local_map, CloudBlockPtr last_target_cblock,
                        float local_map_radius = 80,
                        int max_num_pts = 20000,
                        bool map_based_dynamic_removal_on = false,
                        float dynamic_removal_center_radius = 30.0,
                        float dynamic_dist_thre_min = 0.3,
                        float dynamic_dist_thre_max = 3.0,
                        float near_dist_thre = 0.03);

    bool mapBasedDynamicCloseRemoval(CloudBlockPtr local_map, CloudBlockPtr last_target_cblock,
                                     float center_radius, float dynamic_dist_thre_min, float dynamic_dist_thre_max, float near_dist_thre);

    //keep the points meet dist ~ (near_dist_thre, dynamic_dist_thre_min) U (dynamic_dist_thre_max, +inf)
    //filter the points meet dist ~ (0, near_dist_thre] U [dynamic_dist_thre_min, dynamic_dist_thre_max]
    bool mapScanFeaturePtsDistanceRemoval(PointCloudXYZI::Ptr feature_pts, const PcTreePtr map_kdtree, float center_radius,
                                          float dynamic_dist_thre_min = FLT_MAX, float dynamic_dist_thre_max = FLT_MAX, float near_dist_thre = 0.0);

    bool judgeNewSubmap(float &accu_tran, float &accu_rot, int &accu_frame,
                        float max_accu_tran = 30.0, float max_accu_rot = 90.0, int max_accu_frame = 150);


};


#endif //_INCLUDE_MAP_MANAGER_H