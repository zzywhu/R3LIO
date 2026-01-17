//
// This file is for the map related operations for Lidar Odometry and Lidar SLAM
// Compulsory Dependent 3rd Libs: PCL (>1.7), glog, gflags
// Author: Yue Pan @ ETH Zurich
//

#include "LoopCloser/BlockManager.h"

#include <pcl/kdtree/kdtree_flann.h>
#include <boost/make_shared.hpp>
#include <pcl/filters/voxel_grid.h>


bool BlockManager::updateLocalMap(CloudBlockPtr localMap, CloudBlockPtr curBlock,
                                  float local_map_radius,
                                  int max_num_pts,
                                  bool map_based_dynamic_removal_on,
                                  float dynamic_removal_center_radius,
                                  float dynamic_dist_thre_min,
                                  float dynamic_dist_thre_max,
                                  float near_dist_thre)
{
    Eigen::Matrix4d transT12; //from the last local_map to the target frame
    transT12 = localMap->_poseLo.inverse() * curBlock->_poseLo ;//T12

    //to avoid building kd-tree twice , do the map based filtering at first before the transforming
    //curBlock->transformFeature(transT12);//p1=T12*p2  (ptarget)

    PointCloudXYZI::Ptr pcDownTrans(new PointCloudXYZI);
    pcl::transformPointCloud(*curBlock->_pcDown, *pcDownTrans, transT12);
    std::vector<PointWithCov> pvTransList;
    for(auto& pv:curBlock->_pvList)
    {
        PointWithCov pvTrans;
        pvTrans.pl=transT12.block<3,3>(0,0)*pv.pl+transT12.block<3,1>(0,3);
        pvTrans.pi=transT12.block<3,3>(0,0)*pv.pi+transT12.block<3,1>(0,3);
        pvTransList.emplace_back(pvTrans);
    }

    dynamic_dist_thre_max = max_(dynamic_dist_thre_max, dynamic_dist_thre_min + 0.1);
    if (map_based_dynamic_removal_on)
        mapBasedDynamicCloseRemoval(localMap, curBlock, dynamic_removal_center_radius,
                                    dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);


    localMap->_pcDown->points.insert(localMap->_pcDown->points.end(), pcDownTrans->points.begin(), pcDownTrans->points.end());
    localMap->_pvList.insert(localMap->_pvList.end(), pvTransList.begin(), pvTransList.end());

    // float keypoint_close_dist_thre = 8.0;
   // localMap->appendFeature(*curBlock);//plocal=plocal+ptarget
    //localMap->_poseLo = curBlock->_poseLo;//Tw2

    //local_map->freeTree();
    //local_map->freeRawCloud();//_pcRaw, _pcDown, pc_unground

    return true;
}

//the map consturction and updateing strategy is now the most important part for our application
//and also for loop closure and pose graph optimization

//map based active object removal
//for each non-boundary unground points of current frame, check its nearest neighbor in the local map, if the distance
//to the nearest neighbor is larger than a threshold, then this point would be regarded as a part of an active object
//we will filter the points whose distance is (0, near_dist_thre] U [dynamic_dist_thre_min, dynamic_dist_thre_max] to its nearest neighbor in the local map
bool BlockManager::mapBasedDynamicCloseRemoval(CloudBlockPtr local_map, CloudBlockPtr last_target_cblock,
                                               float center_radius, float dynamic_dist_thre_min, float dynamic_dist_thre_max, float near_dist_thre)
//feature_pts_down is used to append into the local map [dynamic_dist_thre used]
//feature_pts is used to do the next registration (act as target point cloud) [dynamic_dist_thre_down used]
{
    //kd-tree already built
    //suppose the relative velocity of self and the coming vehcile is 180 km/m (50 m/s), then the distance would be 5m /ms, you should set the threshold according to the velocity of the vehicle
    //dynamic_dist_thre_max = 3.0;
    PointCloudXYZI::Ptr pcDown(new PointCloudXYZI());

    double extended_dist = 4.0;
    CFilter<PointType> cf;
    cf.dist_filter(local_map->_pcDown, pcDown, center_radius + extended_dist);

    PcTreePtr downCloudTree(new PcTree());
    downCloudTree->setInputCloud(pcDown);

#pragma omp parallel sections
    {
#pragma omp section
        {
            //dynamic + close points
            mapScanFeaturePtsDistanceRemoval(last_target_cblock->_pcDown, downCloudTree,
                                             center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
        }
//#pragma omp section
//        {
//            //dynamic + close points
//            mapScanFeaturePtsDistanceRemoval(last_target_cblock->_pcRaw, local_map->_cloudTree,
//                                                  center_radius, dynamic_dist_thre_min, dynamic_dist_thre_max, near_dist_thre);
//
//        }
    }

    return true;
}

// filter (0, near_dist_thre) U (dynamic_dist_thre_min, dynamic_dist_thre_max)
// keep (near_dist_thre, dynamic_dist_thre_min) U (dynamic_dist_thre_max, +inf)
bool BlockManager::mapScanFeaturePtsDistanceRemoval(PointCloudXYZI::Ptr feature_pts, const PcTreePtr map_kdtree, float center_radius,
                                                    float dynamic_dist_thre_min, float dynamic_dist_thre_max, float near_dist_thre)
{
    if (feature_pts->points.size() <= 10)
        return false;

    PointCloudXYZI::Ptr feature_pts_temp(new PointCloudXYZI);
    int i;

    //#pragma omp parallel for private(i) //Multi-thread

    std::vector<float> distances_square;
    std::vector<int> search_indices;
    for (i = 0; i < feature_pts->points.size(); i++)
    {
        if (feature_pts->points[i].x * feature_pts->points[i].x +
            feature_pts->points[i].y * feature_pts->points[i].y >center_radius * center_radius)
            feature_pts_temp->points.push_back(feature_pts->points[i]);
        else
        {
            map_kdtree->nearestKSearch(feature_pts->points[i], 1, search_indices,distances_square);//search nearest neighbor

            if ((distances_square[0] > near_dist_thre * near_dist_thre && distances_square[0] < dynamic_dist_thre_min * dynamic_dist_thre_min) ||
                distances_square[0] > dynamic_dist_thre_max * dynamic_dist_thre_max) // the distance should not be too close to keep the map more uniform
                feature_pts_temp->points.push_back(feature_pts->points[i]);
            // else
            //     LOG(INFO) << "Filter out the point, dist [" << std::sqrt(distances_square[0]) << "].";

            std::vector<float>().swap(distances_square);
            std::vector<int>().swap(search_indices);
        }
    }
    feature_pts_temp->points.swap(feature_pts->points);

    return true;
}


// Divide the strip into several submaps according to multiple rules (judge if a new submap would be created)
// Rules: consecutive frame number, accumulated translation (using), accumilated heading angle (using) ...
bool BlockManager::judgeNewSubmap(float &accu_tran, float &accu_rot, int &accu_frame,
                                  float max_accu_tran, float max_accu_rot, int max_accu_frame)
{
    // LOG(INFO) << "Submap division criterion is: \n"
    //           << "1. Frame Number <= " << max_accu_frame
    //           << " , 2. Translation <= " << max_accu_tran
    //           << "m , 3. Rotation <= " << max_accu_rot << " degree.";

    if (accu_tran > max_accu_tran || accu_rot > max_accu_rot || (accu_frame > max_accu_frame && accu_tran>0.5))
//    if (accu_tran > max_accu_tran || accu_rot > max_accu_rot || accu_frame > max_accu_frame)
    {
        //recount from begining
        accu_tran = 0.0;
        accu_rot = 0.0;
        accu_frame = 0;
        return true;
    }
    else
        return false;
}


