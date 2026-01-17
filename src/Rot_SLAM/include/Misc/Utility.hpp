//
// Created by w on 2022/9/14.
//

#ifndef SRC_UTILITY_HPP
#define SRC_UTILITY_HPP

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/point_representation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/transformation_estimation_svd.h>

//Eigen
#include <Eigen/Core>

#include <vector>
#include <list>
#include <chrono>
#include <limits>
#include <time.h>
#include <mutex>

#include "CloudBlock.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define CLOUD_UTILITY_API __declspec(dllexport)
#else
#define CLOUD_UTILITY_API __declspec(dllimport)
#endif
#else
#define CLOUD_UTILITY_API
#endif

template<typename PointT>
class CLOUD_UTILITY_API CloudUtility
{
public:
    //Get Center of a Point Cloud
    void get_cloud_cpt(const typename pcl::PointCloud<PointT>::Ptr &cloud, CenterPoint &cp)
    {
        double cx = 0, cy = 0, cz = 0;
        int point_num = cloud->points.size();

        for (int i = 0; i < point_num; i++)
        {
            cx += cloud->points[i].x / point_num;
            cy += cloud->points[i].y / point_num;
            cz += cloud->points[i].z / point_num;
        }
        cp.x = cx;
        cp.y = cy;
        cp.z = cz;
    }

    //Get Bound of a Point Cloud
    void get_cloud_bbx(const typename pcl::PointCloud<PointT>::Ptr &cloud, Bounds &bound)
    {
        double min_x = DBL_MAX;
        double min_y = DBL_MAX;
        double min_z = DBL_MAX;
        double max_x = -DBL_MAX;
        double max_y = -DBL_MAX;
        double max_z = -DBL_MAX;

        for (int i = 0; i < cloud->points.size(); i++)
        {
            if (min_x > cloud->points[i].x)
                min_x = cloud->points[i].x;
            if (min_y > cloud->points[i].y)
                min_y = cloud->points[i].y;
            if (min_z > cloud->points[i].z)
                min_z = cloud->points[i].z;
            if (max_x < cloud->points[i].x)
                max_x = cloud->points[i].x;
            if (max_y < cloud->points[i].y)
                max_y = cloud->points[i].y;
            if (max_z < cloud->points[i].z)
                max_z = cloud->points[i].z;
        }
        bound.min_x = min_x;
        bound.max_x = max_x;
        bound.min_y = min_y;
        bound.max_y = max_y;
        bound.min_z = min_z;
        bound.max_z = max_z;
    }

    //Get Bound and Center of a Point Cloud
    void get_cloud_bbx_cpt(const typename pcl::PointCloud<PointT>::Ptr &cloud, Bounds &bound, CenterPoint &cp)
    {
        get_cloud_bbx(cloud, bound);
        cp.x = 0.5 * (bound.min_x + bound.max_x);
        cp.y = 0.5 * (bound.min_y + bound.max_y);
        cp.z = 0.5 * (bound.min_z + bound.max_z);
    }

    void get_intersection_bbx(Bounds &bbx_1, Bounds &bbx_2, Bounds &bbx_intersection, float bbx_boundary_pad = 2.0)
    {
        bbx_intersection.min_x = max_(bbx_1.min_x, bbx_2.min_x) - bbx_boundary_pad;
        bbx_intersection.min_y = max_(bbx_1.min_y, bbx_2.min_y) - bbx_boundary_pad;
        bbx_intersection.min_z = max_(bbx_1.min_z, bbx_2.min_z) - bbx_boundary_pad;
        bbx_intersection.max_x = min_(bbx_1.max_x, bbx_2.max_x) + bbx_boundary_pad;
        bbx_intersection.max_y = min_(bbx_1.max_y, bbx_2.max_y) + bbx_boundary_pad;
        bbx_intersection.max_z = min_(bbx_1.max_z, bbx_2.max_z) + bbx_boundary_pad;
    }

    void merge_bbx(std::vector<Bounds> &bbxs, Bounds &bbx_merged)
    {
        bbx_merged.min_x = DBL_MAX;
        bbx_merged.min_y = DBL_MAX;
        bbx_merged.min_z = DBL_MAX;
        bbx_merged.max_x = -DBL_MAX;
        bbx_merged.max_y = -DBL_MAX;
        bbx_merged.max_z = -DBL_MAX;

        for (int i = 0; i < bbxs.size(); i++)
        {
            bbx_merged.min_x = min_(bbx_merged.min_x, bbxs[i].min_x);
            bbx_merged.min_y = min_(bbx_merged.min_y, bbxs[i].min_y);
            bbx_merged.min_z = min_(bbx_merged.min_z, bbxs[i].min_z);
            bbx_merged.max_x = max_(bbx_merged.max_x, bbxs[i].max_x);
            bbx_merged.max_y = max_(bbx_merged.max_y, bbxs[i].max_y);
            bbx_merged.max_z = max_(bbx_merged.max_z, bbxs[i].max_z);
        }
    }

    //Get Bound of Subsets of a Point Cloud
    void get_sub_bbx(typename pcl::PointCloud<PointT>::Ptr &cloud, std::vector<int> &index, Bounds &bound)
    {
        typename pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>);
        for (int i = 0; i < index.size(); i++)
        {
            temp_cloud->push_back(cloud->points[index[i]]);
        }
        get_cloud_bbx(temp_cloud, bound);
    }

protected:
private:
};

#endif //SRC_UTILITY_HPP
