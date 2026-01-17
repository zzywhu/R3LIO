#pragma once

#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/keypoints/iss_3d.h>//iss特征点
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#ifdef TEASER_ON
#include "teaser/geometry.h"
#endif
#include "Common.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define FPFH_ESTIMATION_API __declspec(dllexport)
#else
#define FPFH_ESTIMATION_API __declspec(dllimport)
#endif
#else
#define FPFH_ESTIMATION_API
#endif

using FPFHCloud = pcl::PointCloud<pcl::FPFHSignature33>;
using FPFHCloudPtr = pcl::PointCloud<pcl::FPFHSignature33>::Ptr;

double FPFH_ESTIMATION_API computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud);

void FPFH_ESTIMATION_API extractFeature(PointCloudXYZI *plBuff, const int &nScans,
                                        PointCloudXYZI &cornerPointsSharp,
                                        PointCloudXYZI &surfPointsFlat,
                                        int nSeg=6);

void FPFH_ESTIMATION_API extractISSKeyPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoints,
                         pcl::PointIndicesPtr kptIndices,
                         double salientRadiusRatios=6,double nonMaxRadiusRatios=4, double res=0);

pcl::PointCloud<pcl::Normal>::Ptr FPFH_ESTIMATION_API getNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud, double normals_radius);

pcl::PointCloud<pcl::FPFHSignature33>::Ptr FPFH_ESTIMATION_API getFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                           pcl::PointCloud<pcl::Normal>::Ptr normals,
                                                                           double feature_radius);

class FPFH_ESTIMATION_API FPFHEstimation
{
public:
    FPFHEstimation()
            : _fpfhEstimation(new pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>)
    {};
#ifdef TEASER_ON
    /**
     * Compute FPFH features.
     *
     * @return A shared pointer to the FPFH feature point cloud
     * @param input_cloud
     * @param normal_search_radius Radius for estimating normals
     * @param fpfh_search_radius Radius for calculating FPFH (needs to be at least normalSearchRadius)
     */
    FPFHCloudPtr computeFPFHFeatures(const teaser::PointCloud &input_cloud,
                                     double normal_search_radius = 0.03,
                                     double fpfh_search_radius = 0.05);

    FPFHCloudPtr computeFPFHFeatures(const teaser::PointCloud &input_cloud,
                                     pcl::PointIndicesPtr issIndices,
                                     double normal_search_radius = 0.03,
                                     double fpfh_search_radius = 0.05);
#endif
    /**
     * Return the pointer to the underlying pcl::FPFHEstimation object
     * @return
     */
    inline pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr
    getImplPointer() const
    {
        return _fpfhEstimation;
    }

private:
    // pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr _fpfhEstimation;
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr _fpfhEstimation;

    /**
     * Wrapper function for the corresponding PCL function.
     * @param output_cloud
     */
    void compute(pcl::PointCloud<pcl::FPFHSignature33> &output_cloud)  {_fpfhEstimation->compute(output_cloud);}

    /**
     * Wrapper function for the corresponding PCL function.
     * @param input_cloud
     */
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud)  {_fpfhEstimation->setInputCloud(input_cloud);}

    /**
     * Wrapper function for the corresponding PCL function.
     * @param input_normals
     */
    void setInputNormals(pcl::PointCloud<pcl::Normal>::Ptr input_normals) {_fpfhEstimation->setInputNormals(input_normals);}

    /**
     * Wrapper function for the corresponding PCL function.
     * @param search_method
     */
    void setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method)    {_fpfhEstimation->setSearchMethod(search_method);}

    /**
     * Wrapper function for the corresponding PCL function.
     */
    void setRadiusSearch(double r) { _fpfhEstimation->setRadiusSearch(r); }

    void setIndices(pcl::PointIndicesPtr issIndices){ _fpfhEstimation->setIndices(issIndices); }


};


