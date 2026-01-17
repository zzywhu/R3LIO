/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <flann/flann.hpp>
#ifdef TEASER_ON
#include "teaser/geometry.h"
#endif
#include "Feature/FeatureExtractor.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define MATCHER_API __declspec(dllexport)`
#else
#define MATCHER_API __declspec(dllimport)
#endif
#else
#define MATCHER_API
#endif

class MATCHER_API Matcher
{
public:
    typedef std::vector<Eigen::VectorXf> Feature;
    typedef flann::Index<flann::L2<float>> KDTree;

    // New methods
    // Public methods:
    // 1. calculateCorrespondences
    //    input: source point cloud, target point cloud
    //    output: correspondences
    Matcher() = default;
#ifdef TEASER_ON
    bool findFeatureCorrespondenceFPFH(teaser::PointCloud &srcCloud, teaser::PointCloud& tgtCloud,
                                       std::vector<std::pair<int, int>>& correspondences, double normal_search_radius=0.02,
                                       double fpfh_search_radius=0.04, bool use_absolute_scale = false, bool use_crosscheck = true,
                                       bool use_tuple_test = false, float tuple_scale = 0.95);

    bool findFeatureCorrespondenceFPFH(teaser::PointCloud &tgtCloud, teaser::PointCloud& srcCloud,
                                       teaser::PointCloud& tgtKpts, teaser::PointCloud& srcKpts,
                                       pcl::PointIndicesPtr tgtKptIndices,  pcl::PointIndicesPtr srcKptIndices,
                                       std::vector<std::pair<int, int>>& correspondences,
                                       double normal_search_radius=0.02,
                                       double fpfh_search_radius=0.04, bool use_absolute_scale = false, bool use_crosscheck = true,
                                       bool use_tuple_test = false, float tuple_scale = 0.95);

    /**
     * Calculate correspondences based on given features and point clouds.
     * @param source_points
     * @param target_points
     * @param use_absolute_scale
     * @param use_crosscheck
     * @param use_tuple_test
     * @return
     */
    std::vector<std::pair<int, int>> calculateCorrespondences(teaser::PointCloud &source_points, teaser::PointCloud &target_points,
                                                              FPFHCloud &source_features, FPFHCloud &target_features,
                                                              bool use_absolute_scale = true, bool use_crosscheck = true,
                                                              bool use_tuple_test = true, float tuple_scale = 0);
#endif

private:
    template<typename T>
    void buildKDTree(const std::vector<T> &data, KDTree *tree);

    template<typename T>
    void searchKDTree(KDTree *tree, const T &input, std::vector<int> &indices,
                      std::vector<float> &dists, int nn);

    void advancedMatching(bool use_crosscheck, bool use_tuple_test, float tuple_scale);

    void normalizePoints(bool use_absolute_scale);

private:
    std::vector<std::pair<int, int>>                                            _corres;
#ifdef TEASER_ON
    std::vector<teaser::PointCloud>                                             _pointcloud;
#endif
    std::vector<Feature>                                                        _features;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >    _means; // for normalization
    float                                                                       _globalScale;
};


