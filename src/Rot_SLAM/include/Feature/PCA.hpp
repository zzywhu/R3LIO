//
// This file is used for the  Principle Component Analysis (PCA) and related feature calculation of Point Cloud.
// Dependent 3rd Libs: PCL (>1.7)
// By Yue Pan et al.

//add cuda parallel computation here

#ifndef _INCLUDE_PCA_HPP_
#define _INCLUDE_PCA_HPP_

//pcl
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>

#include <vector>

#include "Misc/Utility.hpp"
#include "PCA.h"
template <typename PointT>
bool PrincipleComponentAnalysis<PointT>::get_normal_pcar(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                         float radius,
                                                         pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
    // Create the normal estimation class, and pass the input dataset to it;
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setNumberOfThreads(omp_get_max_threads()); //More threads sometimes would not speed up the procedure
    ne.setInputCloud(in_cloud);
    // Create an empty kd-tree representation, and pass it to the normal estimation object;
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

    ne.setSearchMethod(tree);
    // Use all neighbors in a sphere of radius;
    ne.setRadiusSearch(radius);
    // Compute the normal
    ne.compute(*normals);
    check_normal(normals);
    return true;
}

template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_pc_normal_pcar(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                             float radius,
                                                             pcl::PointCloud<pcl::PointNormal>::Ptr &pointnormals)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    bool normal_ready = get_normal_pcar(in_cloud, radius, normals);
    if (normal_ready)
    {
        // Concatenate the XYZ and normal fields*
        pcl::concatenateFields(*in_cloud, *normals, *pointnormals);
        return true;
    }
    else
        return false;
}

template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_normal_pcak(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                          int K,
                                                          pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
    // Create the normal estimation class, and pass the input dataset to it;
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setNumberOfThreads(omp_get_max_threads()); //More threads sometimes would not speed up the procedure
    ne.setInputCloud(in_cloud);
    // Create an empty kd-tree representation, and pass it to the normal estimation object;
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    // Use all neighbors in a sphere of radius;
    ne.setKSearch(K);
    // Compute the normal
    ne.compute(*normals);
    check_normal(normals);
    return true;
}

template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_pc_normal_pcak(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                             int K,
                                                             pcl::PointCloud<pcl::PointNormal>::Ptr &pointnormals)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    bool normal_ready = get_normal_pcak(in_cloud, K, normals);
    if (normal_ready)
    {
        // Concatenate the XYZ and normal fields*
        pcl::concatenateFields(*in_cloud, *normals, *pointnormals);
        return true;
    }
    else
        return false;
}

/**
    * \brief Principle Component Analysis (PCA) of the Point Cloud with fixed search radius
    * \param[in] in_cloud is the input Point Cloud (XYZI) Pointer
    * \param[in]     radius is the neighborhood search radius (m) for KD Tree
    * \param[out]features is the pca_feature_t vector of all the points from the Point Cloud
    */
// radius neighborhood
template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                             std::vector<pca_feature_t> &features,
                                                             float radius)
{
    //LOG(INFO) << "input cloud size is " << in_cloud->points.size();
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(in_cloud);
    features.resize(in_cloud->size());

    omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for //Multi-thread
    for (int i = 0; i < in_cloud->points.size(); i++)
    {
        std::vector<int> search_indices; //point index Vector
        std::vector<float> distances;	//distance Vector
        std::vector<int>().swap(search_indices);
        std::vector<float>().swap(distances);

        tree.radiusSearch(in_cloud->points[i], radius, search_indices, distances); //KD tree
        features[i].pt.x = in_cloud->points[i].x;
        features[i].pt.y = in_cloud->points[i].y;
        features[i].pt.z = in_cloud->points[i].z;
        features[i].ptId = i;
        features[i].pt_num = search_indices.size();

        get_pca_feature(in_cloud, search_indices, features[i]);
        assign_normal(in_cloud->points[i], features[i]);
    }

    return true;
}

// K neighborhood
template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                             std::vector<pca_feature_t> &features,
                                                             int K)
{
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(in_cloud);
    features.resize(in_cloud->size());

    omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for //Multi-thread
    for (int i = 0; i < in_cloud->points.size(); i++)
    {
        std::vector<int> search_indices; //point index Vector
        std::vector<float> distances;	//distance Vector
        std::vector<int>().swap(search_indices);
        std::vector<float>().swap(distances);

        tree.nearestKSearch(in_cloud->points[i], K, search_indices, distances); //KD tree
        features[i].pt.x = in_cloud->points[i].x;
        features[i].pt.y = in_cloud->points[i].y;
        features[i].pt.z = in_cloud->points[i].z;
        features[i].ptId = i;
        features[i].pt_num = search_indices.size();
        get_pca_feature(in_cloud, search_indices, features[i]);
        assign_normal(in_cloud->points[i], features[i]);
    }

    return true;
}

// R - K neighborhood (without already built-kd tree)
//within the radius, we would select the nearest K points for calculating PCA
template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                             std::vector<pca_feature_t> &features,
                                                             float radius, int nearest_k, int min_k, int pca_down_rate,
                                                             bool distance_adaptive_on, float unit_dist)
{
    std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
    //LOG(INFO) << "input cloud size is " << in_cloud->points.size();
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(in_cloud);
    features.resize(in_cloud->size());

    std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_build_tree = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

    double time_neighbor_search_total = 0;
    double time_pca_total = 0;

    omp_set_num_threads(min_(6, omp_get_max_threads()));
#pragma omp parallel for //Multi-thread
    for (int i = 0; i < in_cloud->points.size(); i += pca_down_rate)
    {
        std::chrono::steady_clock::time_point tic_1 = std::chrono::steady_clock::now();

        std::vector<int> search_indices_used; //points would be stored in sequence (from the closest point to the farthest point within the neighborhood)
        std::vector<int> search_indices;	  //point index Vector
        std::vector<float> squared_distances; //distance Vector

        float neighborhood_r = radius;
        int neighborhood_k = nearest_k;

        if (distance_adaptive_on) //deprecated
        {
            double dist = std::sqrt(in_cloud->points[i].x * in_cloud->points[i].x +
                                    in_cloud->points[i].y * in_cloud->points[i].y);

            neighborhood_r = max_(radius, dist / unit_dist * radius);
            //neighborhood_k = min_(nearest_k, (int)(unit_dist / dist * nearest_k));
        }
        //nearest_k=0 --> the knn is disabled, only the rnn is used
        tree.radiusSearch(i, neighborhood_r, search_indices, squared_distances, neighborhood_k);

        features[i].pt.x = in_cloud->points[i].x;
        features[i].pt.y = in_cloud->points[i].y;
        features[i].pt.z = in_cloud->points[i].z;
        features[i].ptId = i;
        features[i].pt_num = search_indices.size();

        std::chrono::steady_clock::time_point toc_1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_neighbor_search = std::chrono::duration_cast<std::chrono::duration<double>>(toc_1 - tic_1);

        //deprecated
        features[i].close_to_query_point.resize(search_indices.size());
        for (int j = 0; j < search_indices.size(); j++)
        {
            if (squared_distances[j] < 0.64 * radius * radius) // 0.5^(2/3)
                features[i].close_to_query_point[j] = true;
            else
                features[i].close_to_query_point[j] = false;
        }

        get_pca_feature(in_cloud, search_indices, features[i]);

        if (features[i].pt_num > min_k)
            assign_normal(in_cloud->points[i], features[i]);

        std::vector<int>().swap(search_indices);
        std::vector<int>().swap(search_indices_used);
        std::vector<float>().swap(squared_distances);

        std::chrono::steady_clock::time_point toc_2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_pca = std::chrono::duration_cast<std::chrono::duration<double>>(toc_2 - toc_1);

        time_neighbor_search_total += time_neighbor_search.count() * 1000.0; //in ms
        time_pca_total += time_pca.count() * 1000.0;						 //in ms
    }

    // LOG(INFO) << "Build kd-tree in [" << time_build_tree.count() * 1000.0 << "] ms\n"
    // 		  << "Neighborhood searching in [" << time_neighbor_search_total << "] ms\n"
    // 		  << "PCA calculation in [" << time_pca_total << "] ms\n";

    return true;
}

// R - K neighborhood (with already built-kd tree)
//within the radius, we would select the nearest K points for calculating PCA
template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_pc_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                             std::vector<pca_feature_t> &features, typename pcl::KdTreeFLANN<PointT>::Ptr &tree,
                                                             float radius, int nearest_k, int min_k, int pca_down_rate,
                                                             bool distance_adaptive_on, float unit_dist)
{
    //LOG(INFO) << "[" << in_cloud->points.size() << "] points used for PCA, pca down rate is [" << pca_down_rate << "]";
    features.resize(in_cloud->points.size());

    omp_set_num_threads(min_(6, omp_get_max_threads()));
#pragma omp parallel for												 //Multi-thread
    for (int i = 0; i < in_cloud->points.size(); i += pca_down_rate) //faster way
    {
        // if (i % pca_down_rate == 0) {//this way is much slower
        std::vector<int> search_indices_used; //points would be stored in sequence (from the closest point to the farthest point within the neighborhood)
        std::vector<int> search_indices;	  //point index vector
        std::vector<float> squared_distances; //distance vector

        float neighborhood_r = radius;
        int neighborhood_k = nearest_k;

        if (distance_adaptive_on)
        {
            double dist = std::sqrt(in_cloud->points[i].x * in_cloud->points[i].x +
                                    in_cloud->points[i].y * in_cloud->points[i].y +
                                    in_cloud->points[i].z * in_cloud->points[i].z);
            if (dist > unit_dist)
            {
                neighborhood_r = std::sqrt(dist / unit_dist) * radius;
                //neighborhood_k = (int)(unit_dist / dist * nearest_k));
            }
        }
        //nearest_k=0 --> the knn is disabled, only the rnn is used
        tree->radiusSearch(i, neighborhood_r, search_indices, squared_distances, neighborhood_k);

        features[i].pt.x = in_cloud->points[i].x;
        features[i].pt.y = in_cloud->points[i].y;
        features[i].pt.z = in_cloud->points[i].z;
        features[i].ptId = i;
        features[i].pt_num = search_indices.size();

        //deprecated
        features[i].close_to_query_point.resize(search_indices.size());
        for (int j = 0; j < search_indices.size(); j++)
        {
            if (squared_distances[j] < 0.64 * radius * radius) // 0.5^(2/3)
                features[i].close_to_query_point[j] = true;
            else
                features[i].close_to_query_point[j] = false;
        }

        get_pca_feature(in_cloud, search_indices, features[i]);

        if (features[i].pt_num > min_k)
            assign_normal(in_cloud->points[i], features[i]);
        std::vector<int>().swap(search_indices);
        std::vector<int>().swap(search_indices_used);
        std::vector<float>().swap(squared_distances);
    }
    //}
    return true;
}

template <typename PointT>
void PrincipleComponentAnalysis<PointT>:: calculate_normal_inconsistency(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                                         std::vector<pca_feature_t> &features)
{
    for (int i = 0; i < in_cloud->points.size(); i++)
    {
        double n_x = 0, n_y = 0, n_z = 0;

        for (int j = 0; j < features[i].neighbor_indices.size(); j++)
        {
            n_x += std::abs(in_cloud->points[features[i].neighbor_indices[j]].normal_x);
            n_y += std::abs(in_cloud->points[features[i].neighbor_indices[j]].normal_y);
            n_z += std::abs(in_cloud->points[features[i].neighbor_indices[j]].normal_z);
        }

        Eigen::Vector3d n_mean;
        Eigen::Vector3d n_self;
        n_mean << n_x / features[i].pt_num, n_y / features[i].pt_num, n_z / features[i].pt_num;
        n_mean.normalize();

        n_self << in_cloud->points[i].normal_x, in_cloud->points[i].normal_y, in_cloud->points[i].normal_z;

        features[i].normal_diff_ang_deg = 180.0 / M_PI * std::acos(std::abs(n_mean.dot(n_self))); // n1.norm()=n2.norm()=1
        // if (i % 10 == 0)
        // 	LOG(INFO) << features[i].normal_diff_ang_deg;
    }
}

/**
    * \brief Use PCL to accomplish the Principle Component Analysis (PCA)
    * of one point and its neighborhood
    * \param[in] in_cloud is the input Point Cloud Pointer
    * \param[in] search_indices is the neighborhood points' indices of the search point.
    * \param[out]feature is the pca_feature_t of the search point.
    */
template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: get_pca_feature(typename pcl::PointCloud<PointT>::Ptr in_cloud,
                                                          std::vector<int> &search_indices,
                                                          pca_feature_t &feature)
{
    int pt_num = search_indices.size();

    if (pt_num <= 3)
        return false;

    typename pcl::PointCloud<PointT>::Ptr selected_cloud(new pcl::PointCloud<PointT>());
    for (int i = 0; i < pt_num; ++i)
        selected_cloud->points.push_back(in_cloud->points[search_indices[i]]);

    pcl::PCA<PointT> pca_operator;
    pca_operator.setInputCloud(selected_cloud);

    // Compute eigen values and eigen vectors
    Eigen::Matrix3f eigen_vectors = pca_operator.getEigenVectors();
    Eigen::Vector3f eigen_values = pca_operator.getEigenValues();

    feature.vectors.principalDirection = eigen_vectors.col(0);
    feature.vectors.normalDirection = eigen_vectors.col(2);

    feature.vectors.principalDirection.normalize();
    feature.vectors.normalDirection.normalize();

    feature.values.lamada1 = eigen_values(0);
    feature.values.lamada2 = eigen_values(1);
    feature.values.lamada3 = eigen_values(2);

    if ((feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3) == 0)
        feature.curvature = 0;
    else
        feature.curvature = feature.values.lamada3 / (feature.values.lamada1 + feature.values.lamada2 + feature.values.lamada3);

    // feature.linear_2 = (sqrt(feature.values.lamada1) - sqrt(feature.values.lamada2)) / sqrt(feature.values.lamada1);
    // feature.planar_2 = (sqrt(feature.values.lamada2) - sqrt(feature.values.lamada3)) / sqrt(feature.values.lamada1);
    // feature.spherical_2 = sqrt(feature.values.lamada3) / sqrt(feature.values.lamada1);
    feature.linear_2 = ((feature.values.lamada1) - (feature.values.lamada2)) / (feature.values.lamada1);
    feature.planar_2 = ((feature.values.lamada2) - (feature.values.lamada3)) / (feature.values.lamada1);
    feature.spherical_2 = (feature.values.lamada3) / (feature.values.lamada1);

    search_indices.swap(feature.neighbor_indices);
    return true;
}

//is_plane_feature (true: assign point normal as pca normal vector, false: assign point normal as pca primary direction vector)
template <typename PointT>
bool PrincipleComponentAnalysis<PointT>:: assign_normal(PointT &pt, pca_feature_t &pca_feature, bool is_plane_feature)
{
    if (is_plane_feature)
    {
        pt.normal_x = pca_feature.vectors.normalDirection.x();
        pt.normal_y = pca_feature.vectors.normalDirection.y();
        pt.normal_z = pca_feature.vectors.normalDirection.z();
        pt.normal[3] = pca_feature.planar_2; //planrity
    }
    else
    {
        pt.normal_x = pca_feature.vectors.principalDirection.x();
        pt.normal_y = pca_feature.vectors.principalDirection.y();
        pt.normal_z = pca_feature.vectors.principalDirection.z();
        pt.normal[3] = pca_feature.linear_2; //linarity
    }
    return true;
}


template<typename T>
bool estiNormVector(Eigen::Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num)
{
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++)
    {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);

    for (int j = 0; j < point_num; j++)
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
            return false;

    normvec.normalize();
    return true;
}

template<typename T>
bool estiPlane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
            return false;

    return true;
}



#endif //_INCLUDE_PCA_HPP_
