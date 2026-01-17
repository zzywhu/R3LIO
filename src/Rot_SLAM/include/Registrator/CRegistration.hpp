//
// This file is for the general implements of all kinds of registration methods
// The following registration methods are involoved:
// ICP (PCL), GICP, VGICP, NDT (Kenji Koide), FPFH-SAC(PCL), BSC-SAC (Zhen Dong et al.), TEASER (Heng Yang et al.), MMLLS-ICP (Yue Pan et al.)
// Dependent 3rd Libs: PCL (>1.7), TEASER++ (optional), Sophus (optional)
// By Yue Pan
//

#ifndef _INCLUDE_COMMON_REG_HPP
#define _INCLUDE_COMMON_REG_HPP

#include <math.h>

//pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls_weighted.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/ia_ransac.h>

#if TEASER_ON
//teaser++ (global registration)
#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/certification.h>

#endif
#include "Feature/CFilter.hpp"
#include "Feature/PCA.hpp"
#include "Feature/FeatureExtractor.h"
#include "Matcher/Matcher.h"
#include "Misc/Utility.hpp"
#include "pcl/super4pcs.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define CREGISTRATION_API __declspec(dllexport)
#else
#define CREGISTRATION_API __declspec(dllimport)
#endif
#else
#define CREGISTRATION_API
#endif

enum TransformEstimationType
{
	SVD,
	LM,
	LLS
};
enum CorresEstimationType
{
	NN,
	NS
}; //NN: Nearest Neighbor ; NS: Normal Shooting
enum DistMetricType
{
	Point2Point,
	Point2Plane,
	Plane2Plane
};

template <typename PointT>
class CREGISTRATION_API CRegistration : public CloudUtility<PointT>
{
  public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloudXYZ;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tgtCloudXYZ;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features1;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features2;
    pcl::PointCloud<pcl::Normal>::Ptr normals1;
    pcl::PointCloud<pcl::Normal>::Ptr normals2;

    CRegistration()
    {
        srcCloudXYZ.reset(new pcl::PointCloud<pcl::PointXYZ>());
        tgtCloudXYZ.reset(new pcl::PointCloud<pcl::PointXYZ>());
        features1.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
        features2.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
        normals1.reset(new pcl::PointCloud<pcl::Normal>());
        normals2.reset(new pcl::PointCloud<pcl::Normal>());
    }
	//General implement of icp registration algorithm in pcl (baseline method)
	//(with different distance metrics, correspondence estimation, transformation estimation methods and parameters)
	//radius NN neighborhood search
	double icp_registration(const typename pcl::PointCloud<PointT>::Ptr &SourceCloud,
							const typename pcl::PointCloud<PointT>::Ptr &TargetCloud,
							typename pcl::PointCloud<PointT>::Ptr &TransformedSource,
							Eigen::Matrix4d &transformationS2T,
							DistMetricType metrics, CorresEstimationType ce, TransformEstimationType te,
							bool use_reciprocal_correspondence, bool use_trimmed_rejector,
							int max_iter, float thre_dis, float neighbor_radius)
	{
		clock_t t0, t1, t2;
		t0 = clock();

		double mae_nn;
		pcl::registration::CorrespondenceRejectorVarTrimmed::Ptr trimmed_cr(new pcl::registration::CorrespondenceRejectorVarTrimmed);

		switch (metrics)
		{
		case Point2Point:
		{
			t1 = clock();
			pcl::IterativeClosestPoint<PointT, PointT> icp;

			icp.setInputSource(SourceCloud);
			icp.setInputTarget(TargetCloud);

			typename pcl::registration::TransformationEstimationSVD<PointT, PointT, float>::Ptr te_svd(new pcl::registration::TransformationEstimationSVD<PointT, PointT, float>);
			typename pcl::registration::TransformationEstimationLM<PointT, PointT, float>::Ptr te_lm(new pcl::registration::TransformationEstimationLM<PointT, PointT, float>);

			switch (te)
			{
			case SVD:
				icp.setTransformationEstimation(te_svd); //Use SVD
				break;
			case LM:
				icp.setTransformationEstimation(te_lm); //Use L-M Non-Linear Optimization
				break;
			default: //Default svd
				break;
			}

			// Use Reciprocal Correspondences or not? [a -> b && b -> a]
			icp.setUseReciprocalCorrespondences(use_reciprocal_correspondence);

			// Trimmed or not?
			if (use_trimmed_rejector)
				icp.addCorrespondenceRejector(trimmed_cr);
			else
				icp.setMaxCorrespondenceDistance(thre_dis);

			// Converge criterion ( 'Or' Relation )
			// Set the maximum number of iterations [ n>x ] (criterion 1)
			icp.setMaximumIterations(max_iter); //Most likely to happen
			// Set the transformation difference threshold [delta_t<sqrt(x) or delta_ang<arccos(1-x)] (criterion 2)
			icp.setTransformationEpsilon(1e-8); //Quite hard to happen
			// Set the relative RMS difference between two consecutive iterations [ RMS(n)-RMS(n+1)<x*RMS(n) ] (criterion 3)
			icp.setEuclideanFitnessEpsilon(1e-5); //Quite hard to happen

			icp.align(*TransformedSource);

			transformationS2T = icp.getFinalTransformation();

			t2 = clock();

			if (use_trimmed_rejector)
				thre_dis = trimmed_cr->getTrimmedDistance();
			printf("Estimated trimmed distance threshold is %lf.\n", thre_dis);

			mae_nn = icp.getFitnessScore(thre_dis); //Get the Mean Absolute Error (MAE) after registration calculated from Nearest Neighbor Search

			// Commented these out if you don't want to output the registration log
			std::cout << "SCloud point # " << SourceCloud->points.size() << " , TCloud point # " << TargetCloud->points.size() << std::endl;
			std::cout << "Point to Point ICP done in " << float(t2 - t0) / CLOCKS_PER_SEC << " s" << std::endl;
			std::cout << "The fitness score of this registration is " << mae_nn << std::endl
				 << transformationS2T << std::endl;

			break;
		}
		case Point2Plane:
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr SourceCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::PointCloud<pcl::PointXYZ>::Ptr TargetCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
			copyPointCloud(*SourceCloud, *SourceCloudXYZ);
			copyPointCloud(*TargetCloud, *TargetCloudXYZ);
			// In this case, The Cloud's Normal hasn't been calculated yet.

			pcl::PointCloud<pcl::PointNormal>::Ptr SourceNormal(new pcl::PointCloud<pcl::PointNormal>());
			pcl::PointCloud<pcl::PointNormal>::Ptr TargetNormal(new pcl::PointCloud<pcl::PointNormal>());
			pcl::PointCloud<pcl::PointNormal>::Ptr TransformedSourceN(new pcl::PointCloud<pcl::PointNormal>());

			//Estimate Normal Multi-thread
			PrincipleComponentAnalysis<pcl::PointXYZ> pca_estimator;

			//Radius search
			pca_estimator.get_pc_normal_pcar(SourceCloudXYZ, neighbor_radius, SourceNormal);
			pca_estimator.get_pc_normal_pcar(TargetCloudXYZ, neighbor_radius, TargetNormal);
			//Or
			//KNN search
			//pca_estimator.get_pc_normal_pcak(SourceCloud, covariance_K, SourceNormal);
			//pca_estimator.get_pc_normal_pcak(TargetCloud, covariance_K, TargetNormal);
			t1 = clock();

			pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;

			icp.setInputSource(SourceNormal);
			icp.setInputTarget(TargetNormal);

			if (ce == NS) //Normal Shooting
			{
				pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointNormal, pcl::PointNormal, pcl::PointNormal>::Ptr ns_est(new pcl::registration::CorrespondenceEstimationNormalShooting<pcl::PointNormal, pcl::PointNormal, pcl::PointNormal>);
				ns_est->setInputSource(SourceNormal);
				ns_est->setSourceNormals(SourceNormal);
				ns_est->setInputTarget(TargetNormal);
				ns_est->setKSearch(5);
				icp.setCorrespondenceEstimation(ns_est);
			}

			pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal, float>::Ptr te_lmn(new pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal, float>);
			pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal, float>::Ptr te_lls(new pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal, float>);
			pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<pcl::PointNormal, pcl::PointNormal, float>::Ptr te_lls_weight(new pcl::registration::TransformationEstimationPointToPlaneLLSWeighted<pcl::PointNormal, pcl::PointNormal, float>);
			switch (te)
			{
			case LLS:
				icp.setTransformationEstimation(te_lls); //Use Linear Least Square
				break;
			case LM:
				icp.setTransformationEstimation(te_lmn); //Use L-M Non-Linear Optimization
				break;
			default: //Default lls
				break;
			}

			// Use Reciprocal Correspondences or not? [a -> b && b -> a]
			icp.setUseReciprocalCorrespondences(use_reciprocal_correspondence);

			// Trimmed or not?
			if (use_trimmed_rejector)
				icp.addCorrespondenceRejector(trimmed_cr);
			else
				icp.setMaxCorrespondenceDistance(thre_dis);

			// Converge criterion ( 'Or' Relation )
			// Set the maximum number of iterations [ n>x ] (criterion 1)
			icp.setMaximumIterations(max_iter); //Most likely to happen
			// Set the transformation difference threshold [delta_t<sqrt(x) or delta_ang<arccos(1-x)] (criterion 2)
			icp.setTransformationEpsilon(1e-8); //Quite hard to happen
			// Set the relative RMS difference between two consecutive iterations [ RMS(n)-RMS(n+1)<x*RMS(n) ] (criterion 3)
			icp.setEuclideanFitnessEpsilon(1e-5); //Quite hard to happen

			icp.align(*TransformedSourceN);

			transformationS2T = icp.getFinalTransformation();

			t2 = clock();

			if (use_trimmed_rejector)
				thre_dis = trimmed_cr->getTrimmedDistance();
			printf("Estimated trimmed distance threshold is %lf.\n", thre_dis);

			mae_nn = icp.getFitnessScore(thre_dis); //Get the Mean Absolute Error (MAE) after registration calculated from Nearest Neighbor Search

			// Commented these out if you don't want to output the registration log
			std::cout << "SCloud point # " << SourceCloud->points.size() << " , TCloud point # " << TargetCloud->points.size() << std::endl;
			std::cout << "Point-to-Plane ICP done in " << float(t2 - t0) / CLOCKS_PER_SEC << " s" << std::endl;
			std::cout << "Normal Estimation in " << float(t1 - t0) / CLOCKS_PER_SEC << " s, "
				 << "registration in " << float(t2 - t1) / CLOCKS_PER_SEC << " s." << std::endl;
            std::cout << "The fitness score of this registration is " << mae_nn << std::endl
				 << transformationS2T << std::endl;

			copyPointCloud(*TransformedSourceN, *TransformedSource);
			pcl::PointCloud<pcl::PointNormal>().swap(*TransformedSourceN); // Free the Memory
			pcl::PointCloud<pcl::PointNormal>().swap(*SourceNormal);	   // Free the Memory
			pcl::PointCloud<pcl::PointNormal>().swap(*TargetNormal);	   // Free the Memory
			pcl::PointCloud<pcl::PointXYZ>().swap(*SourceCloudXYZ);		   // Free the Memory
			pcl::PointCloud<pcl::PointXYZ>().swap(*TargetCloudXYZ);		   // Free the Memory

			break;
		}
		case Plane2Plane:
		{
			t1 = clock();
			pcl::GeneralizedIterativeClosestPoint<PointT, PointT> icp;

			// Set the number of points used to calculated the covariance of a point
			// icp.setCorrespondenceRandomness(covariance_K);
			icp.setCorrespondenceRandomness(10);

			icp.setInputSource(SourceCloud);
			icp.setInputTarget(TargetCloud);

			// Use Reciprocal Correspondences or not? [a -> b && b -> a]
			icp.setUseReciprocalCorrespondences(use_reciprocal_correspondence);

			// Trimmed or not?
			if (use_trimmed_rejector)
				icp.addCorrespondenceRejector(trimmed_cr);
			else
				icp.setMaxCorrespondenceDistance(thre_dis);

			icp.setMaximumOptimizerIterations(10);

			// Converge criterion ( 'Or' Relation )
			// Set the maximum number of iterations [ n>x ] (criterion 1)
			icp.setMaximumIterations(max_iter); //Most likely to happen
			// Set the transformation difference threshold [delta_t<sqrt(x) or delta_ang<arccos(1-x)] (criterion 2)
			icp.setTransformationEpsilon(1e-8); //Quite hard to happen
			// Set the relative RMS difference between two consecutive iterations [ RMS(n)-RMS(n+1)<x*RMS(n) ] (criterion 3)
			icp.setEuclideanFitnessEpsilon(1e-5); //Quite hard to happen

			icp.align(*TransformedSource);

			transformationS2T = icp.getFinalTransformation();

			t2 = clock();

			if (use_trimmed_rejector)
				thre_dis = trimmed_cr->getTrimmedDistance();
			printf("Estimated trimmed distance threshold is %lf.\n", thre_dis);

			mae_nn = icp.getFitnessScore(thre_dis); //Get the Mean Absolute Error (MAE) after registration calculated from Nearest Neighbor Search

			// Commented these out if you don't want to output the registration log
			std::cout << "SCloud point # " << SourceCloud->points.size() << " , TCloud point # " << TargetCloud->points.size() << std::endl;
			std::cout << "Plane-to-Plane ICP done in " << float(t2 - t0) / CLOCKS_PER_SEC << " s" << std::endl;
			std::cout << "The fitness score of this registration is " << mae_nn << std::endl
				 << transformationS2T << std::endl;
			break;
		}
		default:
			return -1;
		}

		return mae_nn;
	}

	/**
		* \brief Estimated the approximate overlapping ratio for Cloud1 considering Cloud2
		* \param[in]  Cloud1 : A pointer of the Point Cloud used for overlap ratio calculation
		* \param[in]  Cloud2 : A pointer of the Point Cloud overlapped with Cloud1
		* \param[out] thre_dis : It acts as the search radius of overlapping estimation
		* \return : The estimated overlap ratio [from 0 to 1]
		*/
	float get_overlap_ratio(const typename pcl::PointCloud<PointT>::Ptr &Cloud1,
							const typename pcl::PointCloud<PointT>::Ptr &Cloud2,
							float thre_dis)
	{
		int overlap_point_num = 0;
		float overlap_ratio;

		pcl::search::KdTree<PointT> kdtree;
		kdtree.setInputCloud(Cloud2);

		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;

		for (int i = 0; i < Cloud1->size(); i++)
		{
			if (kdtree.radiusSearch(Cloud1->points[i], thre_dis, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
				overlap_point_num++;
		}

		overlap_ratio = (0.01 + overlap_point_num) / Cloud1->size();
		//cout << "The estimated approximate overlap ratio of Cloud 1 is " << overlap_ratio << endl;
		std::cout << "The estimated approximate overlap ratio of Cloud 1 is " << overlap_ratio;

		return overlap_ratio;
	}



	//NCC: neighborhood category context descriptor
	bool find_feature_correspondence_ncc(const typename pcl::PointCloud<PointT>::Ptr &target_kpts, const typename pcl::PointCloud<PointT>::Ptr &source_kpts,
										 typename pcl::PointCloud<PointT>::Ptr &target_corrs, typename pcl::PointCloud<PointT>::Ptr &source_corrs,
										 bool fixed_num_corr = false, int corr_num = 2000, bool reciprocal_on = true)
										 // to enable reciprocal correspondence, you need to disable fixed_num_corr.
										 // once fixed_num_cor is enabled, reciprocal correspondence would be automatically disabled
	{
		int target_kpts_num = target_kpts->points.size();
		int source_kpts_num = source_kpts->points.size();
		float dist_margin_thre = 0.0;

		std::cout << "[" << target_kpts_num << "] key points in target point cloud and [" << source_kpts_num << "] key points in source point cloud.";

		if (target_kpts_num < 10 || source_kpts_num < 10)
		{
			std::cerr << "Too few key points\n";
			return false;
		}

		std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

		//first get descriptor
		//std::vector<std::vector<int>> target_kpts_descriptors;
		std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> target_kpts_descriptors;

		float intensity_min = FLT_MAX;
		float intensity_max = 0;

		for (int i = 0; i < target_kpts_num; i++)
		{
			float cur_i = target_kpts->points[i].intensity;

			intensity_min = min_(intensity_min, cur_i);
			intensity_max = max_(intensity_max, cur_i);
		}

		for (int i = 0; i < target_kpts_num; i++)
		{
			Eigen::VectorXf temp_descriptor(11);
			int temp_descriptor_close = (int)target_kpts->points[i].normal[0];
			int temp_descriptor_far = (int)target_kpts->points[i].normal[1];
			// neighborhood category with its distance to the query point
			temp_descriptor(0) = temp_descriptor_close / 1000000;
			temp_descriptor(1) = (temp_descriptor_close % 1000000) / 10000;
			temp_descriptor(2) = (temp_descriptor_close % 10000) / 100;
			temp_descriptor(3) = temp_descriptor_close % 100;
			temp_descriptor(4) = temp_descriptor_far / 1000000;
			temp_descriptor(5) = (temp_descriptor_far % 1000000) / 10000;
			temp_descriptor(6) = (temp_descriptor_far % 10000) / 100;
			temp_descriptor(7) = temp_descriptor_far % 100;
			// other properties
			float cur_i = target_kpts->points[i].intensity;
			temp_descriptor(8) = (cur_i - intensity_min) / (intensity_max - intensity_min) * 255.0; //[0 - 255] //normalized intensity
			temp_descriptor(9) = target_kpts->points[i].normal[3] * 100;							//[0 - 100] //curvature
			temp_descriptor(10) = target_kpts->points[i].data[3] * 30;								//[0 - 100] //height above ground
			//LOG(INFO) << temp_descriptor[1] << "," << temp_descriptor[2] << "," << temp_descriptor[3] << "," << temp_descriptor[4] << "," << temp_descriptor[5] << "," << temp_descriptor[6] << "," << temp_descriptor[7] << "," << temp_descriptor[8] << "," << temp_descriptor[9] << "," << temp_descriptor[10] << "," << temp_descriptor[11];
			target_kpts_descriptors.push_back(temp_descriptor);
		}

		std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> source_kpts_descriptors;
		for (int i = 0; i < source_kpts_num; i++)
		{
			Eigen::VectorXf temp_descriptor(11);
			int temp_descriptor_close = (int)source_kpts->points[i].normal[0];
			int temp_descriptor_far = (int)source_kpts->points[i].normal[1];
			// neighborhood category with its distance to the query point
			temp_descriptor(0) = temp_descriptor_close / 1000000;
			temp_descriptor(1) = (temp_descriptor_close % 1000000) / 10000;
			temp_descriptor(2) = (temp_descriptor_close % 10000) / 100;
			temp_descriptor(3) = temp_descriptor_close % 100;
			temp_descriptor(4) = temp_descriptor_far / 1000000;
			temp_descriptor(5) = (temp_descriptor_far % 1000000) / 10000;
			temp_descriptor(6) = (temp_descriptor_far % 10000) / 100;
			temp_descriptor(7) = temp_descriptor_far % 100;
			// other properties
			float cur_i = source_kpts->points[i].intensity;
			temp_descriptor(8) = (cur_i - intensity_min) / (intensity_max - intensity_min) * 255.0; //[0 - 255] //normalized intensity
			temp_descriptor(9) = source_kpts->points[i].normal[3] * 100; //[0 - 100] //curvature
			temp_descriptor(10) = source_kpts->points[i].data[3] * 30;   //[0 - 100] //height above ground
			//LOG(INFO) << temp_descriptor[1] << "," << temp_descriptor[2] << "," << temp_descriptor[3] << "," << temp_descriptor[4] << "," << temp_descriptor[5] << "," << temp_descriptor[6] << "," << temp_descriptor[7] << "," << temp_descriptor[8] << "," << temp_descriptor[9] << "," << temp_descriptor[10] << "," << temp_descriptor[11];
			source_kpts_descriptors.push_back(temp_descriptor);
		}

		std::vector<std::vector<float>> dist_table(target_kpts_num);
		for (int i = 0; i < target_kpts_num; i++)
			dist_table[i].resize(source_kpts_num);

		std::vector<std::pair<int, float>> dist_array;

		omp_set_num_threads(min_(6, omp_get_max_threads())); //TODO: speed up
#pragma omp parallel for  //Multi-thread
		for (int i = 0; i < target_kpts_num; i++)
		{
			for (int j = 0; j < source_kpts_num; j++)
			{
				//Method 1. directly use L1 distance (use the features from 0 to 11)
				for (int k = 0; k < 11; k++)
					dist_table[i][j] += std::abs(target_kpts_descriptors[i](k) - source_kpts_descriptors[j](k));

				//Method 2. use cosine similarity instead
				//dist_table[i][j] =
				//target_kpts_descriptors[i].norm() * source_kpts_descriptors[j].norm() / target_kpts_descriptors[i].dot(source_kpts_descriptors[j]);

				//Method 3. use K-L divergence instead (use only the histogram (distribution)
				//for (int k = 0; k < 8; k++)
				//	dist_table[i][j] += 1.0 * target_kpts_descriptors[i](k) * std::log((1.0 * target_kpts_descriptors[i](k) + 0.001) / (1.0 * source_kpts_descriptors[j](k) + 0.001));
			}
		}
		if (!fixed_num_corr)
		{
			//find correspondence
			for (int i = 0; i < target_kpts_num; i++)
			{
				//LOG(INFO) << "keypoint indice: " << target_bscs[0][i].keypointIndex_;
				int min_dist_col_index = 0;
				float min_dist_row = FLT_MAX;
				for (int j = 0; j < source_kpts_num; j++)
				{
					if (dist_table[i][j] < min_dist_row)
					{
						min_dist_row = dist_table[i][j];
						min_dist_col_index = j;
					}
				}
				bool refined_corr = true;
				if (reciprocal_on) //reciprocal nearest neighbor correspondnece
				{
					for (int j = 0; j < target_kpts_num; j++)
					{
						if (min_dist_row > dist_table[j][min_dist_col_index] + dist_margin_thre)
						{
							refined_corr = false;
							break;
						}
					}
				}
				if (refined_corr)
				{
					//LOG(INFO) << "[" << i << "] - [" << min_dist_col_index << "]:" << min_dist_row;
					target_corrs->points.push_back(target_kpts->points[i]);
					source_corrs->points.push_back(source_kpts->points[min_dist_col_index]);
				}
			}
		}
		else //fixed num correspondence
		{
			for (int i = 0; i < target_kpts_num; i++)
			{
				for (int j = 0; j < source_kpts_num; j++)
				{
					std::pair<int, float> temp_pair;
					temp_pair.first = i * source_kpts_num + j;
					temp_pair.second = dist_table[i][j];
					dist_array.push_back(temp_pair);
				}
			}
			std::sort(dist_array.begin(), dist_array.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) { return a.second < b.second; });
			corr_num = min_(corr_num, dist_array.size()); //take the k shortest distance

			std::vector<int> count_target_kpt(target_kpts_num, 0);
			std::vector<int> count_source_kpt(source_kpts_num, 0);

			int max_corr_num = 6;

			for (int k = 0; k < corr_num; k++)
			{
				int index = dist_array[k].first;
				int i = index / source_kpts_num;
				int j = index % source_kpts_num;

				if (count_target_kpt[i] > max_corr_num || count_source_kpt[j] > max_corr_num) //we only keep the first max_corr_num candidate correspondence of a single point in either source or target point cloud
					continue;

				count_target_kpt[i]++;
				count_source_kpt[j]++;

				target_corrs->points.push_back(target_kpts->points[i]);
				source_corrs->points.push_back(source_kpts->points[j]);
			}
		}

		std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

		//free memory
		std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>().swap(target_kpts_descriptors);
		std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>().swap(source_kpts_descriptors);
		std::vector<std::vector<float>>().swap(dist_table);
		std::vector<std::pair<int, float>>().swap(dist_array);

		std::cout << "[" << source_corrs->points.size() << "] correspondences found in [" << time_used.count() * 1000.0 << "] ms";

		return true;
	}



    //coarse global registration using RANSAC
	int coarse_reg_ransac(const typename pcl::PointCloud<PointT>::Ptr &targetPts,
						  const typename pcl::PointCloud<PointT>::Ptr &sourcePts,
						  Eigen::Matrix4d &tran_mat, float noise_bound = 0.2, int min_inlier_num = 8, int max_iter_num = 20000,
                          double salientRadiusRatio=6,double nonMaxRadiusRatio=4,
                          double normalRadiusRatio=8, double fpfhRadiusRatio=16)
	{
        std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
#ifdef TEASER_ON
        pcl::PointIndicesPtr srcKptIndices(new pcl::PointIndices), tgtKptIndices(new pcl::PointIndices);
        pcl::PointCloud<pcl::PointXYZ>::Ptr srcKeyPoints(new pcl::PointCloud<pcl::PointXYZ>()), tgtKeyPoints(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>()),tgtCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*sourcePts, *srcCloudXYZ);
        pcl::copyPointCloud(*targetPts, *tgtCloudXYZ);

        extractISSKeyPoints(srcCloudXYZ,srcKeyPoints,srcKptIndices,salientRadiusRatio,nonMaxRadiusRatio);
        extractISSKeyPoints(tgtCloudXYZ,tgtKeyPoints,tgtKptIndices,salientRadiusRatio,nonMaxRadiusRatio);

        teaser::PointCloud srcTeaserCloud, tgtTeaserCloud;
        for (size_t i = 0; i < srcCloudXYZ->size(); ++i)
            srcTeaserCloud.push_back({static_cast<float>(srcCloudXYZ->points[i].x),
                                      static_cast<float>(srcCloudXYZ->points[i].y),
                                      static_cast<float>(srcCloudXYZ->points[i].z)});
        for (size_t i = 0; i < tgtCloudXYZ->size(); ++i)
            tgtTeaserCloud.push_back({static_cast<float>(tgtCloudXYZ->points[i].x),
                                      static_cast<float>(tgtCloudXYZ->points[i].y),
                                      static_cast<float>(tgtCloudXYZ->points[i].z)});
        teaser::PointCloud srcTeaserKeyPoints, tgtTeaserKeyPoints;
        for (size_t i = 0; i < srcKeyPoints->size(); ++i)
            srcTeaserKeyPoints.push_back({static_cast<float>(srcKeyPoints->points[i].x),
                                          static_cast<float>(srcKeyPoints->points[i].y),
                                          static_cast<float>(srcKeyPoints->points[i].z)});
        for (size_t i = 0; i < tgtKeyPoints->size(); ++i)
            tgtTeaserKeyPoints.push_back({static_cast<float>(tgtKeyPoints->points[i].x),
                                          static_cast<float>(tgtKeyPoints->points[i].y),
                                          static_cast<float>(tgtKeyPoints->points[i].z)});
//        std::cout<<"Src key points size:"<<srcTeaserKeyPoints.size()<<std::endl;
//        std::cout<<"Dst key points size:"<<tgtTeaserKeyPoints.size()<<std::endl;

        Matcher featMatcher;
        std::vector<std::pair<int, int>> correspondences;
        double cloudRes=max(computeCloudResolution(tgtCloudXYZ),computeCloudResolution(srcCloudXYZ));
        featMatcher.findFeatureCorrespondenceFPFH(tgtTeaserCloud, srcTeaserCloud,
                                                  tgtTeaserKeyPoints, srcTeaserKeyPoints,
                                                  tgtKptIndices,srcKptIndices,
                                                  correspondences,
                                                  normalRadiusRatio*cloudRes, fpfhRadiusRatio*cloudRes);
        if (correspondences.size() <= 3)
        {
            std::cerr << "too few correspondences";
            return (-1);
        }

        typename pcl::PointCloud<PointT>::Ptr targetCloud(new pcl::PointCloud<PointT>());
        typename pcl::PointCloud<PointT>::Ptr sourceCloud(new pcl::PointCloud<PointT>());
        for(auto corr:correspondences)
        {
            targetCloud->push_back(targetPts->points[corr.first]);
            sourceCloud->push_back(sourcePts->points[corr.second]);
        }
#else
		typename pcl::PointCloud<PointT>::Ptr targetCloud= targetPts;
		typename pcl::PointCloud<PointT>::Ptr sourceCloud= sourcePts;
#endif

        int N = targetPts->points.size();

	    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ransacRej;
		ransacRej.setInputSource(sourceCloud);
		ransacRej.setInputTarget(targetCloud);
		ransacRej.setInlierThreshold(noise_bound);
		ransacRej.setMaximumIterations(max_iter_num);
		ransacRej.setRefineModel(true);//false

		boost::shared_ptr<pcl::Correspondences> init_corres(new pcl::Correspondences);
        for (int i=0; i< N; i++)
		{
			pcl::Correspondence cur_corr;
			cur_corr.index_query=i;
			cur_corr.index_match=i;
			init_corres->push_back(cur_corr);
		}

		boost::shared_ptr<pcl::Correspondences> final_corres(new pcl::Correspondences);

        ransacRej.setInputCorrespondences(init_corres);
        ransacRej.getCorrespondences(*final_corres);

		std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

		std::cout << "----------------------------------------------------------------------------";
		std::cout << "Begin RANSAC global coarse registration with [" << N << "] pairs of correspondence";
		std::cout << "RANSAC global coarse registration done in [" << time_used.count() * 1000.0 << "] ms."<<std::endl;
		std::cout << "[" << final_corres->size() << "] inlier correspondences found."<<std::endl;

		if(final_corres->size() >= min_inlier_num)
		{
            Eigen::Matrix4f best_tran =ransacRej.getBestTransformation();

		    tran_mat = best_tran.cast<double>();

            std::cout << "Estimated transformation by RANSAC is :\n"
					  << tran_mat;

            if (final_corres->size() >= 2 * min_inlier_num)
				return (1); //reliable
			else
				return (0); //need check
		}
		else
		{
            std::cerr << "RANSAC failed";
			return (-1);
		}
	}

	//coarse global registration using TEASER ++  (faster and more robust to outlier than RANSAC)

    int  coarse_reg_teaser(typename pcl::PointCloud<PointT>::Ptr &targetPts,
                           typename pcl::PointCloud<PointT>::Ptr &sourcePts,
                           Eigen::Matrix4d &tran_mat, float noise_bound = 0.2, int min_inlier_num = 8,
                           double salientRadiusRatio=6,double nonMaxRadiusRatio=4,
                           double normalRadiusRatio=8, double fpfhRadiusRatio=16)
    {
        //reference: https://github.com/MIT-SPARK/TEASER-plusplus
        //TEASER: Fast and Certifiable Point Cloud Registration, TRO, Heng Yang et al.

#if TEASER_ON

        int teaser_state = 0; //(failed: -1, successful[need check]: 0, successful[reliable]: 1)
        pcl::PointIndicesPtr srcKptIndices(new pcl::PointIndices), tgtKptIndices(new pcl::PointIndices);
        pcl::PointCloud<pcl::PointXYZ>::Ptr srcKeyPoints(new pcl::PointCloud<pcl::PointXYZ>()), tgtKeyPoints(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>()),tgtCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*sourcePts, *srcCloudXYZ);
        pcl::copyPointCloud(*targetPts, *tgtCloudXYZ);

//        pcl::PCDWriter pcdWriter;
//        pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/srcBlock.pcd", *srcCloudXYZ);
//        pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/dstBlock.pcd", *tgtCloudXYZ);
        std::cout<<"Src points size:"<<srcCloudXYZ->size()<<std::endl;
        std::cout<<"Dst points size:"<<tgtCloudXYZ->size()<<std::endl;
        extractISSKeyPoints(srcCloudXYZ,srcKeyPoints,srcKptIndices,salientRadiusRatio,nonMaxRadiusRatio);
        extractISSKeyPoints(tgtCloudXYZ,tgtKeyPoints,tgtKptIndices,salientRadiusRatio,nonMaxRadiusRatio);

        teaser::PointCloud srcTeaserCloud, tgtTeaserCloud;
        for (size_t i = 0; i < srcCloudXYZ->size(); ++i)
            srcTeaserCloud.push_back({static_cast<float>(srcCloudXYZ->points[i].x),
                                      static_cast<float>(srcCloudXYZ->points[i].y),
                                      static_cast<float>(srcCloudXYZ->points[i].z)});
        for (size_t i = 0; i < tgtCloudXYZ->size(); ++i)
            tgtTeaserCloud.push_back({static_cast<float>(tgtCloudXYZ->points[i].x),
                                      static_cast<float>(tgtCloudXYZ->points[i].y),
                                      static_cast<float>(tgtCloudXYZ->points[i].z)});
        teaser::PointCloud srcTeaserKeyPoints, tgtTeaserKeyPoints;
        for (size_t i = 0; i < srcKeyPoints->size(); ++i)
            srcTeaserKeyPoints.push_back({static_cast<float>(srcKeyPoints->points[i].x),
                                          static_cast<float>(srcKeyPoints->points[i].y),
                                          static_cast<float>(srcKeyPoints->points[i].z)});
        for (size_t i = 0; i < tgtKeyPoints->size(); ++i)
            tgtTeaserKeyPoints.push_back({static_cast<float>(tgtKeyPoints->points[i].x),
                                          static_cast<float>(tgtKeyPoints->points[i].y),
                                          static_cast<float>(tgtKeyPoints->points[i].z)});
//        std::cout<<"Src key points size:"<<srcTeaserKeyPoints.size()<<std::endl;
//        std::cout<<"Dst key points size:"<<tgtTeaserKeyPoints.size()<<std::endl;

        Matcher featMatcher;
        std::vector<std::pair<int, int>> correspondences;
        double cloudRes=max(computeCloudResolution(tgtCloudXYZ),computeCloudResolution(srcCloudXYZ));
        featMatcher.findFeatureCorrespondenceFPFH(tgtTeaserCloud, srcTeaserCloud,
                                                  tgtTeaserKeyPoints, srcTeaserKeyPoints,
                                                  tgtKptIndices,srcKptIndices,
                                                  correspondences,
                                                  normalRadiusRatio*cloudRes, fpfhRadiusRatio*cloudRes);
        if (correspondences.size() <= 3)
        {
            std::cerr << "too few correspondences";
            return (-1);
        }

        // Run TEASER++ registration
        // Prepare solver parameters
        teaser::RobustRegistrationSolver::Params params;
        params.noise_bound = noise_bound;
        params.cbar2 = 1.0;
        params.estimate_scaling = false;
        params.rotation_max_iterations = 100;
        params.rotation_gnc_factor = 1.4;
        params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
        params.use_max_clique = true;
        params.kcore_heuristic_threshold = 0.5;
        params.rotation_cost_threshold = 0.005; //1e-6

        // Solve with TEASER++
        std::cout << "----------------------------------------------------------------------------";
        std::cout << "Begin TEASER global coarse registration with [" << correspondences.size() << "] pairs of correspondence";
        teaser::RobustRegistrationSolver solver(params);
        std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
//		solver.solve(src, tgt);
        solver.solve(srcTeaserCloud, tgtTeaserCloud, correspondences);
        std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);

        std::cout << "TEASER global coarse registration done in [" << time_used.count() * 1000.0 << "] ms.";

        auto solution = solver.getSolution();
        std::vector<int> inliers;
        //inliers = solver.getTranslationInliers();
        inliers = solver.getRotationInliers();

        std::cout << "[" << inliers.size() << "] inlier correspondences found.";


        if (solution.valid && inliers.size() >= min_inlier_num)
        {
            tran_mat.setIdentity();
            tran_mat.block<3, 3>(0, 0) = solution.rotation;
            tran_mat.block<3, 1>(0, 3) = solution.translation;

            std::cout << "Estimated transformation by TEASER is :\n"<< tran_mat<<std::endl;

            // certificate the result here
            // teaser::DRSCertifier::Params cer_params;
            // teaser::DRSCertifier certifier(cer_params);
            // auto certification_result = certifier.certify(tran_mat,src,dst, theta);

            if (inliers.size() >= 2 * min_inlier_num)
                return (1); //reliable
            else
                return (0); //need check
        }
        else
        {
            std::cerr << "TEASER failed";
            return (-1);
        }

#endif
        return (-1);
    }


    //coarse global registration using RANSAC
    int coarse_reg_s4pcs(const typename pcl::PointCloud<PointT>::Ptr &targetPts,
                         const typename pcl::PointCloud<PointT>::Ptr &sourcePts,
                         Eigen::Matrix4d &tran_mat, float noise_bound = 0.2,
                         double salientRadiusRatio=6,double nonMaxRadiusRatio=4, double cloudRes=0)
    {
        std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();
        pcl::PointIndicesPtr srcKptIndices(new pcl::PointIndices), tgtKptIndices(new pcl::PointIndices);
        pcl::PointCloud<pcl::PointXYZ>::Ptr srcKeyPoints(new pcl::PointCloud<pcl::PointXYZ>()), tgtKeyPoints(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr srcCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>()),tgtCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*sourcePts, *srcCloudXYZ);
        pcl::copyPointCloud(*targetPts, *tgtCloudXYZ);

        extractISSKeyPoints(srcCloudXYZ,srcKeyPoints,srcKptIndices,salientRadiusRatio,nonMaxRadiusRatio,cloudRes);
        extractISSKeyPoints(tgtCloudXYZ,tgtKeyPoints,tgtKptIndices,salientRadiusRatio,nonMaxRadiusRatio,cloudRes);

        //std::cout<<"Src key points size:"<<srcTeaserKeyPoints.size()<<std::endl;
        //std::cout<<"Dst key points size:"<<tgtTeaserKeyPoints.size()<<std::endl;
        if(cloudRes==0)
             cloudRes = std::max(computeCloudResolution(tgtCloudXYZ), computeCloudResolution(srcCloudXYZ));
        std::cout<<"Cloud resolution: "<<cloudRes<<std::endl;
        std::cout << "running S4PCS..." << std::endl;
        pcl::Super4PCS<pcl::PointXYZ, pcl::PointXYZ> s4pcs;
        pcl::PointCloud<pcl::PointXYZ> final;
        s4pcs.setInputSource(srcKeyPoints);
        s4pcs.setInputTarget(tgtKeyPoints);
        s4pcs.options_.delta = cloudRes;
        s4pcs.options_.configureOverlap(0.5);
        //register
        s4pcs.align(final);

        tran_mat = s4pcs.getFinalTransformation().template cast<double>();

        double score = s4pcs.getFitnessScore();
        std::cout<<"s4pcs FitnessScore:"<<score<<std::endl;

        if(score<=noise_bound)
            return 1;
        return -1;
    }

    void coarse_reg_SACIA(const typename pcl::PointCloud<PointT>::Ptr &srcCloud,
                          const typename pcl::PointCloud<PointT>::Ptr &tgtCloud,
                          Eigen::Matrix4f &transMatrix,
                          const float minSearchDist, const float maxSearchDist, const float maxIteration)
    {
        //srcCloudXYZ.reset(new pcl::PointCloud<pcl::PointXYZ>());
        //tgtCloudXYZ.reset(new pcl::PointCloud<pcl::PointXYZ>());
//        features1.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
//        features2.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
//        normals1.reset(new pcl::PointCloud<pcl::Normal>());
//        normals2.reset(new pcl::PointCloud<pcl::Normal>());
        // compute normals
        pcl::copyPointCloud(*srcCloud, *srcCloudXYZ);
        pcl::copyPointCloud(*tgtCloud, *tgtCloudXYZ);

//    pcl::PCDWriter pcdWriter;
//    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/srcCloudXYZ.pcd", *srcCloudXYZ);
//    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/tgtCloudXYZ.pcd", *tgtCloudXYZ);

        float normalRadius = 2;
        //pcl::PointCloud<pcl::Normal>::Ptr normals1 = getNormals(srcCloudXYZ, normalRadius);
        //pcl::PointCloud<pcl::Normal>::Ptr normals2 = getNormals(tgtCloudXYZ, normalRadius);
        normals1 = getNormals(srcCloudXYZ, normalRadius);
        normals2 = getNormals(tgtCloudXYZ, normalRadius);
        std::cout<<normals1->size()<<std::endl;
        std::cout<<normals2->size()<<std::endl;
        // compute local features
        float featureRadius = 5;
        //pcl::PointCloud<pcl::FPFHSignature33>::Ptr features1 = getFeatures(srcCloudXYZ, normals1, featureRadius);
        //pcl::PointCloud<pcl::FPFHSignature33>::Ptr features2 = getFeatures(tgtCloudXYZ, normals2, featureRadius);
        features1 = getFeatures(srcCloudXYZ, normals1, featureRadius);
        features2 = getFeatures(tgtCloudXYZ, normals2, featureRadius);

        pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
        Eigen::Matrix4f final_transformation;
        sac_ia.setInputSource(srcCloudXYZ);
        sac_ia.setSourceFeatures(features1);
        sac_ia.setInputTarget(tgtCloudXYZ);
        sac_ia.setTargetFeatures(features2);
        sac_ia.setMaximumIterations(100);
        sac_ia.setMinSampleDistance(minSearchDist);
        sac_ia.setMaxCorrespondenceDistance(maxSearchDist);
        pcl::PointCloud<pcl::PointXYZ> finalcloud;
        sac_ia.align(finalcloud);
        sac_ia.getCorrespondenceRandomness();

        transMatrix = sac_ia.getFinalTransformation();
    }

	// registration basic interface
	double base_align(typename pcl::Registration<PointT, PointT>::Ptr registration,
					  const typename pcl::PointCloud<PointT>::Ptr &target_cloud,
					  const typename pcl::PointCloud<PointT>::Ptr &source_cloud,
					  typename pcl::PointCloud<PointT>::Ptr &transformed_source,
					  Eigen::Matrix4d &Trans)
	{
		std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

		registration->setInputTarget(target_cloud);
		registration->setInputSource(source_cloud);

		registration->align(*transformed_source);
		Trans = registration->getFinalTransformation().template cast<double>();

		double fitness_score = registration->getFitnessScore();

		std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
		std::cout << "registration time : " << time_used.count() * 1000.0 << "[ms]";
		std::cout << "fitness score: " << fitness_score;
		std::cout << "Transformation:\n " << Trans;

		return fitness_score;
	}

	//for voxel based fast gicp (by koide3)
	double base_align(typename pcl::Registration<PointT, PointT, double>::Ptr registration,
					  const typename pcl::PointCloud<PointT>::Ptr &target_cloud,
					  const typename pcl::PointCloud<PointT>::Ptr &source_cloud,
					  typename pcl::PointCloud<PointT>::Ptr &transformed_source,
					  Eigen::Matrix4d &Trans, bool only_estimate_source_covariance = false)
	{
		std::chrono::steady_clock::time_point tic = std::chrono::steady_clock::now();

		registration->setInputTarget(target_cloud); //covariance is calculated here
		registration->setInputSource(source_cloud);

		std::chrono::steady_clock::time_point toc_1 = std::chrono::steady_clock::now();

		registration->align(*transformed_source);
		Trans = registration->getFinalTransformation().template cast<double>();

		double fitness_score = registration->getFitnessScore();

		std::chrono::steady_clock::time_point toc_2 = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_used_covariance = std::chrono::duration_cast<std::chrono::duration<double>>(toc_1 - tic);
		std::chrono::duration<double> time_used_align = std::chrono::duration_cast<std::chrono::duration<double>>(toc_2 - toc_1);
		std::cout << "covariance calculation done in [" << time_used_covariance.count() * 1000.0 << "] ms.";
		std::cout << "registration done in [" << time_used_align.count() * 1000.0 << "] ms.";
		std::cout << "fitness score: " << fitness_score;
		std::cout << "Transformation:\n " << Trans;

		return fitness_score;
	}

	//pertubate the point cloud with a small translation
	void pertubate_cloud(const typename pcl::PointCloud<PointT>::Ptr &cloud_in,
						 typename pcl::PointCloud<PointT>::Ptr &cloud_out,
						 float pertubate_value, std::vector<float> &pertubate_vector)
	{
		pertubate_vector.resize(3);
		pertubate_vector[0] = 0.5 * pertubate_value - pertubate_value * ((double)rand() / RAND_MAX);
		pertubate_vector[1] = 0.5 * pertubate_value - pertubate_value * ((double)rand() / RAND_MAX);
		pertubate_vector[2] = 0.5 * pertubate_value - pertubate_value * ((double)rand() / RAND_MAX);

		for (size_t i = 0; i < cloud_in->size(); i++)
		{
			PointT pt;
			pt.x = cloud_in->points[i].x + pertubate_vector[0];
			pt.y = cloud_in->points[i].y + pertubate_vector[1];
			pt.z = cloud_in->points[i].z + pertubate_vector[2];
			cloud_out->push_back(pt);
		}
        std::cout << "The pertubation vector is:  X " << pertubate_vector[0] << " , Y " << pertubate_vector[1] << " , Z " << pertubate_vector[2];
	}


	//Target point cloud: block1
	//Source point cloud: block2
	bool assign_source_target_cloud(const CloudBlockPtr &block_1, const CloudBlockPtr &block_2, Constraint &registrationCons)
	{
        registrationCons.block1 = block_1; //target
        registrationCons.block2 = block_2; //source
        return true;
	}

	//interface for the implement of basic icp algorithm using pcl
	int pcl_icp(Constraint &registration_cons,
				int max_iter_num, float dis_thre_unit,
				DistMetricType metrics, CorresEstimationType ce, TransformEstimationType te,
				bool use_reciprocal_correspondence, bool use_trimmed_rejector, float neighbor_radius = 2.0,
				Eigen::Matrix4d initial_guess = Eigen::Matrix4d::Identity(), bool apply_intersection_filter = true,
				float fitness_score_thre = 10.0)
	{
		CFilter<PointT> cfilter;

		int process_code = 0;

		typename pcl::PointCloud<PointT>::Ptr cloud_t_down(new pcl::PointCloud<PointT>);
		typename pcl::PointCloud<PointT>::Ptr cloud_s_down(new pcl::PointCloud<PointT>);
		typename pcl::PointCloud<PointT>::Ptr cloud_s_guess(new pcl::PointCloud<PointT>);
		typename pcl::PointCloud<PointT>::Ptr cloud_s_tran(new pcl::PointCloud<PointT>);

        registration_cons.block1->cloneCloud(cloud_t_down, true); //target
        registration_cons.block2->cloneCloud(cloud_s_down, true); //source

		Bounds intersection_bbx, source_guess_bbx;
		bool apply_source_initial_guess = false;
		if (!initial_guess.isIdentity(1e-6))
		{
			//Transform the Source pointcloud
			pcl::transformPointCloudWithNormals(*cloud_s_down, *cloud_s_guess, initial_guess);
			cfilter.get_cloud_bbx(cloud_s_guess, source_guess_bbx);
			cfilter.get_intersection_bbx(registration_cons.block1->_localBound, source_guess_bbx, intersection_bbx);
			std::cout << "Apply initial guess transformation\n"
					  << initial_guess;
			apply_source_initial_guess = true;
		}
		else
			cfilter.get_intersection_bbx(registration_cons.block1->_localBound, registration_cons.block2->_localBound, intersection_bbx);

		if (apply_intersection_filter)
			cfilter.get_cloud_pair_intersection(intersection_bbx, cloud_t_down, cloud_s_guess);

		Eigen::Matrix4d Trans_t_sg;
		double fitness_score;
		fitness_score = icp_registration(cloud_s_guess, cloud_t_down,
										 cloud_s_tran, Trans_t_sg, metrics, ce, te,
										 use_reciprocal_correspondence, use_trimmed_rejector,
										 max_iter_num, dis_thre_unit, neighbor_radius);

		if (fitness_score > fitness_score_thre)
			process_code = -3;
		else
			process_code = 1;

		if (apply_source_initial_guess)
			Trans_t_sg = Trans_t_sg * initial_guess;

		registration_cons.Trans1_2 = Trans_t_sg;

		pcl::PointCloud<PointT>().swap(*cloud_s_guess);
		pcl::PointCloud<PointT>().swap(*cloud_s_tran);
		pcl::PointCloud<PointT>().swap(*cloud_t_down);
		pcl::PointCloud<PointT>().swap(*cloud_s_down);

		return process_code;
	}


  protected:
  private:
	void batch_transform_feature_points(typename pcl::PointCloud<PointT>::Ptr pc_ground, typename pcl::PointCloud<PointT>::Ptr pc_pillar,
										typename pcl::PointCloud<PointT>::Ptr pc_beam, typename pcl::PointCloud<PointT>::Ptr pc_facade,
										typename pcl::PointCloud<PointT>::Ptr pc_roof, typename pcl::PointCloud<PointT>::Ptr pc_vertex,
										Eigen::Matrix4d &Tran)
	{
		pcl::transformPointCloudWithNormals(*pc_ground, *pc_ground, Tran);
		pcl::transformPointCloudWithNormals(*pc_pillar, *pc_pillar, Tran);
		pcl::transformPointCloudWithNormals(*pc_beam, *pc_beam, Tran);
		pcl::transformPointCloudWithNormals(*pc_facade, *pc_facade, Tran);
		pcl::transformPointCloudWithNormals(*pc_roof, *pc_roof, Tran);
		pcl::transformPointCloudWithNormals(*pc_vertex, *pc_vertex, Tran);
	}

	//Time complexity of kdtree (in this case, the target point cloud [n points] is used for construct the tree while each point in source point cloud acts as a query point)
	//build tree: O(nlogn) ---> so it's better to build the tree only once
	//searching 1-nearest neighbor: O(logn) in average ---> so we can bear a larger number of target points
	bool determine_corres(typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
						  const typename pcl::search::KdTree<PointT>::Ptr &target_kdtree, float dis_thre,
						  boost::shared_ptr<pcl::Correspondences> &Corr_f, bool normal_shooting_on, bool normal_check = true,
						  float angle_thre_degree = 40, bool duplicate_check = true, int K_filter_distant_point = 500)
	{
		int K_min = 3;
		float filter_dis_times = 2.5;
		int normal_shooting_candidate_count = 10;

		typename pcl::registration::CorrespondenceEstimation<PointT, PointT> corr_est; //for nearest neighbor searching

		//CorrespondenceEstimationNormalShooting computes correspondences as points in the target cloud which have minimum distance to normals computed on the input cloud
		typename pcl::registration::CorrespondenceEstimationNormalShooting<PointT, PointT, PointT> corr_est_ns; //for normal shooting searching

		pcl::registration::CorrespondenceRejectorDistance corr_rej_dist;

		// Add a median distance rejector (deprecated)
		// pcl::registration::CorrespondenceRejectorMedianDistance::Ptr corr_rej_med (new pcl::registration::CorrespondenceRejectorMedianDistance);
		// rej_med->setMedianFactor (4.0);
		// reg.addCorrespondenceRejector (rej_med);

		boost::shared_ptr<pcl::Correspondences> Corr(new pcl::Correspondences);

		typename pcl::PointCloud<PointT>::Ptr Source_Cloud_f(new pcl::PointCloud<PointT>);
		typename pcl::PointCloud<PointT>::Ptr Target_Cloud_f(new pcl::PointCloud<PointT>); //target point cloud would never change

		if (Source_Cloud->points.size() >= K_min &&
			Target_Cloud->points.size() >= K_min)
		{
			if (normal_shooting_on) // Normal Shooting
			{
				corr_est_ns.setInputSource(Source_Cloud);
				corr_est_ns.setInputTarget(Target_Cloud);
				corr_est_ns.setSourceNormals(Source_Cloud);
				corr_est_ns.setSearchMethodTarget(target_kdtree, true);					  //saving the time of rebuilding kd-tree
				corr_est_ns.setKSearch(normal_shooting_candidate_count);				  // Among the K nearest neighbours find the one with minimum perpendicular distance to the normal
				corr_est_ns.determineCorrespondences(*Corr, filter_dis_times * dis_thre); //base on KDtreeNSearch
																						  //corr_est_ns.determineReciprocalCorrespondences(*Corr);
			}
			else //Nearest Neighbor
			{
				corr_est.setInputCloud(Source_Cloud);
				corr_est.setInputTarget(Target_Cloud);
				corr_est.setSearchMethodTarget(target_kdtree, true);				   //saving the time of rebuilding kd-tree
				corr_est.determineCorrespondences(*Corr, filter_dis_times * dis_thre); //base on KDtreeNSearch
																					   //corr_est.determineReciprocalCorrespondences(*Corr);
			}

			// std::chrono::steady_clock::time_point toc = std::chrono::steady_clock::now();
			// std::chrono::duration<double> determine_corr_time = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic);
			// LOG(WARNING) << "Correspondence time 1: [" << determine_corr_time.count() * 1000 << "] ms";

			//Filter outlier source points (they would not appear throughout the registration anymore)
#if 1
			if (Source_Cloud->points.size() >= K_filter_distant_point)
			{
				int count = 0;

				//duplicate check -> just keep one source point corresponding to one target point
				std::vector<unsigned int> duplicate_check_table(Target_Cloud->points.size(), 0);

				for (auto iter = Corr->begin(); iter != Corr->end();)
				{
					int s_index, t_index;
					s_index = (*iter).index_query;
					t_index = (*iter).index_match;

					if (t_index != -1)
					{
						if (duplicate_check && duplicate_check_table[t_index] > 0)
							iter = Corr->erase(iter);
						else
						{
							duplicate_check_table[t_index]++;

							Source_Cloud_f->points.push_back(Source_Cloud->points[s_index]);
							//Target_Cloud_f->points.push_back(Target_Cloud->points[t_index]);
							(*iter).index_query = count;
							//(*iter).index_match = count;
							count++;
							iter++;
						}
					}
					else
						iter++;
				}
				Corr->resize(count);

				Source_Cloud_f->points.swap(Source_Cloud->points);
				//Target_Cloud_f->points.swap(Target_Cloud->points);
				std::vector<unsigned int>().swap(duplicate_check_table);
			}
#endif
			corr_rej_dist.setInputCorrespondences(Corr);
			corr_rej_dist.setMaximumDistance(dis_thre);
			corr_rej_dist.getCorrespondences(*Corr_f);

			if (normal_check) //only for planar points
			{
				int count = 0;
				//Normal direction consistency check
				for (auto iter = Corr_f->begin(); iter != Corr_f->end();)
				{
					int s_index, t_index;
					s_index = (*iter).index_query;
					t_index = (*iter).index_match;

					if (t_index != -1)
					{

						Eigen::Vector3d n1;
						Eigen::Vector3d n2;
						n1 << Source_Cloud->points[s_index].normal[0], Source_Cloud->points[s_index].normal[1], Source_Cloud->points[s_index].normal[2];
						n2 << Target_Cloud->points[t_index].normal[0], Target_Cloud->points[t_index].normal[1], Target_Cloud->points[t_index].normal[2];

						float cos_intersection_angle = std::abs(n1.dot(n2)); // n1.norm()=n2.norm()=1

						if (cos_intersection_angle < cos(angle_thre_degree / 180.0 * M_PI))
						{
							count++;
							iter = Corr_f->erase(iter);
						}
						else
							iter++;
					}
					else
						iter++;
				}
				//LOG(INFO) << count << " correspondences are rejected by normal check";
			}
		}
		else
			return 0;
		return 1;
	}

	bool add_corre_points(const typename pcl::PointCloud<PointT>::Ptr &sc, const typename pcl::PointCloud<PointT>::Ptr &tc, boost::shared_ptr<pcl::Correspondences> &corrs,
						  const typename pcl::PointCloud<PointT>::Ptr &pc_sc_temp, const typename pcl::PointCloud<PointT>::Ptr &pc_tc_temp)
	{
		for (int i = 0; i < (*corrs).size(); i++)
		{
			int s_index, t_index;
			s_index = (*corrs)[i].index_query;
			t_index = (*corrs)[i].index_match;

			if (t_index != -1)
			{
				pc_sc_temp->points.push_back(sc->points[s_index]);
				pc_tc_temp->points.push_back(tc->points[t_index]);
			}
		}
		return 1;
	}

	void update_corr_dist_thre(float &dis_thre_ground, float &dis_thre_pillar, float &dis_thre_beam,
							   float &dis_thre_facade, float &dis_thre_roof, float &dis_thre_vertex,
							   float dis_thre_update_rate, float dis_thre_min)

	{
		dis_thre_ground = max_(1.0 * dis_thre_ground / dis_thre_update_rate, dis_thre_min);
		dis_thre_facade = max_(1.0 * dis_thre_facade / dis_thre_update_rate, dis_thre_min);
		dis_thre_roof = max_(1.0 * dis_thre_roof / dis_thre_update_rate, dis_thre_min);
		dis_thre_pillar = max_(1.0 * dis_thre_pillar / dis_thre_update_rate, dis_thre_min);
		dis_thre_beam = max_(1.0 * dis_thre_beam / dis_thre_update_rate, dis_thre_min);
		dis_thre_vertex = max_(1.0 * dis_thre_vertex / dis_thre_update_rate, dis_thre_min);
	}

	//brief: entrance to mulls transformation estimation
	bool multi_metrics_lls_tran_estimation(const typename pcl::PointCloud<PointT>::Ptr &Source_Ground, const typename pcl::PointCloud<PointT>::Ptr &Target_Ground, boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
										   const typename pcl::PointCloud<PointT>::Ptr &Source_Pillar, const typename pcl::PointCloud<PointT>::Ptr &Target_Pillar, boost::shared_ptr<pcl::Correspondences> &Corr_Pillar,
										   const typename pcl::PointCloud<PointT>::Ptr &Source_Beam, const typename pcl::PointCloud<PointT>::Ptr &Target_Beam, boost::shared_ptr<pcl::Correspondences> &Corr_Beam,
										   const typename pcl::PointCloud<PointT>::Ptr &Source_Facade, const typename pcl::PointCloud<PointT>::Ptr &Target_Facade, boost::shared_ptr<pcl::Correspondences> &Corr_Facade,
										   const typename pcl::PointCloud<PointT>::Ptr &Source_Roof, const typename pcl::PointCloud<PointT>::Ptr &Target_Roof, boost::shared_ptr<pcl::Correspondences> &Corr_Roof,
										   const typename pcl::PointCloud<PointT>::Ptr &Source_Vertex, const typename pcl::PointCloud<PointT>::Ptr &Target_Vertex, boost::shared_ptr<pcl::Correspondences> &Corr_Vertex,
										   Vector6d &unknown_x, Matrix6d &cofactor_matrix, int iter_num, std::string weight_strategy, float z_xy_balance_ratio = 1.0,
										   float pt2pt_residual_window = 0.1, float pt2pl_residual_window = 0.1, float pt2li_residual_window = 0.1)
	{
		Matrix6d ATPA;
		Vector6d ATPb;
		ATPA.setZero();
		ATPb.setZero();

		//Deal with weight (contribution of each correspondence in the transformation estimation)
		float w_ground = 1.0, w_facade = 1.0, w_roof = 1.0, w_pillar = 1.0, w_beam = 1.0, w_vertex = 1.0; //initialization

		int m1 = (*Corr_Ground).size() + (*Corr_Roof).size();
		int m2 = (*Corr_Facade).size();
		int m3 = (*Corr_Pillar).size();
		int m4 = (*Corr_Beam).size();
		int m5 = (*Corr_Vertex).size();

		if (weight_strategy[0] == '1') //x,y,z directional balanced weighting (guarantee the observability of the scene)
		{
			w_ground = max_(0.01, z_xy_balance_ratio * (m2 + 2 * m3 - m4) / (0.0001 + 2.0 * m1)); // x <-> y <-> z
			w_roof = w_ground;
			w_facade = 1.0;
			w_pillar = 1.0;
			w_beam = 1.0;
			w_vertex = 1.0;
		}

		bool dist_weight = false;
		bool residual_weight = false;
		bool intensity_weight = false;
		int iter_thre = 2; //the residual based weighting would only be applied after this number of iteration
		if (weight_strategy[1] == '1' && iter_num > iter_thre) //weight according to residual
			residual_weight = true;
		if (weight_strategy[2] == '1') //weight according to distance
			dist_weight = true;
		if (weight_strategy[3] == '1') //weight according to intensity
			intensity_weight = true;

		//point to plane
		pt2pl_lls_summation(Source_Ground, Target_Ground, Corr_Ground, ATPA, ATPb, iter_num, w_ground, dist_weight, residual_weight, intensity_weight, pt2pl_residual_window);
		pt2pl_lls_summation(Source_Facade, Target_Facade, Corr_Facade, ATPA, ATPb, iter_num, w_facade, dist_weight, residual_weight, intensity_weight, pt2pl_residual_window);
		pt2pl_lls_summation(Source_Roof, Target_Roof, Corr_Roof, ATPA, ATPb, iter_num, w_roof, dist_weight, residual_weight, intensity_weight, pt2pl_residual_window);
		//point to line
		pt2li_lls_pri_direction_summation(Source_Pillar, Target_Pillar, Corr_Pillar, ATPA, ATPb, iter_num, w_pillar, dist_weight, residual_weight, intensity_weight, pt2li_residual_window);
		pt2li_lls_pri_direction_summation(Source_Beam, Target_Beam, Corr_Beam, ATPA, ATPb, iter_num, w_beam, dist_weight, residual_weight, intensity_weight, pt2li_residual_window);
		//point to point
		pt2pt_lls_summation(Source_Vertex, Target_Vertex, Corr_Vertex, ATPA, ATPb, iter_num, w_vertex, dist_weight, residual_weight, intensity_weight, pt2pt_residual_window);

		//ATPA is a symmetric matrix
		ATPA.coeffRef(6) = ATPA.coeffRef(1);
		ATPA.coeffRef(12) = ATPA.coeffRef(2);
		ATPA.coeffRef(13) = ATPA.coeffRef(8);
		ATPA.coeffRef(18) = ATPA.coeffRef(3);
		ATPA.coeffRef(19) = ATPA.coeffRef(9);
		ATPA.coeffRef(20) = ATPA.coeffRef(15);
		ATPA.coeffRef(24) = ATPA.coeffRef(4);
		ATPA.coeffRef(25) = ATPA.coeffRef(10);
		ATPA.coeffRef(26) = ATPA.coeffRef(16);
		ATPA.coeffRef(27) = ATPA.coeffRef(22);
		ATPA.coeffRef(30) = ATPA.coeffRef(5);
		ATPA.coeffRef(31) = ATPA.coeffRef(11);
		ATPA.coeffRef(32) = ATPA.coeffRef(17);
		ATPA.coeffRef(33) = ATPA.coeffRef(23);
		ATPA.coeffRef(34) = ATPA.coeffRef(29);

		//LOG(INFO) << "ATPA=" << std::endl << ATPA;
		//LOG(INFO) << "ATPb=" << std::endl << ATPb;

		// Solve A*x = b  x= (ATPA)^(-1)ATPb
		// x: tx ty tz alpha beta gamma (alpha beta gamma corresponding to roll, pitch and yaw)
		// the approximated rotation matrix is
		// |   1    -gamma   beta  |
		// | gamma     1    -alpha |
		// | -beta   alpha     1   |
		//reference: A Review of Point Cloud Registration Algorithms for Mobile Robotics, Appendix

		unknown_x = ATPA.inverse() * ATPb;

		Eigen::Vector3d euler_angle(unknown_x(3), unknown_x(4), unknown_x(5));
		Eigen::Matrix3d Jacobi;
		get_quat_euler_jacobi(euler_angle, Jacobi);

		//Qxx=(ATPA)^-1
		//information matrix = Dxx^(-1)=Qxx^(-1)/(sigma_post)^2=ATPA/(sigma_post)^2
		cofactor_matrix = ATPA.inverse();

		//convert to the cofactor matrix with regard to quaternion from euler angle
		cofactor_matrix.block<3, 3>(3, 3) = Jacobi * cofactor_matrix.block<3, 3>(3, 3) * Jacobi.transpose();
		cofactor_matrix.block<3, 3>(0, 3) = cofactor_matrix.block<3, 3>(0, 3) * Jacobi.transpose();
		cofactor_matrix.block<3, 3>(3, 0) = Jacobi * cofactor_matrix.block<3, 3>(3, 0);

		return 1;
	}

	//Linearization of Rotation Matrix
	//R = I + (alpha, beta, gamma) ^
	//  = | 1      -gamma    beta |
	//    | gamma   1       -alpha|
	//    |-beta    alpha     1   |

	//point-to-point LLS
	bool pt2pt_lls_summation(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
							 boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
							 float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false,
							 bool intensity_weight_or_not = false,
							 float residual_window_size = 0.1)
	{
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;

			if (t_index != -1)
			{

				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;

				float pi = Source_Cloud->points[s_index].intensity;
				float qi = Target_Cloud->points[t_index].intensity;

				float dx = px - qx;
				float dy = py - qy;
				float dz = pz - qz;

				float wx, wy, wz;
				wx = weight;

				float dist = std::sqrt(qx * qx + qy * qy + qz * qz);

				if (dist_weight_or_not)
					wx = wx * get_weight_by_dist_adaptive(dist, iter_num);
				//wx = wx * get_weight_by_dist(dist);

				if (residual_weight_or_not)
					wx = wx * get_weight_by_residual(std::sqrt(dx * dx + dy * dy + dz * dz), residual_window_size);
				if (intensity_weight_or_not)
					wx = wx * get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

				wy = wx;
				wz = wx;

				// unknown x: [tx ty tz alpha beta gama]

				//    0  1  2  3  4  5
				//    6  7  8  9 10 11
				//   12 13 14 15 16 17
				//   18 19 20 21 22 23
				//   24 25 26 27 28 29
				//   30 31 32 33 34 35

				ATPA.coeffRef(0) += wx;
				ATPA.coeffRef(1) += 0;
				ATPA.coeffRef(2) += 0;
				ATPA.coeffRef(3) += 0;
				ATPA.coeffRef(4) += wx * pz;
				ATPA.coeffRef(5) += (-wx * py);
				ATPA.coeffRef(7) += wy;
				ATPA.coeffRef(8) += 0;
				ATPA.coeffRef(9) += (-wy * pz);
				ATPA.coeffRef(10) += 0;
				ATPA.coeffRef(11) += wy * px;
				ATPA.coeffRef(14) += wz;
				ATPA.coeffRef(15) += wz * py;
				ATPA.coeffRef(16) += (-wz * px);
				ATPA.coeffRef(17) += 0;
				ATPA.coeffRef(21) += wy * pz * pz + wz * py * py;
				ATPA.coeffRef(22) += (-wz * px * py);
				ATPA.coeffRef(23) += (-wy * px * pz);
				ATPA.coeffRef(28) += wx * pz * pz + wz * px * px;
				ATPA.coeffRef(29) += (-wx * py * pz);
				ATPA.coeffRef(35) += wx * py * py + wy * px * px;

				ATPb.coeffRef(0) += (-wx * dx);
				ATPb.coeffRef(1) += (-wy * dy);
				ATPb.coeffRef(2) += (-wz * dz);
				ATPb.coeffRef(3) += wy * pz * dy - wz * py * dz;
				ATPb.coeffRef(4) += wz * px * dz - wx * pz * dx;
				ATPb.coeffRef(5) += wx * py * dx - wy * px * dy;
			}
		}

		return 1;
	}

	//point-to-plane LLS
	bool pt2pl_lls_summation(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
							 boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
							 float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false,
							 bool intensity_weight_or_not = false,
							 float residual_window_size = 0.1)
	{
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;

			if (t_index != -1)
			{
				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;
				float ntx = Target_Cloud->points[t_index].normal_x;
				float nty = Target_Cloud->points[t_index].normal_y;
				float ntz = Target_Cloud->points[t_index].normal_z;

				float pi = Source_Cloud->points[s_index].intensity;
				float qi = Target_Cloud->points[t_index].intensity;

				float w = weight;

				float a = ntz * py - nty * pz;
				float b = ntx * pz - ntz * px;
				float c = nty * px - ntx * py;

				float d = ntx * qx + nty * qy + ntz * qz - ntx * px - nty * py - ntz * pz;

				float dist = std::sqrt(qx * qx + qy * qy + qz * qz);

				if (dist_weight_or_not)
					w = w * get_weight_by_dist_adaptive(dist, iter_num);
				//w = w * get_weight_by_dist(dist);

				if (residual_weight_or_not)
					w = w * get_weight_by_residual(std::abs(d), residual_window_size);
				//w = w * get_weight_by_residual_general(std::abs(d), residual_window_size, 1.0);

				if (intensity_weight_or_not)
					w = w * get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

				(*Corr)[i].weight = w;


				//    0  1  2  3  4  5
				//    6  7  8  9 10 11
				//   12 13 14 15 16 17
				//   18 19 20 21 22 23
				//   24 25 26 27 28 29
				//   30 31 32 33 34 35

				ATPA.coeffRef(0) += w * ntx * ntx;
				ATPA.coeffRef(1) += w * ntx * nty;
				ATPA.coeffRef(2) += w * ntx * ntz;
				ATPA.coeffRef(3) += w * a * ntx;
				ATPA.coeffRef(4) += w * b * ntx;
				ATPA.coeffRef(5) += w * c * ntx;
				ATPA.coeffRef(7) += w * nty * nty;
				ATPA.coeffRef(8) += w * nty * ntz;
				ATPA.coeffRef(9) += w * a * nty;
				ATPA.coeffRef(10) += w * b * nty;
				ATPA.coeffRef(11) += w * c * nty;
				ATPA.coeffRef(14) += w * ntz * ntz;
				ATPA.coeffRef(15) += w * a * ntz;
				ATPA.coeffRef(16) += w * b * ntz;
				ATPA.coeffRef(17) += w * c * ntz;
				ATPA.coeffRef(21) += w * a * a;
				ATPA.coeffRef(22) += w * a * b;
				ATPA.coeffRef(23) += w * a * c;
				ATPA.coeffRef(28) += w * b * b;
				ATPA.coeffRef(29) += w * b * c;
				ATPA.coeffRef(35) += w * c * c;

				ATPb.coeffRef(0) += w * d * ntx;
				ATPb.coeffRef(1) += w * d * nty;
				ATPb.coeffRef(2) += w * d * ntz;
				ATPb.coeffRef(3) += w * d * a;
				ATPb.coeffRef(4) += w * d * b;
				ATPb.coeffRef(5) += w * d * c;
			}
		}

		return 1;
	}

	//point-to-line LLS (calculated using primary direction vector), used now
	//the normal vector here actually stores the primary direcyion vector (for easier calculation of the residual)
	bool pt2li_lls_pri_direction_summation(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
										   boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
										   float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false,
										   bool intensity_weight_or_not = false,
										   float residual_window_size = 0.1)
	{
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;

			if (t_index != -1)
			{
				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;

				//primary direction (in this case, we save the primary direction of linear feature points in the normal vector)
				float vx = Target_Cloud->points[t_index].normal_x;
				float vy = Target_Cloud->points[t_index].normal_y;
				float vz = Target_Cloud->points[t_index].normal_z;

				//LOG(INFO) << nx << "," << ny<< "," <<nz;

				float pi = Source_Cloud->points[s_index].intensity;
				float qi = Target_Cloud->points[t_index].intensity;

				float dx = px - qx;
				float dy = py - qy;
				float dz = pz - qz;

				Eigen::Matrix<double, 3, 6> Amat;
				Eigen::Matrix<double, 3, 1> bvec;
				Eigen::Matrix<double, 3, 1> evec;
				Eigen::Matrix<double, 3, 3> Imat;
				Eigen::Matrix<double, 3, 3> Wmat;
				Imat.setIdentity();
				Wmat.setIdentity();

				Amat(0, 0) = 0;
				Amat(0, 1) = -vz;
				Amat(0, 2) = vy;
				Amat(0, 3) = vy * py + vz * pz;
				Amat(0, 4) = -vy * px;
				Amat(0, 5) = -vz * px;
				Amat(1, 0) = vz;
				Amat(1, 1) = 0;
				Amat(1, 2) = -vx;
				Amat(1, 3) = -vx * py;
				Amat(1, 4) = vz * pz + vx * px;
				Amat(1, 5) = -vz * py;
				Amat(2, 0) = -vy;
				Amat(2, 1) = vx;
				Amat(2, 2) = 0;
				Amat(2, 3) = -vx * pz;
				Amat(2, 4) = -vy * pz;
				Amat(2, 5) = vx * px + vy * py;

				bvec(0, 0) = -vy * dz + vz * dy;
				bvec(1, 0) = -vz * dx + vx * dz;
				bvec(2, 0) = -vx * dy + vy * dx;

				//evec = (Amat * (Amat.transpose() * Amat).inverse() * Amat.transpose() - Imat) * bvec; //posterior residual
				//we'd like to directly use the prior residual
				float ex = std::abs(bvec(0, 0));
				float ey = std::abs(bvec(1, 0));
				float ez = std::abs(bvec(2, 0));
				float ed = std::sqrt(ex * ex + ey * ey + ez * ez);

				float wx, wy, wz, w;
				wx = weight;

				float dist = std::sqrt(qx * qx + qy * qy + qz * qz);

				if (dist_weight_or_not)
					//wx *= get_weight_by_dist(dist);
					wx *= get_weight_by_dist_adaptive(dist, iter_num);

				if (intensity_weight_or_not)
					wx *= get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

				if (residual_weight_or_not)
				{
					// wx *= get_weight_by_residual(ex, residual_window_size);
					// wy *= get_weight_by_residual(ey, residual_window_size);
					// wz *= get_weight_by_residual(ez, residual_window_size);

					wx = wx * get_weight_by_residual(ed, residual_window_size); //original huber
																				// wx = wx * get_weight_by_residual_general(ed, residual_window_size, 1.0);
				}
				wy = wx;
				wz = wx;
				(*Corr)[i].weight = wx;

				Wmat(0, 0) = std::sqrt(wx);
				Wmat(1, 1) = std::sqrt(wy);
				Wmat(2, 2) = std::sqrt(wz);

				Amat = Wmat * Amat;
				bvec = Wmat * bvec;

				for (int j = 0; j < 6; j++)
				{
					for (int k = j; k < 6; k++)
						ATPA(j, k) += ((Amat.block<3, 1>(0, j)).transpose() * (Amat.block<3, 1>(0, k)));
				}
				for (int j = 0; j < 6; j++)
					ATPb.coeffRef(j) += ((Amat.block<3, 1>(0, j)).transpose() * (bvec));
			}
		}
		return 1;
	}


	bool ground_3dof_lls_tran_estimation(const typename pcl::PointCloud<PointT>::Ptr &Source_Ground,
										 const typename pcl::PointCloud<PointT>::Ptr &Target_Ground,
										 boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
										 Eigen::Vector3d &unknown_x, Eigen::Matrix3d &cofactor_matrix,
										 int iter_num, std::string weight_strategy)
	{
		Eigen::Matrix3d ATPA;
		Eigen::Vector3d ATPb;
		ATPA.setZero();
		ATPb.setZero();

		bool dist_weight = false;
		bool residual_weight = false;
		bool intensity_weight = false;

		if (weight_strategy[1] == '1') //weight according to residual
			residual_weight = true;

		if (weight_strategy[2] == '1') //weight according to distance
			dist_weight = true;

		if (weight_strategy[3] == '1') //weight according to intensity
			intensity_weight = true;

		pt2pl_ground_3dof_lls_summation(Source_Ground, Target_Ground, Corr_Ground, ATPA, ATPb, iter_num, 1.0, dist_weight, residual_weight, intensity_weight);

		//ATPA is a symmetric matrix
		//    0  1  2
		//   [3] 4  5
		//   [6][7] 8
		ATPA.coeffRef(3) = ATPA.coeffRef(1);
		ATPA.coeffRef(6) = ATPA.coeffRef(2);
		ATPA.coeffRef(7) = ATPA.coeffRef(5);

		// Solve A*x = b  x= (ATPA)^(-1)ATPb
		// x: tx ty tz alpha beta gamma
		unknown_x = ATPA.inverse() * ATPb;

		return 1;
	}

	//ground 3dof : roll, pitch, z
	bool pt2pl_ground_3dof_lls_summation(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
										 boost::shared_ptr<pcl::Correspondences> &Corr, Eigen::Matrix3d &ATPA, Eigen::Vector3d &ATPb, int iter_num,
										 float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false, bool intensity_weight_or_not = false,
										 float residual_window_size = 0.1)
	{
		//unknown : roll (alpha), picth (beta) and tz
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;

			if (t_index != -1)
			{

				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;
				float ntx = Target_Cloud->points[t_index].normal_x;
				float nty = Target_Cloud->points[t_index].normal_y;
				float ntz = Target_Cloud->points[t_index].normal_z;

				float pi = Source_Cloud->points[s_index].intensity;
				float qi = Target_Cloud->points[t_index].intensity;

				float w = weight;

				float a = ntz * py - nty * pz;
				float b = ntx * pz - ntz * px;

				float d = ntx * qx + nty * qy + ntz * qz - ntx * px - nty * py - ntz * pz;

				float dist = std::sqrt(qx * qx + qy * qy + qz * qz);
				if (dist_weight_or_not)
					w = w * get_weight_by_dist_adaptive(dist, iter_num);
				//w = w * get_weight_by_dist(dist);

				if (residual_weight_or_not)
					w = w * get_weight_by_residual(std::abs(d), residual_window_size);

				if (intensity_weight_or_not)
					w = w * get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

				//    0  1  2
				//    3  4  5
				//    6  7  8

				(*Corr)[i].weight = w;

				ATPA.coeffRef(0) += w * a * a;
				ATPA.coeffRef(1) += w * a * b;
				ATPA.coeffRef(2) += w * a * ntz;
				ATPA.coeffRef(4) += w * b * b;
				ATPA.coeffRef(5) += w * b * ntz;
				ATPA.coeffRef(8) += w * ntz * ntz;

				ATPb.coeffRef(0) += w * d * a;
				ATPb.coeffRef(1) += w * d * b;
				ATPb.coeffRef(2) += w * d * ntz;
			}
		}

		return 1;
	}

	//point-to-line LLS (calculated using normal vector) , Deprecated
	bool pt2li_lls_summation(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
							 boost::shared_ptr<pcl::Correspondences> &Corr, Matrix6d &ATPA, Vector6d &ATPb, int iter_num,
							 float weight, bool dist_weight_or_not = false, bool residual_weight_or_not = false,
							 bool intensity_weight_or_not = false,
							 float residual_window_size = 0.1)
	{
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;

			if (t_index != -1)
			{
				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;
				float ntx = Target_Cloud->points[t_index].normal[0];
				float nty = Target_Cloud->points[t_index].normal[1];
				float ntz = Target_Cloud->points[t_index].normal[2];
				float nsx = Source_Cloud->points[s_index].normal[0];
				float nsy = Source_Cloud->points[s_index].normal[1];
				float nsz = Source_Cloud->points[s_index].normal[2];

				float pi = Source_Cloud->points[s_index].intensity;
				float qi = Target_Cloud->points[t_index].intensity;

				float dx = px - qx;
				float dy = py - qy;
				float dz = pz - qz;

				// norm (nt * ns)
				float nx = nty * nsz - ntz * nsy;
				float ny = ntz * nsx - ntx * nsz;
				float nz = ntx * nsy - nty * nsx;
				float nd = sqrt(nx * nx + ny * ny + nz * nz);
				nx /= nd;
				ny /= nd;
				nz /= nd; //normalize

				double nxy = nx * ny;
				double nxz = nx * nz;
				double nyz = ny * nz;
				double nx2 = nx * nx;
				double ny2 = ny * ny;
				double nz2 = nz * nz;
				double px2 = px * px;
				double py2 = py * py;
				double pz2 = pz * pz;
				double pxy = px * py;
				double pxz = px * pz;
				double pyz = py * pz;

				float d = std::sqrt((nxz * dz - nz2 * dx - ny2 * dx + nxy * dy) * (nxz * dz - nz2 * dx - ny2 * dx + nxy * dy) +
									(-nz2 * dy + nyz * dz + nxy * dx - nx2 * dy) * (-nz2 * dy + nyz * dz + nxy * dx - nx2 * dy) +
									(nyz * dy - ny2 * dz - nx2 * dz + nxz * dx) * (nyz * dy - ny2 * dz - nx2 * dz + nxz * dx));

				// float ex = std::abs(nxz * dz - nz2 * dx - ny2 * dx + nxy * dy);
				// float ey = std::abs(-nz2 * dy + nyz * dz + nxy * dx - nx2 * dy);
				// float ez = std::abs(nyz * dy - ny2 * dz - nx2 * dz + nxz * dx);

				float wx, wy, wz;
				wx = weight;

				float dist = std::sqrt(qx * qx + qy * qy + qz * qz);
				if (dist_weight_or_not)
					wx = wx * get_weight_by_dist_adaptive(dist, iter_num);
				//wx = wx * get_weight_by_dist(dist);

				if (intensity_weight_or_not)
					wx = wx * get_weight_by_intensity(pi + 0.0001, qi + 0.0001);

				if (residual_weight_or_not)
				{
					wx = wx * get_weight_by_residual(d, residual_window_size);
					// wy = wy * get_weight_by_residual(std::abs(evec(1, 0)), residual_window_size);
					// wz = wz * get_weight_by_residual(std::abs(evec(2, 0)), residual_window_size);
				}

				wy = wx;
				wz = wx;

				(*Corr)[i].weight = wx;

				//    0  1  2  3  4  5
				//    6  7  8  9 10 11
				//   12 13 14 15 16 17
				//   18 19 20 21 22 23
				//   24 25 26 27 28 29
				//   30 31 32 33 34 35

				ATPA.coeffRef(0) += (wy * nz2 + wz * ny2);
				ATPA.coeffRef(1) += (-wz * nxy);
				ATPA.coeffRef(2) += (-wy * nxz);
				ATPA.coeffRef(3) += (-wy * nxz * py + wz * nxy * pz);
				ATPA.coeffRef(4) += (wy * nxz * px + wy * nz2 * pz + wz * ny2 * pz);
				ATPA.coeffRef(5) += (-wy * nz2 * py - wz * ny2 * py + wz * nxy * px);
				ATPA.coeffRef(7) += (wx * nz2 + wz * nx2);
				ATPA.coeffRef(8) += (-wx * nyz);
				ATPA.coeffRef(9) += (-wx * nz2 * pz - wx * nyz * py - wz * nz2 * pz);
				ATPA.coeffRef(10) += (wx * nyz * px - wz * nxy * pz);
				ATPA.coeffRef(11) += (wx * nz2 * px + wz * nxy * py + wz * nx2 * px);
				ATPA.coeffRef(14) += (wx * ny2 + wy * nx2);
				ATPA.coeffRef(15) += (wx * nyz * pz + wx * ny2 * py + wy * nx2 * py);
				ATPA.coeffRef(16) += (-wx * ny2 * px - wy * nx2 * px - wy * nxz * pz);
				ATPA.coeffRef(17) += (-wx * nyz * px + wy * nxz * py);
				ATPA.coeffRef(21) += (wx * (nz2 * pz2 + ny2 * py2 + 2 * nyz * pyz) + wy * nx2 * py2 + wz * nx2 * pz2);
				ATPA.coeffRef(22) += (-wx * (nyz * pxz + ny2 * pxy) - wy * (nx2 * pxy + nxz * pyz) + wz * nxy * pz2);
				ATPA.coeffRef(23) += (-wx * (nz2 * pxz + nyz * pxy) + wy * nxz * py2 - wz * (nxy * pyz + nx2 * pxz));
				ATPA.coeffRef(28) += (wx * ny2 * px2 + wy * (nx2 * px2 + nz2 * pz2 + 2 * nxz * pxz) + wz * ny2 * pz2);
				ATPA.coeffRef(29) += (wx * nyz * px2 - wy * (nxz * pxy + nz2 * pyz) - wz * (ny2 * pyz + nxy * pxz));
				ATPA.coeffRef(35) += (wx * nz2 * px2 + wy * nz2 * py2 + wz * (ny2 * py2 + nx2 * px2 + 2 * nxy * pxy));

				ATPb.coeffRef(0) += (wy * (nxz * dz - nz2 * dx) + wz * (-ny2 * dx + nxy * dy));
				ATPb.coeffRef(1) += (wx * (-nz2 * dy + nyz * dz) + wz * (nxy * dx - nx2 * dy));
				ATPb.coeffRef(2) += (wx * (nyz * dy - ny2 * dz) + wy * (-nx2 * dz + nxz * dx));
				ATPb.coeffRef(3) += (wx * (nz * pz * ny * py) * (nz * dy - ny * dz) + wy * nx * py * (-nx * dz + nz * dx) + wz * nx * pz * (-ny * dx + nx * dy));
				ATPb.coeffRef(4) += (wx * ny * px * (-nz * dy + ny * dz) + wy * (nx * px + nz * pz) * (nx * dz - nz * dx) + wz * ny * pz * (-ny * dx + nx * dy));
				ATPb.coeffRef(5) += (wx * nz * px * (-nz * dy + ny * dz) + wy * nz * py * (-nx * dz + nz * dx) + wz * (ny * py + nx * px) * (ny * dx - nx * dy));
			}
		}

		return 1;
	}

	//calculate residual v
	bool get_multi_metrics_lls_residual(const typename pcl::PointCloud<PointT>::Ptr &Source_Ground, const typename pcl::PointCloud<PointT>::Ptr &Target_Ground, boost::shared_ptr<pcl::Correspondences> &Corr_Ground,
										const typename pcl::PointCloud<PointT>::Ptr &Source_Pillar, const typename pcl::PointCloud<PointT>::Ptr &Target_Pillar, boost::shared_ptr<pcl::Correspondences> &Corr_Pillar,
										const typename pcl::PointCloud<PointT>::Ptr &Source_Beam, const typename pcl::PointCloud<PointT>::Ptr &Target_Beam, boost::shared_ptr<pcl::Correspondences> &Corr_Beam,
										const typename pcl::PointCloud<PointT>::Ptr &Source_Facade, const typename pcl::PointCloud<PointT>::Ptr &Target_Facade, boost::shared_ptr<pcl::Correspondences> &Corr_Facade,
										const typename pcl::PointCloud<PointT>::Ptr &Source_Roof, const typename pcl::PointCloud<PointT>::Ptr &Target_Roof, boost::shared_ptr<pcl::Correspondences> &Corr_Roof,
										const typename pcl::PointCloud<PointT>::Ptr &Source_Vertex, const typename pcl::PointCloud<PointT>::Ptr &Target_Vertex, boost::shared_ptr<pcl::Correspondences> &Corr_Vertex,
										const Vector6d &transform_x, double &sigma_square_post, double sigma_thre = 0.2)
	{
		double VTPV = 0;
		int obeservation_count = 0;

		pt2pl_lls_residual(Source_Ground, Target_Ground, Corr_Ground, transform_x, VTPV, obeservation_count);
		pt2pl_lls_residual(Source_Facade, Target_Facade, Corr_Facade, transform_x, VTPV, obeservation_count);
		pt2pl_lls_residual(Source_Roof, Target_Roof, Corr_Roof, transform_x, VTPV, obeservation_count);
		pt2li_lls_residual(Source_Pillar, Target_Pillar, Corr_Pillar, transform_x, VTPV, obeservation_count);
		pt2li_lls_residual(Source_Beam, Target_Beam, Corr_Beam, transform_x, VTPV, obeservation_count);
		pt2pt_lls_residual(Source_Vertex, Target_Vertex, Corr_Vertex, transform_x, VTPV, obeservation_count);

		sigma_square_post = VTPV / (obeservation_count - 6); //   VTPV/(n-t) , t is the neccessary observation number (dof), here, t=6

		std::cout << "The posterior unit weight standard deviation (m) is " << sqrt(sigma_square_post);

		if (sqrt(sigma_square_post) < sigma_thre)
			return 1;
		else
			return 0;
	}

	bool pt2pt_lls_residual(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
							boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count)
	{
		//point-to-plane distance metrics
		//3 observation equation for 1 pair of correspondence
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;

			if (t_index != -1)
			{
				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;

				float dx = px - qx;
				float dy = py - qy;
				float dz = pz - qz;

				Eigen::Matrix<double, 3, 6> A_Matrix;
				Eigen::Matrix<double, 3, 1> b_vector;
				Eigen::Matrix<double, 3, 1> residual_vector;

				A_Matrix << 1, 0, 0, 0, pz, -py,
					0, 1, 0, -pz, 0, px,
					0, 0, 1, py, -px, 0;
				b_vector << -dx, -dy, -dz;

				residual_vector = A_Matrix * transform_x - b_vector;

				VTPV += (*Corr)[i].weight * (residual_vector(0) * residual_vector(0) + residual_vector(1) * residual_vector(1) + residual_vector(2) * residual_vector(2));

				observation_count += 3;
			}
		}

		return 1;
	}

	bool pt2pl_lls_residual(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
							boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count)
	{
		//point-to-plane distance metrics
		//1 observation equation for 1 pair of correspondence
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;
			if (t_index != -1)
			{
				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;
				float ntx = Target_Cloud->points[t_index].normal_x;
				float nty = Target_Cloud->points[t_index].normal_y;
				float ntz = Target_Cloud->points[t_index].normal_z;

				float a = ntz * py - nty * pz;
				float b = ntx * pz - ntz * px;
				float c = nty * px - ntx * py;
				float d = ntx * qx + nty * qy + ntz * qz - ntx * px - nty * py - ntz * pz;

				float residual = ntx * transform_x(0) + nty * transform_x(1) + ntz * transform_x(2) + a * transform_x(3) + b * transform_x(4) + c * transform_x(5) - d;

				//LOG(INFO) << "final weight (pt-pl): " << (*Corr)[i].weight;

				VTPV += (*Corr)[i].weight * residual * residual;

				observation_count++;
			}
		}

		return 1;
	}

	//primary vector stored as point normal
	bool pt2li_lls_residual(const typename pcl::PointCloud<PointT>::Ptr &Source_Cloud, const typename pcl::PointCloud<PointT>::Ptr &Target_Cloud,
							boost::shared_ptr<pcl::Correspondences> &Corr, const Vector6d &transform_x, double &VTPV, int &observation_count)
	{
		//point-to-line distance metrics
		//3 observation equation for 1 pair of correspondence
		for (int i = 0; i < (*Corr).size(); i++)
		{
			int s_index, t_index;
			s_index = (*Corr)[i].index_query;
			t_index = (*Corr)[i].index_match;
			if (t_index != -1)
			{
				float px = Source_Cloud->points[s_index].x;
				float py = Source_Cloud->points[s_index].y;
				float pz = Source_Cloud->points[s_index].z;
				float qx = Target_Cloud->points[t_index].x;
				float qy = Target_Cloud->points[t_index].y;
				float qz = Target_Cloud->points[t_index].z;
				float vx = Target_Cloud->points[t_index].normal_x; //actually primary directional vector
				float vy = Target_Cloud->points[t_index].normal_y;
				float vz = Target_Cloud->points[t_index].normal_z;

				float dx = px - qx;
				float dy = py - qy;
				float dz = pz - qz;

				Eigen::Matrix<double, 3, 6> A_Matrix;
				Eigen::Matrix<double, 3, 1> b_vector;
				Eigen::Matrix<double, 3, 1> residual_vector;

				A_Matrix << 0, vz, -vy, -vz * pz - vy * py, vy * px, vz * px,
					-vz, 0, vx, vx * py, -vx * px - vz * pz, vz * py,
					vy, -vx, 0, vx * pz, vy * pz, -vy * py - vx * px;
				b_vector << -vz * dy + vy * dz, -vx * dz + vz * dx, -vy * dx + vx * dy;

				residual_vector = A_Matrix * transform_x - b_vector;

				//LOG(INFO) << "final weight (pt-li): " << (*Corr)[i].weight;

				//VTPV is the sum of square of the residuals
				VTPV += (*Corr)[i].weight * (residual_vector(0) * residual_vector(0) + residual_vector(1) * residual_vector(1) + residual_vector(2) * residual_vector(2));

				observation_count += 3;
			}
		}
		return 1;
	}

	// the following lines contain the various weighting functions: distance weight, intensity-compatible weight and residual weight

	// Intuition: three part, near , medium , far
	// near used to control translation
	// far used to control rotation
	// medium used to control both
	//TODO: change the distance weight according to the iteration number (mathematic deducing)
	float get_weight_by_dist_adaptive(float dist, int iter_num, float unit_dist = 30.0, float b_min = 0.7, float b_max = 1.3, float b_step = 0.05)
	{
		float b_current = min_(b_min + b_step * iter_num, b_max);
		float temp_weight = b_current + (1.0 - b_current) * dist / unit_dist;
		temp_weight = max_(temp_weight, 0.01);
		return temp_weight;
	}

	//standard
	inline float get_weight_by_dist(float dist, float unit_dist = 60.0, float base_value = 0.7) //unit_dist = 60.0 (is just a multiplier constant)
	{
		return (base_value + (1 - base_value) * dist / unit_dist);
		//return (base_value + (1 - base_value) * unit_dist / dist);
	}

	inline float get_weight_by_intensity(float intensity_1, float intensity_2, float base_value = 0.6, float intensity_scale = 255.0)
	{
		float intensity_diff_ratio = std::fabs(intensity_1 - intensity_2) / intensity_scale;
		float intensity_weight = std::exp(-1.0 * intensity_diff_ratio);
		return intensity_weight;
		//return (base_value + (1 - base_value) * min_(intensity_1 / intensity_2, intensity_2 / intensity_1));
	}

	//By huber loss function
	inline float get_weight_by_residual(float res, float huber_thre = 0.05, int delta = 1) //test different kind of robust kernel function here
	{
		//return ((res > huber_thre) ? (std::sqrt(1.0 + (res / huber_thre - 1.0)) / (res / huber_thre)) : (1.0));
		//return ((res > huber_thre) ? (1.0 + (res / huber_thre - 1.0) / (res / huber_thre) / (res / huber_thre)) : (1.0)); //this function (naive huber) is better
		//return ((res > huber_thre) ? (huber_thre / res) : (1.0));

		//Huber Loss
		//y=0.5*x^2        , x<d
		//y=0.5*d^2+|x|-d  , x>=d
		//d= 1, |x|= res/huber_thre
		//weight=(0.5*d^2+|x|-d)/(0.5*x^2) = (2*res*huber_thre-huber_thre*huber_thre)/res/res)
		return ((res > huber_thre) ? ((2 * res * huber_thre + (delta * delta - 2 * delta) * (huber_thre * huber_thre)) / res / res) : (1.0));
	}

	//general function for m-estimation
	float get_weight_by_residual_general(float res, float thre = 0.05, float alpha = 2.0) //test different kind of robust kernel function here
	{
		float weight;
		res = res / thre;
		if (alpha == 2)
			weight = 1.0;
		else if (alpha == 0)
			weight = 2.0 / (res * res + 2.0);
		else
			weight = 1.0 * std::pow((res * res / std::abs(alpha - 2.0) + 1.0), (alpha * 0.5 - 1.0));

		return weight;
	}

	//roll - pitch - yaw rotation (x - y' - z'') ----> this is for our tiny angle approximation
	bool construct_trans_a(const double &tx, const double &ty, const double &tz,
						   const double &alpha, const double &beta, const double &gamma,
						   Eigen::Matrix4d &transformation_matrix)
	{
		// Construct the transformation matrix from rotation and translation
		transformation_matrix = Eigen::Matrix<double, 4, 4>::Zero();
		// From euler angle to rotation matrix

		transformation_matrix(0, 0) = std::cos(gamma) * std::cos(beta);
		transformation_matrix(0, 1) = -std::sin(gamma) * std::cos(alpha) + std::cos(gamma) * std::sin(beta) * std::sin(alpha);
		transformation_matrix(0, 2) = std::sin(gamma) * std::sin(alpha) + std::cos(gamma) * std::sin(beta) * std::cos(alpha);
		transformation_matrix(1, 0) = std::sin(gamma) * std::cos(beta);
		transformation_matrix(1, 1) = std::cos(gamma) * std::cos(alpha) + std::sin(gamma) * std::sin(beta) * std::sin(alpha);
		transformation_matrix(1, 2) = -std::cos(gamma) * std::sin(alpha) + std::sin(gamma) * std::sin(beta) * std::cos(alpha);
		transformation_matrix(2, 0) = -std::sin(beta);
		transformation_matrix(2, 1) = std::cos(beta) * std::sin(alpha);
		transformation_matrix(2, 2) = std::cos(beta) * std::cos(alpha);

		transformation_matrix(0, 3) = tx;
		transformation_matrix(1, 3) = ty;
		transformation_matrix(2, 3) = tz;
		transformation_matrix(3, 3) = 1.0;

		return 1;
	}

	bool construct_trans_b(const float &tx, const float &ty, const float &tz,
						   const float &alpha, const float &beta, const float &gamma,
						   Eigen::Matrix4d &transformation_matrix)
	{
		// Construct the transformation matrix from rotation and translation
		transformation_matrix.setIdentity();

		//tiny angle simplified version
		transformation_matrix(0, 1) = -gamma;
		transformation_matrix(0, 2) = beta;
		transformation_matrix(1, 0) = gamma;
		transformation_matrix(1, 2) = -alpha;
		transformation_matrix(2, 0) = -beta;
		transformation_matrix(2, 1) = alpha;

		transformation_matrix(0, 3) = static_cast<double>(tx);
		transformation_matrix(1, 3) = static_cast<double>(ty);
		transformation_matrix(2, 3) = static_cast<double>(tz);
		transformation_matrix(3, 3) = static_cast<double>(1);

		return 1;
	}

	//Brief: calculate the Jacobi Matrix of the imaginary part of a quaternion (q1,q2,q3) with regard to its corresponding euler angle (raw,pitch,yaw)
	//for converting the euler angle variance-covariance matrix to quaternion variance-covariance matrix using variance-covariance propagation law
	//Log: Pay attetion to the euler angle order here. I originally use yaw, pitch, roll (z, y', x'') here, the correct one should be roll, pitch, yaw (x, y', z'')
	//reference:
	//1 . http://easyspin.org/easyspin/documentation/eulerangles.html
	//2 . https://en.wikipedia.org/wiki/Euler_angles
	bool get_quat_euler_jacobi(const Eigen::Vector3d &euler_angle, Eigen::Matrix3d &Jacobi, bool xyz_sequence_or_not = true)
	{
		float sin_half_roll, cos_half_roll, sin_half_pitch, cos_half_pitch, sin_half_yaw, cos_half_yaw;

		sin_half_roll = sin(0.5 * euler_angle(0));
		sin_half_pitch = sin(0.5 * euler_angle(1));
		sin_half_yaw = sin(0.5 * euler_angle(2));
		cos_half_roll = cos(0.5 * euler_angle(0));
		cos_half_pitch = cos(0.5 * euler_angle(1));
		cos_half_yaw = cos(0.5 * euler_angle(2));

		//roll pitch yaw (x, y', z'') , used
		if(xyz_sequence_or_not)
		{
			Jacobi(0, 0) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw);
			Jacobi(0, 1) = 0.5 * (-sin_half_roll * sin_half_pitch * cos_half_yaw - cos_half_roll * cos_half_pitch * sin_half_yaw);
			Jacobi(0, 2) = 0.5 * (-sin_half_roll * cos_half_pitch * sin_half_yaw - cos_half_roll * sin_half_pitch * cos_half_yaw);

			Jacobi(1, 0) = 0.5 * (-sin_half_roll * sin_half_pitch * cos_half_yaw + cos_half_roll * cos_half_pitch * sin_half_yaw);
			Jacobi(1, 1) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw - sin_half_roll * sin_half_pitch * sin_half_yaw);
			Jacobi(1, 2) = 0.5 * (-cos_half_roll * sin_half_pitch * sin_half_yaw + sin_half_roll * cos_half_pitch * cos_half_yaw);

			Jacobi(2, 0) = 0.5 * (-sin_half_roll * cos_half_pitch * sin_half_yaw - cos_half_roll * sin_half_pitch * cos_half_yaw);
			Jacobi(2, 1) = 0.5 * (-cos_half_roll * sin_half_pitch * sin_half_yaw - sin_half_roll * cos_half_pitch * cos_half_yaw);
			Jacobi(2, 2) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw);
		}
		else //yaw pitch roll (z, y', x'') , not used
		{
		    Jacobi(0, 0) = 0.5 * (-sin_half_roll * cos_half_pitch * sin_half_yaw - cos_half_roll * sin_half_pitch * cos_half_yaw);
			Jacobi(0, 1) = 0.5 * (-cos_half_roll * sin_half_pitch * sin_half_yaw - sin_half_roll * cos_half_pitch * cos_half_yaw);
			Jacobi(0, 2) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw + sin_half_roll * sin_half_pitch * sin_half_yaw);

			Jacobi(1, 0) = 0.5 * (cos_half_roll * cos_half_pitch * sin_half_yaw - sin_half_roll * sin_half_pitch * cos_half_yaw);
			Jacobi(1, 1) = 0.5 * (-sin_half_roll * sin_half_pitch * sin_half_yaw + cos_half_roll * cos_half_pitch * cos_half_yaw);
			Jacobi(1, 2) = 0.5 * (sin_half_roll * cos_half_pitch * cos_half_yaw - cos_half_roll * sin_half_pitch * sin_half_yaw);

			Jacobi(2, 0) = 0.5 * (cos_half_roll * cos_half_pitch * cos_half_yaw - sin_half_roll * sin_half_pitch * sin_half_yaw);
			Jacobi(2, 1) = 0.5 * (-sin_half_roll * sin_half_pitch * cos_half_yaw + cos_half_roll * cos_half_pitch * sin_half_yaw);
			Jacobi(2, 2) = 0.5 * (-sin_half_roll * cos_half_pitch * sin_half_yaw + cos_half_roll * sin_half_pitch * cos_half_yaw);
		}
		return 1;
	}

	bool get_translation_in_station_coor_sys(const Eigen::Matrix4d &T_world, const CenterPoint &station_in_world, Eigen::Vector3d &t_station)
	{
		Eigen::Vector3d t_ws(station_in_world.x, station_in_world.y, station_in_world.z);
		Eigen::Vector3d t_w(T_world(0, 3), T_world(1, 3), T_world(2, 3));
		Eigen::Matrix3d R_w = T_world.block<3, 3>(0, 0);

		t_station = t_w + R_w * t_ws - t_ws;

		return 1;
	}

	void apply_cloudclock_cp_local_shift(CloudBlockPtr &block, float shift_x, float shift_y, float shift_z)
	{
		if (block->_stationPositionAvailable)
		{
			block->_localStation.x += shift_x;
			block->_localStation.y += shift_y;
			block->_localStation.z += shift_z;
		}
		else
		{
			block->_localCenter.x += shift_x;
			block->_localCenter.y += shift_y;
			block->_localCenter.z += shift_z;
		}
	}

	//this function is for speed up the registration process when the point number is a bit too big
	bool keep_less_source_pts(typename pcl::PointCloud<PointT>::Ptr &pc_ground_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_pillar_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_beam_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_facade_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_roof_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_vertex_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_ground_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_pillar_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_beam_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_facade_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_roof_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_vertex_sc,
							  int ground_down_rate = 4, int facade_down_rate = 2, int target_down_rate = 2)
	{
		CFilter<PointT> cfilter;

		cfilter.random_downsample_pcl(pc_ground_tc, (int)(pc_ground_tc->points.size() / target_down_rate));
		cfilter.random_downsample_pcl(pc_facade_tc, (int)(pc_facade_tc->points.size() / target_down_rate));

		cfilter.random_downsample_pcl(pc_ground_sc, (int)(pc_ground_tc->points.size() / ground_down_rate));
		cfilter.random_downsample_pcl(pc_facade_sc, (int)(pc_facade_tc->points.size() / facade_down_rate));
		cfilter.random_downsample_pcl(pc_pillar_sc, (int)(pc_pillar_tc->points.size()));
		cfilter.random_downsample_pcl(pc_beam_sc, (int)(pc_beam_tc->points.size()));
		cfilter.random_downsample_pcl(pc_roof_sc, (int)(pc_roof_tc->points.size()));
		cfilter.random_downsample_pcl(pc_vertex_sc, (int)(pc_vertex_tc->points.size()));
		return true;
	}

	bool intersection_filter( Constraint &registration_cons,
							  typename pcl::PointCloud<PointT>::Ptr &pc_ground_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_pillar_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_beam_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_facade_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_roof_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_vertex_tc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_ground_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_pillar_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_beam_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_facade_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_roof_sc,
							  typename pcl::PointCloud<PointT>::Ptr &pc_vertex_sc,
							  float bbx_pad = 1.0)
	{
		CFilter<PointT> cfilter;
		Bounds intersection_bbx, source_init_guess_bbx_merged;
		std::vector<Bounds> source_init_guess_bbxs(3);
		cfilter.get_cloud_bbx(pc_ground_sc, source_init_guess_bbxs[0]);
		cfilter.get_cloud_bbx(pc_pillar_sc, source_init_guess_bbxs[1]);
		cfilter.get_cloud_bbx(pc_facade_sc, source_init_guess_bbxs[2]);
		cfilter.merge_bbx(source_init_guess_bbxs, source_init_guess_bbx_merged);
		cfilter.get_intersection_bbx(registration_cons.block1->_localBound, source_init_guess_bbx_merged, intersection_bbx, bbx_pad);
		cfilter.get_cloud_pair_intersection(intersection_bbx,
											pc_ground_tc, pc_pillar_tc, pc_beam_tc, pc_facade_tc, pc_roof_tc, pc_vertex_tc,
											pc_ground_sc, pc_pillar_sc, pc_beam_sc, pc_facade_sc, pc_roof_sc, pc_vertex_sc);
		std::cout << "Intersection local bounding box filtering done";
        return true;
	}

	//Coordinate system covertation related functions
	//Brief: Using the Gauss-Newton Least Square Method to solve 4 Degree of Freedom Transformation from no less than 2 points
	//cp_number is the control points' number for LLS calculation, the rest of points are used to check the accuracy;
	bool coord_system_tran_4dof_lls(const std::vector<std::vector<double>> &coordinatesA, const std::vector<std::vector<double>> &coordinatesB,
									Eigen::Matrix4d &TransMatrixA2B, int cp_number, double theta0_degree) //X Y Z yaw
	{
		Vector4d transAB;
		Vector4d temp_trans;

		int iter_num = 0;

		double theta, theta0, dtheta, eps; // in rad
		double yaw_degree;				   // in degree
		double tx, ty, tz;				   // in meter
		double RMSE_check, sum_squaredist;
		int pointnumberA, pointnumberB, pointnumbercheck;
		//std::vector <std::vector<double>> coordinatesAT;

		//cout << "Input the approximate yaw angle in degree" << endl;
		//cin >> theta0_degree;

		dtheta = 9999;
		eps = 1e-9;

		theta0 = theta0_degree / 180 * M_PI; //Original Guess

		pointnumberA = coordinatesA.size();
		pointnumberB = coordinatesB.size();

		sum_squaredist = 0;

		if (cp_number < 2)
		{
            std::cout << "Error ! Not enough control point number ..." << std::endl;
			return 0;
		}

		while (abs(dtheta) > eps)
		{

			MatrixXd A_;
			VectorXd b_;

			A_.resize(cp_number * 3, 4);
			b_.resize(cp_number * 3, 1);

			for (int j = 0; j < cp_number; j++)
			{
				// A Matrix
				A_(j * 3, 0) = -coordinatesA[j][0] * sin(theta0) - coordinatesA[j][1] * cos(theta0);
				A_(j * 3, 1) = 1;
				A_(j * 3, 2) = 0;
				A_(j * 3, 3) = 0;

				A_(j * 3 + 1, 0) = coordinatesA[j][0] * cos(theta0) - coordinatesA[j][1] * sin(theta0);
				A_(j * 3 + 1, 1) = 0;
				A_(j * 3 + 1, 2) = 1;
				A_(j * 3 + 1, 3) = 0;

				A_(j * 3 + 2, 0) = 0;
				A_(j * 3 + 2, 1) = 0;
				A_(j * 3 + 2, 2) = 0;
				A_(j * 3 + 2, 3) = 1;

				//b Vector
				b_(j * 3, 0) = coordinatesB[j][0] - coordinatesA[j][0] * cos(theta0) + coordinatesA[j][1] * sin(theta0);
				b_(j * 3 + 1, 0) = coordinatesB[j][1] - coordinatesA[j][0] * sin(theta0) - coordinatesA[j][1] * cos(theta0);
				b_(j * 3 + 2, 0) = coordinatesB[j][2] - coordinatesA[j][2];
			}

			//x=(ATPA)-1(ATPb)
			temp_trans = ((A_.transpose() * A_).inverse()) * A_.transpose() * b_;
			dtheta = temp_trans(0, 0);

			theta0 += dtheta;

			iter_num++;

			//cout << "Result for iteration " << iter_num << " is " << endl
			//<< temp_trans(1, 0) << " , " << temp_trans(2, 0) << " , " << temp_trans(3, 0) << " , " << theta0 * 180 / M_PI <<endl;
		}

		transAB = temp_trans;

		theta = theta0;
		yaw_degree = theta * 180 / M_PI;

		tx = transAB(1, 0);
		ty = transAB(2, 0);
		tz = transAB(3, 0);

		std::cout.setf(std::ios::showpoint);
		std::cout.precision(12);

        std::cout << "Calculated by Linear Least Square" << std::endl
			 << "Converged in " << iter_num << " iterations ..." << std::endl
			 << "Station B 's Coordinate and Orientation in A's System is:" << std::endl
			 << "X: " << tx << " m" << std::endl
			 << "Y: " << ty << " m" << std::endl
			 << "Z: " << tz << " m" << std::endl
			 << "yaw: " << yaw_degree << " degree" << std::endl;

		TransMatrixA2B(0, 0) = cos(theta);
		TransMatrixA2B(0, 1) = -sin(theta);
		TransMatrixA2B(0, 2) = 0;
		TransMatrixA2B(0, 3) = tx;

		TransMatrixA2B(1, 0) = sin(theta);
		TransMatrixA2B(1, 1) = cos(theta);
		TransMatrixA2B(1, 2) = 0;
		TransMatrixA2B(1, 3) = ty;

		TransMatrixA2B(2, 0) = 0;
		TransMatrixA2B(2, 1) = 0;
		TransMatrixA2B(2, 2) = 1;
		TransMatrixA2B(2, 3) = tz;

		TransMatrixA2B(3, 0) = 0;
		TransMatrixA2B(3, 1) = 0;
		TransMatrixA2B(3, 2) = 0;
		TransMatrixA2B(3, 3) = 1;

        std::cout << "The Transformation Matrix from Coordinate System A to B is: " << std::endl
			 << TransMatrixA2B << std::endl;

		// Checking
		if (pointnumberA >= pointnumberB)
			pointnumbercheck = pointnumberB;
		else
			pointnumbercheck = pointnumberA;

		if (pointnumbercheck <= cp_number)
            std::cout << "Not enough points for check ..." << std::endl;
		else
		{
			pointnumbercheck -= cp_number;
			for (int j = 0; j < pointnumbercheck; j++)
			{
				double X_tran, Y_tran, Z_tran, squaredist;
				X_tran = cos(theta) * coordinatesA[j + cp_number][0] - sin(theta) * coordinatesA[j + cp_number][1] + tx;
				Y_tran = sin(theta) * coordinatesA[j + cp_number][0] + cos(theta) * coordinatesA[j + cp_number][1] + ty;
				Z_tran = coordinatesA[j + cp_number][2] + tz;

				squaredist = (X_tran - coordinatesB[j + cp_number][0]) * (X_tran - coordinatesB[j + cp_number][0]) +
							 (Y_tran - coordinatesB[j + cp_number][1]) * (Y_tran - coordinatesB[j + cp_number][1]) +
							 (Z_tran - coordinatesB[j + cp_number][2]) * (Z_tran - coordinatesB[j + cp_number][2]);
				sum_squaredist += squaredist;
			}

			RMSE_check = sqrt(sum_squaredist / pointnumbercheck);

            std::cout << "Calculated from " << pointnumbercheck << " points, the RMSE is " << RMSE_check << std::endl;
		}

		return 1;
	}

	//Brief: 6DOF transofrmation estimation useing SVD
	bool coord_system_tran_6dof_svd(const std::vector<std::vector<double>> &coordinatesA, const std::vector<std::vector<double>> &coordinatesB, Matrix4d &TransMatrixA2B, int cp_number) //X Y Z roll pitch yaw
	{
		Matrix4d transAB2D;
		pcl::PointCloud<PointT> Points2D_A, Points2D_B;
		double ZAB_mean, ZAB_sum;
		int pointnumberA, pointnumberB, pointnumbercheck;
		double RMSE_check, sum_squaredist;

		pointnumberA = coordinatesA.size();
		pointnumberB = coordinatesB.size();
		ZAB_sum = 0;
		sum_squaredist = 0;

		for (size_t i = 0; i < cp_number; i++)
		{
			PointT PtA, PtB;
			PtA.x = coordinatesA[i][0];
			PtA.y = coordinatesA[i][1];
			PtA.z = coordinatesA[i][2];

			PtB.x = coordinatesB[i][0];
			PtB.y = coordinatesB[i][1];
			PtB.z = coordinatesB[i][2];

			Points2D_A.push_back(PtA);
			Points2D_B.push_back(PtB);
			ZAB_sum += (coordinatesB[i][2] - coordinatesA[i][2]);
		}

		ZAB_mean = ZAB_sum / cp_number;

		if (cp_number < 2)
		{
            std::cout << "Error ! Not enough control point number ..." << std::endl;
			return 0;
		}

		pcl::registration::TransformationEstimationSVD<PointT, PointT> svd_estimator;
		svd_estimator.estimateRigidTransformation(Points2D_A, Points2D_B, transAB2D);

		TransMatrixA2B = transAB2D.cast<double>();

		double tx, ty, tz, yaw_rad, yaw_degree;

		tx = TransMatrixA2B(0, 3);
		ty = TransMatrixA2B(1, 3);
		tz = TransMatrixA2B(2, 3);
		yaw_rad = acos(TransMatrixA2B(0, 0));
		if (TransMatrixA2B(1, 0) < 0)
			yaw_rad = -yaw_rad;
		yaw_degree = yaw_rad / M_PI * 180;

        std::cout << "Calculated by SVD" << std::endl
			 << "Station B 's Coordinate and Orientation in A's System is:" << std::endl
			 << "X: " << tx << " m" << std::endl
			 << "Y: " << ty << " m" << std::endl
			 << "Z: " << tz << " m" << std::endl
			 << "yaw: " << yaw_degree << " degree" << std::endl;

        std::cout << "The Transformation Matrix from Coordinate System A to B is: " << std::endl
			 << TransMatrixA2B << std::endl;

		// Checking
		if (pointnumberA >= pointnumberB)
			pointnumbercheck = pointnumberB;
		else
			pointnumbercheck = pointnumberA;

		if (pointnumbercheck <= cp_number)
            std::cout << "Not enough points for check ..." << std::endl;
		else
		{
			pointnumbercheck -= cp_number;
			for (int j = 0; j < pointnumbercheck; j++)
			{
				double X_tran, Y_tran, Z_tran, squaredist;
				X_tran = TransMatrixA2B(0, 0) * coordinatesA[j + cp_number][0] + TransMatrixA2B(0, 1) * coordinatesA[j + cp_number][1] + TransMatrixA2B(0, 2) * coordinatesA[j + cp_number][2] + tx;
				Y_tran = TransMatrixA2B(1, 0) * coordinatesA[j + cp_number][0] + TransMatrixA2B(1, 1) * coordinatesA[j + cp_number][1] + TransMatrixA2B(1, 2) * coordinatesA[j + cp_number][2] + ty;
				Z_tran = TransMatrixA2B(2, 0) * coordinatesA[j + cp_number][0] + TransMatrixA2B(2, 1) * coordinatesA[j + cp_number][1] + TransMatrixA2B(2, 2) * coordinatesA[j + cp_number][2] + tz;

				squaredist = (X_tran - coordinatesB[j + cp_number][0]) * (X_tran - coordinatesB[j + cp_number][0]) +
							 (Y_tran - coordinatesB[j + cp_number][1]) * (Y_tran - coordinatesB[j + cp_number][1]) +
							 (Z_tran - coordinatesB[j + cp_number][2]) * (Z_tran - coordinatesB[j + cp_number][2]);
				sum_squaredist += squaredist;
			}

			RMSE_check = sqrt(sum_squaredist / pointnumbercheck);

            std::cout << "Calculated from " << pointnumbercheck << " points, the RMSE is " << RMSE_check << std::endl;
		}
	}

	//Brief: 4 parameters geo-coordiante transformation (tx,ty,r_xy,s)
	bool coord_system_tran_4dof(const std::vector<std::vector<double>> &coordinatesA, const std::vector<std::vector<double>> &coordinatesB, std::vector<double> &transpara, int cp_number) // X Y yaw scale
	{
		double tx, ty, a, b; // 4 parameters
		double s, rot_rad, rot_degree;
		double RMSE_check, sum_squaredist;
		int pointnumberA, pointnumberB, pointnumbercheck;
		Vector4d transAB;
		transpara.resize(5);

		pointnumberA = coordinatesA.size();
		pointnumberB = coordinatesB.size();

		sum_squaredist = 0;

		if (cp_number < 3)
		{
            std::cout << "Error ! Not enough control point number ..." << std::endl;
			return 0;
		}

		MatrixXd A_;
		VectorXd b_;

		A_.resize(cp_number * 2, 4);
		b_.resize(cp_number * 2, 1);

		for (int j = 0; j < cp_number; j++)
		{
			// A Matrix
			A_(j * 2, 0) = 1;
			A_(j * 2, 1) = 0;
			A_(j * 2, 2) = coordinatesA[j][0];
			A_(j * 2, 3) = -coordinatesA[j][1];

			A_(j * 2 + 1, 0) = 0;
			A_(j * 2 + 1, 1) = 1;
			A_(j * 2 + 1, 2) = coordinatesA[j][1];
			A_(j * 2 + 1, 3) = coordinatesA[j][0];

			//b Vector
			b_(j * 2, 0) = coordinatesB[j][0];
			b_(j * 2 + 1, 0) = coordinatesB[j][1];
		}
		transAB = ((A_.transpose() * A_).inverse()) * A_.transpose() * b_;

		tx = transAB(0, 0);
		ty = transAB(1, 0);
		a = transAB(2, 0);
		b = transAB(3, 0);
		s = sqrt(a * a + b * b);

		transpara[0] = tx;
		transpara[1] = ty;
		transpara[2] = s;
		transpara[3] = b / s; //sin (ang)
		transpara[4] = a / s; //cos (ang)

		std::cout.setf(std::ios::showpoint);
		std::cout.precision(12);

        std::cout << "Estimated Transformation From A to B" << std::endl
			 << "tx: " << tx << " m" << std::endl
			 << "ty: " << ty << " m" << std::endl
			 << "scale: " << s << std::endl;

		// Checking
		if (pointnumberA >= pointnumberB)
			pointnumbercheck = pointnumberB;
		else
			pointnumbercheck = pointnumberA;

		if (pointnumbercheck <= cp_number)
            std::cout << "Not enough points for check ..." << std::endl;
		else
		{
			pointnumbercheck -= cp_number;
			for (int j = 0; j < pointnumbercheck; j++)
			{
				double X_tran, Y_tran, squaredist;
				X_tran = transpara[2] * transpara[4] * coordinatesA[j + cp_number][0] - transpara[2] * transpara[3] * coordinatesA[j + cp_number][1] + transpara[0];
				Y_tran = transpara[2] * transpara[3] * coordinatesA[j + cp_number][0] + transpara[2] * transpara[4] * coordinatesA[j + cp_number][1] + transpara[1];
				squaredist = (X_tran - coordinatesB[j + cp_number][0]) * (X_tran - coordinatesB[j + cp_number][0]) +
							 (Y_tran - coordinatesB[j + cp_number][1]) * (Y_tran - coordinatesB[j + cp_number][1]);
				sum_squaredist += squaredist;
			}

			RMSE_check = sqrt(sum_squaredist / pointnumbercheck);

            std::cout << "Calculated from " << pointnumbercheck << " points, the RMSE is " << RMSE_check << std::endl;
		}

		return 1;
	}

	//Brief: 7 parameters geo-coordiante transformation (tx,ty,tz,rx,ry,rz,s)
	bool coord_system_tran_7dof(const std::vector<std::vector<double>> &coordinatesA,
								const std::vector<std::vector<double>> &coordinatesB, std::vector<double> &transpara, int cp_number) // X Y Z roll pitch yaw scale
	{
		double RMSE_check, sum_squaredist;
		int pointnumberA, pointnumberB, pointnumbercheck;
		VectorXd transAB;
		transAB.resize(7);
		transpara.resize(7);

		pointnumberA = coordinatesA.size();
		pointnumberB = coordinatesB.size();

		sum_squaredist = 0;

		if (cp_number < 4)
		{
            std::cout << "Error ! Not enough control point number ..." << std::endl;
			return 0;
		}

		MatrixXd A_;
		VectorXd b_;

		A_.resize(cp_number * 3, 7);
		b_.resize(cp_number * 3, 1);

		for (int j = 0; j < cp_number; j++)
		{
			// A Matrix   tx ty tz rx ry rz s
			A_(j * 3, 0) = 1;
			A_(j * 3, 1) = 0;
			A_(j * 3, 2) = 0;
			A_(j * 3, 3) = 0;
			A_(j * 3, 4) = -coordinatesA[j][2];
			A_(j * 3, 5) = coordinatesA[j][1];
			A_(j * 3, 6) = coordinatesA[j][0];

			A_(j * 3 + 1, 0) = 0;
			A_(j * 3 + 1, 1) = 1;
			A_(j * 3 + 1, 2) = 0;
			A_(j * 3 + 1, 3) = coordinatesA[j][2];
			A_(j * 3 + 1, 4) = 0;
			A_(j * 3 + 1, 5) = -coordinatesA[j][0];
			A_(j * 3 + 1, 6) = coordinatesA[j][1];

			A_(j * 3 + 2, 0) = 0;
			A_(j * 3 + 2, 1) = 0;
			A_(j * 3 + 2, 2) = 1;
			A_(j * 3 + 2, 3) = -coordinatesA[j][1];
			A_(j * 3 + 2, 4) = coordinatesA[j][0];
			A_(j * 3 + 2, 5) = 0;
			A_(j * 3 + 2, 6) = coordinatesA[j][2];

			//b Vector
			b_(j * 3, 0) = coordinatesB[j][0];
			b_(j * 3 + 1, 0) = coordinatesB[j][1];
			b_(j * 3 + 2, 0) = coordinatesB[j][2];
		}
		transAB = ((A_.transpose() * A_).inverse()) * A_.transpose() * b_;

		transpara[0] = transAB(0);
		transpara[1] = transAB(1);
		transpara[2] = transAB(2);
		transpara[3] = transAB(3);
		transpara[4] = transAB(4);
		transpara[5] = transAB(5);
		transpara[6] = transAB(6);

		std::cout.setf(std::ios::showpoint);
		std::cout.precision(10);

        std::cout << "Estimated Transformation From A to B" << std::endl
			 << "tx: " << transpara[0] << " m" << std::endl
			 << "ty: " << transpara[1] << " m" << std::endl
			 << "tz: " << transpara[2] << " m" << std::endl
			 << "rx: " << transpara[3] << std::endl
			 << "ry: " << transpara[4] << std::endl
			 << "rz: " << transpara[5] << std::endl
			 << "scale: " << transpara[6] << std::endl;

		// Checking
		if (pointnumberA >= pointnumberB)
			pointnumbercheck = pointnumberB;
		else
			pointnumbercheck = pointnumberA;

		if (pointnumbercheck <= cp_number)
            std::cout << "Not enough points for check ..." << std::endl;
		else
		{
			pointnumbercheck -= cp_number;
			for (int j = 0; j < pointnumbercheck; j++)
			{
				double X_tran, Y_tran, Z_tran, squaredist;
				X_tran = transpara[0] + transpara[6] * coordinatesA[j + cp_number][0] + transpara[5] * coordinatesA[j + cp_number][1] - transpara[4] * coordinatesA[j + cp_number][2];
				Y_tran = transpara[1] + transpara[6] * coordinatesA[j + cp_number][1] - transpara[5] * coordinatesA[j + cp_number][0] + transpara[3] * coordinatesA[j + cp_number][2];
				Z_tran = transpara[2] + transpara[6] * coordinatesA[j + cp_number][2] + transpara[4] * coordinatesA[j + cp_number][0] - transpara[3] * coordinatesA[j + cp_number][1];
				squaredist = (X_tran - coordinatesB[j + cp_number][0]) * (X_tran - coordinatesB[j + cp_number][0]) +
							 (Y_tran - coordinatesB[j + cp_number][1]) * (Y_tran - coordinatesB[j + cp_number][1]) +
							 (Z_tran - coordinatesB[j + cp_number][2]) * (Z_tran - coordinatesB[j + cp_number][2]);
				sum_squaredist += squaredist;
			}

			RMSE_check = sqrt(sum_squaredist / pointnumbercheck);

            std::cout << "Calculated from " << pointnumbercheck << " points, the RMSE is " << RMSE_check << std::endl;
		}

		return 1;
	}
};


#endif //_INCLUDE_COMMON_REG_HPP
