//
// Created by w on 2022/8/28.
//


#ifndef FAST_GICP_FAST_VGICP_VOXEL_HPP
#define FAST_GICP_FAST_VGICP_VOXEL_HPP

#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <pcl/point_cloud.h>
#include "VoxelMapper.h"
#include "Misc/TicToc.h"
enum class VoxelAccumulationMode
{
    ADDITIVE, ADDITIVE_WEIGHTED, MULTIPLICATIVE
};

static std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
neighborOffsets(NeighborSearchMethod search_method)
{
    switch (search_method)
    {
        default:
            std::cerr << "unsupported neighbor search method" << std::endl;
            abort();
        case NeighborSearchMethod::DIRECT1:
            return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>{Eigen::Vector3i(0, 0, 0)};
        case NeighborSearchMethod::DIRECT7:
            return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
                    {
                            Eigen::Vector3i(0, 0, 0),
                            Eigen::Vector3i(1, 0, 0),
                            Eigen::Vector3i(-1, 0, 0),
                            Eigen::Vector3i(0, 1, 0),
                            Eigen::Vector3i(0, -1, 0),
                            Eigen::Vector3i(0, 0, 1),
                            Eigen::Vector3i(0, 0, -1)
                    };
        case NeighborSearchMethod::DIRECT27:
            break;
    }

    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> offsets27;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                offsets27.push_back(Eigen::Vector3i(i - 1, j - 1, k - 1));
    return offsets27;
}

class Vector3iHash
{
public:
    size_t operator()(const Eigen::Vector3i &x) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, x[0]);
        boost::hash_combine(seed, x[1]);
        boost::hash_combine(seed, x[2]);
        return seed;
    }
};

struct GaussianVoxel
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<GaussianVoxel>;

    GaussianVoxel()
    {
        _numPoints = 0;
        _mean.setZero();
        _cov.setZero();
    }

    virtual ~GaussianVoxel()
    { _cloud.clear(); }

    virtual void append(const Eigen::Vector4d &mean_, const Eigen::Matrix4d &cov_) = 0;

    virtual void append(const PointType &pt) = 0;

    virtual void finalize() = 0;

    virtual void calcCovariances(KD_TREE *kdTree, int kCorrespondences = 20) = 0;

    virtual void calcCovariances(pcl::PointCloud<PointType>::Ptr treeCloud, int kCorrespondences = 20) = 0;

public:
    int _numPoints;
    Eigen::Vector4d _mean;
    Eigen::Matrix4d _cov;
    PointVector _cloud;
};

struct MultiplicativeGaussianVoxel : GaussianVoxel
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MultiplicativeGaussianVoxel() : GaussianVoxel()
    {}

    virtual ~MultiplicativeGaussianVoxel()
    {}

    virtual void append(const Eigen::Vector4d &mean_, const Eigen::Matrix4d &cov_) override
    {
        _numPoints++;
        Eigen::Matrix4d cov_inv = cov_;
        cov_inv(3, 3) = 1;
        cov_inv = cov_inv.inverse().eval();

        _cov += cov_inv;
        _mean += cov_inv * mean_;

    }

    virtual void append(const PointType &pt)
    {
        _cloud.emplace_back(pt);
    }

    virtual void finalize() override
    {
        _cov(3, 3) = 1;
        _mean[3] = 1;

        _cov = _cov.inverse().eval();
        _mean = (_cov * _mean).eval();
    }

    virtual void calcCovariances(KD_TREE *kdTree, int kCorrespondences = 20) override
    {}

    virtual void calcCovariances(pcl::PointCloud<PointType>::Ptr treeCloud, int kCorrespondences = 20) override
    {}
};

struct AdditiveGaussianVoxel : GaussianVoxel
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AdditiveGaussianVoxel() : GaussianVoxel()
    {}

    virtual ~AdditiveGaussianVoxel()
    {}

    virtual void append(const Eigen::Vector4d &mean_, const Eigen::Matrix4d &cov_) override
    {
        _numPoints++;
        _mean += mean_;
        _cov += cov_;
    }


    virtual void append(const PointType &pt)
    {
        _cloud.emplace_back(pt);
        _numPoints++;
    }

    virtual void finalize() override
    {
        _mean /= _numPoints;
        _cov /= _numPoints;
    }

    virtual void calcCovariances(KD_TREE *kdTree, int kCorrespondences = 20) override
    {
        _cov.setZero();
        _mean.setZero();
        const int cloudSize=_cloud.size();
        Eigen::Matrix4d* cov= new Eigen::Matrix4d[cloudSize];
        Eigen::Vector4d* mean=new  Eigen::Vector4d[cloudSize];

#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
        for (int i = 0; i < cloudSize; i++)
        {
            std::vector<float> kSqDistances;
            PointVector pointsNear;
            kdTree->Nearest_Search(_cloud[i], kCorrespondences, pointsNear, kSqDistances);

            Eigen::Matrix<double, 4, -1> neighbors(4, kCorrespondences);
            for (int j = 0; j < pointsNear.size(); j++)
                neighbors.col(j) = Eigen::Vector4d(pointsNear[j].x, pointsNear[j].y, pointsNear[j].z, 1.);

            neighbors.colwise() -= neighbors.rowwise().mean().eval();
            Eigen::Matrix4d covariance = neighbors * neighbors.transpose() / kCorrespondences;

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance.block<3, 3>(0, 0),Eigen::ComputeFullU | Eigen::ComputeFullV);
            cov[i] = Eigen::Matrix4d::Zero();
            cov[i].template block<3, 3>(0, 0) = svd.matrixU() * Eigen::Vector3d(1, 1, 1e-3).asDiagonal() * svd.matrixV().transpose();
            mean[i] = Eigen::Vector4d(_cloud[i].x, _cloud[i].y, _cloud[i].z, 1.);
        }
        for(int i=0;i<cloudSize;i++)
        {
            _cov+=cov[i];
            _mean+=mean[i];
        }
        _cov /= cloudSize;
        _mean /= cloudSize;
		delete cov, mean;
    }

    virtual void calcCovariances(pcl::PointCloud<PointType>::Ptr treeCloud, int kCorrespondences = 20) override
    {
        _cov.setZero();
        _mean.setZero();
        const int cloudSize=_cloud.size();
		Eigen::Matrix4d* cov = new Eigen::Matrix4d[cloudSize];
		Eigen::Vector4d* mean = new  Eigen::Vector4d[cloudSize];

        pcl::search::KdTree<PointType> kdtree;
        if (kdtree.getInputCloud() != treeCloud)
            kdtree.setInputCloud(treeCloud);

#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
        for (int i = 0; i < cloudSize; i++)
        {
            std::vector<float> kSqDistances;
            std::vector<int> kIndices;
            kdtree.nearestKSearch(_cloud[i], kCorrespondences, kIndices, kSqDistances);

            Eigen::Matrix<double, 4, -1> neighbors(4, kCorrespondences);
            for (int j = 0; j < kIndices.size(); j++)
                neighbors.col(j) = Eigen::Vector4d(treeCloud->points[kIndices[j]].x,treeCloud->points[kIndices[j]].y, treeCloud->points[kIndices[j]].z, 1.);

            neighbors.colwise() -= neighbors.rowwise().mean().eval();
            Eigen::Matrix4d covariance = neighbors * neighbors.transpose() / kCorrespondences;

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
            cov[i] = Eigen::Matrix4d::Zero();
            cov[i].template block<3, 3>(0, 0) = svd.matrixU() * Eigen::Vector3d(1, 1, 1e-3).asDiagonal() * svd.matrixV().transpose();
            mean[i] = Eigen::Vector4d(_cloud[i].x, _cloud[i].y, _cloud[i].z, 1.);
        }
        for(int i=0;i<cloudSize;i++)
        {
            _cov+=cov[i];
            _mean+=mean[i];
        }
        _cov /= cloudSize;
        _mean /= cloudSize;
		delete cov, mean;
    }
};

template<typename PointType>
class GuassinVoxelMatchInfo
{
public:
    GuassinVoxelMatchInfo(const PointType &pv, std::shared_ptr<GaussianVoxel> &matchedVoxel)
    {
        _pv = pv;
        _matchedVoxel = matchedVoxel;
    }
public:
    PointType _pv;
    GaussianVoxel::Ptr _matchedVoxel = nullptr;
};

template<typename PointType>
class OctoTreeMatchInfo
{
public:
    OctoTreeMatchInfo(const PointType &pv, std::shared_ptr<OctoTree> &matchedVoxel)
    {
        _pv = pv;
        _matchedVoxel = matchedVoxel;
    }
public:
    PointType _pv;
    std::shared_ptr<OctoTree> _matchedVoxel = nullptr;
};


template<typename PointT>
class GaussianVoxelMap
{
    using VoxelMap = std::unordered_map<
            Eigen::Vector3i,
            GaussianVoxel::Ptr,
            Vector3iHash,
            std::equal_to<Eigen::Vector3i>,
            Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, GaussianVoxel::Ptr>>>;
public:
    GaussianVoxelMap(double resolution, VoxelAccumulationMode mode = VoxelAccumulationMode::ADDITIVE) :
            _voxelResolution(resolution),
            _voxelMode(mode)
    {}

    void setVoxelResolution(const double& res){_voxelResolution=res;}

    void setVoxelMode(const VoxelAccumulationMode& mode){_voxelMode=mode;}

    Eigen::Vector3i voxelCoord(const Eigen::Vector3d &x) const
    {
        return (x.array() / _voxelResolution - 0.5).floor().template cast<int>();
    }

    Eigen::Vector4d voxelOrigin(const Eigen::Vector3i &coord) const
    {
        Eigen::Vector3d origin = (coord.template cast<double>().array() + 0.5) * _voxelResolution;
        return Eigen::Vector4d(origin[0], origin[1], origin[2], 1.0f);
    }

    GaussianVoxel::Ptr lookupVoxel(const Eigen::Vector3i &coord) const
    {
        auto found = _voxelMap.find(coord);
        if (found == _voxelMap.end())
        {
            return nullptr;
        }

        return found->second;
    }

    void updateVoxelMap(const PointVector &cloud)
    {
        for (int i = 0; i < cloud.size(); i++)
        {
            Eigen::Vector3i coord = voxelCoord(Eigen::Vector3d(cloud[i].x, cloud[i].y, cloud[i].z));

            auto found = _voxelMap.find(coord);
            if (found == _voxelMap.end())
            {
                GaussianVoxel::Ptr voxel;
                switch (_voxelMode)
                {
                    case VoxelAccumulationMode::ADDITIVE:
                    case VoxelAccumulationMode::ADDITIVE_WEIGHTED:
                        voxel = std::shared_ptr<AdditiveGaussianVoxel>(new AdditiveGaussianVoxel);
                        break;
                    case VoxelAccumulationMode::MULTIPLICATIVE:
                        voxel = std::shared_ptr<MultiplicativeGaussianVoxel>(new MultiplicativeGaussianVoxel);
                        break;
                }
                found = _voxelMap.insert(found, std::make_pair(coord, voxel));
            }

            auto &voxel = found->second;
            voxel->append(cloud.at(i));
        }

    }

    void getCorrespondences(const std::vector<PointWithCov> pvList,
                            std::vector<GuassinVoxelMatchInfo<PointWithCov>> &matchList,
                            NeighborSearchMethod searchMethod)
    {
        matchList.clear();
        auto offsets = neighborOffsets(searchMethod);

        std::vector<std::vector<GuassinVoxelMatchInfo<PointWithCov>>> corrs(MP_PROC_NUM);
        for (auto &c : corrs)
            c.reserve((pvList.size() * offsets.size()) / MP_PROC_NUM);
//        static std::ofstream matchOfs = std::ofstream("/home/w/Data/fastlio_match.txt", std::ios::trunc | std::ios::in);
//        static int iter = 0;
//        matchOfs << "Iter: " << iter++ << std::endl;

#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
        for (int i = 0; i < pvList.size(); i++)
        {
            //matchOfs << "matched id: " << i << std::endl;
            Eigen::Vector3i coord = voxelCoord(Eigen::Vector3d(pvList[i].pw.x(), pvList[i].pw.y(), pvList[i].pw.z()));
            for (const auto &offset : offsets)
            {
                auto voxel = lookupVoxel(coord + offset);
                if (voxel != nullptr)
                {
                    corrs[omp_get_thread_num()].emplace_back(pvList[i], voxel);
//                    const Eigen::Vector3d mean_A = pvList[i].pw;
//                    const auto &cov_A = pvList[i].neighCov;
//                    matchOfs << "meanA trans:" << mean_A.transpose() << std::endl;
//                    matchOfs << "CovA:" << cov_A.transpose() << std::endl;
//                    matchOfs << "voxel meanB: " << voxel->_mean.transpose() << std::endl;
//                    matchOfs << "voxel cov: " << voxel->_cov << std::endl;
                }
            }
        }

        matchList.reserve(pvList.size() * offsets.size());
        for (const auto &c : corrs)
            matchList.insert(matchList.end(), c.begin(), c.end());

    }

    void getCorrespondences(const typename pcl::PointCloud<PointT>::Ptr inCloudPtr,
                            std::vector<GuassinVoxelMatchInfo<PointT>> &matchList,
                            NeighborSearchMethod searchMethod)
    {
        matchList.clear();
        auto offsets = neighborOffsets(searchMethod);

        std::vector<std::vector<GuassinVoxelMatchInfo<PointT>>> corrs(MP_PROC_NUM);
        for (auto &c : corrs)
            c.reserve((inCloudPtr->size() * offsets.size()) / MP_PROC_NUM);
//        static std::ofstream matchOfs = std::ofstream("/home/w/Data/fastlio_match.txt", std::ios::trunc | std::ios::in);
//        static int iter = 0;
//        matchOfs << "Iter: " << iter++ << std::endl;

#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
        for (int i = 0; i < inCloudPtr->size(); i++)
        {
            //matchOfs << "matched id: " << i << std::endl;
            Eigen::Vector3i coord = voxelCoord(Eigen::Vector3d(inCloudPtr->points[i].x, inCloudPtr->points[i].y, inCloudPtr->points[i].z));
            for (const auto &offset : offsets)
            {
                auto voxel = lookupVoxel(coord + offset);
                if (voxel != nullptr)
                    corrs[omp_get_thread_num()].template emplace_back(inCloudPtr->points[i], voxel);
            }
        }

        matchList.reserve(inCloudPtr->size() * offsets.size());
        for (const auto &c : corrs)
            matchList.insert(matchList.end(), c.begin(), c.end());

    }

    void clear()
    { _voxelMap.clear(); }

private:
    double _voxelResolution;
    VoxelAccumulationMode _voxelMode;

    VoxelMap _voxelMap;


};


#endif
