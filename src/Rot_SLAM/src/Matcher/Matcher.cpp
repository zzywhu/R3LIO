#include <Eigen/Core>
#include <flann/flann.hpp>
#ifdef TEASER_ON
#include <teaser/geometry.h>
#endif
#include "Matcher/Matcher.h"


#ifdef TEASER_ON
bool Matcher::findFeatureCorrespondenceFPFH(teaser::PointCloud &tgtCloud, teaser::PointCloud& srcCloud,
                                            std::vector<std::pair<int, int>>& correspondences,
                                            double normal_search_radius, double fpfh_search_radius,
                                            bool use_absolute_scale, bool use_crosscheck,
                                            bool use_tuple_test, float tuple_scale)
{
    FPFHEstimation fpfh;
    auto objDescriptors = fpfh.computeFPFHFeatures(srcCloud, normal_search_radius, fpfh_search_radius);
    auto sceneDescriptors = fpfh.computeFPFHFeatures(tgtCloud, normal_search_radius, fpfh_search_radius);
    std::cout<<"Src desc size:"<<objDescriptors->size();
    std::cout<<"Dst desc size:"<<sceneDescriptors->size();
    correspondences = calculateCorrespondences(srcCloud, tgtCloud, *objDescriptors, *sceneDescriptors,
                                               use_absolute_scale, use_crosscheck, use_tuple_test, tuple_scale);
    std::cout<<"Found correspondences:"<<correspondences.size()<<std::endl;
    return true;
}

bool Matcher::findFeatureCorrespondenceFPFH(teaser::PointCloud& tgtCloud, teaser::PointCloud& srcCloud,
                                            teaser::PointCloud& tgtKpts, teaser::PointCloud& srcKpts,
                                            pcl::PointIndicesPtr tgtKptIndices,  pcl::PointIndicesPtr srcKptIndices,
                                            std::vector<std::pair<int, int>>& correspondences,
                                            double normal_search_radius, double fpfh_search_radius,
                                            bool use_absolute_scale, bool use_crosscheck,
                                            bool use_tuple_test, float tuple_scale)
{
    FPFHEstimation fpfh;
    auto objDescriptors = fpfh.computeFPFHFeatures(srcCloud, srcKptIndices, normal_search_radius, fpfh_search_radius);
    auto sceneDescriptors = fpfh.computeFPFHFeatures(tgtCloud, tgtKptIndices,normal_search_radius, fpfh_search_radius);
    std::cout<<"Src desc size:"<<objDescriptors->size();
    std::cout<<"Dst desc size:"<<sceneDescriptors->size();
    correspondences = calculateCorrespondences(srcKpts, tgtKpts, *objDescriptors, *sceneDescriptors,
                                               use_absolute_scale, use_crosscheck, use_tuple_test, tuple_scale);
    std::cout<<"Found correspondences:"<<correspondences.size()<<std::endl;
    return true;
}


std::vector<std::pair<int, int>> Matcher::calculateCorrespondences(teaser::PointCloud &source_points, teaser::PointCloud &target_points,
                                                                   FPFHCloud &source_features, FPFHCloud &target_features, bool use_absolute_scale,
                                                                   bool use_crosscheck, bool use_tuple_test, float tuple_scale)
{

    Feature cloud_features;
    _pointcloud.push_back(source_points);
    _pointcloud.push_back(target_points);

    // It compute the _globalScale required to set correctly the search radius
    normalizePoints(use_absolute_scale);

    for (auto &f : source_features)
    {
        Eigen::VectorXf fpfh(33);
        for (int i = 0; i < 33; i++)
            fpfh(i) = f.histogram[i];
        cloud_features.push_back(fpfh);
    }
    _features.push_back(cloud_features);

    cloud_features.clear();
    for (auto &f : target_features)
    {
        Eigen::VectorXf fpfh(33);
        for (int i = 0; i < 33; i++)
            fpfh(i) = f.histogram[i];
        cloud_features.push_back(fpfh);
    }
    _features.push_back(cloud_features);

    advancedMatching(use_crosscheck, use_tuple_test, tuple_scale);

    return _corres;
}


void Matcher::normalizePoints(bool use_absolute_scale)
{
    int num = 2;
    float scale = 0;

    _means.clear();

    for (int i = 0; i < num; ++i)
    {
        float max_scale = 0;

        // compute mean
        Eigen::Vector3f mean;
        mean.setZero();

        int npti = _pointcloud[i].size();
        for (int ii = 0; ii < npti; ++ii)
        {
            Eigen::Vector3f p(_pointcloud[i][ii].x, _pointcloud[i][ii].y, _pointcloud[i][ii].z);
            mean = mean + p;
        }
        mean = mean / npti;
        _means.push_back(mean);

        for (int ii = 0; ii < npti; ++ii)
        {
            _pointcloud[i][ii].x -= mean(0);
            _pointcloud[i][ii].y -= mean(1);
            _pointcloud[i][ii].z -= mean(2);
        }

        // compute scale
        for (int ii = 0; ii < npti; ++ii)
        {
            Eigen::Vector3f p(_pointcloud[i][ii].x, _pointcloud[i][ii].y, _pointcloud[i][ii].z);
            float temp = p.norm(); // because we extract mean in the previous stage.
            if (temp > max_scale)
            {
                max_scale = temp;
            }
        }

        if (max_scale > scale)
        {
            scale = max_scale;
        }
    }

    // mean of the scale variation
    if (use_absolute_scale)
    {
        _globalScale = 1.0f;
    }
    else
    {
        _globalScale = scale; // second choice: we keep the maximum scale.
    }

    if (_globalScale != 1.0f)
    {
        for (int i = 0; i < num; ++i)
        {
            int npti = _pointcloud[i].size();
            for (int ii = 0; ii < npti; ++ii)
            {
                _pointcloud[i][ii].x /= _globalScale;
                _pointcloud[i][ii].y /= _globalScale;
                _pointcloud[i][ii].z /= _globalScale;
            }
        }
    }
}

void Matcher::advancedMatching(bool use_crosscheck, bool use_tuple_test, float tuple_scale)
{

    int fi = 0; // source idx
    int fj = 1; // destination idx

    bool swapped = false;

    if (_pointcloud[fj].size() > _pointcloud[fi].size())
    {
        int temp = fi;
        fi = fj;
        fj = temp;
        swapped = true;
    }

    int nPti = _pointcloud[fi].size();
    int nPtj = _pointcloud[fj].size();

    ///////////////////////////
    /// Build FLANNTREE
    ///////////////////////////
    KDTree feature_tree_i(flann::KDTreeSingleIndexParams(15));
    buildKDTree(_features[fi], &feature_tree_i);

    KDTree feature_tree_j(flann::KDTreeSingleIndexParams(15));
    buildKDTree(_features[fj], &feature_tree_j);

    std::vector<int> corres_K, corres_K2;
    std::vector<float> dis;
    std::vector<int> ind;

    std::vector<std::pair<int, int>> corres;
    std::vector<std::pair<int, int>> corres_cross;
    std::vector<std::pair<int, int>> corres_ij;
    std::vector<std::pair<int, int>> corres_ji;

    ///////////////////////////
    /// INITIAL MATCHING
    ///////////////////////////
    std::vector<int> i_to_j(nPti, -1);
    for (int j = 0; j < nPtj; j++)
    {
        searchKDTree(&feature_tree_i, _features[fj][j], corres_K, dis, 1);
        int i = corres_K[0];
        if (i_to_j[i] == -1)
        {
            searchKDTree(&feature_tree_j, _features[fi][i], corres_K, dis, 1);
            int ij = corres_K[0];
            i_to_j[i] = ij;
        }
        corres_ji.push_back(std::pair<int, int>(i, j));
    }

    for (int i = 0; i < nPti; i++)
    {
        if (i_to_j[i] != -1)
            corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
    }

    int ncorres_ij = corres_ij.size();
    int ncorres_ji = corres_ji.size();

    // corres = corres_ij + corres_ji;
    for (int i = 0; i < ncorres_ij; ++i)
        corres.push_back(std::pair<int, int>(corres_ij[i].first, corres_ij[i].second));
    for (int j = 0; j < ncorres_ji; ++j)
        corres.push_back(std::pair<int, int>(corres_ji[j].first, corres_ji[j].second));

    ///////////////////////////
    /// CROSS CHECK
    /// input : corres_ij, corres_ji
    /// output : corres
    ///////////////////////////
    if (use_crosscheck)
    {
        std::cout << "CROSS CHECK" << std::endl;
        // build data structure for cross check
        corres.clear();
        corres_cross.clear();
        std::vector<std::vector<int>> Mi(nPti);
        std::vector<std::vector<int>> Mj(nPtj);

        int ci, cj;
        for (int i = 0; i < ncorres_ij; ++i)
        {
            ci = corres_ij[i].first;
            cj = corres_ij[i].second;
            Mi[ci].push_back(cj);
        }
        for (int j = 0; j < ncorres_ji; ++j)
        {
            ci = corres_ji[j].first;
            cj = corres_ji[j].second;
            Mj[cj].push_back(ci);
        }

        // cross check
        for (int i = 0; i < nPti; ++i)
        {
            for (int ii = 0; ii < Mi[i].size(); ++ii)
            {
                int j = Mi[i][ii];
                for (int jj = 0; jj < Mj[j].size(); ++jj)
                {
                    if (Mj[j][jj] == i)
                    {
                        corres.push_back(std::pair<int, int>(i, j));
                        corres_cross.push_back(std::pair<int, int>(i, j));
                    }
                }
            }
        }
    }
    else
    {
        std::cout << "Skipping Cross Check." << std::endl;
    }

    ///////////////////////////
    /// TUPLE CONSTRAINT
    /// input : corres
    /// output : corres
    ///////////////////////////
    if (use_tuple_test && tuple_scale != 0)
    {
        std::cout << "TUPLE CONSTRAINT" << std::endl;
        srand(time(NULL));
        int rand0, rand1, rand2;
        int idi0, idi1, idi2;
        int idj0, idj1, idj2;
        float scale = tuple_scale;
        int ncorr = corres.size();
        int number_of_trial = ncorr * 100;
        std::vector<std::pair<int, int>> corres_tuple;

        for (int i = 0; i < number_of_trial; i++)
        {
            rand0 = rand() % ncorr;
            rand1 = rand() % ncorr;
            rand2 = rand() % ncorr;

            idi0 = corres[rand0].first;
            idj0 = corres[rand0].second;
            idi1 = corres[rand1].first;
            idj1 = corres[rand1].second;
            idi2 = corres[rand2].first;
            idj2 = corres[rand2].second;

            // collect 3 points from i-th fragment
            Eigen::Vector3f pti0 = {_pointcloud[fi][idi0].x, _pointcloud[fi][idi0].y,
                                    _pointcloud[fi][idi0].z};
            Eigen::Vector3f pti1 = {_pointcloud[fi][idi1].x, _pointcloud[fi][idi1].y,
                                    _pointcloud[fi][idi1].z};
            Eigen::Vector3f pti2 = {_pointcloud[fi][idi2].x, _pointcloud[fi][idi2].y,
                                    _pointcloud[fi][idi2].z};

            float li0 = (pti0 - pti1).norm();
            float li1 = (pti1 - pti2).norm();
            float li2 = (pti2 - pti0).norm();

            // collect 3 points from j-th fragment
            Eigen::Vector3f ptj0 = {_pointcloud[fj][idj0].x, _pointcloud[fj][idj0].y,
                                    _pointcloud[fj][idj0].z};
            Eigen::Vector3f ptj1 = {_pointcloud[fj][idj1].x, _pointcloud[fj][idj1].y,
                                    _pointcloud[fj][idj1].z};
            Eigen::Vector3f ptj2 = {_pointcloud[fj][idj2].x, _pointcloud[fj][idj2].y,
                                    _pointcloud[fj][idj2].z};

            float lj0 = (ptj0 - ptj1).norm();
            float lj1 = (ptj1 - ptj2).norm();
            float lj2 = (ptj2 - ptj0).norm();

            if ((li0 * scale < lj0) && (lj0 < li0 / scale) && (li1 * scale < lj1) &&
                (lj1 < li1 / scale) && (li2 * scale < lj2) && (lj2 < li2 / scale))
            {
                corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
                corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
                corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
            }
        }
        corres.clear();

        for (size_t i = 0; i < corres_tuple.size(); ++i)
            corres.push_back(std::pair<int, int>(corres_tuple[i].first, corres_tuple[i].second));
    }
    else
    {
        std::cout << "Skipping Tuple Constraint." << std::endl;
    }

    if (swapped)
    {
        std::vector<std::pair<int, int>> temp;
        for (size_t i = 0; i < corres.size(); i++)
            temp.push_back(std::pair<int, int>(corres[i].second, corres[i].first));
        corres.clear();
        corres = temp;
    }
    _corres = corres;

    ///////////////////////////
    /// ERASE DUPLICATES
    /// input : _corres
    /// output : _corres
    ///////////////////////////
    std::sort(_corres.begin(), _corres.end());
    _corres.erase(std::unique(_corres.begin(), _corres.end()), _corres.end());
}
#endif

template<typename T>
void Matcher::buildKDTree(const std::vector<T> &data, Matcher::KDTree *tree)
{
    int rows, dim;
    rows = (int) data.size();
    dim = (int) data[0].size();
    std::vector<float> dataset(rows * dim);
    flann::Matrix<float> dataset_mat(&dataset[0], rows, dim);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < dim; j++)
            dataset[i * dim + j] = data[i][j];
    KDTree temp_tree(dataset_mat, flann::KDTreeSingleIndexParams(15));
    temp_tree.buildIndex();
    *tree = temp_tree;
}

template<typename T>
void Matcher::searchKDTree(Matcher::KDTree *tree, const T &input, std::vector<int> &indices,
                           std::vector<float> &dists, int nn)
{
    int rows_t = 1;
    int dim = input.size();

    std::vector<float> query;
    query.resize(rows_t * dim);
    for (int i = 0; i < dim; i++)
        query[i] = input(i);
    flann::Matrix<float> query_mat(&query[0], rows_t, dim);

    indices.resize(rows_t * nn);
    dists.resize(rows_t * nn);
    flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
    flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);

    tree->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
}

