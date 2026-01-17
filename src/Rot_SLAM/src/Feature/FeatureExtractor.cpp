#include <pcl/features/normal_3d.h>

#ifdef TEASER_ON
#include "teaser/utils.h"
#endif

#include "FeatureExtractor.h"


double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
    double resolution = 0.0;
    int numberOfPoints = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (!std::isfinite((*cloud)[i].x) )
            continue;

        // Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
        if (nres == 2)
        {
            resolution += sqrt(squaredDistances[1]);
            ++numberOfPoints;
        }
    }
    if (numberOfPoints != 0)
        resolution /= numberOfPoints;

    return resolution;
}

void extractISSKeyPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr keyPoints,
                         pcl::PointIndicesPtr kptIndices,
                         double salientRadiusRatios, double nonMaxRadiusRatios, double res)
{
    // compute resolution of cloud
    if(res==0)
    {
        res = computeCloudResolution(cloud);
        std::cout<<"resolution: "<<res<<std::endl;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> issExtractor;

    issExtractor.setMinNeighbors(5);
    issExtractor.setThreshold21(0.975);
    issExtractor.setThreshold32(0.975);
    issExtractor.setNumberOfThreads(4);

    issExtractor.setInputCloud(cloud);
    issExtractor.setSearchMethod(tree);
    issExtractor.setSalientRadius(salientRadiusRatios*res);  // 0.5
    issExtractor.setNonMaxRadius(nonMaxRadiusRatios*res);
    issExtractor.compute(*keyPoints);

    kptIndices->indices = issExtractor.getKeypointsIndices()->indices;
    kptIndices->header = issExtractor.getKeypointsIndices()->header;
}

float cloudCurvature[400000];
void extractFeature(PointCloudXYZI *plBuff, const int &nScans,
                    PointCloudXYZI &cornerPointsSharp,
                    PointCloudXYZI &surfPointsFlat,
                    int nSeg)
{
    cornerPointsSharp.clear();
    surfPointsFlat.clear();
    int cloudSortInd[400000];
    int cloudNeighborPicked[400000];
    std::vector<int> scanStartInd(nScans, 0);
    std::vector<int> scanEndInd(nScans, 0);

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < nScans; i++)
    {
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += plBuff[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }
    const int cloudSize = laserCloud->size();
    for (int i = 5; i < cloudSize - 5; i++)
    {
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x +
                      laserCloud->points[i - 2].x + laserCloud->points[i - 1].x -
                      10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x +
                      laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y +
                      laserCloud->points[i - 2].y + laserCloud->points[i - 1].y -
                      10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y +
                      laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z +
                      laserCloud->points[i - 2].z + laserCloud->points[i - 1].z -
                      10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z +
                      laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;
    }

    for (int i = 0; i < nScans; i++)
    {
        if (scanEndInd[i] - scanStartInd[i] < 50)
            continue;
        for (int j = 0; j < nSeg; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / nSeg;
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / nSeg - 1;

            std::sort(cloudSortInd + sp, cloudSortInd + ep + 1,[](int &i, int &j){ return cloudCurvature[i] < cloudCurvature[j]; });

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {
                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                    else
                        break;

                    cloudNeighborPicked[ind] = 1;

                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1)
                {
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 5)
                        break;

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            break;

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
        }
    }
    std::cout<<"Surf/Line cloud size:"<<surfPointsFlat.size()<<"/"<<cornerPointsSharp.size()<<std::endl;
}

pcl::PointCloud<pcl::Normal>::Ptr getNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud, double normals_radius)
{
    pcl::PointCloud<pcl::Normal>::Ptr normalsPtr(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setInputCloud(incloud);
    norm_est.setRadiusSearch(normals_radius);
    norm_est.compute(*normalsPtr);
    return normalsPtr;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr getFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                       pcl::PointCloud<pcl::Normal>::Ptr normals,
                                                       double feature_radius)
{
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features = pcl::PointCloud<pcl::FPFHSignature33>::Ptr(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method_ptr = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
    fpfh_est.setInputCloud(cloud);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setSearchMethod(search_method_ptr);
    fpfh_est.setRadiusSearch(feature_radius);
    fpfh_est.compute(*features);
    return features;
}


#ifdef TEASER_ON
FPFHCloudPtr FPFHEstimation::computeFPFHFeatures(const teaser::PointCloud &input_cloud,
                                                 double normal_search_radius,
                                                 double fpfh_search_radius)
{

    // Intermediate variables
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    FPFHCloudPtr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto &i : input_cloud)
    {
        pcl::PointXYZ p(i.x, i.y, i.z);
        pcl_input_cloud->push_back(p);
    }

    // Estimate normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(pcl_input_cloud);
    normalEstimation.setRadiusSearch(normal_search_radius);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    // Estimate FPFH
    setInputCloud(pcl_input_cloud);
    setInputNormals(normals);
    setSearchMethod(kdtree);
    setRadiusSearch(fpfh_search_radius);
    compute(*descriptors);

    return descriptors;
}

FPFHCloudPtr FPFHEstimation::computeFPFHFeatures(const teaser::PointCloud &input_cloud,
                                                 pcl::PointIndicesPtr issIndices,
                                                 double normal_search_radius,
                                                 double fpfh_search_radius)
{
    // Intermediate variables
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    FPFHCloudPtr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto &i : input_cloud)
    {
        pcl::PointXYZ p(i.x, i.y, i.z);
        pcl_input_cloud->push_back(p);
    }

    // Estimate normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(pcl_input_cloud);
    normalEstimation.setRadiusSearch(normal_search_radius);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    // Estimate FPFH
    setInputCloud(pcl_input_cloud);
    setInputNormals(normals);
    setSearchMethod(kdtree);
    setRadiusSearch(fpfh_search_radius);
    setIndices(issIndices);
    compute(*descriptors);

    return descriptors;
}
#endif
