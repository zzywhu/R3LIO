//
// Created by w on 2022/11/23.
//

#include "RegistrationLoam.h"
using namespace std;

LoamTracker::Config LoamTracker::_config;

LoamTracker::LoamTracker()
{
    _isDegenerate=false;
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);


    downSizeFilterCorner.setLeafSize(_config._mappingCornerLeafSize, _config._mappingCornerLeafSize, _config._mappingCornerLeafSize);
    downSizeFilterSurf.setLeafSize(_config._mappingSurfLeafSize, _config._mappingSurfLeafSize, _config._mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(_config._mappingSurfLeafSize, _config._mappingSurfLeafSize, _config._mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(_config._surroundingKeyframeDensity, _config._surroundingKeyframeDensity, _config._surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    allocateMemory();
}

void LoamTracker::allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
    laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
    laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
    laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

    laserCloudOri.reset(new pcl::PointCloud<PointType>());
    coeffSel.reset(new pcl::PointCloud<PointType>());

    laserCloudOriCornerVec.resize(config()._nScans * config()._nHorizonScan);
    coeffSelCornerVec.resize(config()._nScans * config()._nHorizonScan);
    laserCloudOriCornerFlag.resize(config()._nScans * config()._nHorizonScan);
    laserCloudOriSurfVec.resize(config()._nScans * config()._nHorizonScan);
    coeffSelSurfVec.resize(config()._nScans * config()._nHorizonScan);
    laserCloudOriSurfFlag.resize(config()._nScans * config()._nHorizonScan);

    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

    laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
    laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
    nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

    for (int i = 0; i < 6; ++i)
        transformTobeMapped[i] = 0;

    matP.setZero();
}

void LoamTracker::setInitGuess(Eigen::Matrix4d& guess)
{
    Eigen::Affine3d initPose(guess);
    pcl::getTranslationAndEulerAngles(initPose.cast<float>(),
                                      transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                      transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    std::printf("Init lidar guess: %f, %f, %f, %f, %f, %f\n",
                transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
}

Eigen::Affine3f LoamTracker::track(const double &timestamp,
                                   const pcl::PointCloud<PointType>::Ptr &laserCloudCorner,
                                   const pcl::PointCloud<PointType>::Ptr &laserCloudSurf)
{
    // extract time stamp
    timeLaserCloudInfoLast = timestamp;

    // extract info and feature cloud
    laserCloudCornerLast = laserCloudCorner;
    laserCloudSurfLast = laserCloudSurf;

    std::cout << "Corner/Surf input size:" << laserCloudCornerLast->size() << "/" << laserCloudSurfLast->size() << std::endl;

    std::lock_guard<std::mutex> lock(mtx);

    if (timeLaserCloudInfoLast - timeLastProcessing >= _config._mappingProcessInterval)
    {

        timeLastProcessing = timeLaserCloudInfoLast;

        //updateInitialGuess();

        extractSurroundingKeyFrames();

        downsampleCurrentScan();

        scan2MapOptimization();

        saveKeyFramesAndFactor();
    }
    return getLastPose();
}


void LoamTracker::pointAssociateToMap(PointType const *const pi, PointType *const po)
{
    po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
}


pcl::PointCloud<PointType>::Ptr LoamTracker::transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom->x + transCur(0, 1) * pointFrom->y + transCur(0, 2) * pointFrom->z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom->x + transCur(1, 1) * pointFrom->y + transCur(1, 2) * pointFrom->z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom->x + transCur(2, 1) * pointFrom->y + transCur(2, 2) * pointFrom->z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
}


PointTypePose LoamTracker::trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}

//void LoamTracker::updateInitialGuess()
//{
//    static Eigen::Affine3f lastImuTransformation;
//    // initialization
//    if (cloudKeyPoses3D->points.empty())
//    {
//        transformTobeMapped[0] = cloudInfo.imuRollInit;
//        transformTobeMapped[1] = cloudInfo.imuPitchInit;
//        transformTobeMapped[2] = cloudInfo.imuYawInit;
//
//        std::printf("Init guess: %f, %ld, %f, %f, %f", cloudInfo.header, cloudInfo.odomAvailable, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
//
//        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
//        return;
//    }
//
//    // use imu pre-integration estimation for pose guess
//    if (cloudInfo.odomAvailable == true)
//    {
//        transformTobeMapped[0] = cloudInfo.initialGuessRoll;
//        transformTobeMapped[1] = cloudInfo.initialGuessPitch;
//        transformTobeMapped[2] = cloudInfo.initialGuessYaw;
//
//        transformTobeMapped[3] = cloudInfo.initialGuessX;
//        transformTobeMapped[4] = cloudInfo.initialGuessY;
//        transformTobeMapped[5] = cloudInfo.initialGuessZ;
//        std::printf("Init guess: %f, %ld, %f, %f, %f, %f, %f, %f", cloudInfo.header, cloudInfo.odomAvailable, cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch,
//                    cloudInfo.initialGuessYaw, cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ);
//
//        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
//        return;
//    }
//
//    //ROS_ERROR("not recommended");
//    // use imu incremental estimation for pose guess (only rotation)
//    if (cloudInfo.imuAvailable == true)
//    {
//        Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
//        Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
//
//        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
//        Eigen::Affine3f transFinal = transTobe * transIncre;
//        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
//                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
//
//        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
//        return;
//    }
//}

void LoamTracker::extractNearby()
{
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), _config._surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    for (int i = 0; i < (int) pointSearchInd.size(); ++i)
    {
        int id = pointSearchInd[i];
        surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

    // also extract some latest key frames in case the robot rotates in one position
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i)
    {
        if (timeLaserCloudInfoLast - cloudKeyPoses6D->points[i].time < 10.0)
            surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
        else
            break;
    }

    extractCloud(surroundingKeyPosesDS);
}

void LoamTracker::extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
{
    std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
    std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

    laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
    laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

    // extract surrounding map
#pragma omp parallel for num_threads(MP_PROC_NUM)
    for (int i = 0; i < (int) cloudToExtract->size(); ++i)
    {
        if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > _config._surroundingKeyframeSearchRadius)
            continue;
        int thisKeyInd = (int) cloudToExtract->points[i].intensity;
        laserCloudCornerSurroundingVec[i] = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        laserCloudSurfSurroundingVec[i] = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }

    // fuse the map
    laserCloudCornerFromMap->clear();
    laserCloudSurfFromMap->clear();
    for (int i = 0; i < (int) cloudToExtract->size(); ++i)
    {
        *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
        *laserCloudSurfFromMap += laserCloudSurfSurroundingVec[i];
    }

    // Downsample the surrounding corner key frames (or map)
    downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
    downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
    laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
    // Downsample the surrounding surf key frames (or map)
    downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
    downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
}

void LoamTracker::extractSurroundingKeyFrames()
{
    if (cloudKeyPoses3D->points.empty() == true)
        return;
    extractNearby();
}

void LoamTracker::downsampleCurrentScan()
{
    // Downsample cloud from current scan
    laserCloudCornerLastDS->clear();
    downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
    downSizeFilterCorner.filter(*laserCloudCornerLastDS);
    laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

    laserCloudSurfLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
    downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();

    std::cout << "Corner/Surf input down size:" << laserCloudCornerLastDSNum << "/" << laserCloudSurfLastDSNum << std::endl;
}

void LoamTracker::updatePointAssociateToMap()
{
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
}

void LoamTracker::cornerOptimization()
{
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(MP_PROC_NUM)
    for (int i = 0; i < laserCloudCornerLastDSNum; i++)
    {
        PointType pointOri, pointSel, coeff;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        pointOri = laserCloudCornerLastDS->points[i];
        pointAssociateToMap(&pointOri, &pointSel);
        kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

        if (pointSearchSqDis[4] < 1.0)
        {
            float cx = 0, cy = 0, cz = 0;
            for (int j = 0; j < 5; j++)
            {
                cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
            }
            cx /= 5;
            cy /= 5;
            cz /= 5;

            float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
            for (int j = 0; j < 5; j++)
            {
                float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                a11 += ax * ax;
                a12 += ax * ay;
                a13 += ax * az;
                a22 += ay * ay;
                a23 += ay * az;
                a33 += az * az;
            }
            a11 /= 5;
            a12 /= 5;
            a13 /= 5;
            a22 /= 5;
            a23 /= 5;
            a33 /= 5;

            matA1.at<float>(0, 0) = a11;
            matA1.at<float>(0, 1) = a12;
            matA1.at<float>(0, 2) = a13;
            matA1.at<float>(1, 0) = a12;
            matA1.at<float>(1, 1) = a22;
            matA1.at<float>(1, 2) = a23;
            matA1.at<float>(2, 0) = a13;
            matA1.at<float>(2, 1) = a23;
            matA1.at<float>(2, 2) = a33;

            cv::eigen(matA1, matD1, matV1);

            if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
            {

                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
                                  + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
                                  + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

                float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

                float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
                            + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

                float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1))
                             - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))
                             + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                float ld2 = a012 / l12;

                float s = 1 - 0.9 * fabs(ld2);

                coeff.x = s * la;
                coeff.y = s * lb;
                coeff.z = s * lc;
                coeff.intensity = s * ld2;

                if (s > 0.1)
                {
                    laserCloudOriCornerVec[i] = pointOri;
                    coeffSelCornerVec[i] = coeff;
                    laserCloudOriCornerFlag[i] = true;
                }
            }
        }
    }
}

void LoamTracker::surfOptimization()
{
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(MP_PROC_NUM)
    for (int i = 0; i < laserCloudSurfLastDSNum; i++)
    {
        PointType pointOri, pointSel, coeff;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        pointOri = laserCloudSurfLastDS->points[i];
        pointAssociateToMap(&pointOri, &pointSel);
        kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        Eigen::Matrix<float, 5, 3> matA0;
        Eigen::Matrix<float, 5, 1> matB0;
        Eigen::Vector3f matX0;

        matA0.setZero();
        matB0.fill(-1);
        matX0.setZero();

        if (pointSearchSqDis[4] < 1.0)
        {
            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
            }

            matX0 = matA0.colPivHouseholderQr().solve(matB0);

            float pa = matX0(0, 0);
            float pb = matX0(1, 0);
            float pc = matX0(2, 0);
            float pd = 1;

            float ps = sqrt(pa * pa + pb * pb + pc * pc);
            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                         pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                         pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }

            if (planeValid)
            {
                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                                                          + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                coeff.x = s * pa;
                coeff.y = s * pb;
                coeff.z = s * pc;
                coeff.intensity = s * pd2;

                if (s > 0.1)
                {
                    laserCloudOriSurfVec[i] = pointOri;
                    coeffSelSurfVec[i] = coeff;
                    laserCloudOriSurfFlag[i] = true;
                }
            }
        }
    }
}

void LoamTracker::combineOptimizationCoeffs()
{
    // combine corner coeffs
    for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
    {
        if (laserCloudOriCornerFlag[i] == true)
        {
            laserCloudOri->push_back(laserCloudOriCornerVec[i]);
            coeffSel->push_back(coeffSelCornerVec[i]);
        }
    }
    // combine surf coeffs
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
    {
        if (laserCloudOriSurfFlag[i] == true)
        {
            laserCloudOri->push_back(laserCloudOriSurfVec[i]);
            coeffSel->push_back(coeffSelSurfVec[i]);
        }
    }
    // reset flag for next iteration
    std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
}

bool LoamTracker::LMOptimization(int iterCount)
{
     //This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
     //lidar <- camera      ---     camera <- lidar
     //x = z                ---     x = y
     //y = x                ---     y = z
     //z = y                ---     z = x
     //roll = yaw           ---     roll = pitch
     //pitch = roll         ---     pitch = yaw
     //yaw = pitch          ---     yaw = roll

    // lidar -> camera
    float srx = sin(transformTobeMapped[1]);
    float crx = cos(transformTobeMapped[1]);
    float sry = sin(transformTobeMapped[2]);
    float cry = cos(transformTobeMapped[2]);
    float srz = sin(transformTobeMapped[0]);
    float crz = cos(transformTobeMapped[0]);

    int laserCloudSelNum = laserCloudOri->size();
    if (laserCloudSelNum < 50)
    {
        return false;
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

    PointType pointOri, coeff;

    for (int i = 0; i < laserCloudSelNum; i++)
    {
        // lidar -> camera
        pointOri.x = laserCloudOri->points[i].y;
        pointOri.y = laserCloudOri->points[i].z;
        pointOri.z = laserCloudOri->points[i].x;
        // lidar -> camera
        coeff.x = coeffSel->points[i].y;
        coeff.y = coeffSel->points[i].z;
        coeff.z = coeffSel->points[i].x;
        coeff.intensity = coeffSel->points[i].intensity;
        // in camera
        float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x
                    + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y
                    + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

        float ary = ((cry * srx * srz - crz * sry) * pointOri.x
                     + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x
                    + ((-cry * crz - srx * sry * srz) * pointOri.x
                       + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

        float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x
                    + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y
                    + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
        // lidar -> camera
        matA.at<float>(i, 0) = arz;
        matA.at<float>(i, 1) = arx;
        matA.at<float>(i, 2) = ary;
        matA.at<float>(i, 3) = coeff.z;
        matA.at<float>(i, 4) = coeff.x;
        matA.at<float>(i, 5) = coeff.y;
        matB.at<float>(i, 0) = -coeff.intensity;
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0)
    {

        cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

        cv::eigen(matAtA, matE, matV);
        matV.copyTo(matV2);

        _isDegenerate = false;
        float eignThre[6] = {100, 100, 100, 100, 100, 100};
        for (int i = 5; i >= 0; i--)
        {
            if (matE.at<float>(0, i) < eignThre[i])
            {
                for (int j = 0; j < 6; j++)
                    matV2.at<float>(i, j) = 0;
                _isDegenerate = true;
            }
            else
            {
                break;
            }
        }
        matP = matV.inv() * matV2;
    }

    if (_isDegenerate)
    {
        cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
        matX.copyTo(matX2);
        matX = matP * matX2;
    }

    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                        pow(matX.at<float>(4, 0) * 100, 2) +
                        pow(matX.at<float>(5, 0) * 100, 2));

    if (deltaR < 0.05 && deltaT < 0.05)
    {
        return true; // converged
    }
    return false; // keep optimizing
}

void LoamTracker::scan2MapOptimization()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (laserCloudCornerLastDSNum > _config._edgeFeatureMinValidNum && laserCloudSurfLastDSNum > _config._surfFeatureMinValidNum)
    {
        kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

        for (int iterCount = 0; iterCount < 30; iterCount++)
        {
            laserCloudOri->clear();
            coeffSel->clear();

            cornerOptimization();
            surfOptimization();

            combineOptimizationCoeffs();

            if (LMOptimization(iterCount) == true)
                break;
        }

        transformUpdate();
    }
    else
        std::printf("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);

}

void LoamTracker::transformUpdate()
{
    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], _config._rotationTollerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], _config._rotationTollerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], _config._zTollerance);
}

float LoamTracker::constraintTransformation(float value, float limit)
{
    if (value < -limit)
        value = -limit;
    if (value > limit)
        value = limit;
    return value;
}

bool LoamTracker::saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll)  < _config._surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < _config._surroundingkeyframeAddingAngleThreshold &&
        abs(yaw)   < _config._surroundingkeyframeAddingAngleThreshold &&
        sqrt(x*x + y*y + z*z) < _config._surroundingkeyframeAddingDistThreshold)
        return false;

    return true;
}

void LoamTracker::addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

void LoamTracker::saveKeyFramesAndFactor()
{
    if (saveFrame() == false)
        return;

    // odom factor
    addOdomFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    //save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
    thisPose6D.roll  = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserCloudInfoLast;
    cloudKeyPoses6D->push_back(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
    pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

    // save key frame cloud
    cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

}
