#include "LidarProcess.h"

#define RETURN0 0x00
#define RETURN0AND1 0x10
#define MAX_LINE_NUM 64

LidarProcess::Config LidarProcess::_config;

const bool time_list_cut_frame(PointType &x, PointType &y) {
    return (x.curvature < y.curvature);
}

LidarProcess::~LidarProcess() {
}

void LidarProcess::processCutFrameCloud(PPointCloud::Ptr pl_orig, const double &lidTime, std::deque<PointCloudXYZI::Ptr> &pcl_out,
                                        std::deque<double> &time_lidar, const int required_frame_num, int scan_count) {
    switch (_config._timeUnit) {
    case Sec:
        _timeUnitScale = 1.e3f;
        break;
    case Ms:
        _timeUnitScale = 1.f;
        break;
    case Us:
        _timeUnitScale = 1.e-3f;
        break;
    case Ns:
        _timeUnitScale = 1.e-6f;
        break;
    default:
        _timeUnitScale = 1.f;
        break;
    }
    _plFull.clear();

    int plsize = pl_orig->points.size();
    _plFull.reserve(plsize);
    std::cout << "LidarProcess::processCutFrameCloud plsize: " << plsize << std::endl;
    for (int i = 0; i < plsize; i++) {
        PointType added_pt;
        added_pt.normal_x = pl_orig->points[i].ring;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig->points[i].x;
        added_pt.y = pl_orig->points[i].y;
        added_pt.z = pl_orig->points[i].z;
        added_pt.intensity = pl_orig->points[i].intensity;
        added_pt.curvature = (pl_orig->points[i].timestamp - lidTime) * _timeUnitScale; //s to ms
        // if (i % _config._pointFilterNum == 0 && pl_orig->points[i].ring < _config._nScans)
        // {
        //     double dist2=added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z;
        //     double cossita=std::abs(added_pt.z)/sqrt(dist2);
        //     if ( dist2 > (_config._blindMin * _config._blindMin)&&
        //          dist2 < (_config._blindMax * _config._blindMax))
        //          //intensity correction
        //         added_pt.intensity=added_pt.intensity*dist2;
        //         _p
        //         lFull.points.push_back(added_pt);
        // }
        if (pl_orig->points[i].ring < _config._nScans) {
            double dist2 = added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z;
            double cossita = std::abs(added_pt.z) / sqrt(dist2);
            if (dist2 > (_config._blindMin * _config._blindMin) && dist2 < (_config._blindMax * _config._blindMax))
                //intensity correction
                added_pt.intensity = added_pt.intensity * dist2;
            _plFull.points.push_back(added_pt);
        }
    }

    sort(_plFull.points.begin(), _plFull.points.end(), time_list_cut_frame);

    //ms
    double last_frame_end_time = lidTime * _timeUnitScale;
    uint valid_num = 0;
    uint cut_num = 0;
    uint valid_pcl_size = _plFull.points.size();

    int required_cut_num = required_frame_num;

    if (scan_count < 3)
        required_cut_num = 1;

    PointCloudXYZI pcl_cut;
    for (uint i = 1; i < valid_pcl_size; i++) {
        valid_num++;
        _plFull[i].curvature += lidTime * 1000 - last_frame_end_time;
        pcl_cut.push_back(_plFull[i]);

        if (valid_num == (int((cut_num + 1) * valid_pcl_size / required_cut_num) - 1)) {
            cut_num++;
            time_lidar.push_back(last_frame_end_time);
            PointCloudXYZI::Ptr pcl_temp(new PointCloudXYZI());
            *pcl_temp = pcl_cut;
            pcl_out.push_back(pcl_temp);
            last_frame_end_time += _plFull[i].curvature;
            pcl_cut.clear();
            pcl_cut.reserve(valid_pcl_size * 2 / required_frame_num);
        }
    }
}

void LidarProcess::process(PPointCloud::Ptr pl_orig, const double &lidTime, PointCloudXYZI::Ptr &pcl_out) {
    switch (_config._timeUnit) {
    case Sec:
        _timeUnitScale = 1.e3f;
        break;
    case Ms:
        _timeUnitScale = 1.f;
        break;
    case Us:
        _timeUnitScale = 1.e-3f;
        break;
    case Ns:
        _timeUnitScale = 1.e-6f;
        break;
    default:
        _timeUnitScale = 1.f;
        break;
    }

    pcl_out->clear();

    const int &plsize = pl_orig->size();
    if (plsize == 0) return;
    pcl_out->reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * _config._scanRate; // scan angular velocity
    std::vector<bool> is_first(_config._nScans, true);
    std::vector<double> yaw_fp(_config._nScans, 0.0);   // yaw of first scan point
    std::vector<float> yaw_last(_config._nScans, 0.0);  // yaw of last scan point
    std::vector<float> time_last(_config._nScans, 0.0); // last offset time
    /*****************************************************************/
    if (pl_orig->points[plsize - 1].timestamp > 0)
        _isGivenOffsetTime = true;
    else
        _isGivenOffsetTime = false;

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;
        added_pt.normal_x = pl_orig->points[i].ring;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig->points[i].x;
        added_pt.y = pl_orig->points[i].y;
        added_pt.z = pl_orig->points[i].z;
        added_pt.intensity = pl_orig->points[i].intensity;
        added_pt.curvature = (pl_orig->points[i].timestamp - lidTime) * _timeUnitScale; //s to ms
                                                                                        //        std::cout<<std::setprecision(6)<<std::fixed<<added_pt.curvature<<std::endl;
        if (!_isGivenOffsetTime) {
            int layer = pl_orig->points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                // printf("layer: %d; is first: %d", layer, is_first[layer]);
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer])
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            else
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;

            if (added_pt.curvature < time_last[layer])
                added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        // if (i % _config._pointFilterNum == 0 && pl_orig->points[i].ring < _config._nScans) {
        //     double dist2 = added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z;
        //     if (dist2 > (_config._blindMin * _config._blindMin) && dist2 < (_config._blindMax * _config._blindMax))
        //         pcl_out->points.push_back(added_pt);
        // }
        if (pl_orig->points[i].ring < _config._nScans) {
            double dist2 = added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z;
            if (dist2 > (_config._blindMin * _config._blindMin) && dist2 < (_config._blindMax * _config._blindMax))
                pcl_out->points.push_back(added_pt);
        }
    }
}

int LidarProcess::downSampleLocalCloud(PointCloudXYZI::Ptr oriCloud, PointCloudXYZI::Ptr downCloud) {
    ////////////downsample the feature points in a scan ////////////
    pcl::PCLPointCloud2::Ptr oriCloud2(new pcl::PCLPointCloud2()), cloudDown2(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*oriCloud, *oriCloud2);
    //pcl::VoxelGrid<pcl::PCLPointCloud2> downSizeFilter2;
    _downSizeFilter.setInputCloud(oriCloud2);
    _downSizeFilter.filter(*cloudDown2);
    pcl::fromPCLPointCloud2(*cloudDown2, *downCloud);

    //_downSizeFilter.setInputCloud(oriCloud);
    //_downSizeFilter.filter(*downCloud);
    //cout << "[ mapping ]: In num: " << oriCloud->size() << " downsamp: " << downCloud->size()<<std::endl;
    return downCloud->points.size();
}

void LidarProcess::filterCloud(PointCloudXYZI::Ptr oriCloud, PointCloudXYZI::Ptr filteredCloud, int filterNum) {
    ////////////downsample the feature points in a scan ////////////
    for (int i = 0; i < oriCloud->size(); i++) {
        if (i % filterNum == 0)
            filteredCloud->push_back(oriCloud->points[i]);
    }
}

void LidarProcess::projectPointCloud(PointCloudXYZI::Ptr oriCloud, CloudInfo &cloudInfo) {
    cloudInfo.startRingIndex.assign(_config._nScans, 0);
    cloudInfo.endRingIndex.assign(_config._nScans, 0);

    cloudInfo.pointColInd.assign(_config._nScans * _config._nHorizonScan, 0);
    cloudInfo.pointRange.assign(_config._nScans * _config._nHorizonScan, 0);

    pcl::PointCloud<PointType>::Ptr fullCloud(new pcl::PointCloud<PointType>());
    fullCloud->points.resize(_config._nScans * _config._nHorizonScan);
    Eigen::MatrixXf rangeMat;
    rangeMat.resize(_config._nScans, _config._nHorizonScan);
    rangeMat.setConstant(FLT_MAX);

    int cloudSize = oriCloud->points.size();
    for (int i = 0; i < cloudSize; ++i) {
        PointType thisPoint;
        thisPoint.x = oriCloud->points[i].x;
        thisPoint.y = oriCloud->points[i].y;
        thisPoint.z = oriCloud->points[i].z;
        thisPoint.intensity = oriCloud->points[i].intensity;

        int rowIdn = oriCloud->points[i].normal_x;
        if (rowIdn < 0 || rowIdn >= _config._nScans)
            continue;

        int columnIdn = round(oriCloud->points[i].curvature / 1000. / 0.1 * _config._nHorizonScan);
        if (columnIdn >= _config._nHorizonScan)
            columnIdn -= _config._nHorizonScan;
        //        std::cout<<columnIdn<<",";
        if (columnIdn < 0 || columnIdn >= _config._nHorizonScan)
            continue;

        float range = pointDistance(thisPoint);

        if (range < 1.0)
            continue;

        if (rangeMat(rowIdn, columnIdn) != FLT_MAX)
            continue;

        rangeMat(rowIdn, columnIdn) = range;

        int index = columnIdn + rowIdn * _config._nHorizonScan;
        fullCloud->points[index] = thisPoint;
    }

    int count = 0;
    // extract segmented cloud for lidar odometry
    for (int i = 0; i < _config._nScans; ++i) {
        cloudInfo.startRingIndex[i] = count - 1 + 5;
        for (int j = 0; j < _config._nHorizonScan; ++j) {
            if (rangeMat(i, j) != FLT_MAX) {
                // mark the points' column index for marking occlusion later
                cloudInfo.pointColInd[count] = j;
                // save range info
                cloudInfo.pointRange[count] = rangeMat(i, j);
                // save extracted cloud
                cloudInfo.cloud_deskewed.push_back(fullCloud->points[j + i * _config._nHorizonScan]);
                // size of extracted cloud
                ++count;
            }
        }
        cloudInfo.endRingIndex[i] = count - 1 - 5;
    }
    //    for(int r=0;r<rangeMat.rows();r++)
    //    {
    //        for(int c=0;c<rangeMat.cols();c++)
    //        {
    //            std::cout<<std::setprecision(2)<<std::fixed<<rangeMat(r,c)<<",";
    //            if((c + r * rangeMat.cols())%20==0)
    //                std::cout<<std::endl;
    //        }
    //    }
    //    std::cout<<"Extracted cloud size:"<<cloudInfo.cloud_deskewed.size()<<std::endl;
}

void LidarProcess::calculateSmoothness(CloudInfo &cloudInfo) {
    int cloudSize = cloudInfo.cloud_deskewed.points.size();
    for (int i = 5; i < cloudSize - 5; i++) {
        float diffRange = cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4]
                          + cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2]
                          + cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10
                          + cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2]
                          + cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4]
                          + cloudInfo.pointRange[i + 5];

        _cloudCurvature[i] = diffRange * diffRange; //diffX * diffX + diffY * diffY + diffZ * diffZ;

        _cloudNeighborPicked[i] = 0;
        _cloudLabel[i] = 0;
        // cloudSmoothness for sorting
        _cloudSmoothness[i].value = _cloudCurvature[i];
        _cloudSmoothness[i].ind = i;
    }
}

void LidarProcess::markOccludedPoints(CloudInfo &cloudInfo) {
    int cloudSize = cloudInfo.cloud_deskewed.points.size();
    // mark occluded points and parallel beam points
    for (int i = 5; i < cloudSize - 6; ++i) {
        // occluded points
        float depth1 = cloudInfo.pointRange[i];
        float depth2 = cloudInfo.pointRange[i + 1];
        int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

        if (columnDiff < 10) {
            // 10 pixel diff in range image
            if (depth1 - depth2 > 0.3) {
                _cloudNeighborPicked[i - 5] = 1;
                _cloudNeighborPicked[i - 4] = 1;
                _cloudNeighborPicked[i - 3] = 1;
                _cloudNeighborPicked[i - 2] = 1;
                _cloudNeighborPicked[i - 1] = 1;
                _cloudNeighborPicked[i] = 1;
            } else if (depth2 - depth1 > 0.3) {
                _cloudNeighborPicked[i + 1] = 1;
                _cloudNeighborPicked[i + 2] = 1;
                _cloudNeighborPicked[i + 3] = 1;
                _cloudNeighborPicked[i + 4] = 1;
                _cloudNeighborPicked[i + 5] = 1;
                _cloudNeighborPicked[i + 6] = 1;
            }
        }
        // parallel beam
        float diff1 = std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
        float diff2 = std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

        if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
            _cloudNeighborPicked[i] = 1;
    }
}

PointCloudXYZI surfaceCloudScanDS;
void LidarProcess::extractFeatures(PointCloudXYZI::Ptr oriCloud, PointCloudXYZI &surfaceCloud, PointCloudXYZI &cornerCloud) {
    CloudInfo cloudInfo;
    projectPointCloud(oriCloud, cloudInfo);

    calculateSmoothness(cloudInfo);

    markOccludedPoints(cloudInfo);

    cornerCloud.clear();
    surfaceCloud.clear();

    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(_config._surfLeafSize, _config._surfLeafSize, _config._surfLeafSize);

    for (int i = 0; i < _config._nScans; i++) {
        PointCloudXYZI::Ptr surfaceCloudScan(new PointCloudXYZI());

        for (int j = 0; j < 6; j++) {
            int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
            int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

            if (sp >= ep)
                continue;

            std::sort(_cloudSmoothness.begin() + sp, _cloudSmoothness.begin() + ep, by_value());

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--) {
                int ind = _cloudSmoothness[k].ind;
                if (_cloudNeighborPicked[ind] == 0 && _cloudCurvature[ind] > _config._edgeThreshold) {
                    largestPickedNum++;
                    if (largestPickedNum <= 20) {
                        _cloudLabel[ind] = 1;
                        cornerCloud.push_back(cloudInfo.cloud_deskewed.points[ind]);
                    } else
                        break;

                    _cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++) {
                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                        if (columnDiff > 10)
                            break;
                        _cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--) {
                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                        if (columnDiff > 10)
                            break;
                        _cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++) {
                int ind = _cloudSmoothness[k].ind;
                if (_cloudNeighborPicked[ind] == 0 && _cloudCurvature[ind] < _config._surfThreshold) {
                    _cloudLabel[ind] = -1;
                    _cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++) {
                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                        if (columnDiff > 10)
                            break;

                        _cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--) {
                        int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                        if (columnDiff > 10)
                            break;

                        _cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
                if (_cloudLabel[k] <= 0)
                    surfaceCloudScan->push_back(cloudInfo.cloud_deskewed.points[k]);
        }

        downSizeFilter.setInputCloud(surfaceCloudScan);
        downSizeFilter.filter(surfaceCloudScanDS);
        surfaceCloud += surfaceCloudScanDS;

        //surfaceCloud += *surfaceCloudScan;
    }
}
