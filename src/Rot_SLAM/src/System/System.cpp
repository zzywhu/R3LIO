//
// Created by w on 2022/9/26.
//
#include "System.h"
#include "CloudInfo.h"

System::Config System::_config;

void System::laserMapFovSegment() {
    _cubNeedrm.clear();
    V3D posLid = _stateIkfom.pos + _stateIkfom.rot.matrix() * _stateIkfom.offset_T_L_I; // twl=twi+Rwi*til=Twi*til

    if (!_isLocalMapInitialized) {
        for (int i = 0; i < 3; i++) {
            _localMapPoints.vertex_min[i] = posLid(i) - _config._cubeLen / 2.0;
            _localMapPoints.vertex_max[i] = posLid(i) + _config._cubeLen / 2.0;
        }
        _isLocalMapInitialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(posLid(i) - _localMapPoints.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(posLid(i) - _localMapPoints.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * _config._detRange || dist_to_map_edge[i][1] <= MOV_THRESHOLD * _config._detRange)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType newLocalMapPoints, tmp_boxpoints;
    newLocalMapPoints = _localMapPoints;
    float mov_dist = max((_config._cubeLen - 2.0 * MOV_THRESHOLD * _config._detRange) * 0.5 * 0.9,
                         double(_config._detRange * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = _localMapPoints;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * _config._detRange) {
            newLocalMapPoints.vertex_max[i] -= mov_dist;
            newLocalMapPoints.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = _localMapPoints.vertex_max[i] - mov_dist;
            _cubNeedrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * _config._detRange) {
            newLocalMapPoints.vertex_max[i] += mov_dist;
            newLocalMapPoints.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = _localMapPoints.vertex_min[i] + mov_dist;
            _cubNeedrm.push_back(tmp_boxpoints);
        }
    }
    _localMapPoints = newLocalMapPoints;
    double delete_begin = omp_get_wtime();
    if (_cubNeedrm.size() > 0) {
        int kdtree_delete_counter = _ikdtree.Delete_Point_Boxes(_cubNeedrm);
        if (kdtree_delete_counter)
            std::cout << "Kdtree delete size:" << kdtree_delete_counter << std::endl;
    }
}

bool System::collectRasterAngles(const double &begTime, const double &endTime) {
    if ((_motorAngleBuf.front().first > begTime && _measures.tMotorAngles.size() == 0) ||
        //_motorAngleBuf.front().first > endTime||
        _motorAngleBuf.back().first < endTime) {
        //std::cout << _measures.tMotorAngles.size() << std::endl;
        printf("Not enough motor data\n");
        return false;
    }

    if (_measures.tMotorAngles.size() > 0) {
        auto itr = _measures.tMotorAngles.rbegin();
        for (; itr != _measures.tMotorAngles.rend(); itr++) {
            if (itr->first > begTime)
                continue;
            break;
        }
        auto tempMotorAngle = *itr;
        _measures.tMotorAngles.clear();
        _measures.tMotorAngles.push_back(tempMotorAngle);
    }

    for (auto it = _motorAngleBuf.begin(); it != _motorAngleBuf.end(); ++it) {
        //        if (it->first < begTime)
        //            continue;
        _measures.tMotorAngles.push_back(*it);
        if (it->first > endTime)
            break;
    }

    return true;
}

bool System::getRasterAngle(const double &curTime) {
    double popAngle = 0;
    double popTime = 0;
    if (curTime <= _motorAngleBuf.back().first) {
        popTime = _motorAngleBuf.front().first;
        popAngle = _motorAngleBuf.front().second;
        while (_motorAngleBuf.front().first < curTime) {
            popTime = _motorAngleBuf.front().first;
            popAngle = _motorAngleBuf.front().second;
            _motorAngleBuf.pop_front();
        }
        const double &frontAngle = _motorAngleBuf.front().second;
        const double &frontTime = _motorAngleBuf.front().first;
        double deltaAngle = frontAngle - popAngle;
        // std::cout<<"deltaAngle:"<<deltaAngle<<std::endl;
        //if (deltaAngle < 0)
        //    deltaAngle += 2 * M_PI;
        if (deltaAngle < -M_PI)
            deltaAngle += 2 * M_PI;
        else if (deltaAngle > M_PI)
            deltaAngle -= 2 * M_PI;
        _curMotorAngle = popAngle + (deltaAngle * (curTime - popTime) / (frontTime - popTime));
        if (_curMotorAngle < 0)
            _curMotorAngle += 2 * M_PI;
        if (_curMotorAngle > 2 * M_PI)
            _curMotorAngle -= 2 * M_PI;
        if (_initMotorAngle < 0)
            _initMotorAngle = _curMotorAngle;
        if (!_config._isMotorInitialized) {
            Eigen::Matrix4d relTo = Eigen::Matrix4d::Identity();
            double rotAngle = _curMotorAngle - _initMotorAngle;
            if (rotAngle < 0)
                rotAngle += 2 * M_PI;

            relTo.block<3, 3>(0, 0) = AngleAxisToRotationMatrix(Eigen::Vector3d(0, 0, rotAngle)); // To1o2
            std::cout << std::endl
                      << "Motor rot angle:" << RAD2DEG(rotAngle) << std::endl;
            // relTo.block<3, 3>(0, 0) = AngleAxisToRotationMatrix(Eigen::Vector3d(0, 0, _curMotorAngle - _initMotorAngle)); // To1o2
            _relToList.push_back(relTo);
        }
    } else {
        printf("No motor data\n");
        return false;
    }
    return true;
}

int System::matchKdTree(const PointCloudXYZI::Ptr &localCloudPtr, const PointCloudXYZI::Ptr &globalCloudPtr,
                        KD_TREE &kdTree, MatchedInfoList &matches) {
    assert(localCloudPtr->size() == globalCloudPtr->size());
    matches.clear();
    const int localCloudSize = localCloudPtr->size();
    double totalResidual = 0;
    _nearestPoints.resize(localCloudSize);
    //    #ifdef MP_EN
    //        omp_set_num_threads(MP_PROC_NUM);
    //        #pragma omp parallel for
    //    #endif
    for (int i = 0; i < localCloudSize; i++) {
        PointType &pointLocal = localCloudPtr->points[i];
        PointType &pointWorld = globalCloudPtr->points[i];
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        PointVector &pointsNear = _nearestPoints[i];
        kdTree.Nearest_Search(pointWorld, NUM_MATCH_POINTS, pointsNear, pointSearchSqDis);
        if (pointsNear.size() < NUM_MATCH_POINTS || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
            continue;
        VD(4)
        pabcd;
        if (estiPlane(pabcd, pointsNear, 0.1)) {
            float pd2 = pabcd(0) * pointWorld.x + pabcd(1) * pointWorld.y + pabcd(2) * pointWorld.z + pabcd(3);

            V3D pl(pointLocal.x, pointLocal.y, pointLocal.z);
            float s = 1. - 0.9 * fabs(pd2) / sqrt(pl.norm());
            if (s > 0.9) {
                // std::cout<<"pd2:"<<pd2<<std::endl;
                Eigen::Vector4d planeParam;
                planeParam << pabcd(0), pabcd(1), pabcd(2), pabcd(3);
                totalResidual += abs(pd2);
                V3D norm = V3D(pabcd(0), pabcd(1), pabcd(2));
                matches.emplace_back(pl, pabcd, V3D(pointsNear[0].x, pointsNear[0].y, pointsNear[0].z), norm, 1 - 0.75 * fabs(pd2));
            }
        }
    }
    // std::cout<<"Total residual:"<< totalResidual<<std::endl;
    return matches.size();
}

void System::motorMotionCompensation() {
    sort(_measures.lidar->begin(), _measures.lidar->end(), timeComprator);
    /*** undistort each lidar point (backward propagation) ***/
    double dt, curAngle, endAngle;
    auto it_pcl = _measures.lidar->points.end() - 1;
    endAngle = _measures.tMotorAngles.back().second;
    for (auto it_kp = _measures.tMotorAngles.end() - 1; it_kp != _measures.tMotorAngles.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;

        double deltaAngle = tail->second - head->second;
        double deltaTime = tail->first - head->first;
        if (deltaTime < 1e-9)
            continue;
        if (deltaAngle < -M_PI)
            deltaAngle += 2 * M_PI;
        else if (deltaAngle > M_PI)
            deltaAngle -= 2 * M_PI;
        double motorSpeed = deltaAngle / deltaTime;

        double offsetTime = head->first - _measures.lidar_beg_time;
        for (; it_pcl->curvature / double(1000) > offsetTime; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - offsetTime;
            curAngle = motorSpeed * dt + head->second;
            if (it_pcl == _measures.lidar->points.end() - 1)
                endAngle = curAngle;
            double da = curAngle - endAngle;
            if (da < -M_PI)
                da += 2 * M_PI;
            else if (da > M_PI)
                da -= 2 * M_PI;
            /* Transform to the 'end' frame */
            Eigen::Vector3d p(it_pcl->x, it_pcl->y, it_pcl->z);
            p = Eigen::AngleAxisd(da, Eigen::Vector3d::UnitZ()).matrix() * p;
            it_pcl->x = p(0);
            it_pcl->y = p(1);
            it_pcl->z = p(2);
            if (it_pcl == _measures.lidar->points.begin())
                break;
        }
        if (it_pcl == _measures.lidar->points.begin())
            break;
    }
}

void System::motorMotionCompensationZGforSIM() {
    sort(_measures.lidar->begin(), _measures.lidar->end(), timeComprator);
    /*** undistort each lidar point (backward propagation) ***/
    double dt, curAngle, endAngle;
    auto it_pcl = _measures.lidar->points.end() - 1;
    endAngle = _measures.tMotorAngles.back().second;

    //std::ofstream fFrameOfs =std::ofstream(string(ROOT_DIR) + "PCD/"+std::to_string(_lidarBegTime)+".txt", std::ios::trunc | std::ios::in);
    //static std::ofstream fTotalOfs =std::ofstream(string(ROOT_DIR) + "PCD/all.txt", std::ios::trunc | std::ios::in);
    for (auto it_kp = _measures.tMotorAngles.end() - 1; it_kp != _measures.tMotorAngles.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;

        double deltaAngle = tail->second - head->second;
        double deltaTime = tail->first - head->first;
        if (deltaTime < 1e-9)
            continue;
        if (deltaAngle < -M_PI)
            deltaAngle += 2 * M_PI;
        else if (deltaAngle > M_PI)
            deltaAngle -= 2 * M_PI;
        double motorSpeed = deltaAngle / deltaTime;

        double offsetTime = head->first - _measures.lidar_beg_time;
        //std::cout<<"motor:"<<offsetTime<<"-"<<tail->first - _measures.lidar_beg_time<<std::endl;
        std::cout << "angle:" << head->second << "-" << tail->second << std::endl;
        for (; it_pcl->curvature / double(1000) > offsetTime; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - offsetTime;
            curAngle = motorSpeed * dt + head->second;
            //std::cout<<"lid:"<<it_pcl->curvature / double(1000) <<"->"<<curAngle<<std::endl;
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            //_cloudAxisTransfer->CalculatePose(rad2deg(curAngle), R, t);
            //_cloudAxisTransfer->getLid2IMUTrans(curAngle, R, t);
            //std::cout << curAngle << std::endl;
            _cloudAxisTransfer->getLid2IMUTransRefineforSIM(curAngle, R, t);
            Eigen::Vector3d pOri(it_pcl->x, it_pcl->y, it_pcl->z);
            Eigen::Vector3d pTrans = R * pOri + t;
            it_pcl->x = pTrans(0);
            it_pcl->y = pTrans(1);
            it_pcl->z = pTrans(2);

            //fFrameOfs<<std::fixed<<std::setprecision(8)<<pOri(0)<<","<<pOri(1)<<","<<pOri(2)<<","<<rad2deg(curAngle)<<std::endl;
            //fTotalOfs<<std::fixed<<std::setprecision(8)<<pTrans(0)<<","<<pTrans(1)<<","<<pTrans(2)<<","<<rad2deg(curAngle)<<std::endl;
            if (it_pcl == _measures.lidar->points.begin())
                break;
        }
        if (it_pcl == _measures.lidar->points.begin())
            break;
    }
}

void System::motorMotionCompensationZG() {
    sort(_measures.lidar->begin(), _measures.lidar->end(), timeComprator);
    /*** undistort each lidar point (backward propagation) ***/
    double dt, curAngle, endAngle;
    auto it_pcl = _measures.lidar->points.end() - 1;
    endAngle = _measures.tMotorAngles.back().second;

    //std::ofstream fFrameOfs =std::ofstream(string(ROOT_DIR) + "PCD/"+std::to_string(_lidarBegTime)+".txt", std::ios::trunc | std::ios::in);
    //static std::ofstream fTotalOfs =std::ofstream(string(ROOT_DIR) + "PCD/all.txt", std::ios::trunc | std::ios::in);
    for (auto it_kp = _measures.tMotorAngles.end() - 1; it_kp != _measures.tMotorAngles.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;

        double deltaAngle = tail->second - head->second;
        double deltaTime = tail->first - head->first;
        if (deltaTime < 1e-9)
            continue;
        if (deltaAngle < -M_PI)
            deltaAngle += 2 * M_PI;
        else if (deltaAngle > M_PI)
            deltaAngle -= 2 * M_PI;
        double motorSpeed = deltaAngle / deltaTime;

        double offsetTime = head->first - _measures.lidar_beg_time;
        //std::cout<<"motor:"<<offsetTime<<"-"<<tail->first - _measures.lidar_beg_time<<std::endl;
        //std::cout<<"angle:"<<head->second<<"-"<<tail->second<<std::endl;
        for (; it_pcl->curvature / double(1000) > offsetTime; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - offsetTime;
            curAngle = motorSpeed * dt + head->second;
            //std::cout<<"lid:"<<it_pcl->curvature / double(1000) <<"->"<<curAngle<<std::endl;
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            //_cloudAxisTransfer->CalculatePose(rad2deg(curAngle), R, t);
            //_cloudAxisTransfer->getLid2IMUTrans(curAngle, R, t);
            _cloudAxisTransfer->getLid2IMUTransRefine(curAngle, R, t);
            Eigen::Vector3d pOri(it_pcl->x, it_pcl->y, it_pcl->z);
            Eigen::Vector3d pTrans = R * pOri + t;
            it_pcl->x = pTrans(0);
            it_pcl->y = pTrans(1);
            it_pcl->z = pTrans(2);

            //fFrameOfs<<std::fixed<<std::setprecision(8)<<pOri(0)<<","<<pOri(1)<<","<<pOri(2)<<","<<rad2deg(curAngle)<<std::endl;
            //fTotalOfs<<std::fixed<<std::setprecision(8)<<pTrans(0)<<","<<pTrans(1)<<","<<pTrans(2)<<","<<rad2deg(curAngle)<<std::endl;
            if (it_pcl == _measures.lidar->points.begin())
                break;
        }
        if (it_pcl == _measures.lidar->points.begin())
            break;
    }
}

bool System::lidarMotionCompensationWithTimestamp(typename pcl::PointCloud<PointType>::Ptr &cloudInOut,
                                                  const Eigen::Matrix4d &T12,
                                                  float minScanDurationMs) // T12 (from k+1 to k frame)
{
    typename pcl::PointCloud<PointType>::Ptr cloudTemp(new pcl::PointCloud<PointType>);
    double s; // interpolation ratio

    double lastTimestamp = -DBL_MAX;
    double firstTimestamp = DBL_MAX;
    double scanDuration;
    const int totalSize = cloudInOut->points.size();
    // get the first and last time stamp (stored as curvature) of this scan (time stamp unit: ms)
    for (int i = 0; i < totalSize; i++) {
        // LOG(INFO)<<cloudInOut->points[i].curvature;
        lastTimestamp = std::max(lastTimestamp, double(cloudInOut->points[i].curvature));
        firstTimestamp = std::min(firstTimestamp, double(cloudInOut->points[i].curvature));
    }
    scanDuration = lastTimestamp - firstTimestamp;
    if (scanDuration < minScanDurationMs)
        return false;

    Eigen::Quaterniond estimatedQuat12 = Eigen::Quaterniond(T12.block<3, 3>(0, 0));
    Eigen::Vector3d estimatedTranslation12 = T12.block<3, 1>(0, 3);

    for (int i = 0; i < totalSize; i++) {
        s = (cloudInOut->points[i].curvature - firstTimestamp) / scanDuration; // curvature as time stamp //TODO: fix
        // the first point (with earliest timestamp) would change the most
        // while the last point (with latest timestamp) would change the least
        Eigen::Quaterniond dQuat = Eigen::Quaterniond::Identity().slerp(s, estimatedQuat12); // Spherical linear interpolation (slerp) for quaternion
        Eigen::Vector3d dTranslation = s * estimatedTranslation12;
        Eigen::Vector3d tempPoint(cloudInOut->points[i].x, cloudInOut->points[i].y, cloudInOut->points[i].z);
        Eigen::Vector3d undistortPoint = dQuat * tempPoint + dTranslation;
        PointType pt;
        pt.x = undistortPoint(0);
        pt.y = undistortPoint(1);
        pt.z = undistortPoint(2);
        pt.intensity = cloudInOut->points[i].intensity;
        pt.curvature = cloudInOut->points[i].curvature;
        cloudTemp->points.push_back(pt);
    }
    cloudTemp->points.swap(cloudInOut->points);
    return true;
}

void System::predictAndUndistort(const MeasureGroup &meas, PointCloudXYZI::Ptr pclOut) {
    /*** sort point clouds by offset time ***/
    const double &pcl_beg_time = meas.lidar_beg_time;
    sort(pclOut->points.begin(), pclOut->points.end(), timeComprator);
    const double &endOffsetTime = pclOut->points.back().curvature / double(1000);

    MD(DIM_STATE, DIM_STATE)
    F_x, cov_w;

    Eigen::Matrix4d velocity = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d deltaT21 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d curTwl = Eigen::Matrix4d::Identity();
    curTwl.block<3, 3>(0, 0) = _Rwl;
    curTwl.block<3, 1>(0, 3) = _twl;

    if (_isFirstFrame) {
        _dt = 0.1;
        _isFirstFrame = false;
        _timeLastScan = pcl_beg_time;
    } else {
        if (_dt != 0) {
            deltaT21 = curTwl.inverse() * _prevTwl; // T21=T2w*Tw1
            velocity = deltaT21 / _dt;
        } else
            return;
        _dt = pcl_beg_time - _timeLastScan;
        _timeLastScan = pcl_beg_time;
    }

    Eigen::Matrix4d predictTwl = curTwl * (velocity * _dt).inverse(); // Tw3=Tw2*T23(T21/dt21=T32/dt32)
    predictTwl /= predictTwl(3, 3);

    Eigen::Quaterniond estimatedQuat21 = Eigen::Quaterniond(deltaT21.block<3, 3>(0, 0));
    Eigen::Vector3d estimatedTranslation21 = deltaT21.block<3, 1>(0, 3);
    /**CV model： un-distort pcl using linear interpolation **/
    auto itrPt = pclOut->points.end() - 1;
    for (; itrPt != pclOut->points.begin(); itrPt--) {
        double s = (endOffsetTime - itrPt->curvature / double(1000)) / _dt; // curvature as time stamp //TODO: fix
        // the first point (with earliest timestamp) would change the most
        // while the last point (with latest timestamp) would change the least
        Eigen::Quaterniond dQuat21 = Eigen::Quaterniond::Identity().slerp(s, estimatedQuat21); // Spherical linear interpolation (slerp) for quaternion
        Eigen::Vector3d dTranslation21 = s * estimatedTranslation21;
        Eigen::Vector3d tempPoint(itrPt->x, itrPt->y, itrPt->z);
        Eigen::Vector3d undistortPoint = dQuat21 * tempPoint + dTranslation21;

        itrPt->x = undistortPoint(0);
        itrPt->y = undistortPoint(1);
        itrPt->z = undistortPoint(2);
    }
    _prevTwl.block<3, 3>(0, 0) = _Rwl;
    _prevTwl.block<3, 1>(0, 3) = _twl;
    _Rwl = predictTwl.block<3, 3>(0, 0);
    _twl = predictTwl.block<3, 1>(0, 3);
}

void System::initKdTreeMap(PointVector pointCloud) {
    if (!_ikdtree.Root_Node) {
        _ikdtree.Build(pointCloud); // pw
        int featsFromMapNum = _ikdtree.validnum();
        //cout << "[ Init Kdtree ]:  Map num: " << featsFromMapNum << endl;
    }
}

void System::updateKdTreeMap(PointVector &pointsIn) {
    PointVector pointToAdd;
    PointVector pointNoNeedDownsample;
    const int &ptSize = pointsIn.size();
    pointToAdd.reserve(ptSize);
    pointNoNeedDownsample.reserve(ptSize);
    for (int i = 0; i < ptSize; i++) {
        if (!_nearestPoints[i].empty() && _isEKFInited) {
            const PointVector &points_near = _nearestPoints[i];
            bool need_add = true;
            PointType downsample_result, mid_point;
            mid_point.x = floor(pointsIn[i].x / _config._filterSizeMap) * _config._filterSizeMap + 0.5 * _config._filterSizeMap;
            mid_point.y = floor(pointsIn[i].y / _config._filterSizeMap) * _config._filterSizeMap + 0.5 * _config._filterSizeMap;
            mid_point.z = floor(pointsIn[i].z / _config._filterSizeMap) * _config._filterSizeMap + 0.5 * _config._filterSizeMap;
            float dist = calcDist(pointsIn[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * _config._filterSizeMap && fabs(points_near[0].y - mid_point.y) > 0.5 * _config._filterSizeMap && fabs(points_near[0].z - mid_point.z) > 0.5 * _config._filterSizeMap) {
                pointNoNeedDownsample.push_back(pointsIn[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calcDist(points_near[readd_i], mid_point) < dist) {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                pointToAdd.push_back(pointsIn[i]);
        } else
            pointToAdd.push_back(pointsIn[i]);
    }

    PointVector pointAdded1 = _ikdtree.Add_Points(pointToAdd, true);
    PointVector pointAdded2 = _ikdtree.Add_Points(pointNoNeedDownsample, false);

    PointVector().swap(_addedPoints);
    _addedPoints.insert(_addedPoints.end(), pointAdded1.begin(), pointAdded1.end());
    _addedPoints.insert(_addedPoints.end(), pointAdded2.begin(), pointAdded2.end());

    std::cout << "Map added points: " << _addedPoints.size() << std::endl;
}

void System::initVoxelMapWithKdtree(PointVector pointCloud) {
    _ikdtree.Build(pointCloud); // pw
    int featsFromMapNum = _ikdtree.validnum();
    cout << "[ Init Kdtree ]:  Map num: " << featsFromMapNum << endl;
    _voxelMapGaussian->updateVoxelMap(pointCloud);
}

void System::updateVoxelMapWithKdtree(PointVector &pointsIn) {
    PointVector pointToAdd;
    PointVector pointNoNeedDownsample;
    const int &ptSize = pointsIn.size();
    pointToAdd.reserve(ptSize);
    pointNoNeedDownsample.reserve(ptSize);
    _nearestPoints.resize(ptSize);
    for (int i = 0; i < ptSize; i++) {
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        PointVector &pointsNear = _nearestPoints[i];
        _ikdtree.Nearest_Search(pointsIn[i], NUM_MATCH_POINTS, pointsNear, pointSearchSqDis);
        if (!pointsNear.empty()) {
            bool need_add = true;
            PointType downsample_result, mid_point;
            mid_point.x = floor(pointsIn[i].x / _config._filterSizeMap) * _config._filterSizeMap + 0.5 * _config._filterSizeMap;
            mid_point.y = floor(pointsIn[i].y / _config._filterSizeMap) * _config._filterSizeMap + 0.5 * _config._filterSizeMap;
            mid_point.z = floor(pointsIn[i].z / _config._filterSizeMap) * _config._filterSizeMap + 0.5 * _config._filterSizeMap;
            float dist = calcDist(pointsIn[i], mid_point);
            if (fabs(pointsNear[0].x - mid_point.x) > 0.5 * _config._filterSizeMap && fabs(pointsNear[0].y - mid_point.y) > 0.5 * _config._filterSizeMap && fabs(pointsNear[0].z - mid_point.z) > 0.5 * _config._filterSizeMap) {
                pointNoNeedDownsample.push_back(pointsIn[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                if (pointsNear.size() < NUM_MATCH_POINTS)
                    break;
                if (calcDist(pointsNear[readd_i], mid_point) < dist) {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                pointToAdd.push_back(pointsIn[i]);
        } else
            pointToAdd.push_back(pointsIn[i]);
    }

    PointVector pointAdded1 = _ikdtree.Add_Points(pointToAdd, true);
    PointVector pointAdded2 = _ikdtree.Add_Points(pointNoNeedDownsample, false);
    _voxelMapGaussian->updateVoxelMap(pointAdded1);
    _voxelMapGaussian->updateVoxelMap(pointAdded2);

    PointVector().swap(_addedPoints);
    _addedPoints.insert(_addedPoints.end(), pointAdded1.begin(), pointAdded1.end());
    _addedPoints.insert(_addedPoints.end(), pointAdded2.begin(), pointAdded2.end());

    // int nNew = _addedPoints.size();
    // std::cout << "Map added points: " << nNew << std::endl;
}

void System::initVoxelMap() {
    _Rwl = _stateIkfom.rot * _stateIkfom.offset_R_L_I;                            // Rwl=Rwi*Ril
    _twl = _stateIkfom.rot * _stateIkfom.offset_T_L_I.matrix() + _stateIkfom.pos; // twl=Twi*til
    //PointCloudXYZI::Ptr globalSurfCloudPtr(new PointCloudXYZI());
    //transCloud(_localSurfCloudPtr, globalSurfCloudPtr, _Rwl, _twl);

    _curSurfPvList.clear();
    for (size_t i = 0; i < _localSurfCloudPtr->size(); i++) {
        PointWithCov pv;
        pv.featType = Plane;
        pv.pl << _localSurfCloudPtr->points[i].x, _localSurfCloudPtr->points[i].y, _localSurfCloudPtr->points[i].z;
        pv.pi = _stateIkfom.offset_R_L_I * pv.pl + _stateIkfom.offset_T_L_I;
        if (pv.pi[2] == 0)
            pv.pi[2] = 0.001;
        //pv.pw << _Rwl * pv.pl + _twl;
        pv.pw = _stateIkfom.rot * pv.pi + _stateIkfom.pos;
        _curSurfPvList.push_back(pv);
    }
    if (_config._covType == 2)
        calcPointNeighCov(_curSurfPvList, 20);

    for (size_t i = 0; i < _localSurfCloudPtr->size(); i++) {
        if (_config._covType == 1) {
            calcBodyCov(_curSurfPvList[i], _config._rangingCov, _config._angleCov);
            M3D pointCrossmat;
            pointCrossmat << SKEW_SYM_MATRX(_curSurfPvList[i].pl);
            _curSurfPvList[i].obsCov = _Rwl * _curSurfPvList[i].bodyCov * _Rwl.transpose() + _Rwl * pointCrossmat * _kf.get_P().block<3, 3>(3, 3) * pointCrossmat.transpose() * _Rwl.transpose() + _kf.get_P().block<3, 3>(0, 0);
        } else if (_config._covType == 2)
            _curSurfPvList[i].obsCov = _curSurfPvList[i].neighCov.block<3, 3>(0, 0);
    }
    TicToc time1;
    //std::cout << "Inited voxel map num: " << _curSurfPvList.size() << std::endl;
    buildVoxelMap(_curSurfPvList, _frameId, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
                  _config._maxPointsSize, _config._maxPointsSize, _config._minSurfEigenValue,
                  _voxelSurfMap);
    //    std::cout << "Inited voxel map" << std::endl;
    //printf("Init buildVoxelMap time: %f ms\n", time1.toc());
}

void System::updateVoxelMap() {
    //////////////update voxel map//////////////
    for (size_t i = 0; i < _curSurfPvList.size(); i++) {
        _curSurfPvList[i].pi = _stateIkfom.offset_R_L_I * _curSurfPvList[i].pl + _stateIkfom.offset_T_L_I;
        if (_curSurfPvList[i].pi[2] == 0)
            _curSurfPvList[i].pi[2] = 0.001;
        _curSurfPvList[i].pw = _stateIkfom.rot * _curSurfPvList[i].pi + _stateIkfom.pos;
        if (_config._covType == 1) {
            M3D pointCrossmat;
            pointCrossmat << SKEW_SYM_MATRX(_curSurfPvList[i].pl);
            calcBodyCov(_curSurfPvList[i], _config._rangingCov, _config._angleCov);
            _curSurfPvList[i].obsCov = _Rwl * _curSurfPvList[i].bodyCov * _Rwl.transpose() + _Rwl * pointCrossmat * _kf.get_P().block<3, 3>(3, 3) * pointCrossmat.transpose() * _Rwl.transpose() + _kf.get_P().block<3, 3>(0, 0);
        } else if (_config._covType == 2)
            _curSurfPvList[i].obsCov = _curSurfPvList[i].neighCov.block<3, 3>(0, 0);
    }
    std::sort(_curSurfPvList.begin(), _curSurfPvList.end(), varContrast);
    //	TicToc time1;
    ::updateVoxelMap(_curSurfPvList, _frameId, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
                     _config._maxPointsSize, _config._maxPointsSize, _config._minSurfEigenValue,
                     _voxelSurfMap, _boxToDel);
    //	printf("updateVoxelMap time: %f ms\n", time1.toc());
}

bool System::motionCheck() {
    if (_rotDeltas[0] > 0.1 && _posDeltas[0] > 0.1) {
        int nIncreased = 0;
        int nLarge = 0;
        const int size = _rotDeltas.size();
        for (int i = 1; i < size; i++) {
            if (_rotDeltas[i] > _rotDeltas[0] || _posDeltas[i] > _posDeltas[0])
                nIncreased++;
            if (_rotDeltas[i] > 1. || _posDeltas[i] > 1.)
                nLarge++;
        }
        // std::cout<<"nIncreased/nLarge:"<<nIncreased<<"/"<<nLarge<<std::endl;
        if ((float)nIncreased / size >= 0.4 || (float)nLarge / size >= 0.4)
            if (_rotDeltas.back() > 0.5 * _rotDeltas.front())
                return true;
    }
    double matchRatio = (float)_matchSurfSizeList.back() / _localSurfCloudDownPtr->size();
    if ((float)matchRatio < 0.2) {
        std::cerr << "Matches ratio too little: " << matchRatio << std::endl;
        return false;
    }

    for (int i = 1; i < _matchSurfSizeList.size(); i++) {
        int deltaSize = _matchSurfSizeList[i] - _matchSurfSizeList[i - 1];
        if (deltaSize < -0.2 * _cloudSurfDownSize) {
            std::cerr << "Matches decreased: " << deltaSize << std::endl;
            return true;
        }
    }
    return true;
}

void System::motorInitialize() {
    //    {
    //        // undistort and predict by Tol guessed
    //        Eigen::Matrix3d Rol;
    //        Rol<<-0.999995, -0.003316, -0.000012,
    //             -0.000000 , 0.003665 ,-0.999993,
    //        0.003316 ,-0.999988 ,-0.003665;
    //        Eigen::Vector3d tol=Eigen::Vector3d::Zero();
    //        transCloudInMotorAxis(_measures.lidar, _measures.lidar, 0, Rol, tol);
    //        motorMotionCompensation();
    //        //transCloudInMotorAxis(_measures.lidar, _measures.lidar, _curMotorAngle, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    //
    //        Eigen::Matrix3d Rlo=Rol.transpose();
    //        Eigen::Vector3d tlo=-Rlo*tol;
    //        transCloudInMotorAxis(_measures.lidar,_measures.lidar, 0, Rlo, tlo);
    //
    //        //GuessT=Tll(theta0,theta)=Tlo*Too(theta0,theta)*Tol
    //        Eigen::Matrix4d Tol=Eigen::Matrix4d::Identity();
    //        Tol.block<3,3>(0,0)=Rol;
    //        Tol.block<3,1>(0,3)=tol;
    //        Eigen::Matrix4d Tlo=Eigen::Matrix4d::Identity();
    //        Tlo.block<3,3>(0,0)=Rlo;
    //        Tlo.block<3,1>(0,3)=tlo;
    //        Eigen::Matrix4d Too=Eigen::Matrix4d::Identity();
    //        double rotAngle=_curMotorAngle - _initMotorAngle;
    //        if (rotAngle < 0)
    //            rotAngle += 2 * M_PI;
    //
    //        Eigen::Matrix3d Roo=AngleAxisToRotationMatrix(Eigen::Vector3d(0,0,rotAngle));
    //        Too.block<3,3>(0,0)=Roo;
    //
    //        Eigen::Matrix4d guessT=Tlo*Too*Tol;
    //        _Rwl=guessT.block<3,3>(0,0);
    //        _twl=guessT.block<3,1>(0,3);
    //        _stateCur.pos_end = -_Rwl* _stateCur.offset_R_L_I.transpose() * _stateCur.offset_T_L_I +_twl; // twi=-Rwl*Rli*til+twl=Twl*tli
    //        _stateCur.rot_end = _Rwl * _stateCur.offset_R_L_I.transpose();//Rwi=Rwl*Rli
    //    }

    *_localCloudPtr = *_measures.lidar;
    if (processCloudLSQ(_regPCL_VGICP) <= 0)
        return;

    if (_relToList.empty())
        return;
    ////////calculate extrinsic/////////
    Eigen::Matrix4d relTl;
    relTl.block<3, 1>(0, 3) = _twl;
    relTl.block<3, 3>(0, 0) = _Rwl;
    _relTlList.push_back(relTl);
    std::cout << "Lidar R: " << R2ypr(_Rwl).transpose() << std::endl;
    std::cout << "Lidar t: " << _twl.transpose() << std::endl;
    // TicToc calcTime;
    if (calcEXRotation(_relToList, _relTlList, _Rol, _tol, 10, _config._maxInierError)) {
        _tol = Eigen::Vector3d();
        std::ofstream extrinsicOfs = std::ofstream(string(ROOT_DIR) + "result/MotorExtrinsic", std::ios::trunc | std::ios::in);
        extrinsicOfs << std::fixed << std::setprecision(6) << _tol.x() << "," << _tol.y() << "," << _tol.z() << std::endl;
        extrinsicOfs << std::fixed << std::setprecision(6) << Quaterniond(_Rol).coeffs().transpose() << std::endl;
        extrinsicOfs << std::fixed << std::setprecision(6) << _Rol << std::endl;
        System::mutableConfig()._isMotorInitialized = true;
        resetSystem();
        std::cout << "Initialized succeeded!" << std::endl;
        // PAUSE;
    }
    // PAUSE;
    // printf("calculate Tol time: %f ms\n\n", calcTime.toc());
}

void System::loopClosing(state_ikfom &state) {
    if (_config._isLoopEn && _config._isImuInitialized) {
        //store Til
        Eigen::Matrix4d Til = Eigen::Matrix4d::Identity();
        Til.block<3, 3>(0, 0) = state.offset_R_L_I.matrix();
        Til.block<3, 1>(0, 3) = state.offset_T_L_I;
        //std::cout<<Til<<std::endl;
        _TilList.emplace_back(Til);
        //Convert into GNSS coordinate (pg=Tgi*Til*pl)
        PointCloudXYZI::Ptr localCloudPtr(new PointCloudXYZI()), localCloudDownPtr(new PointCloudXYZI());
        _Rwg = state.rot.matrix() * _Rig;             //Rwg=Rwi*Rig
        _twg = state.rot.matrix() * _tig + state.pos; //twg
        Eigen::Matrix3d Rgi = _Rig.inverse();
        Eigen::Vector3d tgi = -_Rig.inverse() * _tig;
        Eigen::Matrix3d Rgl = Rgi * state.offset_R_L_I.matrix();       //Rgl=Rgi*Ril
        Eigen::Vector3d tgl = Rgi * state.offset_T_L_I.matrix() + tgi; //Tgi*til
        transCloud(_localCloudPtr, localCloudPtr, Rgl, tgl);
        transCloud(_localCloudDownPtr, localCloudDownPtr, Rgl, tgl);

        std::vector<PointWithCov> curPvList;
        for (auto pv : _curSurfPvList) {
            pv.pi = Rgi * pv.pi + tgi;
            curPvList.emplace_back(pv);
        }

        if (_measures.gnss) { // input gnss info into loopclosing thread
            processGNSS();
            _measures.gnss = nullptr;
        }
        // input frame info into loopclosing thread
        _loopCloser->getProcessLock().lock();
        _loopCloser->inputFrameData(_frameId, _lidarBegTime,
                                    _Rwg, _twg,
                                    localCloudPtr, localCloudDownPtr,
                                    curPvList,
                                    _imuProcessor->_imuDatas,
                                    false,
                                    _globalGrav);
        _loopCloser->getProcessLock().unlock();

        // Update frontend state
        updateLoopStatus(state);
        _imuProcessor->_imuDatas.clear();
        //        if(_loopCloser->loopCount()>0)
        //            PAUSE;
    }
}

void System::loopClosing(StatesGroup &state) {
    if (_config._isLoopEn && _config._isImuInitialized) {
        //store Til
        Eigen::Matrix4d Til = Eigen::Matrix4d::Identity();
        Til.block<3, 3>(0, 0) = state.offset_R_L_I.matrix();
        Til.block<3, 1>(0, 3) = state.offset_T_L_I;
        _TilList.emplace_back(Til);

        //Convert into GNSS coordinate (pg=Tgi*Til*pl)
        PointCloudXYZI::Ptr localCloudPtr(new PointCloudXYZI()), localCloudDownPtr(new PointCloudXYZI());

        _Rwg = state.rot_end.matrix() * _Rig;                 //Rwg=Rwi*Rig
        _twg = state.rot_end.matrix() * _tig + state.pos_end; //twg
        Eigen::Matrix3d Rgi = _Rig.inverse();
        Eigen::Vector3d tgi = -_Rig.inverse() * _tig;
        Eigen::Matrix3d Rgl = Rgi * state.offset_R_L_I.matrix();       //Rgl=Rgi*Ril
        Eigen::Vector3d tgl = Rgi * state.offset_T_L_I.matrix() + tgi; //Tgi*til
        transCloud(_localCloudPtr, localCloudPtr, Rgl, tgl);
        transCloud(_localCloudDownPtr, localCloudDownPtr, Rgl, tgl);
        std::vector<PointWithCov> curPvList;
        for (auto pv : _curSurfPvList) {
            pv.pi = Rgi * pv.pi + tgi;
            curPvList.emplace_back(pv);
        }
        if (_measures.gnss) { // input gnss info into loopclosing thread
            processGNSS();
            _measures.gnss = nullptr;
        }

        // input frame info into loopclosing thread\
        //_loopCloser->getProcessLock().lock();
        _loopCloser->inputFrameData(_frameId, _lidarBegTime,
                                    _Rwg, _twg,
                                    localCloudPtr, localCloudDownPtr,
                                    curPvList,
                                    _imuProcessor->_imuDatas,
                                    false,
                                    _globalGrav);
        //_loopCloser->getProcessLock().unlock();

        // Update frontend state
        updateLoopStatus(state);
        _imuProcessor->_imuDatas.clear();
        //        if(_loopCloser->loopCount()>0)
        //            PAUSE;
    }
}

void System::updateFrontendMap() {
    std::vector<PointWithCov> totalSurfPvList;
    // Correct new frames pushed in loopclosing thread
    for (auto &frameBlock : _loopCloser->getFrameBuffer()) {
        if (frameBlock->_uniqueId <= _loopCloser->lastCorrectBlockID())
            continue;
        //  std::cout << "Frontend correct buffer block id:" << frameBlock->_uniqueId << std::endl;
        frameBlock->_poseLo = _loopCloser->getLoopTrans() * frameBlock->_poseLo; // Tw2'=Tw1'*T1'2'

        for (auto &pv : frameBlock->_pvList) {
            pv.pw = frameBlock->_poseLo.block<3, 3>(0, 0) * pv.pi + frameBlock->_poseLo.block<3, 1>(0, 3);
            totalSurfPvList.emplace_back(pv);
        }
    }
    // update voxelmap
    if (!totalSurfPvList.empty())
        ::updateVoxelMap(totalSurfPvList, _frameId, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
                         _config._maxPointsSize, _config._maxCovPointsSize, _config._minSurfEigenValue,
                         _loopCloser->getVoxelSurfMap(), _boxToDel);

    if (!_voxelSurfMap.empty()) {
        TicToc time;
        std::cout << "voxel surf map size before loop closing: " << _voxelSurfMap.size() << std::endl;
        for (auto locOctItr : _voxelSurfMap)
            delete locOctItr.second;
        _voxelSurfMap.clear();
        _voxelSurfMap = _loopCloser->getVoxelSurfMap();
        // std::cout << "voxel surf map size after loop closing: " << _voxelSurfMap.size() << std::endl;
        // printf("Delete old voxel map time: %f ms\n", time.toc());
        // std::cout << "Voxel map updated by loop closing" << std::endl;
    }

    _dispMutex.lock();
    if (_ikdtree.Root_Node) {
        TicToc time;
        //update traj in viewer
        _trajPts->clear();
        _trajPropagatPts->clear();
        Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
        gravAlignMat.block<3, 3>(0, 0) = _rotAlign;
        Eigen::Matrix4d Tgi = Eigen::Matrix4d::Identity();
        Tgi.block<3, 3>(0, 0) = _Rig.inverse();
        Tgi.block<3, 1>(0, 3) = -_Rig.inverse() * _tig;
        for (auto block : _loopCloser->getFrameBlocks()) {
            Eigen::Matrix4d Twi = block->_poseLo * Tgi;
            PointType pt;
            pt.x = Twi(0, 3);
            pt.y = Twi(1, 3);
            pt.z = Twi(2, 3);
            _trajPts->push_back(pt); //twi
        }
        pcl::transformPointCloud(*_trajPts, *_trajPts, gravAlignMat);

        // update ikdtree
        // std::cout<<"updating kdtree map on "<<LoopCloser::config()._frontendWinSize<<" frames"<<std::endl;
        const int &blockSize = _loopCloser->getFrameBlocks().size();
        int localBlockSize = LoopCloser::config()._frontendWinSize;
        if (localBlockSize < 0)
            localBlockSize = blockSize;

        int startIdx = std::max(0, blockSize - localBlockSize);
        for (int i = startIdx; i < blockSize; i++) { //add frames cloud after pgo
            auto block = _loopCloser->getFrameBlock(i);
            PointCloudXYZI::Ptr globalPcDown(new PointCloudXYZI());
            pcl::transformPointCloud(*block->_pcDown, *globalPcDown, block->_poseLo);
            if (i == startIdx)
                _ikdtree.Build(globalPcDown->points);
            else
                _ikdtree.Add_Points(globalPcDown->points, true);
        }
        for (auto &block : _loopCloser->getFrameBuffer()) { //add buffer frame cloud in loopclosing thread
            PointCloudXYZI::Ptr globalPcDown(new PointCloudXYZI());
            pcl::transformPointCloud(*block->_pcDown, *globalPcDown, block->_poseLo);
            _ikdtree.Add_Points(globalPcDown->points, true);
        }
        //std::cout<<"idtree map updated"<<std::endl;
        //printf("Update idtree map time: %f ms\n", time.toc());
    }
    _dispMutex.unlock();
}

void System::updateLoopStatus(state_ikfom &state) {
    _loopCloser->getUpdateLock().lock();
    if (_loopCloser->isUpdated()) {
        // Correct new map using frames buffer in loop closing thread
        updateFrontendMap();

        _isLoopCorrected = true;        // For matching points threshold
        _loopCloser->setUpdated(false); //Allow Loop closer to detect frames in buffer

        // Update current state
        Eigen::Matrix4d loopTrans = _loopCloser->getLoopTrans();

        Eigen::Matrix4d curTwg = Eigen::Matrix4d::Identity();
        curTwg.block<3, 3>(0, 0) = state.rot.matrix() * _Rig;             //Rwg=Rwi*Rig
        curTwg.block<3, 1>(0, 3) = state.rot.matrix() * _tig + state.pos; //twg
        Eigen::Matrix4d curTwi = Eigen::Matrix4d::Identity();
        curTwi.block<3, 3>(0, 0) = state.rot.matrix();
        curTwi.block<3, 1>(0, 3) = state.pos;
        Eigen::Matrix4d Tgi = Eigen::Matrix4d::Identity();
        Tgi.block<3, 3>(0, 0) = _Rig.inverse();
        Tgi.block<3, 1>(0, 3) = -_Rig.inverse() * _tig;

        Eigen::Matrix4d transTwg = loopTrans * curTwg;
        Eigen::Matrix4d transTwi = transTwg * Tgi; //_loopCloser->getFrameBlocks().back()->_poseLo*Tgi;//transTwg*Tgi;
        loopTrans = transTwi * curTwi.inverse();

        state.pos = transTwi.block<3, 1>(0, 3);
        state.rot = transTwi.block<3, 3>(0, 0);
        state.vel = loopTrans.block<3, 3>(0, 0) * state.vel;
        _kf.change_x(state);
        _Rwl = state.rot * state.offset_R_L_I; // Rwl=Rwi*Ril
        _twl = state.rot.matrix() * state.offset_T_L_I + state.pos;

        // {//Append current frame corrected into frontend map
        //     transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
        //     if(_config._matchMethod==0)//KdTree
        //         updateKdTreeMap(_globalCloudDownPtr->points);
        //     if(_config._matchMethod==1)//Voxelmap
        //     {
        //         if(_config._covType==2)
        //             _ikdtree.Add_Points(_globalCloudDownPtr->points,true);
        //         updateVoxelMap();
        //     }
        // }
        updateViewer(true); // update viewer
        //_loopCloser->getGlobalCloudDown()->clear();
        // PAUSE;
    }
    _loopCloser->getUpdateLock().unlock();
}

void System::updateLoopStatus(StatesGroup &state) {
    _loopCloser->getUpdateLock().lock();
    if (_loopCloser->isUpdated()) {
        // Correct new map using frames buffer in loop closing thread
        updateFrontendMap();

        _isLoopCorrected = true;        // For matching points threshold
        _loopCloser->setUpdated(false); //Allow Loop closer to detect frames in buffer

        // update current state
        Eigen::Matrix4d loopTrans = _loopCloser->getLoopTrans();

        Eigen::Matrix4d curTwg = Eigen::Matrix4d::Identity();
        curTwg.block<3, 3>(0, 0) = state.rot_end * _Rig;                 //Rwg=Rwi*Rig
        curTwg.block<3, 1>(0, 3) = state.rot_end * _tig + state.pos_end; //twg
        Eigen::Matrix4d curTwi = Eigen::Matrix4d::Identity();
        curTwi.block<3, 3>(0, 0) = state.rot_end;
        curTwi.block<3, 1>(0, 3) = state.pos_end;
        Eigen::Matrix4d Tgi = Eigen::Matrix4d::Identity();
        Tgi.block<3, 3>(0, 0) = _Rig.inverse();
        Tgi.block<3, 1>(0, 3) = -_Rig.inverse() * _tig;

        Eigen::Matrix4d transTwg = loopTrans * curTwg;
        Eigen::Matrix4d transTwi = transTwg * Tgi; //_loopCloser->getFrameBlocks().back()->_poseLo*Tgi;//transTwg*Tgi;
        loopTrans = transTwi * curTwi.inverse();

        state.pos_end = transTwi.block<3, 1>(0, 3);
        state.rot_end = transTwi.block<3, 3>(0, 0);
        state.vel_end = loopTrans.block<3, 3>(0, 0) * state.vel_end;
        _Rwl = state.rot_end * state.offset_R_L_I; // Rwl=Rwi*Ril
        _twl = state.rot_end * state.offset_T_L_I + state.pos_end;

        { //Append current frame corrected into frontend map
            transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
            if (_config._matchMethod == 0) //KdTree
                updateKdTreeMap(_globalCloudDownPtr->points);
            if (_config._matchMethod == 1) //Voxelmap
            {
                if (_config._covType == 2)
                    _ikdtree.Add_Points(_globalCloudDownPtr->points, true);
                updateVoxelMap();
            }
        }
        updateViewer(true); // update viewer
        //_loopCloser->getGlobalCloudDown()->clear();
    }
    _loopCloser->getUpdateLock().unlock();
}

void System::deleteUnstablePoints() {
    PointVector ptToDelList;
    _ikdtree.Delete_Point_Boxes(_boxToDel);
    if (_boxToDel.size() > 0)
        std::cout << "Deleted boxes:" << _boxToDel.size() << std::endl;

    _boxToDel.clear();
}

void System::calibAndRefineLI() {
    _frameNum++;
    if (!_config._isImuInitialized && !_isDataAccumStart && _stateCur.pos_end.norm() > 0.05) {
        printf(BOLDCYAN "[Initialization] Movement detected, data accumulation starts.\n\n\n\n\n" RESET);
        _isDataAccumStart = true;
        _moveStartTime = _lidarEndTime;
    }
    // Refine state
    if (_config._isImuInitialized && (_frameNum % _config._origOdomFreq * _config._cutFrameNum == 0) && !_isOnlineCalibFinish) {
        double onlineCalibDuration = _lidarEndTime - _onlineCalibStartTime;
        onlineCalibDuration = onlineCalibDuration < _config._onlineRefineTime ? onlineCalibDuration : _config._onlineRefineTime;
        cout << "\x1B[2J\x1B[H"; // clear the screen
        if (_config._onlineRefineTime > 0.1)
            printProgress(onlineCalibDuration / _config._onlineRefineTime);
        if (onlineCalibDuration > (_config._onlineRefineTime - 1e-6)) {
            _isOnlineCalibFinish = true;
            cout << endl;
            printState();
            _foutResult << "Refinement result:" << endl;
            fileoutCalibResult(_foutResult, _stateCur, _timediffImuWrtLidar + _config._timeLagIMUWtrLidar);
            _initiatorLI->outputErrorPlot(RESULT_FILE_DIR("IMU_Refinement"));
            cout << endl
                 << "Initialization and refinement result is written to " << endl
                 << BOLDGREEN
                 << RESULT_FILE_DIR("Initialization_result.txt") << RESET << endl;
        }
    }
    // Calibrate lidar to imu and initialize states
    if (!_config._isImuInitialized && !_isDataAccumFinished && _isDataAccumStart) {
        // Push Lidar's Angular velocity and linear velocity
        _initiatorLI->push_Lidar_CalibState(_stateCur.rot_end, _stateCur.bias_g, _stateCur.vel_end, _lidarEndTime);
        // Data Accumulation Sufficience Appraisal
        _isDataAccumFinished = _initiatorLI->data_sufficiency_assess(_jacoRot, _frameNum, _stateCur.bias_g,
                                                                     mutableConfig()._origOdomFreq, mutableConfig()._cutFrameNum);

        if (_isDataAccumFinished) {
            //            std::cout<<"Data accumulation Finished"<<std::endl;
            _initiatorLI->LI_Initialization(mutableConfig()._origOdomFreq, mutableConfig()._cutFrameNum, _timediffImuWrtLidar, _moveStartTime);

            _onlineCalibStartTime = _lidarEndTime;

            // Transfer to FAST-LIO2
            mutableConfig()._isImuInitialized = true;
            _stateCur.offset_R_L_I = _initiatorLI->get_R_LI();
            _stateCur.offset_T_L_I = _initiatorLI->get_T_LI();
            _stateCur.pos_end = -_stateCur.rot_end * _stateCur.offset_R_L_I.transpose() * _stateCur.offset_T_L_I + _stateCur.pos_end; // twi=-Rwl*Rli*til+twl=Twl*tli
            _stateCur.rot_end = _stateCur.rot_end * _stateCur.offset_R_L_I.transpose();
            _stateCur.gravity = _initiatorLI->get_Grav_L0();
            _stateCur.bias_g = _initiatorLI->get_gyro_bias();
            _stateCur.bias_a = _initiatorLI->get_acc_bias();

            if (LidarProcess::config()._lidarType != AVIA)
                mutableConfig()._cutFrameNum = 2;
            // _cutFrameNum = 1;
            mutableConfig()._timeLagIMUWtrLidar = _initiatorLI->get_total_time_lag(); // Compensate IMU's time in the buffer
            for (int i = 0; i < _imuBuffer.size(); i++)
                _imuBuffer[i]->header = _imuBuffer[i]->header - _config._timeLagIMUWtrLidar;
            if (_config._udpateMethod == 0) {
                _imuProcessor->imu_en = _config._isImuInitialized;
                _imuProcessor->LI_init_done = true;
                _imuProcessor->set_mean_acc_norm(_config._meanAccNorm);
                _imuProcessor->set_gyr_cov(V3D(_config._gyrCov, _config._gyrCov, _config._gyrCov));
                _imuProcessor->set_acc_cov(V3D(_config._accCov, _config._accCov, _config._accCov));
                _imuProcessor->set_gyr_bias_cov(V3D(_config._bGyrCov, _config._bGyrCov, _config._bGyrCov));
                _imuProcessor->set_acc_bias_cov(V3D(_config._bAccCov, _config._bAccCov, _config._bAccCov));
            }

            // Output Initialization result
            _foutResult << "Initialization result:" << endl;
            fileoutCalibResult(_foutResult, _stateCur, _timediffImuWrtLidar + _config._timeLagIMUWtrLidar);
            _initiatorLI->outputErrorPlot(RESULT_FILE_DIR("IMU_Initialization"));
        }
    }
}

bool System::assertDegeneracy() {
    return false;
    Eigen::MatrixXd normMat;
    normMat.resize(3, _matchedSurfList.size());
    for (int i = 0; i < _matchedSurfList.size(); i++)
        normMat.block<3, 1>(0, i) = _matchedSurfList[i].normal;
    Eigen::Matrix3d M = normMat * normMat.transpose();
    //std::cout<<"M:"<<M<<std::endl;
    Eigen::EigenSolver<Eigen::Matrix3d> es(M);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    double minEigenValue = evalsReal(evalsMin);
    double midEigenValue = evalsReal(evalsMid);
    double maxEigenValue = evalsReal(evalsMax);
    double ratio1 = minEigenValue / midEigenValue;
    double ratio2 = midEigenValue / maxEigenValue;
    //    std::cout<<"ratio1:"<<ratio1<<std::endl;
    //    std::cout<<"ratio2:"<<ratio2<<std::endl;
    if (ratio1 < 0.05 || ratio2 < 0.05)
        return true;
    return false;
}

void System::mapping() {
    if (_frameId < _config._nSkipFrames)
        return;

    if (_measures.lidar->empty())
        return;

    // Motor data exists: transform local pcl into motor zero angle coordinate(let pl be coord in motor zero axis)
    if (System::config()._isMotorInitialized && _curMotorAngle >= 0) {
        //TicToc time;
        if (_cloudAxisTransfer) // for ZG device
            motorMotionCompensationZG();
        else // other devices(extrinsic consist of lidar->motor(0 deg), motor(0 deg)->imu)
        {
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, 0, _Rol, _tol);
            motorMotionCompensation();
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, _curMotorAngle, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        }
        //printf("Motor undistort time: %f ms\n", time.toc());
    }
    //    ostringstream pathOfs;
    //    pathOfs<<std::setprecision(6)<<std::fixed<<string(ROOT_DIR) + "PCD/"<<_lidarBegTime-_sensorTimeDiff<<".txt";
    //    ReadWriter::outputCloudFile(pathOfs.str(),*_measures.lidar);

    //    static std::ofstream fCloudOfs =std::ofstream(string(ROOT_DIR) + "PCD/all.txt", std::ios::trunc | std::ios::in);
    //    for(auto& pt: _measures.lidar->points)
    //        fCloudOfs<<std::fixed<<std::setprecision(8)<<","<<pt.x<<","<<pt.y<<","<<pt.z<<","<<rad2deg(_curMotorAngle)<<std::endl;

    *_localCloudPtr = *_measures.lidar;
    // merge cloud by motor angle
    if (_frameId < _config._nMergedFrames) //_nMergedFrames=0
        processCloudOnlyMotor();
    // calibrate IMU extrinsic
    else if (!_config._isImuInitialized)
        processCloudESIKF();
    // Mapping without imu
    else if (_measures.imu.empty())
        //processCloudCeres();
        //processCloudLSQ(_regVGICP);
        processCloudVGICP();
    //    // Mapping with imu
    else if (!_measures.imu.empty()) {
        if (_config._udpateMethod == 0)
            processCloudIKFoM();
        //processCloudESIKF();
        else
            processCloudLSQ(_regPCL_VGICP);
        // processCloudVGICP();
        //processCloudLOAM();
    }
}

void System::mapping_sim() {
    if (_frameId < _config._nSkipFrames)
        return;

    if (_measures.lidar->empty())
        return;

    // Motor data exists: transform local pcl into motor zero angle coordinate(let pl be coord in motor zero axis)
    if (System::config()._isMotorInitialized && _curMotorAngle >= 0) {
        //TicToc time;
        if (_cloudAxisTransfer) // for ZG device
            motorMotionCompensationZGforSIM();
        else // other devices(extrinsic consist of lidar->motor(0 deg), motor(0 deg)->imu)
        {
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, 0, _Rol, _tol);
            motorMotionCompensation();
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, _curMotorAngle, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        }
        //printf("Motor undistort time: %f ms\n", time.toc());
    }
    //    ostringstream pathOfs;
    //    pathOfs<<std::setprecision(6)<<std::fixed<<string(ROOT_DIR) + "PCD/"<<_lidarBegTime-_sensorTimeDiff<<".txt";
    //    ReadWriter::outputCloudFile(pathOfs.str(),*_measures.lidar);

    //    static std::ofstream fCloudOfs =std::ofstream(string(ROOT_DIR) + "PCD/all.txt", std::ios::trunc | std::ios::in);
    //    for(auto& pt: _measures.lidar->points)
    //        fCloudOfs<<std::fixed<<std::setprecision(8)<<","<<pt.x<<","<<pt.y<<","<<pt.z<<","<<rad2deg(_curMotorAngle)<<std::endl;

    *_localCloudPtr = *_measures.lidar;
    // merge cloud by motor angle
    if (_frameId < _config._nMergedFrames) //_nMergedFrames=0
        processCloudOnlyMotor();
    // calibrate IMU extrinsic
    else if (!_config._isImuInitialized)
        processCloudESIKF();
    // Mapping without imu
    else if (_measures.imu.empty())
        //processCloudCeres();
        //processCloudLSQ(_regVGICP);
        processCloudVGICP();
    //    // Mapping with imu
    else if (!_measures.imu.empty()) {
        if (_config._udpateMethod == 0)
            processCloudIKFoM();
        //processCloudESIKF();
        else
            processCloudLSQ(_regPCL_VGICP);
        // processCloudVGICP();
        //processCloudLOAM();
    }
}

void System::mapping_calib() {
    if (_frameId < _config._nSkipFrames)
        return;

    if (_measures.lidar->empty())
        return;

    // Motor data exists: transform local pcl into motor zero angle coordinate(let pl be coord in motor zero axis)
    if (System::config()._isMotorInitialized && _curMotorAngle >= 0) {
        //TicToc time;
        if (_cloudAxisTransfer) // for ZG device
            motorMotionCompensationZG();
        else // other devices(extrinsic consist of lidar->motor(0 deg), motor(0 deg)->imu)
        {
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, 0, _Rol, _tol);
            motorMotionCompensation();
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, _curMotorAngle, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        }
        //printf("Motor undistort time: %f ms\n", time.toc());
    }
    //    ostringstream pathOfs;
    //    pathOfs<<std::setprecision(6)<<std::fixed<<string(ROOT_DIR) + "PCD/"<<_lidarBegTime-_sensorTimeDiff<<".txt";
    //    ReadWriter::outputCloudFile(pathOfs.str(),*_measures.lidar);

    //    static std::ofstream fCloudOfs =std::ofstream(string(ROOT_DIR) + "PCD/all.txt", std::ios::trunc | std::ios::in);
    //    for(auto& pt: _measures.lidar->points)
    //        fCloudOfs<<std::fixed<<std::setprecision(8)<<","<<pt.x<<","<<pt.y<<","<<pt.z<<","<<rad2deg(_curMotorAngle)<<std::endl;

    *_localCloudPtr = *_measures.lidar;
    // merge cloud by motor angle
    if (_frameId < _config._nMergedFrames) //_nMergedFrames=0
        processCloudOnlyMotor();
    // calibrate IMU extrinsic
    else if (!_config._isImuInitialized)
        processCloudESIKF();
    // Mapping without imu
    else if (_measures.imu.empty())
        //processCloudCeres();
        //processCloudLSQ(_regVGICP);
        processCloudVGICP();
    //    // Mapping with imu
    else if (!_measures.imu.empty()) {
        if (_config._udpateMethod == 0)
            //processCloudIKFoM();
            processCloudIKFoM_calib();
        //processCloudESIKF();
        else
            processCloudLSQ(_regPCL_VGICP);
        // processCloudVGICP();
        //processCloudLOAM();
    }
}

void System::mapping_undist() {
    if (_frameId < _config._nSkipFrames)
        return;

    if (_measures.lidar->empty())
        return;

    // Motor data exists: transform local pcl into motor zero angle coordinate(let pl be coord in motor zero axis)
    if (System::config()._isMotorInitialized && _curMotorAngle >= 0) {
        //TicToc time;
        if (_cloudAxisTransfer) // for ZG device
            motorMotionCompensationZG();
        else // other devices(extrinsic consist of lidar->motor(0 deg), motor(0 deg)->imu)
        {
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, 0, _Rol, _tol);
            motorMotionCompensation();
            transCloudInMotorAxis(_measures.lidar, _measures.lidar, _curMotorAngle, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        }
        //printf("Motor undistort time: %f ms\n", time.toc());
    }
    *_localCloudPtr = *_measures.lidar;
}
//void System::processGNSS()
//{
//#ifdef GTSAM_ON
//    if(_loopCloser&&!_loopCloser->getFrameBlocks().empty())
//    {
//        while(_loopCloser->getFrameBlocks().back()->_uniqueId!=_frameId)
//            usleep(5000);
//        std::cout<<"last frameblock uniqueID:"<<_loopCloser->getFrameBlocks().back()->_uniqueId<<std::endl;
//        _loopCloser->getProcessLock().lock();
//
//        //constraints framewisePgoEdges;
//        // _loopCloser->constructFrameEdges(framewisePgoEdges);
//        auto lastFrameBlock=_loopCloser->getFrameBlocks().back();
//        _gpsConstraintList.emplace_back(_measures.gnss,lastFrameBlock->_idInStrip);
//
//        static std::vector<double> sourceFramePts,targetFramePts;
//        for (size_t i = 0; i < 3; i++)
//            targetFramePts.emplace_back(lastFrameBlock->_poseLo(i,3));
//        for (size_t i = 0; i < 3; i++)
//            sourceFramePts.emplace_back(_measures.gnss->pos[i]);
//        if(!_isGnssInited)
//        {
//            if(_gpsConstraintList.size()>=config()._gnssInitSize)
//                _isGnssInited=true;
//        }
//        if(_isGnssInited)
//        {
//            double R[9]; double t[3];
//            double s = 1.0;
//            if(AbsouteOrientation::runCalSevenParams(targetFramePts, sourceFramePts, _gpsConstraintList.size(), R, t, s))
//            {
//                _gnssTrans(0, 0) = R[0]; _gnssTrans(0, 1) = R[1]; _gnssTrans(0, 2) = R[2];_gnssTrans(0, 3) = t[0];
//                _gnssTrans(1, 0) = R[3]; _gnssTrans(1, 1) = R[4]; _gnssTrans(1, 2) = R[5];_gnssTrans(1, 3) = t[1];
//                _gnssTrans(2, 0) = R[6]; _gnssTrans(2, 1) = R[7]; _gnssTrans(2, 2) = R[8];_gnssTrans(2, 3) = t[2];
//                std::cout<<"Gnss trans matrix:\n"<<_gnssTrans<<std::endl;
//                std::cout<<"Absoute orientation scale:"<<s<<std::endl;
//            }
//
//            PointCloudXYZI::Ptr gpsCloudTrans(new PointCloudXYZI());
//            for(auto gpsConstraint:_gpsConstraintList)
//            {
//                SensorMsgs::GNSSData::Ptr gpsData=gpsConstraint.first;
//                const int& idInStrip=gpsConstraint.second;
//                V3D gnssPosTrans=_gnssTrans.block<3,3>(0,0)*gpsData->pos+_gnssTrans.block<3,1>(0,3);
//
//                PointType pt;
//                pt.x=gnssPosTrans.x();
//                pt.y=gnssPosTrans.y();
//                pt.z=gnssPosTrans.z();
//                gpsCloudTrans->push_back(pt);
//
//                auto frameBlock=loopCloser()->getFrameBlock(idInStrip);
//                double error=(gnssPosTrans-frameBlock->_poseLo.block<3,1>(0,3)).norm();
//                std::cout<<"Align error:"<<error<<std::endl;
//            }
//            pcl::PCDWriter pcdWriter;
//            pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/gpsCloudTrans.pcd", *gpsCloudTrans);
//
//        }
//        _loopCloser->getProcessLock().unlock();
//
//        PAUSE;
//    }
//#endif
//}

void System::processGNSS() {
#ifdef GTSAM_ON
    if (_loopCloser && !_loopCloser->getFrameBlocks().empty()) {
        if (_lastGNSSData)
            if ((_measures.gnss->pos - _lastGNSSData->pos).norm() < 0.5)
                return;
        _lastGNSSData = _measures.gnss;

        //constraints framewisePgoEdges;
        // _loopCloser->constructFrameEdges(framewisePgoEdges);
        _loopCloser->addGNSSFrameIdx(_frameId);
        _gpsConstraintBuf.emplace_back(_measures.gnss, _frameId);

        static std::vector<double> sourceFramePts, targetFramePts;
        if (!_isGnssInited) {
            for (size_t i = 0; i < 3; i++)
                targetFramePts.emplace_back(_twl(i));
            for (size_t i = 0; i < 3; i++)
                sourceFramePts.emplace_back(_measures.gnss->pos[i]);

            if (_gpsConstraintBuf.size() >= config()._gnssInitSize) {
                double R[9];
                double t[3];
                double s = 1.0;
                if (AbsouteOrientation::runCalSevenParams(targetFramePts, sourceFramePts, _gpsConstraintBuf.size(), R, t, s)) {
                    _gnssTrans(0, 0) = R[0];
                    _gnssTrans(0, 1) = R[1];
                    _gnssTrans(0, 2) = R[2];
                    _gnssTrans(0, 3) = t[0];
                    _gnssTrans(1, 0) = R[3];
                    _gnssTrans(1, 1) = R[4];
                    _gnssTrans(1, 2) = R[5];
                    _gnssTrans(1, 3) = t[1];
                    _gnssTrans(2, 0) = R[6];
                    _gnssTrans(2, 1) = R[7];
                    _gnssTrans(2, 2) = R[8];
                    _gnssTrans(2, 3) = t[2];
                    _isGnssInited = true;
                    targetFramePts.clear();
                    sourceFramePts.clear();
                    //std::cout<<"Gnss trans matrix:\n"<<_gnssTrans<<std::endl;
                    //std::cout<<"Absoute orientation scale:"<<s<<std::endl;
                }
            }
        }
        if (_isGnssInited) {
            static PointCloudXYZI::Ptr gnssCloud(new PointCloudXYZI());
            _dispMutex.lock();
            while (!_gpsConstraintBuf.empty()) {
                SensorMsgs::GNSSData::Ptr gpsData = _gpsConstraintBuf.front().first;
                const int &frameID = _gpsConstraintBuf.front().second;

                SensorMsgs::GNSSData::Ptr gpsDataTrans(new SensorMsgs::GNSSData(*gpsData));
                gpsDataTrans->pos = _gnssTrans.block<3, 3>(0, 0) * gpsData->pos + _gnssTrans.block<3, 1>(0, 3);

                _gpsConstraintBuf.pop_front();
                //std::cout<<"Add GNSS pos contraint to frame "<<frameID<<": "<<gpsDataTrans->pos.transpose()<<std::endl;
                if (_loopCloser->config()._isOutputPCD) {
                    PointType pt;
                    pt.x = gpsDataTrans->pos.x();
                    pt.y = gpsDataTrans->pos.y();
                    pt.z = gpsDataTrans->pos.z();
                    gnssCloud->push_back(pt);
                }

                _loopCloser->inputGNSSData(gpsDataTrans, frameID);
                if (config()._isEnable3DViewer)
                    _gnssQueue.emplace(gpsDataTrans);
            }
            _dispMutex.unlock();
            if (_loopCloser->config()._isOutputPCD) {
                pcl::PCDWriter pcdWriter;
                pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/gnssCloud.pcd", *gnssCloud);
            }

            //            if(_loopCloser->config()._isPgoIncremental)
            //            {
            //                _loopCloser->frameGraphOptimize();
            //                if(!_loopCloser->config()._isCorrectRealTime)
            //                    _loopCloser->correctFrontendMap(_loopCloser->getFrameBlocks());
            //                updateLoopStatus(_stateIkfom);
            //            }
        }
        //PAUSE;
    }
#endif
}

bool System::syncPackages(MeasureGroup &meas) {
    if (_lidarBuffer.empty())
        return false;
    /*** push a lidar scan ***/
    if (!_isLidarPushed) {
        meas.lidar = _lidarBuffer.front();
        meas.lidar_beg_time = _timeBuffer.front();
        if (meas.lidar->points.size() <= _config._minFramePoint) // frame points too little
        {
            _lidarBuffer.pop_front();
            _timeBuffer.pop_front();
            std::cerr << "Skip scan: Too few input point cloud!" << std::endl;
            return false;
        } else {
            if (LidarProcess::config()._lidarType == TIMOO16 && (meas.lidar->points.back().curvature - meas.lidar->points.front().curvature < 0.5 * double(1000) / double(_config._origOdomFreq * _config._cutFrameNum))) {
                std::cerr << "TIMOO lidar timestamp error!\n";
                _lidarBuffer.pop_front();
                _timeBuffer.pop_front();
                return false;
            }
            _scanNum++;
            _lidarEndTime = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        }

        meas.lidar_end_time = _lidarEndTime;

        _isLidarPushed = true;
    }

    collectMotorIMU();

    // ensure motor data exist after current lidar frame
    if (!_motorAngleBuf.empty() && _lastMotorTimestamp < _lidarEndTime)
        return false;
    if (!_imuBuffer.empty() && _config._isMotorInitialized) {
        // ensure imu data exist after current lidar frame
        if (_lastImuTimestamp < _lidarEndTime)
            return false;
        /*** push imu data, and pop from imu buffer ***/
        double imuTime = _imuBuffer.front()->header;
        meas.imu.clear();
        while ((!_imuBuffer.empty()) && (imuTime < _lidarEndTime)) {
            auto &frontIMUData = _imuBuffer.front();
            imuTime = _imuBuffer.front()->header;
            if (imuTime > _lidarEndTime)
                break;
            meas.imu.push_back(_imuBuffer.front());
            if (_imuPreintegration) {
                if (imuTime <= _lidarEndTime) {
                    if (_imuPreintegration->predictByImu(frontIMUData, _statePropagat)) {
                        const gtsam::NavState &curState = _imuPreintegration->getCurState();
                        Pose6D imuPose = set_pose6d(imuTime - _lidarBegTime,
                                                    frontIMUData->linear_acceleration, frontIMUData->angular_velocity,
                                                    curState.velocity(), curState.position(), curState.R());
                        _imuProcessor->pushIMUPose(imuPose); // For cloud undistortion
                        _stateCur = _statePropagat;
                        //printf(REDPURPLE "[Preintergrate] " RESET);
                        //std::cout << "Pos in World Frame   = " << _stateCur.pos_end.transpose() << std::endl;
                        // PAUSE;
                    }

                    // std::cout<<std::setprecision(6)<<std::fixed<<imuTime<<": "<<frontIMUData->orientation.coeffs().transpose()<<std::endl;
                    // _stateCur.rot_end = frontIMUData->orientation.matrix();
                }
            }
            _imuBuffer.pop_front();
        }
    }
    /*** get current motor angle, and pop from motor buffer ***/
    if (!_motorAngleBuf.empty()) {
        if (!collectRasterAngles(meas.lidar_beg_time + meas.lidar->front().curvature / double(1000), meas.lidar_end_time)) {
            _lidarBuffer.pop_front();
            _timeBuffer.pop_front();
            _isLidarPushed = false;
            return false;
        }
        if (!getRasterAngle(meas.lidar_end_time))
            return false;
        //        cout << "Current Motor Angle: " << _curMotorAngle << endl;
    }

    if (!_gpsBuffer.empty()) {
        /*** push imu data, and pop from imu buffer ***/
        while ((!_gpsBuffer.empty())) {
            auto &frontGPSData = _gpsBuffer.front();
            const double gpsTime = _gpsBuffer.front()->header;
            //            std::cout<<"gpsTime:"<<gpsTime<<std::endl;
            //            std::cout<<"lidarEndTime:"<<_lidarEndTime<<std::endl;
            if (gpsTime > _lidarEndTime + 0.1)
                break;
            if (gpsTime > _lidarEndTime - 0.1) {
                meas.gnss = frontGPSData;
                _gpsBuffer.pop_front();
                break;
            }
            _gpsBuffer.pop_front();
        }
    }
    _lidarBuffer.pop_front();
    _timeBuffer.pop_front();
    _isLidarPushed = false;
    //std::cout << std::fixed << std::setprecision(6) << "Lidar end time: " << _lidarEndTime << endl;
    return true;
}

void System::collectMotorIMU() {
    while (!_motorAngleBufAll.empty()) {
        double frontTime = _motorAngleBufAll.front().first + _sensorTimeDiff;
        if (frontTime <= _lidarEndTime + 0.1) {
            if (frontTime < _lastMotorTimestamp) {
                std::cerr << "Motor loop back." << std::endl;
                //_motorAngleBuf.clear();
                _motorAngleBufAll.pop_front();
                continue;
            }
            _motorAngleBuf.push_back(_motorAngleBufAll.front());
            _motorAngleBuf.back().first = frontTime;
            _lastMotorTimestamp = frontTime;

            _motorAngleBufAll.pop_front();
            // cout << "motorTimestamp: " << frontTime << ", _lidarBegTime: " << _lidarBegTime << endl;
        } else
            break;
    }

    while (!_imuBufferAll.empty()) {
        double frontTime = _imuBufferAll.front()->header + _sensorTimeDiff - _timediffImuWrtLidar - _config._timeLagIMUWtrLidar;
        if (frontTime <= _lidarEndTime + 0.1) {
            // IMU Time Compensation
            if (frontTime < _lastImuTimestamp) {
                std::cerr << "IMU loop back, clear IMU buffer." << std::endl;
                _imuBuffer.clear();
            }

            _lastImuTimestamp = frontTime;
            _imuBuffer.push_back(_imuBufferAll.front());
            _imuBuffer.back()->header = frontTime;

            // push all IMU meas into Init_LI
            if (!_config._isImuInitialized && !_isDataAccumFinished)
                _initiatorLI->push_ALL_IMU_CalibState(_imuBufferAll.front(), _config._meanAccNorm);
            _imuBufferAll.pop_front();
            // cout << "imuTimestamp: " << frontTime + _timediffImuWrtLidar + _timeLagIMUWtrLidar << ", _lidarBegTime: " << _lidarBegTime << endl;
        } else
            break;
    }

    while (!_gpsBufferAll.empty()) {
        double frontTime = _gpsBufferAll.front()->header;
        if (frontTime <= _lidarEndTime + 0.1) {
            // IMU Time Compensation
            if (frontTime < _lastGPSTimestamp) {
                std::cerr << "IMU loop back, clear IMU buffer." << std::endl;
                _gpsBuffer.clear();
            }

            _lastGPSTimestamp = frontTime;
            _gpsBuffer.push_back(_gpsBufferAll.front());
            _gpsBufferAll.pop_front();
            // cout << "imuTimestamp: " << frontTime + _timediffImuWrtLidar + _timeLagIMUWtrLidar << ", _lidarBegTime: " << _lidarBegTime << endl;
        } else
            break;
    }
}

double System::getSensorTimeDiff(double lidTime, double imuTime) {
    tm *tm_lidar, *tm_raw;
    tm tm_lidar_new, tm_raw_new;

    time_t lidar_timep = lidTime;
    time_t raw_timep = imuTime;

    tm_lidar = gmtime(&lidar_timep);
    tm_lidar_new.tm_year = tm_lidar->tm_year;
    tm_lidar_new.tm_mon = tm_lidar->tm_mon;
    tm_lidar_new.tm_mday = tm_lidar->tm_mday;
    tm_lidar_new.tm_hour = 0;
    tm_lidar_new.tm_min = 0;
    tm_lidar_new.tm_sec = 0;
    tm_lidar_new.tm_isdst = tm_lidar->tm_isdst;

    tm_raw = gmtime(&raw_timep);
    tm_raw_new.tm_year = tm_raw->tm_year;
    tm_raw_new.tm_mon = tm_raw->tm_mon;
    tm_raw_new.tm_mday = tm_raw->tm_mday;
    tm_raw_new.tm_hour = 0;
    tm_raw_new.tm_min = 0;
    tm_raw_new.tm_sec = 0;
    tm_raw_new.tm_isdst = tm_raw->tm_isdst;

    time_t t_lidar = mktime(&tm_lidar_new) + 28800.;
    time_t t_raw = mktime(&tm_raw_new) - 28800.;
    //std::cout << "t_lidar: " << t_lidar << std::endl;
    // std::cout << "t_raw: " << t_raw << std::endl;

    return t_lidar - t_raw;
}

void System::lidCallback(PPointCloud::Ptr cloudIn, double lidTime) {
    _mtxBuffer.lock();

    if (_isFirstLidarFrame && _config._isTimeSyncEn) {
        _isFirstLidarFrame = false;
        _sensorTimeDiff = getSensorTimeDiff(lidTime, _imuBufferAll.front()->header);
    }

    if (lidTime < _lidarEndTime) {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        _lidarBuffer.clear();
        _timeBuffer.clear();
    }
    _lidarBegTime = lidTime;
    // cout << std::fixed << std::setprecision(6) << "Lidar beg time: " << lidTime << endl;

    if (!_isTimediffSetFlg && abs(_lidarBegTime - _lastImuTimestamp) > 20. && !_imuBuffer.empty()) {
        _isTimediffSetFlg = true;
        _timediffImuWrtLidar = _lastImuTimestamp - _lidarBegTime;
        printf("Self sync IMU and Lidar, time diff is %.10lf \n", _timediffImuWrtLidar);
    }
    if (!_config._isCutFrame) {
        PointCloudXYZI::Ptr cloudOut(new PointCloudXYZI());
        _lidarProcessor->process(cloudIn, lidTime, cloudOut);
        _lidarBuffer.push_back(cloudOut);
        _timeBuffer.push_back(lidTime);
        // std::cout<<"Lidar begin:"<<cloudOut->front().curvature<<std::endl;
        // std::cout<<"Lidar end:"<<cloudOut->back().curvature<<std::endl;
    } else {
        static int scanCount = 0;
        deque<PointCloudXYZI::Ptr> ptrQueue;
        deque<double> timeQueue;

        _lidarProcessor->processCutFrameCloud(cloudIn, lidTime, ptrQueue, timeQueue, _config._cutFrameNum, scanCount++);
        while (!ptrQueue.empty() && !timeQueue.empty()) {
            _lidarBuffer.push_back(ptrQueue.front());
            ptrQueue.pop_front();
            _timeBuffer.push_back(timeQueue.front() / double(1000)); // unit:s
            timeQueue.pop_front();
        }
    }

    _mtxBuffer.unlock();
    _sigBuffer.notify_all();
    if (syncPackages(_measures)) {
        if (!_config._isMotorInitialized)
            motorInitialize();
        else
            mapping();

        std::cout << "Frame id: " << _frameId << std::endl
                  << std::endl;
        _frameId++;
    }
}

void System::loadParams(const std::string &filePath) {
    // load params
    YAML::Node node = YAML::LoadFile(filePath);

    const YAML::Node &common = node["common"];
    _config._imuFilePath = common["imu_file_path"].as<std::string>();
    _config._rasterFilePath = common["raster_file_path"].as<std::string>();
    _config._pcapFilePath = common["pcap_file_path"].as<std::string>();
    _config._lidarCorrectFilePath = common["lidar_correction_file_path"].as<std::string>();
    _config._isEnable3DViewer = common["enable_3d_viewer"].as<bool>();
    _config._isTimeSyncEn = common["time_sync_en"].as<bool>();
    _config._isMotorInitialized = common["is_motor_initialized"].as<bool>();
    _config._isImuInitialized = common["is_imu_initialized"].as<bool>();
    _config._nSkipFrames = common["skip_frames"].as<int>();
    _config._enableGravityAlign = common["enable_gravity_align"].as<bool>();

    const YAML::Node &preprocess = node["preprocess"];
    _config._minFramePoint = preprocess["min_frame_point"].as<int>();
    _config._isFeatExtractEn = preprocess["feature_extract_en"].as<bool>();
    _config._featureExtractSegNum = preprocess["feature_extract_seg_num"].as<int>();
    _config._nMergedFrames = preprocess["n_merge_frames"].as<int>();
    LidarProcess::mutableConfig()._pointFilterNum = preprocess["point_filter_num"].as<int>();
    LidarProcess::mutableConfig()._blindMin = preprocess["blind_min"].as<double>();
    LidarProcess::mutableConfig()._blindMax = preprocess["blind_max"].as<double>();
    LidarProcess::mutableConfig()._lidarType = preprocess["lidar_type"].as<int>();
    LidarProcess::mutableConfig()._nScans = preprocess["scan_line"].as<int>();
    LidarProcess::mutableConfig()._timeUnit = preprocess["timestamp_unit"].as<int>();
    LidarProcess::mutableConfig()._scanRate = preprocess["scan_rate"].as<int>();

    const YAML::Node &initialization = node["initialization"];
    _config._isCutFrame = initialization["cut_frame"].as<bool>();
    _config._cutFrameNum = initialization["cut_frame_num"].as<int>();
    _config._origOdomFreq = initialization["orig_odom_freq"].as<int>();
    _config._meanAccNorm = initialization["mean_acc_norm"].as<double>();
    _config._onlineRefineTime = initialization["online_refine_time"].as<double>();
    _config._dataAccumLength = initialization["data_accum_length"].as<double>();
    _config._rotLICov = initialization["Rot_LI_cov"].as<vector<double>>();
    _config._transLICov = initialization["Trans_LI_cov"].as<vector<double>>();
    _config._maxInierError = initialization["motor_inlier_error"].as<double>();
    _config._imuMaxInitCount = initialization["imu_init_size"].as<int>();
    _config._gnssInitSize = initialization["gnss_init_size"].as<int>();

    const YAML::Node &mapping = node["mapping"];
    _config._udpateMethod = mapping["update_method"].as<int>();
    _config._matchMethod = mapping["match_method"].as<int>();
    _config._nMaxInterations = mapping["max_iteration"].as<int>();
    _config._detRange = mapping["det_range"].as<double>();
    _config._cubeLen = mapping["cube_side_length"].as<double>();
    _config._filterSizeSurf = mapping["filter_size_surf"].as<double>();
    _config._filterSizeMap = mapping["filter_size_map"].as<double>();
    _config._gyrCov = mapping["gyr_cov"].as<double>();
    _config._accCov = mapping["acc_cov"].as<double>();
    _config._bAccCov = mapping["b_acc_cov"].as<double>();
    _config._bGyrCov = mapping["b_gyr_cov"].as<double>();
    _config._timeLagIMUWtrLidar = mapping["time_lag_IMU_wtr_lidar"].as<double>();
    _config._tilVec = mapping["extrinsic_til"].as<vector<double>>();
    _config._RilVec = mapping["extrinsic_Ril"].as<vector<double>>();
    _config._tolVec = mapping["extrinsic_tol"].as<vector<double>>();
    _config._RolVec = mapping["extrinsic_Rol"].as<vector<double>>();
    _config._tigVec = mapping["extrinsic_tig"].as<vector<double>>();
    _config._RigVec = mapping["extrinsic_Rig"].as<vector<double>>();
    _config._isEstiExtrinsic = mapping["extrinsic_est_en"].as<bool>();
    _config._gnssMaxError = mapping["gnss_max_error"].as<double>();

    _config._maxPointsSize = mapping["max_points_size"].as<int>();
    _config._maxCovPointsSize = mapping["max_cov_points_size"].as<int>();
    _config._layerPointSizeList = mapping["layer_point_size"].as<vector<int>>();
    _config._maxLayers = mapping["max_layers"].as<int>();
    _config._voxelLength = mapping["voxel_size"].as<double>();
    _config._minSurfEigenValue = mapping["plannar_threshold"].as<double>();
    _config._rangingCov = mapping["ranging_cov"].as<double>();
    _config._angleCov = mapping["angle_cov"].as<double>();
    _config._covType = mapping["covariance_type"].as<int>();

    const YAML::Node &BundleAdjustment = node["BundleAdjustment"];
    _config._isEnableBA = BundleAdjustment["is_enable_BA"].as<bool>();
    _config._isVerbose = BundleAdjustment["is_verbose"].as<bool>();
    _config._filterNum = BundleAdjustment["filter_num"].as<int>();
    _config._thdNum = BundleAdjustment["thd_num"].as<int>();
    _config._windowSize = BundleAdjustment["window_size"].as<int>();
    _config._marginSize = BundleAdjustment["margin_size"].as<int>();

    const YAML::Node &LoopClosing = node["LoopClosing"];
    _config._isLoopEn = LoopClosing["is_loop_en"].as<bool>();

    LoopCloser::mutableConfig()._poseGraphOptimizationMethod = LoopClosing["PGO_method"].as<std::string>();
    LoopCloser::mutableConfig()._detectLoopMethod = LoopClosing["detect_method"].as<int>();
    LoopCloser::mutableConfig()._detectorConfigPath = LoopClosing["detector_config_path"].as<std::string>();
    LoopCloser::mutableConfig()._coolingSubmapNum = LoopClosing["cooling_submap_num"].as<int>();
    LoopCloser::mutableConfig()._numFrameLargeDrift = LoopClosing["num_frame_large_drift"].as<int>();
    LoopCloser::mutableConfig()._minSubmapIdDiff = LoopClosing["min_submap_id_diff"].as<int>();
    LoopCloser::mutableConfig()._maxSubMapAccuTran = LoopClosing["max_submap_tccu_tran"].as<double>();
    LoopCloser::mutableConfig()._maxSubMapAccuRot = LoopClosing["max_submap_accu_rot"].as<double>();
    LoopCloser::mutableConfig()._maxSubMapAccuFrames = LoopClosing["max_submap_accu_frames"].as<int>();
    LoopCloser::mutableConfig()._normalRadiusRatio = LoopClosing["normal_radius_ratio"].as<double>();
    LoopCloser::mutableConfig()._fpfhRadiusRatio = LoopClosing["fpfh_radius_ratio"].as<double>();
    LoopCloser::mutableConfig()._salientRadiusRatio = LoopClosing["salient_radius_ratio"].as<double>();
    LoopCloser::mutableConfig()._nonMaxRadiusRatio = LoopClosing["nonMax_radius_ratio"].as<double>();
    LoopCloser::mutableConfig()._voxelDownSize = LoopClosing["voxel_down_size"].as<double>();
    LoopCloser::mutableConfig()._vgicpVoxRes = LoopClosing["vgicp_vox_res"].as<double>();
    LoopCloser::mutableConfig()._isFramewisePGO = LoopClosing["is_frame_wise_pgo"].as<bool>();
    LoopCloser::mutableConfig()._isPgoIncremental = LoopClosing["is_pgo_incremental"].as<bool>();
    LoopCloser::mutableConfig()._isStoreFrameBlock = LoopClosing["is_store_frame"].as<bool>();
    LoopCloser::mutableConfig()._frameStoreInterval = LoopClosing["frame_store_interval"].as<int>();
    LoopCloser::mutableConfig()._isCorrectRealTime = LoopClosing["is_correct_realTime"].as<bool>();
    LoopCloser::mutableConfig()._isLoopDetectEn = LoopClosing["loop_detect_enable"].as<bool>();
    LoopCloser::mutableConfig()._isOutputPCD = LoopClosing["is_output_PCD"].as<bool>();
    LoopCloser::mutableConfig()._frontendWinSize = LoopClosing["frontend_win_size"].as<int>();

    LoopCloser::mutableConfig()._voxelLength = config()._voxelLength;
    LoopCloser::mutableConfig()._maxLayers = config()._maxLayers;
    LoopCloser::mutableConfig()._layerPointSizeList = config()._layerPointSizeList;
    LoopCloser::mutableConfig()._maxPointsSize = config()._maxPointsSize;
    LoopCloser::mutableConfig()._maxCovPointsSize = config()._maxCovPointsSize;
    LoopCloser::mutableConfig()._minEigenValue = config()._minSurfEigenValue;
    LoopCloser::mutableConfig()._gnssMaxError = config()._gnssMaxError;

    const YAML::Node &pcdSave = node["pcd_save"];
    _config._isSaveMap = pcdSave["is_save_map"].as<bool>();
    _config._pcdSaveInterval = pcdSave["interval"].as<int>();
}

bool System::initSystem() {
    _imgProcesser.startThread();
    std::cout << "imgprocesser started!" << std::endl;

    if (_config._isEnable3DViewer) {
        initPCLViewer();
        ColourWheel cw;
        for (int i = 0; i < 200; i++)
            _voxelColors.emplace_back(cw.GetUniqueColour());
    }
    if (_config._isLoopEn) {
        _loopCloser = new LoopCloser;
        if (_loopCloser->config()._detectLoopMethod == 1)
            _loopCloser->mutableConfig()._isNeedGravAligned = false;
        else
            _loopCloser->mutableConfig()._isNeedGravAligned = false;

        _loopCloser->startThread();
        std::cout << "LoopClosing Enabled" << std::endl;
    } else
        std::cout << "LoopClosing Disabled" << std::endl;

    _tol << VEC_FROM_ARRAY(_config._tolVec);
    _Rol << MAT_FROM_ARRAY(_config._RolVec);
    _til << VEC_FROM_ARRAY(_config._tilVec);
    _Ril << MAT_FROM_ARRAY(_config._RilVec);
    _tig << VEC_FROM_ARRAY(_config._tigVec);
    _Rig << MAT_FROM_ARRAY(_config._RigVec);
    _stateCur.offset_R_L_I = _Ril;
    _stateCur.offset_T_L_I = _til;
    _statePropagat = _stateCur;

    if (_config._udpateMethod == 0) { //EKF method for state update
        _imuProcessor->set_R_LI_cov(V3D(VEC_FROM_ARRAY(_config._rotLICov)));
        _imuProcessor->set_T_LI_cov(V3D(VEC_FROM_ARRAY(_config._transLICov)));
        _imuProcessor->imu_en = _config._isImuInitialized;
        if (_config._isImuInitialized) {         // IMU extrinsic exists
            _imuProcessor->LI_init_done = false; // To init imu in ImuProcess::Process
            _imuProcessor->set_mean_acc_norm(_config._meanAccNorm);
            _imuProcessor->set_gyr_cov(V3D(_config._gyrCov, _config._gyrCov, _config._gyrCov));
            _imuProcessor->set_acc_cov(V3D(_config._accCov, _config._accCov, _config._accCov));
            _imuProcessor->set_gyr_bias_cov(V3D(_config._bGyrCov, _config._bGyrCov, _config._bGyrCov));
            _imuProcessor->set_acc_bias_cov(V3D(_config._bAccCov, _config._bAccCov, _config._bAccCov));
            _imuProcessor->set_imu_max_init_count(_config._imuMaxInitCount);
            _isOnlineCalibFinish = true;
            //        if ( LidarProcess::config().lidarType != AVIA)
            //            _cutFrameNum = 2;
            //std::cout << "Extrinsic is set, no need to initialization!" << std::endl;
            //printState("Initialized States");
        } else { // For imu extrinsic calibration
            _initiatorLI = std::make_shared<LI_Init>();
            _initiatorLI->data_accum_length = _config._dataAccumLength;
            _imuProcessor->LI_init_done = false;
            _imuProcessor->set_gyr_cov(V3D(_config._gyrCov, _config._gyrCov, _config._gyrCov));
            _imuProcessor->set_acc_cov(V3D(_config._accCov, _config._accCov, _config._accCov));
            _imuProcessor->set_gyr_bias_cov(V3D(_config._bGyrCov, _config._bGyrCov, _config._bGyrCov));
            _imuProcessor->set_acc_bias_cov(V3D(_config._bAccCov, _config._bAccCov, _config._bAccCov));
            _foutResult.open(RESULT_FILE_DIR("Initialization_result.txt"), ios::out);
        }
    } else if (_config._udpateMethod == 1) { //LSQ method for state update
        if (!_config._isImuInitialized) {
            std::cerr << "Update method 0 doesn't support unknown extrinsic!" << std::endl;
            return false;
        }
        _imuPreintegration = shared_ptr<ImuPreintegration>(new ImuPreintegration());
        _imuPreintegration->mutableConfig()._meanAccNorm = _config._meanAccNorm;
        _imuPreintegration->mutableConfig()._imuGyrNoise = _config._gyrCov;
        _imuPreintegration->mutableConfig()._imuAccNoise = _config._accCov;
        _imuPreintegration->mutableConfig()._imuGyrBiasNoise = _config._bGyrCov;
        _imuPreintegration->mutableConfig()._imuAccBiasNoise = _config._bAccCov;
        _imuPreintegration->mutableConfig()._Ril = _Ril;
        _imuPreintegration->mutableConfig()._til = _til;
        _imuPreintegration->init();
        // _regPCL_VGICP.setMaxCorrespondenceDistance(0.05);
    } else { // State update without imu
        if (!_config._isImuInitialized) {
            std::cerr << "Update method doesn't support unknown extrinsic!" << std::endl;
            return false;
        }
    }

    if (config()._isCutFrame)
        mutableConfig()._cutFrameNum = 1;

    _jacoRot.setZero();
    _G.setZero();
    _H_T_H.setZero();
    _I_STATE.setIdentity();

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);

    if (_config._matchMethod == 0)
        _kf.init_dyn_share(get_f, df_dx, df_dw,
                           std::bind(&System::hShareModelKdTree, this, std::placeholders::_1, std::placeholders::_2),
                           _config._nMaxInterations,
                           epsi);
    else if (_config._matchMethod == 1)
        _kf.init_dyn_share(get_f, df_dx, df_dw,
                           std::bind(&System::hShareModelVoxelMap, this, std::placeholders::_1, std::placeholders::_2),
                           _config._nMaxInterations,
                           epsi);

    auto &state = const_cast<state_ikfom &>(_kf.get_x());
    state.offset_R_L_I = _Ril;
    state.offset_T_L_I = _til;
    _globalGrav = state.grav;

    _lidarProcessor->setDownsampleLeafSize(_config._filterSizeSurf, _config._filterSizeSurf, _config._filterSizeSurf);
    _ikdtree.set_downsample_param(_config._filterSizeMap);

    if (_config._udpateMethod >= 1)
        _voxelMapGaussian.reset(new GaussianVoxelMap<PointType>(System::mutableConfig()._voxelLength, VoxelAccumulationMode::ADDITIVE));

    if (!_config._pcapFilePath.empty()) { // Reading lidar data from pcap file
        double startAngle = 0.;
        //_pandarReader = new PandarGeneral(_config._pcapFilePath, std::bind(&System::lidCallback, this, std::placeholders::_1, std::placeholders::_2), static_cast<int>(startAngle * 100 + 0.5), 0, 0, "PandarXT-16", "PandarXT-16", "", false);
        //if (!_pandarReader)
        //{
        //    std::cout << "Created pander reader failed" << std::endl;
        //    return false;
        // }
        //if (0 != _pandarReader->LoadCorrectionFile(_config._lidarCorrectFilePath))
        // {
        //    std::cout << "Loaded correction file from " << _config._lidarCorrectFilePath << " failed" << std::endl;
        //    return false;
        //}
    }

    ReadWriter::cleanDir(std::string(ROOT_DIR) + "PCD");

    std::cout << "Initialize|Lidar type: " << LidarProcess::config()._lidarType << endl;
    std::cout << "Initialize|Pcap path: " << _config._pcapFilePath << endl;
    std::cout << "Initialize|Imu path: " << _config._imuFilePath << endl;
    std::cout << "Initialize|Raster path: " << _config._rasterFilePath << endl;

    return true;
}

void System::resetSystem() {
    System::mutableConfig()._isMotorInitialized = true;
    System::mutableConfig()._isSaveMap = true;
    _frameId = 0;
    // Stop display thread and clear buffer
    if (_dispThread) {
        _isResetShow = true;
        _dispMutex.lock();

        queue<PointCloudXYZI::Ptr>().swap(_localCloudQueue);
        queue<PointCloudXYZI::Ptr>().swap(_localCloudDownQueue);
        queue<PointCloudXYZI::Ptr>().swap(_downCloudMapQueue);
        queue<Eigen::Matrix3d>().swap(_RQueue);
        queue<Eigen::Vector3d>().swap(_tQueue);
        queue<double>().swap(_motorAngleQueue);
        _localCloudDisp->clear();
        _localCloudDownDisp->clear();
        _downCloudMapDisp->clear();
        _RDisp.setIdentity();
        _tDisp.setZero();
        _motorAngleDisp = 0;
        _dispMutex.unlock();
    }

    _relToList.clear();
    _relTlList.clear();
    _initMotorAngle = -1;

    _lidarBuffer.clear();
    _motorAngleBuf.clear();
    _imuBuffer.clear();
    _timeBuffer.clear();

    _Rwl = Eigen::Matrix3d::Identity();
    _twl.setZero();

    _denseCloudMap->clear();
    _downCloudMap->clear();
    _localCloudPtr->clear();
    _localCloudDownPtr->clear();
    _globalCloudDownPtr->clear();
    _nearestPoints.clear();

    // reset kdtree
    _ikdtree.Build(_globalCloudDownPtr->points); // pw
}

bool System::start() {
    if (!initSystem()) {
        std::cerr << "System initializing failed!" << std::endl;
        return false;
    }

    if (!ReadWriter::parseIMUFile(System::config()._imuFilePath, _imuBufferAll)) {
        std::cerr << "Loading IMU data failed!" << std::endl;
        return false;
    }
    if (!ReadWriter::parseMotorFile(System::config()._rasterFilePath, _motorAngleBufAll)) {
        std::cerr << "Loading motor data failed!" << std::endl;
        return false;
    }

    //_pandarReader->Start();
    return true;
}

void System::stop() {
    //if(_pandarReader)
    //{
    //    _pandarReader->Stop();
    //    if(_pandarReader->isFinished())
    //    {
    //        delete _pandarReader;
    //        _pandarReader = nullptr;
    //    }
    //}

    if (!_loopCloser->isFinished()) {
        _loopCloser->requestFinish();
        while (!_loopCloser->isFinished())
            usleep(5000);
    }
    _isFinished = true;
}

bool System::isFinished() {
    //if(_pandarReader)
    //	if (_pandarReader->isFinished())
    //		stop();

    return _isFinished;
}

CRegistration<PointType> cReg;
pcl::PointCloud<PointType>::Ptr lastSubmapCloud(new pcl::PointCloud<PointType>()), curTransCloud(new PointCloudXYZI);
PointCloudXYZI::Ptr curTransCloudDS(new PointCloudXYZI), lastSubmapCloudDS(new pcl::PointCloud<PointType>());
bool System::processContinualTask(const std::string &lastTaskPath) {
    if (!_loopCloser)
        return false;
    if (_loopCloser->getSubMapBlocks().empty())
        return false;

    printf("续扫拼接计算....\n");

    //    std::cout << "读取bin...\n";
    float lastTaskRefPose[6] = {0}; // x,y,z, roll, pitch,yaw
    bool bReadOk = false;
    bReadOk = readLastTaskInfo(lastTaskPath, lastTaskRefPose, lastSubmapCloud);
    if (!bReadOk)
        return false;

    printf("续扫bin文件读取完成....\n");
    PointCloudXYZI::Ptr globalDownCloud(new PointCloudXYZI());

    PointCloudXYZI::Ptr curSubmapCloud(new PointCloudXYZI);
    pcl::transformPointCloud(*_loopCloser->getSubMapBlocks().front()->_pcDown, *curSubmapCloud, _loopCloser->getSubMapBlocks().front()->_poseLo);
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(Eigen::Affine3d(_loopCloser->getSubMapBlocks().front()->_poseLo).cast<float>(), x, y, z, roll, pitch, yaw);

    Eigen::Affine3f transGuess = pcl::getTransformation(lastTaskRefPose[0] - x, lastTaskRefPose[1] - y, lastTaskRefPose[2] - z, 0, 0, 0);
    pcl::transformPointCloud(*curSubmapCloud, *curTransCloud, transGuess);

    // debug output      /****************/
    pcl::PCDWriter pcdWriter;
    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/curTrans.pcd", *curTransCloud);
    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/lastSubmapCloud.pcd", *lastSubmapCloud);

    voxelFilter(lastSubmapCloud, lastSubmapCloudDS, 0.5);
    voxelFilter(curTransCloud, curTransCloudDS, 0.5);

    std::cout << "coarse align...\n";
    Eigen::Matrix4f coarseTransMatrix = Eigen::Matrix4f::Identity();
    cReg.coarse_reg_SACIA(curTransCloudDS, lastSubmapCloudDS, coarseTransMatrix, 0 /*0.5*/, 30 /*30*/, 80);
    printf("续扫coarse拼接完成....\n");
    //        printf("续扫coarse拼接完成....\n");

    pcl::PointCloud<PointType>::Ptr coarseTransCloud(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*curTransCloud, *coarseTransCloud, coarseTransMatrix);
    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/curTrans1.pcd", *coarseTransCloud);

    std::cout << "fine align...\n";

    //    Eigen::Matrix4f  fineTransMatrix =  Eigen::Matrix4f::Identity();
    //    pcl::PointCloud<PointType>::Ptr finalTransCloud(new pcl::PointCloud<PointType>());
    //    registerByGICP(coarseTransCloud, lastSubmapCloud, fineTransMatrix, 1.5, 10, finalTransCloud);
    //    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/curTrans2.pcd", *finalTransCloud);
    //    Eigen::Matrix4d transCur2LastTask = Eigen::Matrix4f(fineTransMatrix*coarseTransMatrix * transGuess.matrix()).cast<double>();

    fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> fastVGICPReg;
    fastVGICPReg.setResolution(1.0);
    fastVGICPReg.setNumThreads(MP_PROC_NUM);
    pcl::PointCloud<pcl::PointXYZ>::Ptr localCloudXYZPtr2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr localCloudXYZPtr1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*coarseTransCloud, *localCloudXYZPtr1);
    pcl::copyPointCloud(*lastSubmapCloud, *localCloudXYZPtr2);
    fastVGICPReg.setInputSource(localCloudXYZPtr1);
    fastVGICPReg.setInputTarget(localCloudXYZPtr2);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    fastVGICPReg.align(*aligned, Eigen::Matrix4d::Identity());
    Eigen::Matrix4d fineTransMatrix = fastVGICPReg.getFinalTransformation();
    bool isConverged = fastVGICPReg.hasConverged();
    if (!isConverged) {
        std::cerr << "Continual work not converged" << std::endl;
        //return false;
    }
    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/curTrans2.pcd", *aligned);

    Eigen::Matrix4d transCur2LastTask = fineTransMatrix * Eigen::Matrix4f(coarseTransMatrix * transGuess.matrix()).cast<double>();

    PointCloudXYZI::Ptr finalTransCloud(new PointCloudXYZI);
    pcl::transformPointCloud(*curSubmapCloud, *finalTransCloud, transCur2LastTask);
    pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/finalTransCloud.pcd", *finalTransCloud);

    //    for(auto & frameBlock : _loopCloser->getFrameBlocks())
    //    {
    //        pcl::PointCloud<PointType>::Ptr frameCloud(new pcl::PointCloud<PointType>());
    //        pcl::io::loadPCDFile(frameBlock->_pcdFilePath, *frameBlock->_pcRaw);
    //        pcl::transformPointCloud(*frameBlock->_pcRaw, *frameCloud, frameBlock->_poseLo);
    //        pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/frame_ori_"+std::to_string(frameBlock->_uniqueId)+".pcd", *frameCloud);
    //    }

    std::cout << transCur2LastTask << std::endl;
    for (auto &frameBlock : _loopCloser->getFrameBlocks())
        frameBlock->_poseLo = transCur2LastTask * frameBlock->_poseLo;

    printf("续扫拼接完成....\n");
    //printf("续扫拼接完成....\n");
    return true;
}

void System::saveLastTaskInfo(const std::string &lastTaskPath) {
    if (_loopCloser->getSubMapBlocks().empty())
        return;
    if (_loopCloser->getSubMapBlocks().size() < 2)
        return;

    auto lastSubmap = _loopCloser->getSubMapBlocks().back();

    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(Eigen::Affine3d(lastSubmap->_poseLo).cast<float>(), x, y, z, roll, pitch, yaw);
    Eigen::Quaterniond rotAlignQuat = Eigen::Quaterniond(_rotAlign);
    FILE *fout = fopen(lastTaskPath.c_str(), "wb");
    if (fout) {
        float rpyXYZ[6] = {x, y, z, roll, pitch, yaw};
        fwrite(rpyXYZ, sizeof(float), 6, fout);
        double gravAlign[4] = {rotAlignQuat.x(), rotAlignQuat.y(), rotAlignQuat.z(), rotAlignQuat.w()};
        fwrite(gravAlign, sizeof(double), 4, fout);
        unsigned long ptsNum = lastSubmap->_pcDown->size();
        fwrite(&ptsNum, sizeof(unsigned long), 1, fout);

        PointCloudXYZI::Ptr globalDownCloud(new PointCloudXYZI());
        pcl::transformPointCloud(*lastSubmap->_pcDown, *globalDownCloud, lastSubmap->_poseLo);
        std::vector<float> xyziData(ptsNum * 4);
        for (unsigned long j = 0; j < ptsNum; j++) {
            xyziData[j * 4 + 0] = globalDownCloud->points[j].x;
            xyziData[j * 4 + 1] = globalDownCloud->points[j].y;
            xyziData[j * 4 + 2] = globalDownCloud->points[j].z;
            xyziData[j * 4 + 3] = globalDownCloud->points[j].intensity;
        }

        fwrite(&xyziData[0], sizeof(float), ptsNum * 4, fout);
        fclose(fout);
    } else
        std::cerr << "Failed openning file: " << lastTaskPath << std::endl;

    //    pcl::PCDWriter pcdWriter;
    //    for(auto & frameBlock : _loopCloser->getFrameBlocks())
    //    {
    //        pcl::PointCloud<PointType>::Ptr frameCloud(new pcl::PointCloud<PointType>());
    //        pcl::io::loadPCDFile(frameBlock->_pcdFilePath, *frameBlock->_pcRaw);
    //        pcl::transformPointCloud(*frameBlock->_pcRaw, *frameCloud, frameBlock->_poseLo);
    //        pcdWriter.writeBinary(string(ROOT_DIR) + "/PCD/frame_"+std::to_string(frameBlock->_uniqueId)+".pcd", *frameCloud);
    //    }
    std::cout << "Last task saved!" << std::endl;
}

bool System::readLastTaskInfo(const std::string &lastTaskPath, float *refXYZ, pcl::PointCloud<PointType>::Ptr &lastSubmap) {
    FILE *fin = fopen(lastTaskPath.c_str(), "rb");
    if (!fin) {
        std::cout << "error: cannot read last task file!\n";
        return false;
    }

    if (fin) {
        float readXYZ[6] = {0};
        fread(readXYZ, sizeof(float), 6, fin);
        memcpy(refXYZ, readXYZ, sizeof(float) * 6);

        double gravAlignData[4] = {0};
        fread(gravAlignData, sizeof(double), 4, fin);
        Eigen::Quaterniond gravAignQuat;
        gravAignQuat.x() = gravAlignData[0];
        gravAignQuat.y() = gravAlignData[1];
        gravAignQuat.z() = gravAlignData[2];
        gravAignQuat.w() = gravAlignData[3];
        _rotAlign = gravAignQuat.matrix();
        // printf("read ref-XYZ: %f %f %f\n", refXYZ[0], refXYZ[1], refXYZ[2]);

        unsigned long ptsNum = 0;
        fread(&ptsNum, sizeof(unsigned long), 1, fin);
        if (ptsNum == 0) {
            fclose(fin);
            std::cout << "warning: last task submap is empty\n";
            return false;
        }

        // std::cout << "读取到子地图点数: " << ptsNum << std::endl;

        lastSubmap->resize(ptsNum);
        for (int i = 0; i < lastSubmap->size(); i++) {
            fread(&lastSubmap->points[i].x, sizeof(float), 1, fin);
            fread(&lastSubmap->points[i].y, sizeof(float), 1, fin);
            fread(&lastSubmap->points[i].z, sizeof(float), 1, fin);
            fread(&lastSubmap->points[i].intensity, sizeof(float), 1, fin);
        }
        fclose(fin);
        //std::cout << "读取bin完成\n";
    }

    return true;
}