
#include "System.h"
#include "CloudInfo.h"

void System::processCloudOnlyMotor() {
    ////////////////Undistort pcl by imu motions////////////////
    if (_imuProcessor->imu_need_init()) {
        _imuProcessor->Process(_measures, _kf, _localCloudPtr);
        _stateIkfom = _kf.get_x(); //udpate state
        return;
    }

    if (_localCloudPtr->empty() || (_localCloudPtr == nullptr)) {
        _firstLidarTime = _measures.lidar_beg_time;
        _onlineCalibStartTime = _measures.lidar_beg_time;
        std::printf("System is not ready, no points stored.!\n");
        return;
    }
    ////////////////Downsample local cloud////////////////
    //TicToc time2;
    if (_config._isFeatExtractEn) {
        _lidarProcessor->extractFeatures(_localCloudPtr, *_localSurfCloudPtr, *_localCornerCloudPtr);
        _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localSurfCloudPtr, _localSurfCloudDownPtr);
        _cloudCornerDownSize = _lidarProcessor->downSampleLocalCloud(_localCornerCloudPtr, _localCornerCloudDownPtr);
        //std::cout<<"Corner/Surf input size:"<<_localCornerCloudPtr->size()<<"/"<<_localSurfCloudPtr->size()<<std::endl;
        std::cout << "Corner/Surf input down size:" << _cloudCornerDownSize << "/" << _cloudSurfDownSize << std::endl;
    } else {
        _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        _localSurfCloudPtr = _localCloudPtr;
        _localSurfCloudDownPtr = _localCloudDownPtr;
        //std::cout<<"Input down size:"<<_cloudSurfDownSize<<std::endl;
    }

    if (_cloudSurfDownSize < 5 && _cloudCornerDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        return;
    }
    //printf("Downsample cloud time: %f ms\n", time2.toc());

    if (!_isInitMap) {
        _globalGrav = Eigen::Vector3d(_stateIkfom.grav);
        if (_config._enableGravityAlign)
            _rotAlign = Eigen::Quaterniond::FromTwoVectors(_globalGrav, V3D(0, 0, -1)).toRotationMatrix(); //Rw'w

        _Rwl = _stateIkfom.rot.matrix() * _stateIkfom.offset_R_L_I;                   // Rwl=Rwi*Ril
        _twl = _stateIkfom.rot.matrix() * _stateIkfom.offset_T_L_I + _stateIkfom.pos; // twl=Twi*til
        transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
        initKdTreeMap(_globalCloudDownPtr->points);

        if (_config._isEnable3DViewer)
            updateViewer();

        if (_config._isSaveMap) {
            updateDenseMap(_globalCloudDownPtr);
            saveMap();
        }
        //_imuProcessor->clear();
        _isInitMap = true;
        return;
    }

    _Rwl = _stateIkfom.rot.matrix() * _stateIkfom.offset_R_L_I;
    _twl = _stateIkfom.rot.matrix() * _stateIkfom.offset_T_L_I + _stateIkfom.pos;
    transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);

    //printState(_stateIkfom);

    //TicToc time4;
    if (_config._isEnable3DViewer) {
        _addedPoints = _ikdtree.Add_Points(_globalCloudDownPtr->points, true);
        updateViewer();
    }

    //printf("Update viewer time: %f ms\n", time4.toc());
    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }
}

void System::processCloudCeres() {
    ////////////Predict pose and undistort pointcloud/////////
    // predictAndUndistort(_measures, _localCloudPtr);
    _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);

    if (_cloudDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        return;
    }
    transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
    ////////// initialize the map kdtree //////////
    if (_ikdtree.Root_Node == nullptr) {
        initKdTreeMap(_globalCloudDownPtr->points);
        if (_config._isEnable3DViewer)
            updateViewer();
        if (_config._isSaveMap) {
            PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
            transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
            updateDenseMap(curWorldCloudPtr);
        }
        return;
    } else {
        //////////Match and optimize lidar relative pose//////////
        for (int i = 0; i < _config._nMaxInterations; i++) {
            updateViewer();
            // PAUSE;

            // TicToc matchTime;
            matchKdTree(_localCloudDownPtr, _globalCloudDownPtr, _ikdtree, _matchInfos);
            // printf("Match time: %f ms\n\n", matchTime.toc());
            // TicToc optTime;
            //poseOptimize(_matchInfos, 0, _Rwl, _twl, 0.5, i==_config._nMaxInterations-1);
            _regICP.clear();
            for (auto &match : _matchInfos)
                _regICP.push_surf(match._pl, match._center, match._norm, match._coeffs);
            _regICP.so3_pose.setQuaternion(Quaterniond(_Rwl));
            _regICP.t_pose = _twl;
            _regICP.damping_iter();
            _Rwl = _regICP.so3_pose.matrix();
            _twl = _regICP.t_pose;
            //            std::cout<< "Rot:" <<RotMtoEuler(_Rwl).transpose() * 57.3<<std::endl;
            //            std::cout<< "Translation:" <<_twl.transpose() * 57.3<<std::endl;
            //            printf("Optimize time: %f ms\n\n", optTime.toc());

            //            Eigen::Matrix4d Twl;
            //            Twl.block<3,3>(0,0)=_Rwl;
            //            Twl.block<3,1>(0,3)=_twl;
            //            Eigen::Matrix4d T21=Twl.inverse()*_prevTwl; //T2w*Tw1=T21
            //            std::cout << "rel R: " << R2ypr(T21.block<3, 3>(0, 0)).transpose() << std::endl;
            //            std::cout << "rel t: " << T21.block<3,1>(0,3).transpose() << std::endl;
            //            lidarMotionCompensationWithTimestamp(_localCloudDownPtr, T21);
            //            if(lidarMotionCompensationWithTimestamp(_localCloudPtr, T21))
            //                 _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr,_localCloudDownPtr);

            transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
        }
    }
    _isEKFInited = true;
    updateKdTreeMap(_globalCloudDownPtr->points);

    if (_config._isSaveMap) {
        PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }

    if (_config._isEnable3DViewer)
        updateViewer();
    // PAUSE;
}

void System::processCloudLSQ(Registration &reg) {
    ////////////////Undistort pcl by imu motions////////////////
    predictAndUndistort(_measures, _localCloudPtr);

    ////////////////Down sample local cloud////////////////
    _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
    if (_cloudDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        return;
    }

    ////////////Initialize voxel map////////////
    if (!_isInitMap) {
        transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl); // pl->pw

        initVoxelMapWithKdtree(_globalCloudDownPtr->points);
        _isInitMap = true;

        ///////for display/////
        // transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl,_twl);
        // initKdTreeMap(_globalCloudDownPtr->points);
        return;
    }

    ///////////////match and update state with map//////////////////
    /*** Calculate local cloud covariances ***/
    std::vector<PointWithCov> pvList;
    for (size_t i = 0; i < _cloudDownSize; i++) {
        PointWithCov pv;
        // pv.pi << _localCloudDownPtr->points[i].x, _localCloudDownPtr->points[i].y, _localCloudDownPtr->points[i].z;
        pv.pl << _localCloudDownPtr->points[i].x, _localCloudDownPtr->points[i].y, _localCloudDownPtr->points[i].z;
        pvList.push_back(pv);
    }

    calcPointNeighCov(pvList, 20);

    _voxelMapGaussian->clear();
    _ikdtree.flatten(_ikdtree.Root_Node, _ikdtree.PCL_Storage, NOT_RECORD);
    _voxelMapGaussian->updateVoxelMap(_ikdtree.PCL_Storage);

    bool isConverged = false;
    for (int iterCount = 0; iterCount < _config._nMaxInterations; iterCount++) {
        /*** Search correspondences ***/
        transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl); // pl->pw
        for (size_t i = 0; i < _cloudDownSize; i++)
            pvList[i].pw << _globalCloudDownPtr->points[i].x, _globalCloudDownPtr->points[i].y, _globalCloudDownPtr->points[i].z;

        RegistrationICP *regPcl = dynamic_cast<RegistrationICP *>(&reg);
        if (regPcl) {
            matchKdTree(_localCloudDownPtr, _globalCloudDownPtr, _ikdtree, _matchInfos);
            regPcl->clear();
            for (auto &match : _matchInfos)
                regPcl->push_surf(match._pl, match._center, match._norm, match._coeffs);
        } else {
            /*** Get correspondences ***/
            std::vector<GuassinVoxelMatchInfo<PointWithCov>> matchVoxelList;
            _voxelMapGaussian->getCorrespondences(pvList, matchVoxelList, NeighborSearchMethod::DIRECT1);

            /*** Calculate voxel map covariances ***/
            for (auto &match : matchVoxelList)
                match._matchedVoxel->calcCovariances(&_ikdtree, 20);

            RegistrationVGICP *regVGICP = dynamic_cast<RegistrationVGICP *>(&reg);
            regVGICP->setMatchedList(matchVoxelList);
        }

        reg._so3Pose.setQuaternion(Quaterniond(_Rwl));
        reg._tPose = _twl;
        reg.damping_iter();

        if ((_twl - reg._tPose).norm() * 100 < 0.015 && (R2ypr(_Rwl) - R2ypr(reg._so3Pose.matrix())).norm() < 0.01)
            isConverged = true;

        _Rwl = reg._so3Pose.matrix();
        _twl = reg._tPose;

        if (isConverged)
            break;
    }

    transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl); // pl->pw

    //////////////update kdtree and voxel map//////////////
    _ikdtree.Add_Points(_globalCloudDownPtr->points, false);
    // updateVoxelMapWithKdtree(_globalCloudDownPtr->points);

    /////////////////viewer update////////////////
    if (_config._isEnable3DViewer)
        updateViewer();

    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }
    // PAUSE;
}

int System::processCloudLSQ(RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> &reg) {
    ////////////////Undistort pcl by imu motions////////////////
    //predictAndUndistort(_measures, _localCloudPtr);
    if (!_imuProcessor->getIMUPoses().empty()) {
        _imuProcessor->imuMotionCompensation(*_localCloudPtr, _stateCur);
        _imuProcessor->clearIMUPose();
    }

    if (_config._isFeatExtractEn) {
        _lidarProcessor->extractFeatures(_localCloudPtr, *_localSurfCloudPtr, *_localCornerCloudPtr);
        _localCloudPtr = _localSurfCloudPtr;
        std::cout << "Corner/Surf input down size:" << _localCornerCloudDownPtr->size() << "/" << _localSurfCloudDownPtr->size() << std::endl;
    }
    _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);

    ////////////////Downsample local cloud////////////////
    if (_cloudDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        if (!_config._isMotorInitialized && !_relToList.empty())
            _relToList.pop_back();
        return -1;
    }
    // Convert imu pose to lidar pose
    _Rwl = _stateCur.rot_end * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
    _twl = _stateCur.rot_end * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til

    ////////////Initialize voxel map////////////
    if (!_isInitMap) {
        transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl); // pl->pw

        initVoxelMapWithKdtree(_globalCloudDownPtr->points);
        //initKdTreeMap();
        _isInitMap = true;

        if (_imuPreintegration)
            _imuPreintegration->optimizeOdometry(_stateCur, _measures.lidar_end_time);

        if (!_config._isMotorInitialized && !_relToList.empty())
            _relToList.pop_back();

        loopClosing(_stateCur);
        return 0;
    }

    ///////////////match and update state with map//////////////////
    /*** Calculate local cloud covariances ***/
    std::vector<PointWithCov> pvList;
    for (size_t i = 0; i < _cloudDownSize; i++) {
        PointWithCov pv;
        // pv.pi << _localCloudDownPtr->points[i].x, _localCloudDownPtr->points[i].y, _localCloudDownPtr->points[i].z;
        pv.pl << _localCloudDownPtr->points[i].x, _localCloudDownPtr->points[i].y, _localCloudDownPtr->points[i].z;
        pvList.push_back(pv);
    }
    calcPointNeighCov(pvList, 20);

    //    _voxelMapGaussian->clear();
    //    _ikdtree.flatten(_ikdtree.Root_Node, _ikdtree.PCL_Storage, NOT_RECORD);
    //    _voxelMapGaussian->updateVoxelMap(_ikdtree.PCL_Storage);

    Eigen::Matrix4d Twl = Eigen::Matrix4d::Identity();
    Twl.block<3, 3>(0, 0) = _Rwl;
    Twl.block<3, 1>(0, 3) = _twl;
    Eigen::Isometry3d guess = Eigen::Isometry3d(Twl);
    reg.setLMLambda(-1.0);

    TicToc time1;
    std::set<GaussianVoxel::Ptr> updatedVoxelSet;
    bool isConverged = true;
    RegistrationPCL_VGICP<pcl::PointXYZ, pcl::PointXYZ> *regPCL_VGICP = dynamic_cast<RegistrationPCL_VGICP<pcl::PointXYZ, pcl::PointXYZ> *>(&reg);
    for (int iterCount = 0; iterCount < 64; iterCount++) {
        if (regPCL_VGICP) {
            /*** Search correspondences ***/
            transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl); // pl->pw
                                                                             //            if(iterCount==0)
                                                                             //            {
                                                                             //                std::cout<<"Iter: "<<iterCount<<std::endl;
                                                                             //                std::cout<<"Pos: "<<_stateCur.pos_end.transpose()<<std::endl;
                                                                             //                updateViewer();
                                                                             //                PAUSE;
                                                                             //            }

            for (size_t i = 0; i < _cloudDownSize; i++)
                pvList[i].pw << _globalCloudDownPtr->points[i].x, _globalCloudDownPtr->points[i].y, _globalCloudDownPtr->points[i].z;
            std::vector<GuassinVoxelMatchInfo<PointWithCov>> matchVoxelList;
            _voxelMapGaussian->getCorrespondences(pvList, matchVoxelList, NeighborSearchMethod::DIRECT1);

            /*** Calculate voxel map covariances ***/
            for (auto &match : matchVoxelList) {
                // std::cout<<match._pv.pl.transpose()<<std::endl;
                if (!updatedVoxelSet.count(match._matchedVoxel)) {
                    updatedVoxelSet.insert(match._matchedVoxel);
                    match._matchedVoxel->calcCovariances(&_ikdtree, 20);
                }
            }

            regPCL_VGICP->setMatchedList(matchVoxelList);
        }

        Eigen::Isometry3d delta;
        if (!reg.step_optimize(guess, delta)) {
            std::cerr << "lm not converged!!" << std::endl;
            isConverged = false;
            break;
        }

        _Rwl = guess.rotation();
        _twl = guess.translation();
        if (reg.is_converged(delta))
            break;
    }
    if (!isConverged) {
        if (!_config._isMotorInitialized && !_relToList.empty())
            _relToList.pop_back();
        return -1;
    }

    _stateCur.pos_end = -_Rwl * _stateCur.offset_R_L_I.transpose() * _stateCur.offset_T_L_I + _twl; // twi=-Rwl*Rli*til+twl=Twl*tli
    _stateCur.rot_end = _Rwl * _stateCur.offset_R_L_I.transpose();                                  //Rwi=Rwl*Rli
    //printState("After VGICP");
    //updateViewer();

    if (_imuPreintegration) {
        _imuPreintegration->optimizeOdometry(_stateCur, _measures.lidar_end_time);
        _Rwl = _stateCur.rot_end * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
        _twl = _stateCur.rot_end * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til

        // printState("After IMU fusion");
    }

    printf("Registration time: %f ms\n\n", time1.toc());

    loopClosing(_stateCur);

    //////////////update kdtree and voxel map//////////////
    transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl); // pl->pw
    updateVoxelMapWithKdtree(_globalCloudDownPtr->points);
    //_ikdtree.Add_Points(_globalCloudDownPtr->points,true);

    /////////////////viewer update////////////////
    if (_config._isEnable3DViewer)
        updateViewer();

    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }
    //PAUSE;
}

void System::processCloudVGICP() {
    static pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelGrid;
    static pcl::PointCloud<pcl::PointXYZ>::Ptr globalMapDownPtr(new pcl::PointCloud<pcl::PointXYZ>());
    static deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> globalCloudQueue;
    ////////////////Undistort pcl by imu motions////////////////
    if (!_imuProcessor->getIMUPoses().empty()) {
        _imuProcessor->imuMotionCompensation(*_localCloudPtr, _statePropagat);
        _imuProcessor->clearIMUPose();
    }
    ////////////////Down sample local cloud////////////////
    //_cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
    _localCloudDownPtr = _localCloudPtr;
    _cloudDownSize = _localCloudPtr->size();
    if (_cloudDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        return;
    }
    // Convert imu pose to lidar pose
    _Rwl = _stateCur.rot_end * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
    _twl = _stateCur.rot_end * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til
    _Twl.block<3, 3>(0, 0) = _Rwl;
    _Twl.block<3, 1>(0, 3) = _twl;

    if (!_isInitMap) {
        _isInitMap = true;

        if (_imuPreintegration)
            _imuPreintegration->optimizeOdometry(_stateCur, _measures.lidar_end_time);

        transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
        ///////for display/////
        if (_config._isEnable3DViewer) {
            initKdTreeMap(_globalCloudDownPtr->points);
            updateViewer();
        }

        pcl::copyPointCloud(*_globalCloudDownPtr, *_globalCloudXYZPtr);
        *_globalMapPtr += *_globalCloudXYZPtr;

        loopClosing(_stateCur);

        return;
    }

    _fastVGICPReg.setResolution(0.2);
    _fastVGICPReg.setNumThreads(8);
    //_fastVGICPReg.setMaximumIterations(200);
    _fastVGICPReg.clearTarget();
    _fastVGICPReg.clearSource();
    //transCloud(_localCloudDownPtr, _globalCloudDownPtr,_Rwl,_twl);
    updateViewer();
    //PAUSE;

    _localCloudXYZPtr->clear();
    pcl::copyPointCloud(*_localCloudDownPtr, *_localCloudXYZPtr);
    _fastVGICPReg.setInputSource(_localCloudXYZPtr);

    voxelGrid.setLeafSize(0.2f, 0.2f, 0.2f);
    voxelGrid.setInputCloud(_globalMapPtr);
    voxelGrid.filter(*globalMapDownPtr);
    //globalMapDownPtr=_globalMapPtr;
    _fastVGICPReg.setInputTarget(globalMapDownPtr);

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    _fastVGICPReg.align(*aligned, _Twl);
    Eigen::Matrix4d relT = _fastVGICPReg.getFinalTransformation();
    //    pcl::transformPointCloud(*_localCloudXYZPtr, *aligned, _Twl);
    //    Eigen::Matrix4d relT=_Twl;

    //    pcl::transformPointCloud(*localCloudXYZPtr, *globalCloudXYZPtr, relT);
    //    *globalMapPtr+=*globalCloudXYZPtr;
    //    _globalMapPtr->clear();
    //    *_globalMapPtr=*aligned;
    //*_globalMapPtr=*globalMapDownPtr+*aligned;
    *_globalMapPtr += *aligned;

    static pcl::PCDWriter pcdWriter;
    string mapSavPath(string(ROOT_DIR) + "PCD/globalMap.pcd");
    pcdWriter.writeBinary(mapSavPath, *_globalMapPtr);
    string framePath(string(ROOT_DIR) + "PCD/frameCloud.pcd");
    pcdWriter.writeBinary(mapSavPath, *_localCloudXYZPtr);

    //    globalCloudQueue.push_back(aligned);
    //    if(globalCloudQueue.size()>25)
    //        globalCloudQueue.pop_front();
    //    for(auto globalCloud:globalCloudQueue)
    //        *_globalMapPtr+=*globalCloud;

    //Twl=relT*Twl;
    //Twl=Twl*relT;
    _Twl = relT;
    _Rwl = _Twl.block<3, 3>(0, 0);
    _twl = _Twl.block<3, 1>(0, 3);

    _stateCur.pos_end = -_Rwl * _stateCur.offset_R_L_I.transpose() * _stateCur.offset_T_L_I + _twl; // twi=-Rwl*Rli*til+twl=Twl*tli
    _stateCur.rot_end = _Rwl * _stateCur.offset_R_L_I.transpose();                                  //Rwi=Rwl*Rli
    //printState("After VGICP");

    _fastVGICPReg.clear();

    if (_imuPreintegration) {
        _imuPreintegration->optimizeOdometry(_stateCur, _measures.lidar_end_time);
        _Rwl = _stateCur.rot_end * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
        _twl = _stateCur.rot_end * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til         // pl->pw
        printState("After IMU fusion");
    }

    /////////////////viewer update////////////////
    if (_config._isEnable3DViewer) {
        _globalCloudDownPtr->clear();
        transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
        _addedPoints = _ikdtree.Add_Points(_globalCloudDownPtr->points, true);
        updateViewer();
    }

    loopClosing(_stateCur);

    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }
    PAUSE;
}

void System::processCloudLOAM() {
    if (!_imuProcessor->getIMUPoses().empty()) {
        _imuProcessor->imuMotionCompensation(*_localCloudPtr, _stateCur);
        _imuProcessor->clearIMUPose();
        std::cout << "Propgate Pos: " << _stateCur.pos_end.transpose() << std::endl;
        std::cout << "Propgate Rot: " << R2ypr(_stateCur.rot_end.transpose()).transpose() << std::endl;
    }
    _lidarProcessor->extractFeatures(_localCloudPtr, *_localSurfCloudPtr, *_localCornerCloudPtr);
    _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);

    // Convert imu pose to lidar pose
    _Rwl = _stateCur.rot_end * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
    _twl = _stateCur.rot_end * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til
    Eigen::Matrix4d Twl = Eigen::Matrix4d::Identity();
    Twl.block<3, 3>(0, 0) = _Rwl;
    Twl.block<3, 1>(0, 3) = _twl;
    _loamTracker.setInitGuess(Twl);

    //    V3D ypr=R2ypr(_Rwl);
    //    std::printf("Init guess: %f, %f, %f, %f, %f, %f\n",_twl[0], _twl[1], _twl[2], DEG2RAD(ypr[2]), DEG2RAD(ypr[1]), DEG2RAD(ypr[0]));

    // updateViewer();
    // PAUSE;

    const Eigen::Affine3f lastPose = _loamTracker.track(_measures.lidar_end_time, _localCornerCloudPtr, _localSurfCloudPtr);
    _twl = lastPose.translation().cast<double>();
    _Rwl = lastPose.rotation().cast<double>();
    _stateCur.pos_end = -_Rwl * _stateCur.offset_R_L_I.transpose() * _stateCur.offset_T_L_I + _twl; // twi=-Rwl*Rli*til+twl=Twl*tli
    _stateCur.rot_end = _Rwl * _stateCur.offset_R_L_I.transpose();                                  //Rwi=Rwl*Rli
    printState("After LOAM tracking");

    if (_imuPreintegration) {
        _imuPreintegration->optimizeOdometry(_stateCur, _measures.lidar_end_time, _loamTracker.isDegenerate());
        printState("After IMU fusion");
        _Rwl = _stateCur.rot_end * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
        _twl = _stateCur.rot_end * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til         // pl->pw
    }

    /////////////////viewer update////////////////
    if (_config._isEnable3DViewer) {
        transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
        if (!_isInitMap) {
            _isInitMap = true;
            initKdTreeMap(_globalCloudDownPtr->points);
        } else
            _ikdtree.Add_Points(_globalCloudDownPtr->points, true);
        updateViewer();
    }

    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }
    // PAUSE;
}

void System::processCloudESIKF() {
    ////////////////Undistort pcl by imu motions////////////////
    // PAUSE;
    //TicToc time1;
    _imuProcessor->Process(_measures, _stateCur, _localCloudPtr);
    _statePropagat = _stateCur;
    if (_imuProcessor->imu_need_init())
        return;
    //printf("Imu process time: %f ms\n", time1.toc());

    //TicToc time2;
    if (_localCloudPtr->empty() || (_localCloudPtr == nullptr)) {
        _firstLidarTime = _measures.lidar_beg_time;
        _onlineCalibStartTime = _measures.lidar_beg_time;
        std::printf("System is not ready, no points stored.!\n");
        return;
    }

    if (_config._isFeatExtractEn) {
        _lidarProcessor->extractFeatures(_localCloudPtr, *_localSurfCloudPtr, *_localCornerCloudPtr);
        // extractFeature(ringBuff, LidarProcess::config()._nScans, *_localCornerCloudPtr, *_localSurfCloudPtr, _config._featureExtractSegNum);
        _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localSurfCloudPtr, _localSurfCloudDownPtr);
        _cloudCornerDownSize = _lidarProcessor->downSampleLocalCloud(_localCornerCloudPtr, _localCornerCloudDownPtr);
        //std::cout<<"Corner/Surf input size:"<<_localCornerCloudPtr->size()<<"/"<<_localSurfCloudPtr->size()<<std::endl;
        std::cout << "Corner/Surf input down size:" << _localCornerCloudDownPtr->size() << "/" << _localSurfCloudDownPtr->size() << std::endl;
    } else {
        _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        _localSurfCloudPtr = _localCloudPtr;
        _localSurfCloudDownPtr = _localCloudDownPtr;
    }

    ////////////////Downsample local cloud////////////////
    if (_cloudSurfDownSize < 5 && _cloudCornerDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        return;
    }

    _isEKFInited = !((_measures.lidar_beg_time - _firstLidarTime) < INIT_TIME);

    ////////////Initialize voxel map////////////
    if (_isEKFInited && !_isInitMap) {
        _globalGrav = _stateCur.gravity;
        if (_config._enableGravityAlign) {
            _rotAlign = Eigen::Quaterniond::FromTwoVectors(_globalGrav, V3D(0, 0, -1)).toRotationMatrix(); //Rw'w
                                                                                                           //            _stateCur.rot_end = _rotAlign * _stateCur.rot_end;//Rw'i=Tw'w*Rwi=Rw'w*Rwi
                                                                                                           //            _stateCur.pos_end = _rotAlign * _stateCur.pos_end;//tw'i=Tw'w*twi=Rw'w*twi+tw'w
        }
        if (_config._matchMethod == 1)
            initVoxelMap();
        ///////for display/////
        if (_config._matchMethod == 0 || _config._isEnable3DViewer) {
            _Rwl = _stateCur.rot_end.matrix() * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
            _twl = _stateCur.rot_end.matrix() * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til
            transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
            initKdTreeMap(_globalCloudDownPtr->points);
        }

        if (_config._isEnable3DViewer)
            updateViewer();

        loopClosing(_stateCur);
        _isInitMap = true;
        return;
    }

    //printf("Downsample cloud time: %f ms\n", time2.toc());
    ///////////////match and update state with map//////////////////
    //TicToc time3;
    if (_config._matchMethod == 0) {
        laserMapFovSegment();
        if (!updateKFKdTree())
            return;
    } else if (_config._matchMethod == 1)
        if (!updateKFVoxelMap())
            return;
    //printf("Update state time: %f ms\n", time3.toc());
    printState();
    //    std::cout<<"Rwl:"<<R2ypr(_Rwl).transpose()<<std::endl;
    //    std::cout<<"twl:"<<_twl.transpose()<<std::endl;

    ///////Loop closing/////
    loopClosing(_stateCur);

    ////////////////Calibrate lidar to imu and refine state//////////////////
    calibAndRefineLI();

    //TicToc time4;
    transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
    if (_config._matchMethod == 0)
        updateKdTreeMap(_globalCloudDownPtr->points);
    if (_config._matchMethod == 1 && _config._isEnable3DViewer)
        _ikdtree.Add_Points(_globalCloudDownPtr->points, true);

    if (_config._isEnable3DViewer)
        updateViewer();
    //printf("Update viewer time: %f ms\n", time4.toc());

    deleteUnstablePoints();

    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }

    //PAUSE;
}

void System::buildsurfmap(pcl::PointCloud<pcl::PointXYZI>::Ptr &densecloud) {
    for (int i = 0; i < _matchedSurfList.size(); i++) {
        //cv::Mat coloredMat(800, 800, CV_8UC3);
        Eigen::Vector3d center = _matchedSurfList[i].center;
        Eigen::Vector3d norm = _matchedSurfList[i].normal;
        Eigen::Vector3d pl = _matchedSurfList[i].pv.pw;
        center = _statePropIkfom.rot.matrix().inverse() * (center - _statePropIkfom.pos);
        pl = _statePropIkfom.rot.matrix().inverse() * (pl - _statePropIkfom.pos);
        norm = _statePropIkfom.rot.matrix().inverse() * norm;
    }
}

void System::imagecreator_calib() {
    for (int i = 0; i < _mapCloudQueue.size(); i++) {
        *_denseCloudMap += *_mapCloudQueue[i];
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense(new pcl::PointCloud<pcl::PointXYZI>);

    Eigen::Matrix3d Rlc;
    Eigen::Vector3d tlc;
    Rlc << -0.0055755, 0.999835, -0.0173053,
        0.0611648, -0.0169322, -0.997984,
        -0.998112, -0.00662274, -0.0610603;
    tlc << -0.0399568, 0.317649, -0.0235294;
    transCloud2(_denseCloudMap, intensityMapdense, Rlc * _statePropIkfom.rot.matrix().inverse(),
                -Rlc * _statePropIkfom.rot.matrix().inverse() * _statePropIkfom.pos + tlc);
    _intensityImg = _imgProcesser.projectDepth(intensityMapdense);
    _denseCloudMap->clear();
}
void System::imagecreatortest() {
    //_sys->_dispMutex.lock();
    //clock_t start=clock();
    //std::cout<<_sys->_mapCloudQueue.size()<<std::endl;
    for (int i = 0; i < _mapCloudQueue.size(); i++) {
        *_denseCloudMap += *_mapCloudQueue[i];
    }

    //clock_t end = clock();
    //double elapsed1 = double(end - start) / CLOCKS_PER_SEC * 1000;
    //pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMap(new pcl::PointCloud<pcl::PointXYZI>);
    //transCloud2(_denseCloudMap, intensityMap,_stateIkfom.rot.matrix().inverse(),-_stateIkfom.rot.matrix().inverse()*_stateIkfom.pos);

    //_intensityImg=_imgProcesser.projectToXZ(intensityMap,_scanlineIdMap,false);
    //cv::imwrite("/home/zzy/SLAM/my_slam_work/src/Rot_intensity_augm_LIVO/image/image.png",_sys->_intensityImg);
    _denseCloudMap->clear();
    // _sys->_dispMutex.unlock();
    //std::cerr << "1 " << std::endl;
    //return;
}

void System::imagecreatoropt() {
    for (int i = 0; i < _mapCloudQueue.size() - 1; i++) {
        *_denseCloudMap += *_mapCloudQueue[i];
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapsparse(new pcl::PointCloud<pcl::PointXYZI>);
    transCloud2(_denseCloudMap, intensityMapdense, _statePropIkfom.rot.matrix().inverse(), -_statePropIkfom.rot.matrix().inverse() * _statePropIkfom.pos);
    _denseCloudMap->clear();
    intensityMapsparse->width = _mapCloudQueue.back()->size();
    intensityMapsparse->height = 1;
    intensityMapsparse->resize(_mapCloudQueue.back()->size());
    for (int i = 0; i < _mapCloudQueue.back()->size(); i++) {
        intensityMapsparse->points[i].x = _mapCloudQueue.back()->points[i].x;
        intensityMapsparse->points[i].y = _mapCloudQueue.back()->points[i].y;
        intensityMapsparse->points[i].z = _mapCloudQueue.back()->points[i].z;
        intensityMapsparse->points[i].intensity = _mapCloudQueue.back()->points[i].intensity;
    }
    //std::this_thread::sleep_for(std::chrono::seconds(2));
    auto data_process_start = std::chrono::high_resolution_clock::now();
    // transCloud3(intensityMapdense, intensityMapdense_left, _imgProcesser.Rleft, Eigen::Vector3d(0,0,0));
    // transCloud3(intensityMapdense, intensityMapdense_right, _imgProcesser.Rright, Eigen::Vector3d(0,0,0));
    // _intensityImg = _imgProcesser.projectPinhole(intensityMapdense, false);
    // _intensityImg_left = _imgProcesser.projectPinhole(intensityMapdense_left, false);
    // _intensityImg_right = _imgProcesser.projectPinhole(intensityMapdense_right, false);
    auto [img, img_left, img_right] = _imgProcesser.projectPinholeall(
        intensityMapdense,
        _imgProcesser.Rleft,
        _imgProcesser.Rright,
        false);
    _intensityImg = img;
    _intensityImg_left = img_left;
    _intensityImg_right = img_right;

    auto data_process_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> data_process_time = data_process_end - data_process_start;
    std::cout << "project time: " << data_process_time.count() << " ms" << std::endl;
    _imgProcesser.inputframedata(intensityMapsparse, _intensityImg, _intensityImg_left, _intensityImg_right);

    if (_mapCloudQueue.size() == 72) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

void System::processCloudIKFoM() {
    ////////////////Undistort pcl by imu motions////////////////
    //TicToc time1;
    _imuProcessor->Process(_measures, _kf, _localCloudPtr);
    _stateIkfom = _kf.get_x(); //udpate state
    _statePropIkfom = _stateIkfom;
    _Rwlprop = _statePropIkfom.rot.matrix() * _statePropIkfom.offset_R_L_I;
    _twlprop = _statePropIkfom.rot.matrix() * _statePropIkfom.offset_T_L_I + _stateIkfom.pos;
    if (_mapCloudQueue.size() > 70) {
        _mapCloudQueue.push_back(_localCloudPtr);
        imagecreatoropt();
    }

    if (_imuProcessor->imu_need_init())
        return;

    if (_localCloudPtr->empty() || (_localCloudPtr == nullptr)) {
        _firstLidarTime = _measures.lidar_beg_time;
        _onlineCalibStartTime = _measures.lidar_beg_time;
        //std::printf("System is not ready, no points stored.!\n");
        return;
    }

    pcl::io::savePCDFileBinary(string(ROOT_DIR) + "PCD/" + std::to_string(_frameId) + ".pcd", *_localCloudPtr);
    //_fTimeOfs<<time1.toc()<<" ";
    //printf("Imu process time: %f ms\n", time1.toc());

    ////////////////Downsample local cloud////////////////
    //TicToc time2;
    if (_config._isFeatExtractEn) {
        _lidarProcessor->extractFeatures(_localCloudPtr, *_localSurfCloudPtr, *_localCornerCloudPtr);
        _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localSurfCloudPtr, _localSurfCloudDownPtr);
        _cloudCornerDownSize = _lidarProcessor->downSampleLocalCloud(_localCornerCloudPtr, _localCornerCloudDownPtr);
        //std::cout<<"Corner/Surf input size:"<<_localCornerCloudPtr->size()<<"/"<<_localSurfCloudPtr->size()<<std::endl;
        std::cout << "Corner/Surf input down size:" << _cloudCornerDownSize << "/" << _cloudSurfDownSize << std::endl;
    } else {
        if (LidarProcess::config()._nScans == 32) {
            PointCloudXYZI::Ptr filterCloudPtr(new PointCloudXYZI());
            _lidarProcessor->filterCloud(_localCloudPtr, filterCloudPtr, 3);
            //std::cout<<filterCloudPtr->size()<<std::endl;
            _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(filterCloudPtr, _localCloudDownPtr);
        } else if (LidarProcess::config()._nScans == 16)
            _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        else {
            std::cerr << "Wrong scanlines" << std::endl;
            return;
        }
        _localSurfCloudPtr = _localCloudPtr;
        _localSurfCloudDownPtr = _localCloudDownPtr;
        //std::cout<<"Input down size:"<<_cloudSurfDownSize<<std::endl;
    }

    if (_cloudSurfDownSize < 5 && _cloudCornerDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        return;
    }
    //_fTimeOfs<<time2.toc()<<" ";
    //printf("Downsample cloud time: %f ms\n", time2.toc());

    ////////////Initialize voxel map////////////
    _isEKFInited = !((_measures.lidar_beg_time - _firstLidarTime) < INIT_TIME);

    if (!_isInitMap) {
        _globalGrav = Eigen::Vector3d(_stateIkfom.grav.get_vect());
        _rotAlign_traj = Eigen::Quaterniond::FromTwoVectors(_globalGrav, V3D(0, 0, -1)).toRotationMatrix();
        if (_config._enableGravityAlign) {
            _rotAlign = Eigen::Quaterniond::FromTwoVectors(_globalGrav, V3D(0, 0, -1)).toRotationMatrix(); //Rw'w
            //_stateIkfom.rot = _rotAlign * _stateIkfom.rot;//Rw'i=Tw'w*Rwi=Rw'w*Rwi
            //_stateIkfom.pos = _rotAlign * _stateIkfom.pos;//tw'i=Tw'w*twi=Rw'w*twi+tw'w
        }

        if (_config._matchMethod == 1)
            initVoxelMap();

        if (_config._matchMethod == 0 || _config._covType == 2 || _config._isEnable3DViewer) {
            _Rwl = _stateIkfom.rot.matrix() * _stateIkfom.offset_R_L_I;                   // Rwl=Rwi*Ril
            _twl = _stateIkfom.rot.matrix() * _stateIkfom.offset_T_L_I + _stateIkfom.pos; // twl=Twi*til
            transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
            initKdTreeMap(_globalCloudDownPtr->points);
        }

        if (_config._isEnable3DViewer)
            updateViewer();

        if (_config._isSaveMap) {
            updateDenseMap(_globalCloudDownPtr);
            saveMap();
        }
        loopClosing(_stateIkfom);
        _isInitMap = true;

        //printState(_stateIkfom);
        //PAUSE;

        return;
    }

    ///////////////match and update state with map//////////////////
    //TicToc time3;
    if (_config._matchMethod == 0) //Kdtree
    {
        laserMapFovSegment();
        _nearestPoints.resize(_cloudSurfDownSize);
        _globalSurfCloudDownPtr->resize(_cloudSurfDownSize);
    } else if (_config._matchMethod == 1) //VoxelMap
    {
        sort(_localSurfCloudDownPtr->points.begin(), _localSurfCloudDownPtr->points.end(), timeComprator);

        _curSurfPvList.clear();
        for (size_t i = 0; i < _localSurfCloudDownPtr->size(); i++) {
            PointWithCov pv;
            pv.featType = Plane;
            pv.pl << _localSurfCloudDownPtr->points[i].x, _localSurfCloudDownPtr->points[i].y, _localSurfCloudDownPtr->points[i].z;
            _curSurfPvList.push_back(pv);
        }
        /*** init cov calculation***/
        if (_config._covType == 2)
            calcPointNeighCov(_curSurfPvList, 20);

        _voxelCovUpdated.clear();
    }

    double solveTime = 0;

    //update imageprocesser
    if (_imgProcesser._matchimgbuffer.size() == 1 && _imgProcesser._matchimgbuffer_right.size() == 1
        && _imgProcesser._matchimgbuffer_left.size() == 1) {
        _matchImg = _imgProcesser._matchimgbuffer.back();
        _matchImg_left = _imgProcesser._matchimgbuffer_left.back();
        _matchImg_right = _imgProcesser._matchimgbuffer_right.back();
        _matchlinelist = _imgProcesser._matchlinelistbuffer.back();
        _matchlinelist_left = _imgProcesser._matchlinelistbuffer_left.back();
        _matchlinelist_right = _imgProcesser._matchlinelistbuffer_right.back();
        _sparselinecloud = _imgProcesser._sparselinecloudbuffer.back();
        _linelist = _imgProcesser._linelistbuffer.back();
        _linelist_left = _imgProcesser._linelistbuffer_left.back();
        _linelist_right = _imgProcesser._linelistbuffer_right.back();
    }
    _matchlinecloud->clear();

    for (int i = 0; i < _matchlinelist.size(); i++) {
        _matchlinecloud->points.push_back(_matchlinelist[i].p3d);
    }

    for (int i = 0; i < _matchlinelist_left.size(); i++) {
        _matchlinecloud->points.push_back(_matchlinelist_left[i].p3d);
    }
    for (int i = 0; i < _matchlinelist_right.size(); i++) {
        _matchlinecloud->points.push_back(_matchlinelist_right[i].p3d);
    }

    _matchlinecloud->width = _matchlinecloud->size();
    _matchlinecloud->height = 1;

    _kf.update_iterated_dyn_share_modified(0.001, solveTime);
    _stateIkfom = _kf.get_x();

    _Rwl = _stateIkfom.rot.matrix() * _stateIkfom.offset_R_L_I;
    _twl = _stateIkfom.rot.matrix() * _stateIkfom.offset_T_L_I + _stateIkfom.pos;

    loopClosing(_stateIkfom);
    //printState(_stateIkfom);
    Eigen::Vector3d alignedPos = _twl;
    if (_config._enableGravityAlign) {
        alignedPos = _rotAlign_traj * _twl;
    }
    // Convert rotation matrix to quaternion
    Eigen::Quaterniond quat(_Rwl);
    quat.normalize();

    // Write to file with timestamp, position (x, y, z), and orientation (qx, qy, qz, qw)
    fout_traj << std::fixed << std::setprecision(5)
              << _measures.lidar_end_time << " "
              << alignedPos(0) << " " << alignedPos(1) << " " << alignedPos(2) << " "
              << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;

    ///////Update map/////
    //TicToc time4;

    //std::cout<<_stateIkfom.pos<<std::endl;

    transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
    PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
    transCloud(_localCloudDownPtr, curWorldCloudPtr, _Rwl, _twl);
    _matchworldlinecloudrgb = convertPointCloudWithLines(_matchlinecloud);
    _matchlinecloud->clear();

    //transCloud(_localCloudDownPtr, curWorldPropCloudPtr, _Rwlprop, _twlprop);

    if (_config._matchMethod == 0) //KdTree
        updateKdTreeMap(_globalCloudDownPtr->points);
    if (_config._matchMethod == 1) //Voxelmap
    {
        // if(_config._covType==2)
        //     _ikdtree.Add_Points(_globalCloudDownPtr->points,true);
        updateVoxelMap();
    }
    std::cout << "Voxel count: " << _voxelSurfMap.size() << std::endl;
    //_fTimeOfs<<time4.toc() << " ";
    //TicToc time5;
    if (_config._isEnable3DViewer) {
        if (_config._matchMethod == 1 && _config._covType == 1) //Voxelmap && obs cov
            _addedPoints = _ikdtree.Add_Points(_globalCloudDownPtr->points, true);
        updateViewer();
    }

    deleteUnstablePoints();

    if (_mapCloudQueue.size() <= 70) {
        _mapCloudQueue.push_back(curWorldCloudPtr);
        imagecreatortest();
    }
    if (_mapCloudQueue.size() > 70) {
        _mapCloudQueue[_mapCloudQueue.size() - 1] = curWorldCloudPtr;
    }

    //imagecreatortest();
    //_fTimeOfs << time5.toc();
    //printf("Update viewer time: %f ms\n", time5.toc());
    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        //PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        //transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }
    //PAUSE;
    //_fTimeOfs<<std::endl;
}

void System::processCloudIKFoM_calib() {
    ////////////////Undistort pcl by imu motions////////////////
    //TicToc time1;
    _imuProcessor->Process(_measures, _kf, _localCloudPtr);
    _stateIkfom = _kf.get_x(); //udpate state
    _statePropIkfom = _stateIkfom;
    _Rwlprop = _statePropIkfom.rot.matrix() * _statePropIkfom.offset_R_L_I;
    _twlprop = _statePropIkfom.rot.matrix() * _statePropIkfom.offset_T_L_I + _stateIkfom.pos;
    // if(_mapCloudQueue.size()>70){
    //     _mapCloudQueue.push_back(_localCloudPtr);
    //     imagecreatoropt();
    // }
    if (_imuProcessor->imu_need_init())
        return;

    if (_localCloudPtr->empty() || (_localCloudPtr == nullptr)) {
        _firstLidarTime = _measures.lidar_beg_time;
        _onlineCalibStartTime = _measures.lidar_beg_time;
        //std::printf("System is not ready, no points stored.!\n");
        return;
    }
    //_fTimeOfs<<time1.toc()<<" ";
    //printf("Imu process time: %f ms\n", time1.toc());

    ////////////////Downsample local cloud////////////////
    //TicToc time2;
    if (_config._isFeatExtractEn) {
        _lidarProcessor->extractFeatures(_localCloudPtr, *_localSurfCloudPtr, *_localCornerCloudPtr);
        _cloudDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localSurfCloudPtr, _localSurfCloudDownPtr);
        _cloudCornerDownSize = _lidarProcessor->downSampleLocalCloud(_localCornerCloudPtr, _localCornerCloudDownPtr);
        //std::cout<<"Corner/Surf input size:"<<_localCornerCloudPtr->size()<<"/"<<_localSurfCloudPtr->size()<<std::endl;
        std::cout << "Corner/Surf input down size:" << _cloudCornerDownSize << "/" << _cloudSurfDownSize << std::endl;
    } else {
        if (LidarProcess::config()._nScans == 32) {
            PointCloudXYZI::Ptr filterCloudPtr(new PointCloudXYZI());
            _lidarProcessor->filterCloud(_localCloudPtr, filterCloudPtr, 3);
            //std::cout<<filterCloudPtr->size()<<std::endl;
            _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(filterCloudPtr, _localCloudDownPtr);
        } else if (LidarProcess::config()._nScans == 16)
            _cloudSurfDownSize = _lidarProcessor->downSampleLocalCloud(_localCloudPtr, _localCloudDownPtr);
        else {
            std::cerr << "Wrong scanlines" << std::endl;
            return;
        }
        _localSurfCloudPtr = _localCloudPtr;
        _localSurfCloudDownPtr = _localCloudDownPtr;
        //std::cout<<"Input down size:"<<_cloudSurfDownSize<<std::endl;
    }

    if (_cloudSurfDownSize < 5 && _cloudCornerDownSize < 5) {
        std::printf("No point, skip this scan!\n");
        return;
    }
    //_fTimeOfs<<time2.toc()<<" ";
    //printf("Downsample cloud time: %f ms\n", time2.toc());

    ////////////Initialize voxel map////////////
    _isEKFInited = !((_measures.lidar_beg_time - _firstLidarTime) < INIT_TIME);

    if (!_isInitMap) {
        _globalGrav = Eigen::Vector3d(_stateIkfom.grav.get_vect());
        _rotAlign_traj = Eigen::Quaterniond::FromTwoVectors(_globalGrav, V3D(0, 0, -1)).toRotationMatrix();
        if (_config._enableGravityAlign) {
            _rotAlign = Eigen::Quaterniond::FromTwoVectors(_globalGrav, V3D(0, 0, -1)).toRotationMatrix(); //Rw'w
            //_stateIkfom.rot = _rotAlign * _stateIkfom.rot;//Rw'i=Tw'w*Rwi=Rw'w*Rwi
            //_stateIkfom.pos = _rotAlign * _stateIkfom.pos;//tw'i=Tw'w*twi=Rw'w*twi+tw'w
        }

        if (_config._matchMethod == 1)
            initVoxelMap();

        if (_config._matchMethod == 0 || _config._covType == 2 || _config._isEnable3DViewer) {
            _Rwl = _stateIkfom.rot.matrix() * _stateIkfom.offset_R_L_I;                   // Rwl=Rwi*Ril
            _twl = _stateIkfom.rot.matrix() * _stateIkfom.offset_T_L_I + _stateIkfom.pos; // twl=Twi*til
            transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
            initKdTreeMap(_globalCloudDownPtr->points);
        }

        if (_config._isEnable3DViewer)
            updateViewer();

        if (_config._isSaveMap) {
            updateDenseMap(_globalCloudDownPtr);
            saveMap();
        }
        loopClosing(_stateIkfom);
        _isInitMap = true;

        //printState(_stateIkfom);
        //PAUSE;

        return;
    }

    ///////////////match and update state with map//////////////////
    //TicToc time3;
    if (_config._matchMethod == 0) //Kdtree
    {
        laserMapFovSegment();
        _nearestPoints.resize(_cloudSurfDownSize);
        _globalSurfCloudDownPtr->resize(_cloudSurfDownSize);
    } else if (_config._matchMethod == 1) //VoxelMap
    {
        sort(_localSurfCloudDownPtr->points.begin(), _localSurfCloudDownPtr->points.end(), timeComprator);

        _curSurfPvList.clear();
        for (size_t i = 0; i < _localSurfCloudDownPtr->size(); i++) {
            PointWithCov pv;
            pv.featType = Plane;
            pv.pl << _localSurfCloudDownPtr->points[i].x, _localSurfCloudDownPtr->points[i].y, _localSurfCloudDownPtr->points[i].z;
            _curSurfPvList.push_back(pv);
        }
        /*** init cov calculation***/
        if (_config._covType == 2)
            calcPointNeighCov(_curSurfPvList, 20);

        _voxelCovUpdated.clear();
    }

    double solveTime = 0;

    //update imageprocesser

    _kf.update_iterated_dyn_share_modified(0.001, solveTime);
    _stateIkfom = _kf.get_x();

    _Rwl = _stateIkfom.rot.matrix() * _stateIkfom.offset_R_L_I;
    _twl = _stateIkfom.rot.matrix() * _stateIkfom.offset_T_L_I + _stateIkfom.pos;
    //_fTimeOfs<<time3.toc()<<" ";
    //printf("Update state time: %f ms\n", time3.toc());
    //printState(_stateIkfom);
    ///////Loop closing/////
    loopClosing(_stateIkfom);
    //printState(_stateIkfom);

    ///////Update map/////
    //TicToc time4;

    //std::cout<<_stateIkfom.pos<<std::endl;

    transCloud(_localCloudDownPtr, _globalCloudDownPtr, _Rwl, _twl);
    PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
    transCloud(_localCloudDownPtr, curWorldCloudPtr, _Rwl, _twl);
    //transCloud3(_matchlinecloud, _matchworldlinecloud, _Rwl, _twl);
    //transCloud(_localCloudDownPtr, curWorldPropCloudPtr, _Rwlprop, _twlprop);

    if (_config._matchMethod == 0) //KdTree
        updateKdTreeMap(_globalCloudDownPtr->points);
    if (_config._matchMethod == 1) //Voxelmap
    {
        // if(_config._covType==2)
        //     _ikdtree.Add_Points(_globalCloudDownPtr->points,true);
        updateVoxelMap();
    }
    std::cout << "Voxel count: " << _voxelSurfMap.size() << std::endl;
    //_fTimeOfs<<time4.toc() << " ";
    //TicToc time5;
    if (_config._isEnable3DViewer) {
        if (_config._matchMethod == 1 && _config._covType == 1) //Voxelmap && obs cov
            _addedPoints = _ikdtree.Add_Points(_globalCloudDownPtr->points, true);
        updateViewer();
    }

    deleteUnstablePoints();

    _mapCloudQueue.push_back(curWorldCloudPtr);
    if (_mapCloudQueue.size() > 150) {
        imagecreator_calib();
    }
    _matchImg = _intensityImg;
    // if(_mapCloudQueue.size()>70){
    //     _mapCloudQueue[_mapCloudQueue.size()-1]=curWorldCloudPtr;
    // }

    //imagecreatortest();
    //_fTimeOfs << time5.toc();
    //printf("Update viewer time: %f ms\n", time5.toc());
    /////////////dense map update and save /////////////
    if (_config._isSaveMap) {
        //PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
        //transCloud(_localCloudPtr, curWorldCloudPtr, _Rwl, _twl);
        updateDenseMap(curWorldCloudPtr);
        saveMap();
    }
    //PAUSE;
    //_fTimeOfs<<std::endl;
}

void System::hShareModelKdTree(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    //    _Rwl = s.rot * s.offset_R_L_I;//Rwl=Rwi*Ril
    //    _twl = s.rot * s.offset_T_L_I + s.pos;//twl=Twi*til
    //    transCloud(_localSurfCloudDownPtr, _globalSurfCloudDownPtr, _Rwl, _twl);//pl->pw
    //    updateViewer(true);
    //    printState(s);
    //    PAUSE;

    PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
    PointCloudXYZI::Ptr corrNormVect(new PointCloudXYZI(100000, 1));
    double totalResidual = 0.0;
    int effctFeatNum = 0;
    //std::cout<<std::setprecision(16)<<std::fixed<<s.rot<<std::endl;
    /** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < _cloudSurfDownSize; i++) {
        PointType &point_body = _localSurfCloudDownPtr->points[i];   // pl
        PointType &point_world = _globalSurfCloudDownPtr->points[i]; // pw

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);                              // pl
        V3D p_global(s.rot.matrix() * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // pw=Twi*Til*pl
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &pointsNear = _nearestPoints[i];
        if (ekfom_data.converge) {
            /** Find the closest surfaces in the map **/
            _ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, pointsNear, pointSearchSqDis);
            _isPointSelectedSurf[i] = pointsNear.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!_isPointSelectedSurf[i]) continue;

        VF(4)
        pabcd;
        _isPointSelectedSurf[i] = false;
        if (estiPlane(pabcd, pointsNear, 0.1f)) {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9) {
                _isPointSelectedSurf[i] = true;
                _normvec->points[i].x = pabcd(0);
                _normvec->points[i].y = pabcd(1);
                _normvec->points[i].z = pabcd(2);
                _normvec->points[i].intensity = pd2;
                //  std::cout<<std::setprecision(16)<<std::fixed<<pd2<<std::endl;
                _resLast[i] = abs(pd2);
            }
        }
    }
    effctFeatNum = 0;

    for (int i = 0; i < _cloudSurfDownSize; i++) {
        if (_isPointSelectedSurf[i]) {
            laserCloudOri->points[effctFeatNum] = _localSurfCloudDownPtr->points[i];
            corrNormVect->points[effctFeatNum] = _normvec->points[i];
            totalResidual += _resLast[i];
            effctFeatNum++;
        }
    }

    if (effctFeatNum < 1) {
        ekfom_data.valid = false;
        std::printf("No Effective Points! \n");
        return;
    }

    double res_mean_last = totalResidual / effctFeatNum;
    //    std::cout<<std::setprecision(12)<<std::fixed<<"matched size:"<<effctFeatNum<<std::endl;
    //    std::cout<<"mean residual:"<<res_mean_last<<std::endl;

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effctFeatNum, 12); // 23
    ekfom_data.h.resize(effctFeatNum);
    ekfom_data.R = MatrixXd::Zero(effctFeatNum, 1);
    for (int i = 0; i < effctFeatNum; i++) {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z); // pl
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; // pi=Til*pl
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corrNormVect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (_config._isEstiExtrinsic) {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        } else
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
        ekfom_data.R(i) = 0.001;
    }
    //    std::cout<<std::setprecision(16)<<std::fixed<<ekfom_data.h_x<<std::endl;
}

void System::hShareModelVoxelMap(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    double totalResidual = 0.0;
    //std::cout<<s.offset_T_L_I<<std::endl;
    _Rwl = s.rot.matrix() * s.offset_R_L_I;         //Rwl=Rwi*Ril
    _twl = s.rot.matrix() * s.offset_T_L_I + s.pos; //twl=Twi*til
    //transCloud(_localSurfCloudDownPtr, _globalSurfCloudDownPtr, _Rwl, _twl);//pl->pw
    // updateViewer(true);
    // printState(s);
    // PAUSE;

    for (size_t i = 0; i < _curSurfPvList.size(); i++) {
        _curSurfPvList[i].pi = s.offset_R_L_I * _curSurfPvList[i].pl + s.offset_T_L_I;
        if (_curSurfPvList[i].pi[2] == 0)
            _curSurfPvList[i].pi[2] = 0.001;
        _curSurfPvList[i].pw = s.rot.matrix() * _curSurfPvList[i].pi + s.pos;
        if (_config._covType == 1) {
            M3D pointCrossmat;
            pointCrossmat << SKEW_SYM_MATRX(_curSurfPvList[i].pl);
            calcBodyCov(_curSurfPvList[i], _config._rangingCov, _config._angleCov);
            _curSurfPvList[i].obsCov = _Rwl * _curSurfPvList[i].bodyCov * _Rwl.transpose() + _Rwl * pointCrossmat * _kf.get_P().block<3, 3>(3, 3) * pointCrossmat.transpose() * _Rwl.transpose() + _kf.get_P().block<3, 3>(0, 0);
        } else if (_config._covType == 2)
            _curSurfPvList[i].obsCov = _curSurfPvList[i].neighCov.block<3, 3>(0, 0);
    }

    std::vector<V3D> nonMatchList;
    bool isStrict = true;
    if (_config._covType == 0 || _config._covType == 2 || _isLoopCorrected) {
        isStrict = false;
        _isLoopCorrected = false;
    }

    //		TicToc time1;
    //std::cout<<_config._radius_k<<std::endl;
    buildResidualListOmp(_voxelSurfMap, _config._voxelLength, 3.0, _config._maxLayers, _curSurfPvList, _matchedSurfList, nonMatchList, isStrict, _config._radius_k);
    //		printf("buildResidualListOmp time: %f ms\n", time1.toc());
    _Rl2l = _Rwlprop.conjugate() * _Rwl;
    _tl2l = _Rwlprop.conjugate() * _twl - _Rwlprop.conjugate() * _twlprop;
    if (_config._isUseIntensity) {
        transCloud3(_sparselinecloud, _curlinecloud, _Rl2l, _tl2l);
        _imgProcesser.buildmatchlinelist(_matchlinelist, _curlinecloud, _linelist);
    }
    const int matchSurfSize = _matchedSurfList.size();
    int matchlinesize = _matchlinelist.size();
    if (matchSurfSize < 1) {
        ekfom_data.valid = false;
        std::printf("No Effective Points! \n");
        return;
    }

    //if(assertDegeneracy())
    //{
    //    std::cerr<<"Environment degenerate!"<<std::endl;
    //            std::cout<<"Iter:"<<iterCount<<std::endl;
    //            _Rwl = _stateCur.rot_end.matrix() * _stateCur.offset_R_L_I;//Rwl=Rwi*Ril
    //            _twl = _stateCur.rot_end.matrix() * _stateCur.offset_T_L_I + _stateCur.pos_end;//twl=Twi*til
    // updateViewer();
    // PAUSE;
    //  }
    if (_config._isUseIntensity) {
        ekfom_data.h_x = MatrixXd::Zero(matchSurfSize + matchlinesize, 12); // 23
        ekfom_data.h.resize(matchSurfSize + matchlinesize);
        ekfom_data.R = MatrixXd::Zero(matchSurfSize + matchlinesize, 1);
    }
    if (!_config._isUseIntensity) {
        ekfom_data.h_x = MatrixXd::Zero(matchSurfSize, 12); // 23
        ekfom_data.h.resize(matchSurfSize);
        ekfom_data.R = MatrixXd::Zero(matchSurfSize, 1);
        matchlinesize = 0;
    }

    static int frameIdForTmpFix = 0;
    frameIdForTmpFix++;
    for (int i = 0; i < matchSurfSize + matchlinesize; i++) {
        /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = _matchedSurfList[i].pv.pw - _matchedSurfList[i].center;
        J_nq.block<1, 3>(0, 3) = -_matchedSurfList[i].normal;
        double dist = _matchedSurfList[i].normal.dot(_matchedSurfList[i].pv.pw) + _matchedSurfList[i].d;
        V3D normVec(_matchedSurfList[i].normal);

        if (_config._covType == 0)
            _matchedSurfList[i].R = 0.001;
        else if (_config._covType == 1) {
            double sigma_l = J_nq * _matchedSurfList[i].plane_cov * J_nq.transpose();
            _matchedSurfList[i].R = (sigma_l + normVec.transpose() * _Rwl * _matchedSurfList[i].pv.bodyCov * _Rwl.transpose() * normVec);
            //_matchedSurfList[i].R=0.001;
            //std::cout<<_matchedSurfList[i].R<<std::endl;
        } else if (_config._covType == 2) {
            // Update voxel plane cov by kdtree
            OctoTree *voxel = _matchedSurfList[i].voxel_correspondence;
            PlaneParams plane = *voxel->_planePtr;
            if (!_voxelCovUpdated.count(voxel)) {
                if (!voxel->_tempPoints.empty() && (voxel->_cloudCov.isZero() || voxel->_tempPoints.size() < 100)) {
                    std::vector<PointWithCov> downVoxelPoints;
                    downSamplingVoxel(voxel->_tempPoints, downVoxelPoints, 0.01);
                    if (downVoxelPoints.size() < voxel->_minFeatUpdateThreshold)
                        downVoxelPoints = voxel->_tempPoints;

                    voxel->calcNeighCov(downVoxelPoints, &_ikdtree); // Calculate voxel point neighbor cov by kdtree
                    for (size_t i = 0; i < downVoxelPoints.size(); i++)
                        downVoxelPoints[i].obsCov = downVoxelPoints[i].neighCov.block<3, 3>(0, 0);
                    //voxel->updatePlaneCov(downVoxelPoints, voxel->_planePtr);//update voxel plane cov by points' neighbor cov
                    voxel->initPlane(downVoxelPoints, &plane); //update voxel plane cov by points' neighbor cov
                                                               //                        voxel->calcNeighCov(voxel->_tempPoints, &_ikdtree);
                                                               //                        for (size_t i = 0; i < voxel->_tempPoints.size(); i++)
                                                               //                            voxel->_tempPoints[i].obsCov = voxel->_tempPoints[i].neighCov.block<3, 3>(0, 0);
                                                               //                        voxel->initPlane(voxel->_tempPoints, voxel->_planePtr);
                }
                _voxelCovUpdated.insert(voxel);
            }
            _Rwl = s.rot.matrix() * s.offset_R_L_I;         //Rwl=Rwi*Ril
            _twl = s.rot.matrix() * s.offset_T_L_I + s.pos; //twl=Twi*til
            double sigma_l = J_nq * plane.planeCov * J_nq.transpose();
            _matchedSurfList[i].R = sigma_l + normVec.transpose() * _Rwl * _matchedSurfList[i].pv.neighCov.block<3, 3>(0, 0) * _Rwl.transpose() * normVec;
        }

        //            _matchedSurfList[i].R=0.001;
        //R_inv(i) = 1.0 / _matchedSurfList[i].R;

        // std::cout<< R_inv(i)<<std::endl;

        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(_matchedSurfList[i].pv.pl);
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(_matchedSurfList[i].pv.pi);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * normVec);
        V3D A(point_crossmat * C);
        if (_config._isEstiExtrinsic && _config._isImuInitialized) {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*normVec);
            ekfom_data.h_x.block<1, 12>(i, 0) << VEC_FROM_ARRAY(normVec), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        } else
            ekfom_data.h_x.block<1, 12>(i, 0) << VEC_FROM_ARRAY(normVec), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        if (i >= matchSurfSize) {
            double a = _matchlinelist[i - matchSurfSize].a;
            double b = _matchlinelist[i - matchSurfSize].b;
            double c = _matchlinelist[i - matchSurfSize].c;
            double fx = 400.0;
            double fy = 400.0;
            Eigen::Vector3d pl2 = Eigen::Vector3d(_matchlinelist[i - matchSurfSize].p3d.x, _matchlinelist[i - matchSurfSize].p3d.y, _matchlinelist[i - matchSurfSize].p3d.z);
            V3D pl = _Rl2l.conjugate() * pl2 - _Rl2l.conjugate() * _tl2l;
            double x = pl2(0);
            double y = pl2(1);
            double z = pl2(2);
            V3D normVec((a * fx * y - b * fy * z) / (x * x), -a * fx / x, b * fy / x);
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(pl);
            M3D point_be_crossmat;
            V3D C(_Rwlprop * normVec);
            V3D A(point_crossmat * _Rwl.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(i, 0) << VEC_FROM_ARRAY(C), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.R(i) = 0.1;
            ekfom_data.h(i) = -_matchlinelist[i - matchSurfSize].dist;
        }
        /*** Measuremnt: distance to the closest surface/corner ***/
        if (i < matchSurfSize) {
            ekfom_data.R(i) = _matchedSurfList[i].R;
            ekfom_data.h(i) = -dist;
        }
        //totalResidual += fabs(dist);
    }
    //double meanResidual = totalResidual / (matchSurfSize+matchlinesize);
    //std::cout<<"Matched feat num:" << matchSurfSize<<"+"<<matchlinesize<< std::endl;
    //std::cout<<"Mean Residual:" << meanResidual << std::endl;
}

bool System::updateKFVoxelMap() {
    PointCloudXYZI::Ptr corrNormVect(new PointCloudXYZI(100000, 1));
    double totalResidual = 0.0;
    VD(DIM_STATE)
    solution;
    int rematchNum = 0;

    // sort(_localCloudDownPtr->points.begin(), _localCloudDownPtr->points.end(), timeComprator);
    sort(_localSurfCloudDownPtr->points.begin(), _localSurfCloudDownPtr->points.end(), timeComprator);

    /*** init cov calculation***/
    _curSurfPvList.clear();
    for (size_t i = 0; i < _localSurfCloudDownPtr->size(); i++) {
        PointWithCov pv;
        pv.featType = Plane;
        pv.pl << _localSurfCloudDownPtr->points[i].x, _localSurfCloudDownPtr->points[i].y, _localSurfCloudDownPtr->points[i].z;
        _curSurfPvList.push_back(pv);
    }
    if (_config._covType == 2)
        calcPointNeighCov(_curSurfPvList, 20);

    std::set<OctoTree *> voxelUpdated;
    _rotDeltas.clear();
    _posDeltas.clear();
    _matchSurfSizeList.clear();
    _matchCornerSizeList.clear();
    /*** iterated state estimation ***/
    for (int iterCount = 0; iterCount < _config._nMaxInterations; iterCount++) {
        corrNormVect->clear();
        totalResidual = 0.0;

        /** LiDAR match based on 3 sigma criterion **/
        //         _Rwl = _stateCur.rot_end.matrix() * _stateCur.offset_R_L_I;//Rwl=Rwi*Ril
        //         _twl = _stateCur.rot_end.matrix() * _stateCur.offset_T_L_I + _stateCur.pos_end;//twl=Twi*til
        //         transCloud(_localSurfCloudDownPtr, _globalSurfCloudDownPtr, _Rwl, _twl);//pl->pw
        //         updateViewer();
        //std::cout<<"Iter:"<<iterCount<<std::endl;
        //         printState();
        //PAUSE;

        for (size_t i = 0; i < _curSurfPvList.size(); i++) {
            _curSurfPvList[i].pi = _stateCur.offset_R_L_I * _curSurfPvList[i].pl + _stateCur.offset_T_L_I;
            if (_curSurfPvList[i].pi[2] == 0)
                _curSurfPvList[i].pi[2] = 0.001;
            _curSurfPvList[i].pw = _stateCur.rot_end * _curSurfPvList[i].pi + _stateCur.pos_end;

            if (_config._covType == 1) {
                M3D pointCrossmat;
                pointCrossmat << SKEW_SYM_MATRX(_curSurfPvList[i].pi);
                calcBodyCov(_curSurfPvList[i], _config._rangingCov, _config._angleCov);
                _curSurfPvList[i].obsCov = _stateCur.rot_end * _curSurfPvList[i].bodyCov * _stateCur.rot_end.transpose() + _stateCur.rot_end * (-pointCrossmat) * _stateCur.cov.block<3, 3>(0, 0) * (-pointCrossmat.transpose()) * _stateCur.rot_end.transpose() + _stateCur.cov.block<3, 3>(3, 3);
            } else if (_config._covType == 2)
                _curSurfPvList[i].obsCov = _curSurfPvList[i].neighCov.block<3, 3>(0, 0);
        }

        //if (_isLoopCorrected && iterCount == 0)
        //{
        //     updateViewer();
        // std::cout<<"Iter:"<<iterCount<<std::endl;
        // printState();
        // PAUSE;
        // _isLoopCorrected=false;
        //}

        std::vector<V3D> nonMatchList;
        bool isStrict = true;
        if (_config._covType == 0 || _config._covType == 2 || _isLoopCorrected)
            isStrict = false;
        //		TicToc time1;
        buildResidualListOmp(_voxelSurfMap, _config._voxelLength, 3.0, _config._maxLayers, _curSurfPvList, _matchedSurfList, nonMatchList, isStrict, _config._radius_k);
        //		printf("buildResidualListOmp time: %f ms\n", time1.toc());

        PointType pb;
        PointType pw;
        PointType norm;
        const int matchSurfSize = _matchedSurfList.size();
        _matchSurfSizeList.emplace_back(matchSurfSize);
        for (int i = 0; i < matchSurfSize; i++) {
            pb.x = _matchedSurfList[i].point(0);
            pb.y = _matchedSurfList[i].point(1);
            pb.z = _matchedSurfList[i].point(2);
            norm.x = _matchedSurfList[i].normal(0);
            norm.y = _matchedSurfList[i].normal(1);
            norm.z = _matchedSurfList[i].normal(2);

            pointBodyToWorld(_stateCur, &pb, &pw);
            float dis = (pw.x * norm.x + pw.y * norm.y + pw.z * norm.z + _matchedSurfList[i].d);
            norm.intensity = dis;
            corrNormVect->push_back(norm);
            totalResidual += fabs(dis);
            // totalResidual += fabs(pw.x * norm.x + pw.y * norm.y + pw.z * norm.z);
        }
        double meanResidual = totalResidual / matchSurfSize;
        //        std::cout<<"Matched feat num:"<<matchSurfSize<<std::endl;
        //        std::cout<<"Mean Residual:"<<meanResidual<<std::endl;

        MatrixXd K(DIM_STATE, matchSurfSize);
        MatrixXd Hsub(matchSurfSize, 12);
        MatrixXd Hsub_T_R_inv(12, matchSurfSize);
        VectorXd R_inv(matchSurfSize);
        VectorXd meas_vec(matchSurfSize);

        Hsub.setZero();
        Hsub_T_R_inv.setZero();
        meas_vec.setZero();

        if (assertDegeneracy()) {
            std::cerr << "Environment degenerate!" << std::endl;
            //            std::cout<<"Iter:"<<iterCount<<std::endl;
            //            _Rwl = _stateCur.rot_end.matrix() * _stateCur.offset_R_L_I;//Rwl=Rwi*Ril
            //            _twl = _stateCur.rot_end.matrix() * _stateCur.offset_T_L_I + _stateCur.pos_end;//twl=Twi*til
            updateViewer();
            PAUSE;
        }

        for (int i = 0; i < matchSurfSize; i++) {
            /*** get the normal vector of closest surface/corner ***/
            const PointType &norm_p = corrNormVect->points[i];
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = _matchedSurfList[i].pv.pw - _matchedSurfList[i].center;
            J_nq.block<1, 3>(0, 3) = -_matchedSurfList[i].normal;
            V3D normVec(norm_p.x, norm_p.y, norm_p.z);

            if (_config._covType == 0)
                _matchedSurfList[i].R = 0.001;
            else if (_config._covType == 1) {
                double sigma_l = J_nq * _matchedSurfList[i].plane_cov * J_nq.transpose();
                _matchedSurfList[i].R = (sigma_l + normVec.transpose() * _stateCur.rot_end * _matchedSurfList[i].pv.bodyCov * _stateCur.rot_end.transpose() * normVec);
            } else if (_config._covType == 2) {
                // Update voxel plane cov by kdtree
                OctoTree *voxel = _matchedSurfList[i].voxel_correspondence;
                PlaneParams plane = *voxel->_planePtr;
                if (!voxelUpdated.count(voxel)) {
                    if (!voxel->_tempPoints.empty() && (voxel->_cloudCov.isZero() || voxel->_tempPoints.size() < 100)) {
                        std::vector<PointWithCov> downVoxelPoints;
                        downSamplingVoxel(voxel->_tempPoints, downVoxelPoints, 0.01);
                        voxel->calcNeighCov(downVoxelPoints, &_ikdtree); // Calculate voxel point neighbor cov by kdtree
                        for (size_t i = 0; i < downVoxelPoints.size(); i++)
                            downVoxelPoints[i].obsCov = downVoxelPoints[i].neighCov.block<3, 3>(0, 0);

                        //voxel->updatePlaneCov(downVoxelPoints, voxel->_planePtr);//update voxel plane cov by points' neighbor cov
                        voxel->initPlane(downVoxelPoints, &plane); //update voxel plane cov by points' neighbor cov

                        //                        voxel->calcNeighCov(voxel->_tempPoints, &_ikdtree);
                        //                        for (size_t i = 0; i < voxel->_tempPoints.size(); i++)
                        //                            voxel->_tempPoints[i].obsCov = voxel->_tempPoints[i].neighCov.block<3, 3>(0, 0);
                        //                        voxel->initPlane(voxel->_tempPoints, voxel->_planePtr);
                    }
                    voxelUpdated.insert(voxel);
                }

                double sigma_l = J_nq * plane.planeCov * J_nq.transpose();
                _matchedSurfList[i].R = sigma_l + normVec.transpose() * _Rwl * _matchedSurfList[i].pv.neighCov.block<3, 3>(0, 0) * _Rwl.transpose() * normVec;
            }

            //            _matchedSurfList[i].R=0.001;
            R_inv(i) = 1.0 / _matchedSurfList[i].R;
            // std::cout<< R_inv(i)<<std::endl;

            /*** calculate the Measurement Jacobian matrix H ***/
            M3D pointCrossmat;
            pointCrossmat << SKEW_SYM_MATRX(_matchedSurfList[i].pv.pi);
            V3D A(pointCrossmat * _stateCur.rot_end.transpose() * normVec);
            if (_config._isImuInitialized && _config._isEstiExtrinsic) {
                M3D point_this_L_cross;
                point_this_L_cross << SKEW_SYM_MATRX(_matchedSurfList[i].pv.pl);
                V3D H_R_LI = point_this_L_cross * _stateCur.offset_R_L_I.transpose() * _stateCur.rot_end.transpose() * normVec;
                V3D H_T_LI = _stateCur.rot_end.transpose() * normVec;
                Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(H_R_LI), VEC_FROM_ARRAY(H_T_LI);
            } else
                Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, 0, 0, 0, 0, 0, 0; // H

            Hsub_T_R_inv.col(i) = Hsub.row(i).transpose() * R_inv(i); // H' R^
            /*** Measurement: distance to the closest surface/corner ***/
            meas_vec(i) = -norm_p.intensity; // Z
        }

        /*** Iterative Kalman Filter Update ***/
        _H_T_H.block<12, 12>(0, 0) = Hsub_T_R_inv * Hsub;                              // H' R^ H
        MD(DIM_STATE, DIM_STATE) &&K_1 = (_H_T_H + _stateCur.cov.inverse()).inverse(); //(H' R^ H+ P^)^
        K = K_1.block<DIM_STATE, 12>(0, 0) * Hsub_T_R_inv;                             // K = H' R^/ (H' R^ H+ P^)
        auto vec = _statePropagat - _stateCur;                                         // X(k|k-1)
        solution = K * meas_vec + vec - K * Hsub * vec.block<12, 1>(0, 0);             // X(k|k)= X(k|k-1) + K(Z-HX(k|k-1))

        //        std::cout<<K<<std::endl;

        V3D rotAdd = solution.block<3, 1>(0, 0);
        V3D tAdd = solution.block<3, 1>(3, 0);
        double RDelta = rotAdd.norm() * 57.3;
        double tDelta = tAdd.norm();
        // std::cout<<"RDelta/tDelta: "<<RDelta<<"/"<<tDelta<<std::endl;

        _rotDeltas.push_back(RDelta);
        _posDeltas.push_back(tDelta);

        // state update
        _stateCur += solution;

        bool isEKFConverged = false;
        if ((RDelta < 0.01) && (tDelta * 100 < 0.015))
            isEKFConverged = true;

        if (isEKFConverged || ((rematchNum == 0) && (iterCount == (_config._nMaxInterations - 2))))
            rematchNum++;

        /*** Convergence Judgements and Covariance Update ***/
        if (rematchNum >= 2 || (iterCount == _config._nMaxInterations - 1)) {
            if (_isEKFInited) {
                /*** Covariance Update ***/
                _G.setZero();
                _G.block<DIM_STATE, 12>(0, 0) = K * Hsub;
                _stateCur.cov = (_I_STATE - _G) * _stateCur.cov; // P(k|k)=（I-KH）P
                //                std::cout << _stateCur.cov << std::endl;
                break;
            }
        }
    }

    //    if (!motionCheck())
    //    {
    //        std::cout << "Invalid motion" << std::endl;
    //        _stateCur = _statePropagat;
    //        // PAUSE;
    //        return false;
    //    }
    //////////////update voxel map//////////////
    _Rwl = _stateCur.rot_end.matrix() * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
    _twl = _stateCur.rot_end.matrix() * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til
    transCloud(_localSurfCloudDownPtr, _globalSurfCloudDownPtr, _Rwl, _twl);        // pl->pw

    for (size_t i = 0; i < _curSurfPvList.size(); i++) {
        _curSurfPvList[i].pi = _stateCur.offset_R_L_I * _curSurfPvList[i].pl + _stateCur.offset_T_L_I;
        if (_curSurfPvList[i].pi[2] == 0)
            _curSurfPvList[i].pi[2] = 0.001;
        _curSurfPvList[i].pw = _stateCur.rot_end * _curSurfPvList[i].pi + _stateCur.pos_end;
        if (_config._covType == 1) {
            M3D pointCrossmat;
            pointCrossmat << SKEW_SYM_MATRX(_curSurfPvList[i].pi);
            calcBodyCov(_curSurfPvList[i], _config._rangingCov, _config._angleCov);
            _curSurfPvList[i].obsCov = _stateCur.rot_end * _curSurfPvList[i].bodyCov * _stateCur.rot_end.transpose() + _stateCur.rot_end * (-pointCrossmat) * _stateCur.cov.block<3, 3>(0, 0) * (-pointCrossmat.transpose()) * _stateCur.rot_end.transpose() + _stateCur.cov.block<3, 3>(3, 3);
        } else if (_config._covType == 2)
            _curSurfPvList[i].obsCov = _curSurfPvList[i].neighCov.block<3, 3>(0, 0);
    }
    std::sort(_curSurfPvList.begin(), _curSurfPvList.end(), varContrast);
    //	TicToc time1;
    ::updateVoxelMap(_curSurfPvList, _frameId, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
                     _config._maxPointsSize, _config._maxPointsSize, _config._minSurfEigenValue,
                     _voxelSurfMap, _boxToDel);
    //	printf("updateVoxelMap time: %f ms\n", time1.toc());
    return true;
}

bool System::updateKFKdTree() {
    PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
    PointCloudXYZI::Ptr corrNormVect(new PointCloudXYZI(100000, 1));
    double totalResidual = 0.0;
    int effctFeatNum = 0;
    bool isNearestSearch = true;
    bool isEKFStop = false;
    bool isEKFConverged = false;
    VD(DIM_STATE)
    solution;
    V3D rot_add, T_add, vel_add, gyr_add;
    int rematchNum = 0;
    _nearestPoints.resize(_cloudSurfDownSize);

    for (int iterCount = 0; iterCount < _config._nMaxInterations; iterCount++) {
        laserCloudOri->clear();
        corrNormVect->clear();
        totalResidual = 0.0;

        //        _Rwl=_stateCur.rot_end.matrix()*_stateCur.offset_R_L_I;//Rwl=Rwi*Ril
        //        _twl=_stateCur.rot_end.matrix()*_stateCur.offset_T_L_I+_stateCur.pos_end;//twl=Twi*til
        //        updateViewer();
        //        std::cout<<"Iter:"<<iterCount<<std::endl;
        //        printState();
        //        PAUSE;

        /** closest surface search and residual computation **/
#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int i = 0; i < _cloudSurfDownSize; i++) {
            PointType &point_body = _localSurfCloudDownPtr->points[i]; // pl
            V3D p_body(point_body.x, point_body.y, point_body.z);

            // transform to world frame
            PointType point_world; // pw
            pointBodyToWorld(_stateCur, &point_body, &point_world);
            vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
            auto &pointsNear = _nearestPoints[i];

            if (isNearestSearch) {
                _ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, pointsNear, pointSearchSqDis, 5);
                if (pointsNear.size() < NUM_MATCH_POINTS)
                    _isPointSelectedSurf[i] = false;
                else
                    _isPointSelectedSurf[i] = !(pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5);
            }
            _resLast[i] = -1000.0f;
            if (!_isPointSelectedSurf[i])
                continue;

            _isPointSelectedSurf[i] = false;
            VD(4)
            pabcd;
            pabcd.setZero();
            if (estiPlane(pabcd, pointsNear, 0.1)) //(planeValid)
            {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

                if (s > 0.9)
                //if(fabs(pd2)< 0.3*p_body.norm())
                {
                    _isPointSelectedSurf[i] = true;
                    _normvec->points[i].x = pabcd(0);
                    _normvec->points[i].y = pabcd(1);
                    _normvec->points[i].z = pabcd(2);
                    _normvec->points[i].intensity = pd2;
                    _resLast[i] = abs(pd2);
                    totalResidual += _resLast[i];
                }
            }
        }
        effctFeatNum = 0;
        for (int i = 0; i < _cloudSurfDownSize; i++) {
            if (_isPointSelectedSurf[i]) {
                laserCloudOri->points[effctFeatNum] = _localSurfCloudDownPtr->points[i];
                corrNormVect->points[effctFeatNum] = _normvec->points[i];
                effctFeatNum++;
            }
        }

        if (effctFeatNum < 1) {
            std::printf("No Effective Points! \n");
            return false;
        }
        double meanResidual = totalResidual / effctFeatNum;
        //        std::cout<<"matched feat num:"<<effctFeatNum<<std::endl;
        //        std::cout<<"Mean Residual:"<<meanResidual<<std::endl;

        /*** Computation of Measurement Jacobian matrix H and measurents vector ***/
        MatrixXd Hsub(effctFeatNum, 12);
        MatrixXd Hsub_T_R_inv(12, effctFeatNum);
        VectorXd R_inv(effctFeatNum);
        VectorXd meas_vec(effctFeatNum);

        Hsub.setZero();
        Hsub_T_R_inv.setZero();
        meas_vec.setZero();

        for (int i = 0; i < effctFeatNum; i++) {
            const PointType &laser_p = laserCloudOri->points[i];
            V3D pl(laser_p.x, laser_p.y, laser_p.z);

            V3D pi = _stateCur.offset_R_L_I * pl + _stateCur.offset_T_L_I;
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(pi);

            /*** get the normal vector of closest surface/corner ***/
            const PointType &norm_p = corrNormVect->points[i];
            V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

            R_inv(i) = 1000;
            laserCloudOri->points[i].intensity = sqrt(R_inv(i));

            /*** calculate the Measurement Jacobian matrix H ***/
            if (_config._isImuInitialized) {
                M3D point_this_L_cross;
                point_this_L_cross << SKEW_SYM_MATRX(pl);
                V3D H_R_LI = point_this_L_cross * _stateCur.offset_R_L_I.transpose() * _stateCur.rot_end.transpose() * norm_vec;
                V3D H_T_LI = _stateCur.rot_end.transpose() * norm_vec;
                V3D A(point_crossmat * _stateCur.rot_end.transpose() * norm_vec);
                Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(H_R_LI), VEC_FROM_ARRAY(H_T_LI);
            } else {
                V3D A(point_crossmat * _stateCur.rot_end.transpose() * norm_vec);
                Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z, 0, 0, 0, 0, 0, 0; // H
            }

            Hsub_T_R_inv.col(i) = Hsub.row(i).transpose() * 1000; //H' R^
            /*** Measurement: distance to the closest surface/corner ***/
            meas_vec(i) = -norm_p.intensity; // Z
        }

        MatrixXd K(DIM_STATE, effctFeatNum);

        isEKFStop = false;
        isEKFConverged = false;

        /*** Iterative Kalman Filter Update ***/
        _H_T_H.block<12, 12>(0, 0) = Hsub_T_R_inv * Hsub;                              //H' R^ H
        MD(DIM_STATE, DIM_STATE) &&K_1 = (_H_T_H + _stateCur.cov.inverse()).inverse(); //(H' R^ H+ P^)^
        K = K_1.block<DIM_STATE, 12>(0, 0) * Hsub_T_R_inv;                             //K = H' R^/ (H' R^ H+ P^)
        auto vec = _statePropagat - _stateCur;                                         //X(k|k-1)
        solution = K * meas_vec + vec - K * Hsub * vec.block<12, 1>(0, 0);             //X(k|k)= X(k|k-1) + K(Z-HX(k|k-1))
        //state update
        _stateCur += solution;

        rot_add = solution.block<3, 1>(0, 0);
        T_add = solution.block<3, 1>(3, 0);
        if ((rot_add.norm() * 57.3 < 0.01) && (T_add.norm() * 100 < 0.015))
            isEKFConverged = true;

        //        double deltaR = rot_add.norm() * 57.3;
        //        double deltaT = T_add.norm() * 100;

        /*** Rematch Judgement ***/
        isNearestSearch = false;
        if (isEKFConverged || ((rematchNum == 0) && (iterCount == (_config._nMaxInterations - 2)))) {
            isNearestSearch = true;
            //            if (iterCount == NUM_MAX_ITERATIONS - 2)
            //                iterCount = max(0,iterCount-NUM_MAX_ITERATIONS/2);
            rematchNum++;
        }

        /*** Convergence Judgements and Covariance Update ***/
        if (!isEKFStop && (rematchNum >= 2 || (iterCount == _config._nMaxInterations - 1))) {
            /*** Covariance Update ***/
            _G.setZero();
            _G.block<DIM_STATE, 12>(0, 0) = K * Hsub;
            _stateCur.cov = (_I_STATE - _G) * _stateCur.cov; //P(k|k)=（I-KH）P
            isEKFStop = true;
        }

        if (isEKFStop)
            break;
    }

    _Rwl = _stateCur.rot_end.matrix() * _stateCur.offset_R_L_I;                     // Rwl=Rwi*Ril
    _twl = _stateCur.rot_end.matrix() * _stateCur.offset_T_L_I + _stateCur.pos_end; // twl=Twi*til
    return true;
}
