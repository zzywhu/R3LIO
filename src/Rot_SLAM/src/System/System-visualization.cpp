//
// Created by w on 2022/9/26.
//
#include "System.h"
#include "CloudInfo.h"


template<typename T>
void clearQueue(std::queue<T> &queue){std::queue<T>{}.swap(queue);}

void System::convertToRGBCloud(PointCloudXYZI &cloudIn, pcl::PointCloud<pcl::PointXYZRGB> &cloudRGB)
{
    static int topColor[3] = {255, 0, 0};
    static int midColor[3] = {0, 255, 0};
    static int bottomColor[3] = {0, 0, 255};

    static int r0 = topColor[0] - bottomColor[0];
    static int g0 = topColor[1] - bottomColor[1];
    static int b0 = topColor[2] - bottomColor[2];

    static int r1 = midColor[0] - bottomColor[0];
    static int g1 = midColor[1] - bottomColor[1];
    static int b1 = midColor[2] - bottomColor[2];

    static int r2 = topColor[0] - midColor[0];
    static int g2 = topColor[1] - midColor[1];
    static int b2 = topColor[2] - midColor[2];

    double midz = (_maxDispZ + _minDispZ) / 2;
    const int cloudSize = cloudIn.points.size();
    cloudRGB.resize(cloudSize);
    for (size_t i = 0; i < cloudSize; ++i)
    {
        cloudRGB.points[i].x = cloudIn.points[i].x;
        cloudRGB.points[i].y = cloudIn.points[i].y;
        cloudRGB.points[i].z = cloudIn.points[i].z;
        float k1 = (cloudIn.points[i].z - _minDispZ) / (_maxDispZ - _minDispZ);
        cloudRGB.points[i].r = bottomColor[0] + r1 * k1;
        cloudRGB.points[i].g = bottomColor[1] + g1 * k1;
        cloudRGB.points[i].b = bottomColor[2] + b1 * k1;
        //        if (cloudIn.points[i].z < _minDispZ)
        //        {
        //            float k1 = (cloudIn.points[i].z - _minDispZ) / (midz - _minDispZ);
        //            cloudRGB.points[i].r = bottomColor[0] + r1 * k1;
        //            cloudRGB.points[i].g = bottomColor[1] + g1 * k1;
        //            cloudRGB.points[i].b = bottomColor[2] + b1 * k1;
        //        }
        //        else
        //        {
        //            float k2 = (cloudRGB.points[i].z - _minDispZ) / (_maxDispZ - midz);
        //            cloudRGB.points[i].r = midColor[0] + r2 * k2;
        //            cloudRGB.points[i].g = midColor[1] + g2 * k2;
        //            cloudRGB.points[i].b = midColor[2] + b2 * k2;
        //        }
    }
}

void System::voxelMap2RGBClouds(const std::unordered_map<VOXEL_LOC, OctoTree *>& voxelMap,PointCloudXYZRGB::Ptr octCloudRGBPtr)
{
    octCloudRGBPtr->clear();
    for (auto &locOct : voxelMap)
    {
        OctoTree *oct = locOct.second;
        if (oct->_planePtr->isPlane)
        {
            const Colour &colour = _voxelColors[oct->_planePtr->id % 200];
            std::vector<PointWithCov> downVoxelPoints;
            downSamplingVoxel(oct->_tempPoints, downVoxelPoints, 0.2);
            for (size_t i = 0; i < downVoxelPoints.size(); i++)
            {
                PointTypeRGB ptRGB;
                ptRGB.x = downVoxelPoints[i].pw.x();
                ptRGB.y = downVoxelPoints[i].pw.y();
                ptRGB.z = downVoxelPoints[i].pw.z();
                ptRGB.r = 255 * colour.red;
                ptRGB.g = 255 * colour.green;
                ptRGB.b = 255 * colour.blue;
                octCloudRGBPtr->push_back(ptRGB);
            }
        }
    }
}

void System::initPCLViewer()
{
    //return;
    _dispThread = new std::thread(std::bind(&System::show, this));
    std::chrono::milliseconds dura(200);
    std::this_thread::sleep_for(dura);
}

void System::resetViewer()
{
    std::lock_guard<std::mutex> lg(_dispMutex);
    _p->removeAllShapes(_vp1);
    _p->removeAllPointClouds(_vp1);

    clearQueue(_localCloudQueue);
    clearQueue(_localCloudDownQueue);
    clearQueue(_localPlaneDownQueue);
    clearQueue(_localLineDownQueue);
    clearQueue(_stateQueue);
    clearQueue(_statePropagatQueue);
    clearQueue(_RQueue);
    clearQueue(_tQueue);
    clearQueue(_motorAngleQueue);
    clearQueue(_downCloudMapQueue);
    clearQueue(_voxelCloudQueue);
    clearQueue(_matchedSurfListQueue);

    _trajPts->clear();
    _trajPropagatPts->clear();
}


void System::updateViewer(bool bForceUpdateMap)
{
    //return;
	static int nTotalAddedPoints=0;
    _dispMutex.lock();
    _localCloudQueue.push(_localCloudPtr->makeShared());
    _localCloudDownQueue.push(_localCloudDownPtr->makeShared());
    _localPlaneDownQueue.push(_localSurfCloudDownPtr->makeShared());
    _localLineDownQueue.push(_localCornerCloudDownPtr->makeShared());
    _stateQueue.push(_stateIkfom);
    _statePropagatQueue.push(_statePropIkfom);
    // Twl=Two*Too'*Tol=Two'*exp(angle*n)*Tol=[Rwo*Rol|Rwo*tol+two]
    //        Eigen::Matrix3d motorRot = AngleAxisToRotationMatrix(Eigen::Vector3d(0, 0, _curMotorAngle));
    //        Eigen::Matrix3d Rwl=_Rwo * motorRot;
    //        Eigen::Vector3d twl=_Rwo * motorRot * _tol+_two;
    _RQueue.push(_Rwl);
    _tQueue.push(_twl);
    _motorAngleQueue.push(_curMotorAngle);

    const BoxPointType &range = _ikdtree.tree_range();
    _minDispZ = range.vertex_min[2];
    _maxDispZ = range.vertex_max[2];
    //_minDispZ=0;
    //_maxDispZ = 7.;
    nTotalAddedPoints+=_addedPoints.size();
    if((nTotalAddedPoints>100000&&(_frameId%25==0))||bForceUpdateMap)
    {
        PointVector().swap(_ikdtree.PCL_Storage);
        _ikdtree.flatten(_ikdtree.Root_Node, _ikdtree.PCL_Storage, NOT_RECORD);
        _downCloudMap->points = _ikdtree.PCL_Storage;
        _downCloudMapQueue.push(_downCloudMap);
        nTotalAddedPoints=0;
        //    PointCloudXYZRGB::Ptr octRGBCloudPtr(new PointCloudXYZRGB);
//    voxelMap2RGBClouds(_voxelSurfMap, octRGBCloudPtr);
//    _voxelCloudQueue.push(octRGBCloudPtr);
    }

    _matchedSurfListQueue.push(_matchedSurfList);
    if(_loopCloser)
        _lcIdxPairsQueue.push(_loopCloser->getLCPairs());

    _dispMutex.unlock();
}

//void System::showKdTree(const int &vp)
//{
//    if(_addedPoints.empty())
//        return;
//    std::string lable = "global" +std::to_string(_frameId)+ std::to_string(vp);
//
//   // _p->removePointCloud(lable);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//    for(auto p:_addedPoints)
//    {
//        pcl::PointXYZRGB pRGB;
//        pRGB.x=p.x;
//        pRGB.y=p.y;
//        pRGB.z=p.z;
//        pRGB.r=255;
//        pRGB.g=255;
//        pRGB.b=255;
//        cloud->push_back(pRGB);
//    }
//    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGBPtr(new pcl::PointCloud<pcl::PointXYZRGB>());
//    //convertToRGBCloud(*_downCloudMapDisp, *cloudRGBPtr);
//
////    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbCloudHandler(cloudRGBPtr);
////    _p->addPointCloud<pcl::PointXYZRGB>(cloudRGBPtr, rgbCloudHandler, lable);
////    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, lable);
//
//    _p->addPointCloud(cloud, lable, vp);
//    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, lable); // 设置点云大小
//
//    _addedPoints.clear();
//}

void System::showKdTree(const int &vp)
{
    if(!_downCloudMapDisp)
        return;
    if(_downCloudMapDisp->empty())
        return;
//    _dispMutex.lock();
//    pcl::VoxelGrid<PointType> voxelFilter;
//    voxelFilter.setLeafSize(0.5,0.5,0.5);
//    voxelFilter.setInputCloud(_downCloudMapDisp->makeShared());
//    voxelFilter.filter(*_downCloudMapDisp);
//    PointCloudXYZI::Ptr downCloudMapDisp=_downCloudMapDisp->makeShared();
//    _dispMutex.unlock();

    transCloud(_downCloudMapDisp, _downCloudMapDisp, _rotAlign, Eigen::Vector3d::Zero());

    std::string lable = "global" + std::to_string(vp);
    _p->removePointCloud(lable);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGBPtr(new pcl::PointCloud<pcl::PointXYZRGB>());
    convertToRGBCloud(*_downCloudMapDisp, *cloudRGBPtr);

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbCloudHandler(cloudRGBPtr);
    _p->addPointCloud<pcl::PointXYZRGB>(cloudRGBPtr, rgbCloudHandler, lable);
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, lable);
    _downCloudMapDisp.reset(new PointCloudXYZI);
//    PointCloudColorHandlerCustom<PointType> mapCloudHandler(_downCloudMapDisp, 255, 255, 255);
//    _p->addPointCloud(_downCloudMapDisp, mapCloudHandler, lable, vp);
}

void System::showLocalDownCloud(const int &vp)
{
    std::string lable = "local" + std::to_string(vp);
    _p->removePointCloud(lable);

//    pcl::VoxelGrid<PointType> voxelFilter;
//    voxelFilter.setLeafSize(0.5,0.5,0.5);
//    voxelFilter.setInputCloud(_localCloudDownDisp->makeShared());
//    voxelFilter.filter(*_localCloudDownDisp);

    const int cloudDownSize = _localCloudDownDisp->size();
    PointCloudXYZI::Ptr curCloudDownPtr(new PointCloudXYZI());
    curCloudDownPtr->resize(cloudDownSize);
    transCloud(_localCloudDownDisp, curCloudDownPtr, _rotAlign*_RDisp, _rotAlign*_tDisp);
    PointCloudColorHandlerCustom<PointType> curCloudHandler(curCloudDownPtr, 255, 0, 0);
    _p->addPointCloud(curCloudDownPtr, curCloudHandler, lable, vp);
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, lable);
}

void System::showLocalFeatCloud(const int &vp)
{
    if(!_config._isFeatExtractEn)
        return;
    std::string lableLine = "LocalLine" + std::to_string(vp);
    std::string lablePlane = "LocalPlane" + std::to_string(vp);
    _p->removePointCloud(lableLine);
    _p->removePointCloud(lablePlane);

    const int planeDownSize = _localPlaneDownDisp->size();
    PointCloudXYZI::Ptr curPlaneDownPtr(new PointCloudXYZI());
    curPlaneDownPtr->resize(planeDownSize);
    transCloud(_localPlaneDownDisp, curPlaneDownPtr, _rotAlign*_RDisp, _rotAlign*_tDisp);
    PointCloudColorHandlerCustom<PointType> curPlaneHandler(curPlaneDownPtr, 255, 255, 0);
    _p->addPointCloud(curPlaneDownPtr, curPlaneHandler, lablePlane, vp);
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, lablePlane);

    const int lineDownSize = _localLineDownDisp->size();
    PointCloudXYZI::Ptr curLineDownPtr(new PointCloudXYZI());
    curLineDownPtr->resize(lineDownSize);
    transCloud(_localLineDownDisp, curLineDownPtr, _rotAlign*_RDisp, _rotAlign*_tDisp);
    PointCloudColorHandlerCustom<PointType> curLineHandler(curLineDownPtr, 0, 255, 255);
    _p->addPointCloud(curLineDownPtr, curLineHandler, lableLine, vp);
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, lableLine);
}

void System::showDenseMapCloud(const int &vp)
{
    std::string lable1 = "local" + std::to_string(vp);
    std::string lable2 = "global" + std::to_string(vp);
    _p->removePointCloud(lable1);
    _p->removePointCloud(lable2);

    PointCloudXYZI::Ptr curWorldCloudPtr(new PointCloudXYZI());
    transCloud(_localCloudDisp, curWorldCloudPtr, _rotAlign*_RDisp, _rotAlign*_tDisp);

    if (!_config._isSaveMap)
        updateDenseMap(curWorldCloudPtr);

    PointCloudColorHandlerCustom<PointType> curCloudHandler(curWorldCloudPtr, 255, 0, 0);
    PointCloudColorHandlerCustom<PointType> mapCloudHandler(_denseCloudMap, 0, 255, 0);
    _p->addPointCloud(curWorldCloudPtr, curCloudHandler, lable1, vp);
    _p->addPointCloud(_denseCloudMap, mapCloudHandler, lable2, vp);
}

void System::updateTrajectory()
{
    Eigen::Matrix4f curPose = Eigen::Matrix4f::Identity();
    // Twl=Two*Tol=[Rwo*Rol|Rwo*tol+two]
    //    curPose.block<3, 3>(0, 0) = AngleAxisToRotationMatrix(Eigen::Vector3f(0, 0, _motorAngleDisp)) * _RDisp.cast<float>();
    curPose.block<3, 3>(0, 0) = (_rotAlign*_RDisp).cast<float>();
    curPose.block<3, 1>(0, 3) = (_rotAlign*_tDisp).cast<float>();
    PointType curPos;
    curPos.x = curPose(0, 3);
    curPos.y = curPose(1, 3);
    curPos.z = curPose(2, 3);
    _trajPts->push_back(curPos);

    Eigen::Matrix4d curPropagatePose = Eigen::Matrix4d::Identity();
    //curPropagatePose.block<3, 3>(0, 0) = _statePropagatDisp.rot_end * _statePropagatDisp.offset_R_L_I;//Rwl=Rwi*Ril
    curPropagatePose.block<3, 1>(0, 3) = _rotAlign*_statePropagatDisp.rot * _statePropagatDisp.offset_T_L_I + _rotAlign*_statePropagatDisp.pos; // twl=Twi*til
    PointType curPropagatPos;
    curPropagatPos.x = curPropagatePose(0, 3);
    curPropagatPos.y = curPropagatePose(1, 3);
    curPropagatPos.z = curPropagatePose(2, 3);
    _trajPropagatPts->push_back(curPropagatPos);

    _dispMutex.lock();
    while(!_gnssQueue.empty())
    {
        _gnssDisp = _gnssQueue.front();
        _gnssQueue.pop();
        V3D curPos=_rotAlign*_gnssDisp->pos;
        PointType curGnssPos;
        curGnssPos.x = curPos(0);
        curGnssPos.y = curPos(1);
        curGnssPos.z = curPos(2);
        _trajGNSSPts->push_back(curGnssPos);
        _gnssDisp=nullptr;
    }
    _dispMutex.unlock();
}

void System::showTrajectory(const int &vp)
{
    Eigen::Affine3f Two = Eigen::Affine3f::Identity();
    _p->updateCoordinateSystemPose("origin", Two);

    // draw Trajectory
    Eigen::Matrix4f curPose = Eigen::Matrix4f::Identity();
    // Twl=Two*Tol=[Rwo*Rol|Rwo*tol+two]
    //    curPose.block<3, 3>(0, 0) = AngleAxisToRotationMatrix(Eigen::Vector3f(0, 0, _motorAngleDisp)) * _RDisp.cast<float>();
    curPose.block<3, 3>(0, 0) = (_rotAlign*_RDisp).cast<float>();
    curPose.block<3, 1>(0, 3) = (_rotAlign*_tDisp).cast<float>();
    _p->updateCoordinateSystemPose("lidar", Eigen::Affine3f(curPose));

    PointCloudColorHandlerCustom<PointType> trajCloudHandler(_trajPts, 255, 0, 0);
    _p->removePointCloud("traj", vp);
    _p->addPointCloud(_trajPts, trajCloudHandler, "traj", vp);
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "traj");

    // draw propagat Trajectory
    PointCloudColorHandlerCustom<PointType> trajPropagatCloudHandler(_trajPropagatPts, 255, 255, 0);
    _p->removePointCloud("trajPropagat", vp);
    _p->addPointCloud(_trajPropagatPts, trajPropagatCloudHandler, "trajPropagat", vp);
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "trajPropagat");

    if(!_trajGNSSPts->empty())
    {
        PointCloudColorHandlerCustom<PointType> trajGNSSCloudHandler(_trajGNSSPts, 0, 255, 0);
        _p->removePointCloud("trajGnss", vp);
        _p->addPointCloud(_trajGNSSPts, trajGNSSCloudHandler, "trajGnss", vp);
        _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "trajGnss");
    }
    //_p->setCameraPosition(curPos.x, curPos.y, curPos.z + 20, 0, 0, 0);
}

void System::showLoopClosure(const int &vp)
{
    if(!_loopCloser)
        return;
    _p->removeAllShapes();
    int idx=0;
    Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
    gravAlignMat.block<3,3>(0,0) = _rotAlign;

    CloudBlockPtrs subMaps = _loopCloser->getSubMapBlocks();
    for(auto idPair:_lcIdxPairsDisp)
    {
        auto pose1=gravAlignMat*subMaps[idPair.first]->_poseLo;
        auto pose2=gravAlignMat*subMaps[idPair.second]->_poseLo;
        pcl::PointXYZ pt1(pose1(0,3),pose1(1,3),pose1(2,3));
        pcl::PointXYZ pt2(pose2(0,3),pose2(1,3),pose2(2,3));

        _p->addLine(pt1, pt2, 0, 1, 0,"line"+to_string(idx),vp);
        idx++;
    }
}

void System::showVoxelMap(const int &vp)
{
    if(_voxelMapCloudDisp->empty())
        return;
    Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
    gravAlignMat.block<3,3>(0,0) = _rotAlign;
    pcl::transformPointCloud(*_voxelMapCloudDisp, *_voxelMapCloudDisp, gravAlignMat);

    //_p->removeAllShapes(vp);
    _p->removeAllPointClouds(vp);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbCloudHandler(_voxelMapCloudDisp);
    _p->addPointCloud<pcl::PointXYZRGB>(_voxelMapCloudDisp, rgbCloudHandler, "RGBVexelCloud", vp);
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2., "RGBVexelCloud");

}

//void System::showMatchedPoints(const int &vp)
//{
//    _p->removePointCloud("match");
//    _p->removePointCloud("PlaneCenter");
//    _p->removePointCloud("Normals");
//    _p->removeAllShapes();
//    PointCloudXYZI::Ptr curWorldCloud(new PointCloudXYZI());
//    const int matchSize = _matchedSurfListDisp.size();
//    if (!matchSize)
//        return;
//    curWorldCloud->resize(matchSize);
//    for (int i = 0; i < matchSize; i++)
//    {
//        V3D pl(_matchedSurfListDisp[i].point.x(), _matchedSurfListDisp[i].point.y(), _matchedSurfListDisp[i].point.z()); // pl
//        V3D pw(_RDisp * pl + _tDisp);
//        curWorldCloud->points[i].x = pw[0];
//        curWorldCloud->points[i].y = pw[1];
//        curWorldCloud->points[i].z = pw[2];
//    }
//    PointCloudColorHandlerCustom<PointType> curCloudHandler(curWorldCloud, 255, 255, 0);
//    _p->addPointCloud(curWorldCloud, curCloudHandler, "match", vp);
//    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, "match");
//
//    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//    normals->resize(matchSize);
//    PointCloudXYZI::Ptr globalPlaneCenters(new PointCloudXYZI());
//    globalPlaneCenters->resize(matchSize);
//    double RMax=FLT_MIN,RMin=FLT_MAX;
//    for(int i=0;i<matchSize;i++)
//    {
//        if(_matchedSurfListDisp[i].R>RMax)
//            RMax=_matchedSurfListDisp[i].R;
//        if(_matchedSurfListDisp[i].R<RMin)
//            RMin=_matchedSurfListDisp[i].R;
//    }
//
//    for(int i=0;i<matchSize;i++)
//    {
//        //if(i%5!=0)
//        //continue;
//        globalPlaneCenters->points[i].normal_x=_matchedSurfListDisp[i].normal.x();
//        globalPlaneCenters->points[i].normal_y=_matchedSurfListDisp[i].normal.y();
//        globalPlaneCenters->points[i].normal_z=_matchedSurfListDisp[i].normal.z();
//        globalPlaneCenters->points[i].x = _matchedSurfListDisp[i].center[0];
//        globalPlaneCenters->points[i].y = _matchedSurfListDisp[i].center[1];
//        globalPlaneCenters->points[i].z = _matchedSurfListDisp[i].center[2];
//        double ratioR=(min(0.01,_matchedSurfListDisp[i].R)-RMin)/(0.01-RMin);
//        _p->addLine(globalPlaneCenters->points[i], curWorldCloud->points[i], 1-ratioR, ratioR, 0,"line"+to_string(i),vp);
//    }
//    _p->addPointCloud<PointType>(globalPlaneCenters, "PlaneCenter",vp);
//    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0.5, "PlaneCenter");
//    _p->addPointCloudNormals<PointType, PointType>(globalPlaneCenters, globalPlaneCenters, 10, 0.05, "Normals");
//
//}

void System::showMatchedPoints(const int &vp)
{
    _p->removePointCloud("match");
    _p->removePointCloud("PlaneCenter");
    //    _p->removeAllShapes();
    PointCloudXYZRGB::Ptr curCloudRGBPtr(new PointCloudXYZRGB());
    const int matchSize = _matchedSurfListDisp.size();
    if (!matchSize)
        return;

    double RMax = FLT_MIN, RMin = FLT_MAX;
    // double totalR=0;
    for (int i = 0; i < matchSize; i++)
    {
        // totalR+=_matchedSurfListDisp[i].R;
        if (_matchedSurfListDisp[i].R < RMin)
            RMin = _matchedSurfListDisp[i].R;
    }
    sort(_matchedSurfListDisp.begin(), _matchedSurfListDisp.end(), [](MatchOctoTreeInfo &x, MatchOctoTreeInfo &y)
         { return x.R < y.R; });
    double medianR = _matchedSurfListDisp[matchSize / 2].R;
    RMax = 2 * medianR - RMin;

    curCloudRGBPtr->resize(matchSize);
    for (int i = 0; i < matchSize; i++)
    {
        V3D pl(_matchedSurfListDisp[i].point.x(), _matchedSurfListDisp[i].point.y(), _matchedSurfListDisp[i].point.z()); // pl
        V3D pw(_RDisp * pl + _tDisp);
        curCloudRGBPtr->points[i].x = pw[0];
        curCloudRGBPtr->points[i].y = pw[1];
        curCloudRGBPtr->points[i].z = pw[2];
        // double ratioR=(min(0.01,_matchedSurfListDisp[i].R)-0)/(0.01-0);
        //  double ratioR=(min(0.1*RMax,_matchedSurfListDisp[i].R)-RMin)/(0.1*RMax-RMin);
        // std::cout<<_matchedSurfListDisp[i].R<<std::endl;
        double ratioR = (_matchedSurfListDisp[i].R - RMin) / (RMax - RMin);
        curCloudRGBPtr->points[i].r = 255 * ratioR;
        curCloudRGBPtr->points[i].g = 255 * (1 - ratioR);
        curCloudRGBPtr->points[i].b = 0;
    }

    Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
    gravAlignMat.block<3,3>(0,0) = _rotAlign;
    pcl::transformPointCloud(*curCloudRGBPtr, *curCloudRGBPtr, gravAlignMat);

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbCloudHandler(curCloudRGBPtr);
    _p->addPointCloud<pcl::PointXYZRGB>(curCloudRGBPtr, rgbCloudHandler, "match");
    _p->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, "match");
}

void System::show()
{
    //return;
    _p = new pcl::visualization::PCLVisualizer("LI_init voxel mapping visualization");
    _p->createViewPort(0.0, 0, 1.0, 1.0, _vp1);
    //_p->setBackgroundColor(0.15f, 0.15f, 0.15f);
    _p->setBackgroundColor(1.f, 1.f, 1.f);
    _p->setCameraPosition(0, 0, 100, 0, 0, 0, -0.7, -0.9, 0.2);
    // _p->setBackgroundColor(1, 1, 1);
    _p->addCoordinateSystem(1., Eigen::Affine3f(), "origin", _vp1);
    _p->addCoordinateSystem(0.5, Eigen::Affine3f(), "lidar", _vp1);
#ifdef WIN32
    _p->getRenderWindow()->GlobalWarningDisplayOff();
#endif
    while (1)
    {
        if (_isResetShow)
        {
            _p->removeAllPointClouds(_vp1);
            _isResetShow = false;
        }

        if (_isDispStop)
            break;
        // std::cout<<"_localCloudDownQueue"<<_localCloudDownQueue.size()<<std::endl;
        _dispMutex.lock();
        if (_localCloudDownQueue.empty())
        {
            _dispMutex.unlock();
            _p->spinOnce(2);
            continue;
        }

        assert(_localCloudDownQueue.size() == _localCloudQueue.size() &&
               _localCloudDownQueue.size() == _RQueue.size() && _localCloudDownQueue.size() == _tQueue.size() &&
               _localCloudDownQueue.size() == _motorAngleQueue.size()); //&&
               //_localCloudDownQueue.size() == _downCloudMapQueue.size());
//        if(_localCloudDownQueue.size()>10)
//        {
//            clearQueue(_localCloudQueue);
//            clearQueue(_localCloudDownQueue);
//            clearQueue(_localPlaneDownQueue);
//            clearQueue(_localLineDownQueue);
//            clearQueue(_downCloudMapQueue);
//            clearQueue(_stateQueue);
//            clearQueue(_statePropagatQueue);
//            clearQueue(_RQueue);
//            clearQueue(_tQueue);
//            clearQueue(_motorAngleQueue);
//            clearQueue(_voxelCloudQueue);
//            clearQueue(_matchedSurfListQueue);
//        _dispMutex.unlock();
//        continue;
//         }

//        while(_localCloudQueue.size()>1)
//            _localCloudQueue.pop();
//        while(_localCloudDownQueue.size()>1)
//            _localCloudDownQueue.pop();
//        while(_localPlaneDownQueue.size()>1)
//            _localPlaneDownQueue.pop();
//        while(_localLineDownQueue.size()>1)
//            _localLineDownQueue.pop();
//        while(_downCloudMapQueue.size()>1)
//            _downCloudMapQueue.pop();
//        while(_stateQueue.size()>1)
//            _stateQueue.pop();
//        while(_statePropagatQueue.size()>1)
//            _statePropagatQueue.pop();
//        while(_RQueue.size()>1)
//            _RQueue.pop();
//        while(_tQueue.size()>1)
//            _tQueue.pop();
//        while(_motorAngleQueue.size()>1)
//            _motorAngleQueue.pop();
//        while(_voxelCloudQueue.size()>1)
//            _voxelCloudQueue.pop();
//        while(_matchedSurfListQueue.size()>1)
//            _matchedSurfListQueue.pop();

        _localCloudDownDisp = _localCloudDownQueue.front();
        _localCloudDisp = _localCloudQueue.front();
        _matchedSurfListDisp = _matchedSurfListQueue.front();
        _stateDisp=_stateQueue.front();
        _statePropagatDisp=_statePropagatQueue.front();
        _RDisp = _RQueue.front();
        _tDisp = _tQueue.front();
        _motorAngleDisp = _motorAngleQueue.front();

        if(!_downCloudMapQueue.empty())
            _downCloudMapDisp = _downCloudMapQueue.front();
        if(!_voxelCloudQueue.empty())
            _voxelMapCloudDisp = _voxelCloudQueue.front();
        if(_loopCloser)
            _lcIdxPairsDisp = _lcIdxPairsQueue.front();
        if (!_localPlaneDownQueue.empty())
        {
            _localPlaneDownDisp = _localPlaneDownQueue.front();
            _localPlaneDownQueue.pop();
        }
        if (!_localLineDownQueue.empty())
        {
            _localLineDownDisp = _localLineDownQueue.front();
            _localLineDownQueue.pop();
        }

        _localCloudDownQueue.pop();
        _localCloudQueue.pop();
        _stateQueue.pop();
        _statePropagatQueue.pop();
        _RQueue.pop();
        _tQueue.pop();
        _motorAngleQueue.pop();
        if(!_downCloudMapQueue.empty())
            _downCloudMapQueue.pop();
        if(!_voxelCloudQueue.empty())
            _voxelCloudQueue.pop();
        _matchedSurfListQueue.pop();
        if(_loopCloser)
            _lcIdxPairsQueue.pop();

        _dispMutex.unlock();

        updateTrajectory();
        if(_frameIdDisp!=_frameId)
        {
            //if(_frameId%50==0||_addedPoints.size()>5000)//||_localCloudQueue.empty())
            showKdTree(_vp1);
            if(_config._isLoopEn)
                showLoopClosure(_vp1);
            //showVoxelMap(_vp1);
            //if(_frameId%2==0&&_localCloudQueue.empty())
            showLocalDownCloud(_vp1);
            showLocalFeatCloud(_vp1);
            showTrajectory(_vp1);
            //showMatchedPoints(_vp1);
            //        if(_isMotorInitialized)
            //            showDenseMapCloud(_vp1);
        }

        _frameIdDisp=_frameId;
        _p->spinOnce(2);
        // std::chrono::milliseconds dura(2);
        // std::this_thread::sleep_for(dura);
    }
}
