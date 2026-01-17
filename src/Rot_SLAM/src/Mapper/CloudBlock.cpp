//
// Created by w on 2022/9/14.
//

#include "CloudBlock.h"


void CloudBlock::init()
{
    _timestamp=-1;
    _lastFrameId=0;
    _uniqueId=0;		  //Unique ID
    _idInStrip=0;	  //ID in the strip

    _stationPositionAvailable=false; //If the approximat
    _poseFixed = false;  //the pose is fixed or not
    _poseStable = false; //the pose is stable or no
    _poseLo.setIdentity();
    _poseOptimized.setIdentity();
    _poseInit.setIdentity();
    _informationMatrixToNext.setIdentity();
    _localStationPose.setIdentity(); //Station pose i
    _pcRaw.reset(new PointCloudXYZI);
    //_pcRawWorld.reset(new PointCloudXYZI);
    _pcDown.reset(new PointCloudXYZI);
    //_pcDownWorld.reset(new PointCloudXYZI);
    _pcDownAligned.reset(new PointCloudXYZI);
}


void CloudBlock::freeRawCloud()
{
    _pcRaw.reset(new PointCloudXYZI());
    _pcDown.reset(new PointCloudXYZI());
    //_pcDownWorld.reset(new PointCloudXYZI());
    _pcDownAligned.reset(new PointCloudXYZI());
}

void CloudBlock::freeAll()
{
    freeRawCloud();
    //_pvList.clear();
    std::vector<PointWithCov>().swap(_pvList);
}

void CloudBlock::cloneMetadata(const CloudBlock &inBlock)
{
    _bound = inBlock._bound;
    _localBound = inBlock._localBound;
    _localCenter = inBlock._localCenter;
    _poseLo = inBlock._poseLo;
    _poseInit = inBlock._poseInit;
    _poseOptimized = inBlock._poseOptimized;
    _gravity=inBlock._gravity;
    _timestamp = inBlock._timestamp;
    _uniqueId = inBlock._uniqueId;
    _lastFrameId = inBlock._lastFrameId;
    _idInStrip = inBlock._idInStrip;
    *_pcDown = *(inBlock._pcDown);
    *_pcDownAligned = *(inBlock._pcDownAligned);
    //*_pcDownWorld = *(inBlock._pcDownWorld);
    *_pcRaw = *(inBlock._pcRaw);
    //*_pcRawWorld = *(inBlock._pcRawWorld);
    _pvList=inBlock._pvList;
    _imuDataList=inBlock._imuDataList;
    _pcdFilePath=inBlock._pcdFilePath;
}

void CloudBlock::cloneCloud(PointCloudXYZI::Ptr &pcOut, bool getPcDone)
{
    if (getPcDone)
    {
        //pcOut->points.insert(pcOut->points.end(), _pcDown->points.begin(), _pcDown->points.end());
        *pcOut+=*_pcDown;
    }
    else
        //pcOut->points.insert(pcOut->points.end(), _pcRaw->points.begin(), _pcRaw->points.end());
        *pcOut+=*_pcRaw;
}

void CloudBlock::appendFeature(const CloudBlock &inBlock)
{
    //*_pcRaw+=*inBlock._pcRaw;
    //*_pcDown+=*inBlock._pcDown;
    //_pcRaw->points.insert(_pcRaw->points.end(), inBlock._pcRaw->points.begin(), inBlock._pcRaw->points.end());
    _pcDown->points.insert(_pcDown->points.end(), inBlock._pcDown->points.begin(), inBlock._pcDown->points.end());
    _pvList.insert(_pvList.end(), inBlock._pvList.begin(), inBlock._pvList.end());
}

void CloudBlock::transformFeature(const Eigen::Matrix4d &transMat)
{
    pcl::transformPointCloud(*_pcDown, *_pcDown, transMat);
   // pcl::transformPointCloud(*_pcRaw, *_pcRaw, transMat);
    for(auto& pv:_pvList)
    {
        pv.pl=transMat.block<3,3>(0,0)*pv.pl+transMat.block<3,1>(0,3);
        pv.pi=transMat.block<3,3>(0,0)*pv.pi+transMat.block<3,1>(0,3);
    }

}
