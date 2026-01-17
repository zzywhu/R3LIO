//
// Created by w on 2022/10/19.
//

#ifndef FAST_LIO_VIEWER_H
#define FAST_LIO_VIEWER_H

#include <thread>
#include <deque>
#include <mutex>

#include "DrawerPCL.h"
#include "Common.h"
#include "VoxelMapper.h"

#ifdef SLAM_SHARED_LIBS
#ifdef SLAM_TOOL_EXPORTS
#define VIEWER_API  __declspec(dllexport)
#else
#define VIEWER_API  __declspec(dllimport)
#endif
#else
#define VIEWER_API
#endif


class VIEWER_API Viewer
{
public:
    Viewer() :
            _drawerPCL(nullptr),
            _isFinishRequested(false),
            _isStopRequested(false),
            _isStopped(true),
            _viewThread(nullptr)
    {}
    ~Viewer()
    {}

    void setDrawerPcl(DrawerPCL* const  drawerPcl) { _drawerPCL = drawerPcl; }
    DrawerPCL* getDrawerPcl() { return _drawerPCL; }

    void setResetShow(const bool& flag ){_isResetShow=flag;}

    std::thread* getViewThreadPtr() { return _viewThread; }

    bool startThread()
    {
        _viewThread = new std::thread(&Viewer::run, this);
        if (!_viewThread)
            return false;
        return true;
    }

    void updateViewer();

    void run();

    bool popBuffer();

    void requestFinish()
    {
        _isFinishRequested = true;
    }

    bool checkFinish()
    {
        return _isFinishRequested;
    }

    void setFinish(){ _isFinished = true;}

    bool isFinished(){return _isFinished;}

    void requestStop()
    {
        if (!_isStopped)
            _isStopRequested = true;
    }

    bool isStopped(){return _isStopped;}

    bool stop()
    {
        if (_isFinishRequested)
            return false;
        else if (_isStopRequested)
        {
            _isStopped = true;
            _isStopRequested = false;
            return true;
        }

        return false;

    }

    void release()
    {
        _isStopped = false;
        _isStopRequested = false;
    }


private:

    DrawerPCL*				                                    _drawerPCL;

    bool                                                        _isResetShow;

    bool					                                    _isFinishRequested;
    bool					                                    _isFinished;

    bool					                                    _isStopped;
    bool					                                    _isStopRequested;

    std::thread*			                                    _viewThread;

#ifdef USE_PANGOLIN
    pangolin::Handler*	                                        _handler;
#endif

    std::queue<PointCloudXYZI::Ptr>                             _localCloudQueue;
    std::queue<PointCloudXYZI::Ptr>                             _localCloudDownQueue;
    std::queue<PointCloudXYZI::Ptr>                             _localPlaneDownQueue;
    std::queue<PointCloudXYZI::Ptr>                             _localLineDownQueue;
    std::queue<PointCloudXYZI::Ptr>                             _downCloudMapQueue;
    std::queue<Eigen::Matrix3d>                                 _RQueue;
    std::queue<Eigen::Vector3d>                                 _tQueue;
    std::queue<double>                                          _motorAngleQueue;
    std::queue<std::unordered_map<VOXEL_LOC, OctoTree *>>       _voxelMapQueue;
    std::queue<std::vector<MatchOctoTreeInfo>>                      _matchedSurfListQueue;

    std::unordered_map<VOXEL_LOC, OctoTree *>                   _voxelMapDisp;
    PointCloudXYZI::Ptr                                         _localCloudDisp;
    PointCloudXYZI::Ptr                                         _localCloudDownDisp;
    PointCloudXYZI::Ptr                                         _localPlaneDownDisp;
    PointCloudXYZI::Ptr                                         _localLineDownDisp;
    PointCloudXYZI::Ptr                                         _downCloudMapDisp;
    Eigen::Matrix3d                                             _RDisp;
    Eigen::Vector3d                                             _tDisp;
    double                                                      _motorAngleDisp;
    std::vector<MatchOctoTreeInfo>                                  _matchedSurfListDisp;

    PointCloudXYZI::Ptr                                         _trajPts;
    std::vector<Colour>                                         _voxelColors;

    std::mutex                                                  _dispMutex;

};


#endif //FAST_LIO_VIEWER_H
