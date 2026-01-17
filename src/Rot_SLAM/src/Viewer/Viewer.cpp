//
// Created by w on 2022/10/19.
//
#include "Viewer.h"


void Viewer::updateViewer()
{
    _dispMutex.lock();
    _dispMutex.unlock();
}

bool Viewer::popBuffer()
{
    _dispMutex.lock();
    assert(_localCloudDownQueue.size() == _localCloudQueue.size() &&
           _localCloudDownQueue.size() == _RQueue.size() && _localCloudDownQueue.size() == _tQueue.size() &&
           _localCloudDownQueue.size() == _motorAngleQueue.size() &&
           _localCloudDownQueue.size() == _downCloudMapQueue.size());

    if (_localCloudDownQueue.empty())
    {
        _dispMutex.unlock();
        return false;
    }

    _localCloudDownDisp = _localCloudDownQueue.front();
    _localCloudDisp = _localCloudQueue.front();
    _matchedSurfListDisp = _matchedSurfListQueue.back();
    _RDisp = _RQueue.front();
    _tDisp = _tQueue.front();
    _motorAngleDisp = _motorAngleQueue.front();
    _downCloudMapDisp = _downCloudMapQueue.front();
    _voxelMapDisp = _voxelMapQueue.front();
    if(!_localPlaneDownQueue.empty())
    {
        _localPlaneDownDisp = _localPlaneDownQueue.front();
        _localPlaneDownQueue.pop();
    }
    if(!_localLineDownQueue.empty())
    {
        _localLineDownDisp = _localLineDownQueue.front();
        _localLineDownQueue.pop();
    }

    _localCloudDownQueue.pop();
    _localCloudQueue.pop();
    _RQueue.pop();
    _tQueue.pop();
    _motorAngleQueue.pop();
    _downCloudMapQueue.pop();
    _voxelMapQueue.pop();
    _matchedSurfListQueue.pop();

    _dispMutex.unlock();
    return true;

}

void Viewer::run()
{
#ifdef USE_PANGOLIN
    pangolin::CreateWindowAndBind("Tracker: Map Viewer", 1024, 768);

        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGl we might need
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show Submap", true, true);
        pangolin::Var<bool> menuShowOdomTraj("menu.Show Odom Traj", true, true);
        pangolin::Var<bool> menuShowLines("menu.Show Lines", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Planes", true, true);

        //pangolin::Var<bool> menu3DPick("menu.Pick mode", false, true);

        //////////////// Define Camera Render Object (for view / scene browsing)////////////////
        pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1024, 768, _viewpointF, _viewpointF, 512, 389, 0.1, 1000),
                                          pangolin::ModelViewLookAt(_viewpointX, _viewpointY, _viewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

        ////////////////Add named OpenGL viewport to window and provide 3D Handler////////////////
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        bool bFollow = true;
        while (1)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            if(popBuffer())
            {}


            pangolin::FinishFrame();


            if (stop())
            {
                while (isStopped())
                {
                    usleep(3000);
                }
            }

            if (checkFinish())
                break;
        }

        setFinish();
#else
    using pcl::visualization::PointCloudColorHandlerCustom;
    int vp1=0;
    pcl::visualization::PCLVisualizer * p = new pcl::visualization::PCLVisualizer("LI_init voxel mapping visualization");
    p->createViewPort(0.0, 0, 1.0, 1.0, vp1);
    p->setBackgroundColor(0.15f, 0.15f, 0.15f);

    p->addCoordinateSystem(1., Eigen::Affine3f(), "origin", vp1);
    p->addCoordinateSystem(0.5, Eigen::Affine3f(), "lidar", vp1);

    while (1)
    {
        if(popBuffer())
        {
            //TODO:show
        }

        p->spinOnce(2);
        // std::chrono::milliseconds dura(2);
        // std::this_thread::sleep_for(dura);
        // std::cout<<"_localCloudDownQueue"<<_localCloudDownQueue.size()<<std::endl;

        if (_isResetShow)
        {
            p->removeAllPointClouds(vp1);
            _isResetShow = false;
        }

        if (stop())
            while (isStopped())
                usleep(3000);

        if (checkFinish())
            break;

    }
#endif //USE_PANGOLIN

}