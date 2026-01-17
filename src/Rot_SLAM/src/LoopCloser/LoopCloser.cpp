//
// Created by w on 2022/9/15.
//

#include "LoopCloser/LoopCloser.h"
#include "Mapper/VoxelVGICP.hpp"
#include "Registrator/LsqRegPCL_impl.hpp"
#include "Registrator/fast_gicp/gicp/fast_gicp.hpp"
#include "Registrator/fast_gicp/gicp/fast_vgicp.hpp"
#include "Registrator/fast_gicp/gicp/impl/fast_gicp_impl.hpp"
#include "Registrator/fast_gicp/gicp/impl/fast_vgicp_impl.hpp"
#include "Registrator/fast_gicp/gicp/impl/lsq_registration_impl.hpp"
#include "So3Math.h"
#include "plog/Log.h"
using namespace std;

LoopCloser::Config LoopCloser::_config;

bool LoopCloser::startThread()
{
    _loopClosingThread = new std::thread(&LoopCloser::run, this);
    if (!_loopClosingThread)
        return false;
    return true;
}

void LoopCloser::release()
{
    std::unique_lock<std::mutex> lock1(_mutexStop);
    std::unique_lock<std::mutex> lock2(_mutexFinish);
    if (_bFinished)
        return;
    _bStopped = false;
    _bStopRequested = false;

    if(!_frameBuffer.empty())
        _frameBuffer.clear();

    std::cout << "Local Mapping released" << std::endl;
}

void LoopCloser::loadDetectorConfig(const std::string &configPath)
{

   // printf("Loading loop test parameters...\n");
    auto yl = yamlLoader(configPath);

    double corr_thres;

    yl.loadOneConfig({"correlation_thres"}, corr_thres);
    _ptrEvaluator = std::make_unique<ContLCDEvaluator>(corr_thres);

    yl.loadOneConfig({"ContourDBConfig", "nnk_"}, _dbConfig.nnk_);
    yl.loadOneConfig({"ContourDBConfig", "max_fine_opt_"}, _dbConfig.max_fine_opt_);
    yl.loadSeqConfig({"ContourDBConfig", "q_levels_"}, _dbConfig.q_levels_);

    yl.loadOneConfig({"ContourDBConfig", "TreeBucketConfig", "max_elapse_"}, _dbConfig.tb_cfg_.max_elapse_);
    yl.loadOneConfig({"ContourDBConfig", "TreeBucketConfig", "min_elapse_"}, _dbConfig.tb_cfg_.min_elapse_);

    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_cell_cnt"}, _dbConfig.cont_sim_cfg_.ta_cell_cnt);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_cell_cnt"}, _dbConfig.cont_sim_cfg_.tp_cell_cnt);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_eigval"}, _dbConfig.cont_sim_cfg_.tp_eigval);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_h_bar"}, _dbConfig.cont_sim_cfg_.ta_h_bar);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_rcom"}, _dbConfig.cont_sim_cfg_.ta_rcom);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_rcom"}, _dbConfig.cont_sim_cfg_.tp_rcom);
    _ptrContourDB = std::make_unique<ContourDB>(_dbConfig);

    yl.loadOneConfig({"thres_lb_", "i_ovlp_sum"}, _thresLb.sim_constell.i_ovlp_sum);
    yl.loadOneConfig({"thres_lb_", "i_ovlp_max_one"}, _thresLb.sim_constell.i_ovlp_max_one);
    yl.loadOneConfig({"thres_lb_", "i_in_ang_rng"}, _thresLb.sim_constell.i_in_ang_rng);
    yl.loadOneConfig({"thres_lb_", "i_indiv_sim"}, _thresLb.sim_pair.i_indiv_sim);
    yl.loadOneConfig({"thres_lb_", "i_orie_sim"}, _thresLb.sim_pair.i_orie_sim);
    yl.loadOneConfig({"thres_lb_", "correlation"}, _thresLb.sim_post.correlation);
    yl.loadOneConfig({"thres_lb_", "area_perc"}, _thresLb.sim_post.area_perc);
    yl.loadOneConfig({"thres_lb_", "neg_est_dist"}, _thresLb.sim_post.neg_est_dist);

    yl.loadOneConfig({"thres_ub_", "i_ovlp_sum"}, _thresUb.sim_constell.i_ovlp_sum);
    yl.loadOneConfig({"thres_ub_", "i_ovlp_max_one"}, _thresUb.sim_constell.i_ovlp_max_one);
    yl.loadOneConfig({"thres_ub_", "i_in_ang_rng"}, _thresUb.sim_constell.i_in_ang_rng);
    yl.loadOneConfig({"thres_ub_", "i_indiv_sim"}, _thresUb.sim_pair.i_indiv_sim);
    yl.loadOneConfig({"thres_ub_", "i_orie_sim"}, _thresUb.sim_pair.i_orie_sim);
    yl.loadOneConfig({"thres_ub_", "correlation"}, _thresUb.sim_post.correlation);
    yl.loadOneConfig({"thres_ub_", "area_perc"}, _thresUb.sim_post.area_perc);
    yl.loadOneConfig({"thres_ub_", "neg_est_dist"}, _thresUb.sim_post.neg_est_dist);

    yl.loadSeqConfig({"ContourManagerConfig", "lv_grads_"}, _cmConfig.lv_grads_);
    yl.loadOneConfig({"ContourManagerConfig", "reso_row_"}, _cmConfig.reso_row_);
    yl.loadOneConfig({"ContourManagerConfig", "reso_col_"}, _cmConfig.reso_col_);
    yl.loadOneConfig({"ContourManagerConfig", "n_row_"}, _cmConfig.n_row_);
    yl.loadOneConfig({"ContourManagerConfig", "n_col_"}, _cmConfig.n_col_);
    yl.loadOneConfig({"ContourManagerConfig", "lidar_height_"}, _cmConfig.lidar_height_);
    yl.loadOneConfig({"ContourManagerConfig", "blind_sq_"}, _cmConfig.blind_sq_);
    yl.loadOneConfig({"ContourManagerConfig", "min_cont_key_cnt_"}, _cmConfig.min_cont_key_cnt_);
    yl.loadOneConfig({"ContourManagerConfig", "min_cont_cell_cnt_"}, _cmConfig.min_cont_cell_cnt_);

    yl.close();
}

void LoopCloser::inputGNSSData(SensorMsgs::GNSSData::Ptr gnssData, const int& frameID)
{
    _gnssMap[frameID] = gnssData;
    _gnssMapBuffer[frameID] = gnssData;
}

void LoopCloser::inputFrameData(int frameId, const double& timeStamp,
                                const M3D& rot, const V3D& pos,
                                PointCloudXYZI::Ptr localCloud,
                                PointCloudXYZI::Ptr localCloudDown,
                                std::vector<PointWithCov>& pvList,
                                std::vector<IMUData>& imuDataList,
                                bool isNormalizeIntensity,
                                V3D gravity)
{
    _mutexBuf.lock();

    //std::cout<<"LoopCloser input lidar frame:"<<frameId<<std::endl;
    //TicToc time;

    CloudBlockPtr frameBlockPtr(new CloudBlock);
    frameBlockPtr->_timestamp=timeStamp;
    frameBlockPtr->_uniqueId=frameId;
    frameBlockPtr->_pcRaw=localCloud->makeShared(); //frame cloud in IMU coordinate
    frameBlockPtr->_pcDown=localCloudDown->makeShared(); //frame downsampled cloud in IMU coordinate
    frameBlockPtr->_pvList=pvList;
    frameBlockPtr->_poseLo.block<3,3>(0, 0)=rot;
    frameBlockPtr->_poseLo.block<3,1>(0, 3)=pos;
    frameBlockPtr->_poseInit= frameBlockPtr->_poseLo;
    frameBlockPtr->_gravity=gravity;
    frameBlockPtr->_imuDataList=imuDataList;

    if (isNormalizeIntensity)
    {
        float min_intensity = FLT_MAX;
        float max_intensity = -FLT_MAX;
        for (int i = 0; i < frameBlockPtr->_pcDown->points.size(); i++)
        {
            min_intensity = min_(min_intensity, frameBlockPtr->_pcDown->points[i].intensity);
            max_intensity = max_(max_intensity, frameBlockPtr->_pcDown->points[i].intensity);
        }
        float intesnity_scale = 255.0 / (max_intensity - min_intensity); //rescale to 0-255
        for (int i = 0; i < frameBlockPtr->_pcDown->points.size(); i++)
            frameBlockPtr->_pcDown->points[i].intensity = (frameBlockPtr->_pcDown->points[i].intensity - min_intensity) * intesnity_scale;
    }

    _frameBuffer.push_back(frameBlockPtr);
    //printf("cloud input closer time: %f ms\n", time.toc());
    _mutexBuf.unlock();
}

void LoopCloser::storeFrameData()
{
    TicToc time;

    _curFrameBlock->_idInStrip=_frameBlocks.size();
    _curFrameBlock->_lastFrameId=_frameBlocks.size();
    if(LoopCloser::mutableConfig()._issavemap)
    {
        //std::cout<<std::string(ROOT_DIR)+"PCD/"+std::to_string(_curFrameBlock->_uniqueId)+".pcd"<<std::endl;
        _curFrameBlock->_pcdFilePath=std::string(ROOT_DIR)+"PCD/"+std::to_string(_curFrameBlock->_uniqueId)+".pcd";
        _pcdWriter.writeBinary(_curFrameBlock->_pcdFilePath, *_curFrameBlock->_pcRaw);
    }
    //add to gtsam graph
    if(!_frameBlocks.empty())
    {
        Eigen::Matrix4d relPose = _frameBlocks.back()->_poseLo.inverse() * _curFrameBlock->_poseLo;//T12=T1w*Tw2
        _relPoses.push_back(relPose);
    }

    _frameBlocks.emplace_back(new CloudBlock(*_curFrameBlock,true));
    _frameIDMap[_curFrameBlock->_uniqueId]=_curFrameBlock->_idInStrip;// for indexing frame in frameblocks by frame id

    if(config()._isPgoIncremental)
    {//PGO incremental for a existed global pose graph
        if (config()._isFramewisePGO)
        {
            if(_frameBlocks.size() == 1)
                _pgOptimizer.addFixedConstraint(_curFrameBlock);
            else
            { //Add adjacent frame Edges to gtsam graph
                constraints adjCons;
                _confinder.add_adjacent_constraint(_frameBlocks, adjCons, _frameBlocks.size());
                _pgOptimizer.addEdgeConstraint(adjCons[0]);
            }
        }
    }

    //save memory
    _frameBlocks.back()->_pcRaw.reset(new PointCloudXYZI);// stored in file
    if(!config()._isFramewisePGO&&!config()._isPgoIncremental)
    {//no need to store cloud for reconstructing map after submap PGO
        std::vector<PointWithCov>().swap(_frameBlocks.back()->_pvList);
        _frameBlocks.back()->_pcDown.reset(new PointCloudXYZI());
    }
    else
    {//no need to store cloud outside frontend window for reconstructing map after frame PGO
        int idx=_frameBlocks.size()-config()._frontendWinSize-1;
        if(idx>=0)
        {
            std::vector<PointWithCov>().swap(_frameBlocks[idx]->_pvList);
            if(!config()._isOutputPCD)
                _frameBlocks[idx]->_pcDown.reset(new PointCloudXYZI());
        }
    }

    // printf("store frame time: %f ms\n", time.toc());
}

bool LoopCloser::detectLoop()
{
    //TicToc time;
    bool isDetected = false;
    if(_isNewSubMap)//first cloud block input
    {
        //_curSubMapBlock = boost::make_shared<CloudBlock>() ;
        _curSubMapBlock->freeAll();
        _curSubMapBlock->_timestamp=_curFrameBlock->_timestamp;
        _curSubMapBlock->_poseLo=_curFrameBlock->_poseLo;
        _curSubMapBlock->_gravity=_curFrameBlock->_gravity;
        _curSubMapBlock->_uniqueId=_frameBlocks.size();
        _isNewSubMap=false;
        //std::cout<<"Loop trans:"<<_loopTrans<<std::endl;
        if(_config._isFramewisePGO||_config._isStoreFrameBlock)
            storeFrameData(); //Force to store when is the first frame of a new submap

        if(config()._isPgoIncremental)
        {//PGO incremental for a existed global pose graph
            if (!config()._isFramewisePGO)
            {
                _keyframeBlocks.emplace_back(_curFrameBlock);
                if (_keyframeBlocks.size() == 1)
                    _pgOptimizer.addFixedConstraint(_curFrameBlock);
                else
                { //Add adjacent frame Edges to gtsam graph
                    constraints adjCons;
                    _confinder.add_adjacent_constraint(_keyframeBlocks, adjCons, _keyframeBlocks.size());
                    _pgOptimizer.addEdgeConstraint(adjCons[0], false);
                }
            }
        }
    }

    _blockMapManager.updateLocalMap(_curSubMapBlock, _curFrameBlock);

    if(_config._isFramewisePGO||_config._isStoreFrameBlock)
        if(_curFrameBlock->_uniqueId%config()._frameStoreInterval==0)
            if(_frameBlocks.back()->_uniqueId!=_curFrameBlock->_uniqueId)
                storeFrameData();

    _curSubMapBlock->_lastFrameId=_frameBlocks.size()-1;//_curFrameBlock->_uniqueId;
    _nAccuFrames+= 1;
    if(_lastFrameBlock)
    {  //update accumulated information
        //_accuTran += calTranslationFromTranmat(relPose);
        _accuTran = (_curFrameBlock->_poseLo.inverse() * _curSubMapBlock->_poseLo).block<3, 1>(0, 3).norm();
        Eigen::Matrix4d relPose = _lastFrameBlock->_poseLo.inverse() * _curFrameBlock->_poseLo;
        _accuRotDeg += calRotationDegFromTranmat(relPose);
    }

    //printf("Update local map time: %f ms\n", time.toc());
    if(_blockMapManager.judgeNewSubmap(_accuTran, _accuRotDeg, _nAccuFrames,
                                       _config._maxSubMapAccuTran,_config._maxSubMapAccuRot,_config._maxSubMapAccuFrames))
    {
        _isNewSubMap=true;
        //time.tic();
        { //down sample submap
            _voxelFilter.setInputCloud(_curSubMapBlock->_pcDown->makeShared());
            _voxelFilter.filter(*_curSubMapBlock->_pcDown); // used for registration while finding loop

            if(config()._isFramewisePGO) // no need to store _pvList for reconstructing map after submap PGO
                std::vector<PointWithCov>().swap(_curSubMapBlock->_pvList);
            else
                downSamplingVoxel(_curSubMapBlock->_pvList,0.1);

            _curSubMapBlock->_pcDown->points.shrink_to_fit();
            _curSubMapBlock->_pvList.shrink_to_fit();
        }
        std::cout<< "Create new submap [" << _subMapBlocks.size() << "]"<<std::endl;
        CloudBlockPtr subMapBlock = boost::make_shared<CloudBlock>(*_curSubMapBlock);
        subMapBlock->_idInStrip = _subMapBlocks.size();
        subMapBlock->_poseInit = subMapBlock->_poseLo;
        if (_subMapBlocks.empty())
            subMapBlock->_poseFixed = true; //fixed the first submap

        if(_config._isNeedGravAligned)
        {
            Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
            gravAlignMat.block<3,3>(0,0) = Eigen::Quaterniond::FromTwoVectors(subMapBlock->_gravity, Eigen::Vector3d {0, 0, -1.0}).toRotationMatrix();
            Eigen::Matrix4d gravCorrPose = gravAlignMat*subMapBlock->_poseLo;
            gravCorrPose.block<3,1>(0,3)=Eigen::Vector3d(0,0,0);
            pcl::transformPointCloud(*subMapBlock->_pcDown, *subMapBlock->_pcDownAligned, gravCorrPose);
            //pcl::transformPointCloud(*subMapBlock->_pcDown, *subMapBlock->_pcDownAligned, subMapBlock->_poseLo);
        }


#if SAVE_MID_FILE
        {  //frame cloud in IMU coordinate
            //PointCloudXYZI::Ptr globalDownCloud(new PointCloudXYZI());
            //pcl::transformPointCloud(*subMapBlock->_pcDown, *globalDownCloud, subMapBlock->_poseLo);
            //_pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/block" + std::to_string(subMapBlock->_idInStrip) + ".pcd", *globalDownCloud);
            //_pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/local_block" + std::to_string(subMapBlock->_idInStrip) + ".pcd", *subMapBlock->_pcDownAligned);
            //_pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/block" + std::to_string(subMapBlock->_idInStrip) + ".pcd", *subMapBlock->_pcRawWorld);
        }
#endif
        _subMapBlocks.push_back(subMapBlock);
        _coolingIndex--;
        
        
        //delete the past registration edges (which are not reliable)
        if(_config._isCorrectRealTime)
            _confinder.cancel_registration_constraint(_pgoEdges, _config._minConstraintConfidence, _config._maxConstraintSigma);
        //add adjacent edge between current submap and the last submap
        _confinder.add_adjacent_constraint(_subMapBlocks, _pgoEdges, _subMapBlocks.size());
        //printf("Create new sub map time: %f ms\n", time.toc());
        

        if (_coolingIndex > 0) //find registration edges and then do pgo
            std::cout<<"No need to detect loop for cooling index: "<<_coolingIndex<<std::endl;
        else
        {
            _config._detectLoopMethod=0;
            _isDetected=false;
            if (_config._isLoopDetectEn)
            {
                if(_config._detectLoopMethod==0){
                    isDetected=detectLoopNN(_subMapBlocks.back(), _subMapBlocks);
                    }
                else if(_config._detectLoopMethod==1)// detect loop closure by contour context method
                    isDetected=detectLoopContourContext(_subMapBlocks.back());
                
            }
        }
    }
    return isDetected;
}

void LoopCloser::run()
{
    _bFinished = false;
    _voxelFilter.setLeafSize(_config._voxelDownSize,_config._voxelDownSize,_config._voxelDownSize);
    while (1)
    {
        _mutexBuf.lock();
        //TicToc time;
        bool hasFrameData=false;
        if(!_frameBuffer.empty() && !_isUpdated) // Only can _frameBuffer pop out while existed blocks are contributed into voxelmap in main thread
        {
            if(!_config._isStoreFrameBlock&&_lastFrameBlock)
                _lastFrameBlock->freeAll();
            _lastFrameBlock=_curFrameBlock;
            _curFrameBlock=_frameBuffer.front();
            _frameBuffer.erase(_frameBuffer.begin());

            //_curFrameBlock->_poseLo=_loopTrans*_curFrameBlock->_poseLo;
            //_cloudUtility.get_cloud_bbx_cpt(_curFrameBlock->_pcDown, _curFrameBlock->_localBound, _curFrameBlock->_localCenter);
            _nAccuOptFrames++;
            hasFrameData=true;
        }
        _mutexBuf.unlock();
        if(hasFrameData)
        {
            _mutexProcess.lock();

            bool isFindLoop=detectLoop();//found loop closure
            bool isGNSSProcessed=processGNSS();

            if(isFindLoop||isGNSSProcessed)
            {
                if(_config._isCorrectRealTime)
                {// correct realtime
                    if(config()._isFramewisePGO)
                        if(config()._isPgoIncremental)
                            frameGraphOptimize();
                        else
                            correctFrameLoop();
                    else
                        if(config()._isPgoIncremental)
                            submapGraphOptimize();
                        else
                            correctSubmapLoop();

                    if(isGNSSProcessed)
                        printGNSSError();
                }
                else if(isGNSSProcessed)
                {// not realtime loop mode, while gnss input, force to correct realtime
                    printGNSSError();
                    if(config()._isFramewisePGO)
                        if(config()._isPgoIncremental)
                            frameGraphOptimize();
                        else
                            correctFrameLoop();
                    else
                        if(config()._isPgoIncremental)
                            submapGraphOptimize();
                        else
                            std::cerr<<"Cannot support this global optimization mode"<<std::endl;
                    correctFrontendMap(getFrameBlocks());

                    printGNSSError();
                }

                // _coolingIndex = std::max(_frameBuffer.size(), (ulong)_config._coolingSubmapNum); // reset loop test cooling count
                _coolingIndex =  _config._coolingSubmapNum; // wating submap count for detecting next loop
                _nAccuOptFrames = 0; // clear accumlated frames count
                _isDetected=true;
                _loopCount++;
            }

            _mutexProcess.unlock();
        }

        if (_frameBuffer.empty() && stop())
        {
            while (isStopped() && !checkFinish())
                usleep(3000);
            if (checkFinish())
                break;
        }
        if (checkFinish())
            break;
        usleep(3000);
    }
    setFinish();
}

bool LoopCloser::detectLoopNN(const CloudBlockPtr block, CloudBlockPtrs &subBlocks)
{
    TicToc time;
    bool isOverallLoopSearchingOn = false;
    int regEdgeCount = 0;
    //calculate bbx (local)
    CFilter<PointType> cf;
    cf.get_cloud_bbx(block->_pcDown, block->_localBound);
    //calculate bbx (global)
    PointCloudXYZI::Ptr globalDownCloud(new PointCloudXYZI());
    pcl::transformPointCloud(*block->_pcDown, *globalDownCloud, block->_poseLo);//pw=Tw1*p1
    cf.get_cloud_bbx(globalDownCloud, block->_bound);

    constraints currentCandiEdges;
    if (_nAccuOptFrames > _config._numFrameLargeDrift && _config._isOverallSearchingOn) //expand the loop closure searching area
    {
        isOverallLoopSearchingOn = true;
        regEdgeCount = _confinder.find_overlap_registration_constraint(subBlocks, block, currentCandiEdges,
                                                                       1.5 * _config._neighborSearchDist, 0.0,
                                                                       _config._minSubmapIdDiff, _config._isSearchNeighbor2d, 3);
    }
    else //standard loop closure searching
        regEdgeCount = _confinder.find_overlap_registration_constraint(subBlocks, block, currentCandiEdges,
                                                                       _config._neighborSearchDist, _config._minIOUThre,
                                                                       _config._minSubmapIdDiff, _config._isSearchNeighbor2d, 3);
    //std::cout<<"Overlaped blocks: "<<regEdgeCount<<std::endl;
    if(regEdgeCount<1)
        return false;

    //_voxelFilter.setLeafSize(_config._voxelDownSize,_config._voxelDownSize,_config._voxelDownSize);

    Constraint& registrationCon=currentCandiEdges[0];
    Eigen::Matrix4d initT12=registrationCon.Trans1_2;

    PointCloudXYZI::Ptr globalDownCloud1(new PointCloudXYZI()), globalDownCloud2(new PointCloudXYZI()), transDownCloud1(new PointCloudXYZI());
    pcl::transformPointCloud(*registrationCon.block1->_pcDown, *globalDownCloud1, registrationCon.block1->_poseLo);
    pcl::transformPointCloud(*registrationCon.block2->_pcDown, *globalDownCloud2, registrationCon.block2->_poseLo);
    //Eigen::Matrix4d Tw2 = registrationCon.block1->_poseLo * initT12;
    //pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);
//        {
//            _pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/condi_block_ori"+to_string(registrationCon.block1->_idInStrip)+".pcd", *globalDownCloud1);
//            _pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/trans_block_ori"+to_string(registrationCon.block2->_idInStrip)+".pcd", *transDownCloud1);
//        }
    Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
    gravAlignMat.block<3,3>(0,0) = Eigen::Quaterniond::FromTwoVectors(registrationCon.block1->_gravity, Eigen::Vector3d {0, 0, -1.0}).toRotationMatrix();
    Eigen::Matrix4d poseAlign1=gravAlignMat*registrationCon.block1->_poseLo;
    Eigen::Matrix4d poseAlign2=gravAlignMat*registrationCon.block2->_poseLo;
    Eigen::Matrix4d poseAlign3=gravAlignMat*getSubMapBlock(registrationCon.block2->_idInStrip-1)->_poseLo;
    double hDiff1= poseAlign1(2,3)-poseAlign2(2,3);
    double hDiff2= poseAlign1(2,3)-poseAlign3(2,3);

    //std::cout<<"hDiff1:"<<hDiff1<<std::endl;
    //std::cout<<"hDiff2:"<<hDiff2<<std::endl;
    if(abs(hDiff1)>1.0&&abs(hDiff1)<1.5&&abs(hDiff2)<1.5)
    {
        initT12(2,3)+=hDiff1;
       // std::cout<<"Correct height diff:"<<hDiff1<<std::endl;
    }
    double dist=(registrationCon.block1->_poseLo.block<3,1>(0,3)-registrationCon.block2->_poseLo.block<3,1>(0,3)).norm();
    //std::cout<<"Loop candidate dist:"<<dist<<std::endl;
    
    Eigen::Matrix4d Tw2 = registrationCon.block1->_poseLo * initT12;
    pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);
#if SAVE_MID_FILE
    {
        _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/condi_block"+to_string(registrationCon.block1->_idInStrip)+".pcd", *globalDownCloud1);
        _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/condi_block"+to_string(registrationCon.block2->_idInStrip)+".pcd", *globalDownCloud2);
        _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/trans_block_init"+
                               to_string(registrationCon.block2->_idInStrip)+
                               "-"+
                               to_string(registrationCon.block1->_idInStrip)+".pcd", *transDownCloud1);
    }
#endif
    //printf("kNN searching time: %f ms\n", time.toc());
    //time.tic();
    fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> fastVGICPReg;
    fastVGICPReg.setResolution(_config._vgicpVoxRes);
    fastVGICPReg.setMaxCorrespondenceDistance(dist);
    fastVGICPReg.setNumThreads(MP_PROC_NUM);
    pcl::PointCloud<pcl::PointXYZ>::Ptr localCloudXYZPtr2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr localCloudXYZPtr1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*registrationCon.block1->_pcDown, *localCloudXYZPtr1);
    pcl::copyPointCloud(*registrationCon.block2->_pcDown, *localCloudXYZPtr2);
    fastVGICPReg.setInputSource(localCloudXYZPtr2);
    fastVGICPReg.setInputTarget(localCloudXYZPtr1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
    fastVGICPReg.align(*aligned, initT12);
    registrationCon.Trans1_2=fastVGICPReg.getFinalTransformation();
    registrationCon.sigma = sqrt(fastVGICPReg.compute_error(Eigen::Isometry3d(registrationCon.Trans1_2)));//马氏距离
    bool isConverged=fastVGICPReg.hasConverged();
    if(!isConverged)
    {
        //std::cout<<"VGICP reg not converged!"<<std::endl;
        //nWrongLoop++;
        return false;
    }

//        RegistrationPCL_VGICP<PointType, PointType> regVGICP;
//        bool isCoverged = regVGICP.reg(downCloud2,downCloud1,
//                                       initT12, registrationCon.Trans1_2);
//        registrationCon.sigma = sqrt(regVGICP.compute_error(Eigen::Isometry3d(registrationCon.Trans1_2)));//马氏距离
//    std::cout<<"VGICP Error:"<<registrationCon.sigma<<std::endl;

    //Verify loop block clouds after registration
    Tw2 = registrationCon.block1->_poseLo * registrationCon.Trans1_2;
    pcl::transformPointCloud(*registrationCon.block1->_pcDown, *globalDownCloud1, registrationCon.block1->_poseLo);
    pcl::transformPointCloud(*registrationCon.block2->_pcDown, *globalDownCloud2, registrationCon.block2->_poseLo);
    pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);

    std::pair<float, float> distScore = _confinder.getFitnessScoreDist(globalDownCloud1, transDownCloud1);
    if (distScore.first < 0.5 || distScore.second > 0.25 || registrationCon.sigma>100)
    {
        isConverged=false;
        //std::cerr<<"ratio/dist:"<<distScore.first<<"/"<<distScore.second<<std::endl;
        //std::cout << "Coarse register..."<<std::endl;
    }
//    if (isConverged)
//    {
//        Eigen::Matrix4d diffTrans=initT12.inverse()*registrationCon.Trans1_2;
//        if(diffTrans.block<3,1>(0,3).norm()>0.04)
//        {
//            std::cout<<"pose changed:"<<std::endl;
//            std::cout<<diffTrans.block<3,1>(0,3).norm();
//            std::cout<<RotMtoEuler(Eigen::Matrix3d(diffTrans.block<3,3>(0,0))).transpose()<<std::endl;
//            isConverged=false;
//        }
//    }
    if (isConverged)
    {

#if SAVE_MID_FILE
        _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/trans_block"+
                               to_string(registrationCon.block2->_idInStrip)+
                               "-"+
                               to_string(registrationCon.block1->_idInStrip)+".pcd", *transDownCloud1);
#endif
//            std::cout<<"Trans1_2 after vgicp registration:\n"<<currentCandiEdges[j].Trans1_2<<std::endl;
       // std::cout<<"VGICP reg done successful!"<<std::endl;
        registrationCon.information_matrix=100*registrationCon.information_matrix;
        _pgoEdges.push_back(registrationCon);

        //Ad frame reg edge
        if(config()._isPgoIncremental)
        {
            Constraint regFrameConn;
            regFrameConn.block1 = _frameBlocks[registrationCon.block1->_uniqueId];
            regFrameConn.block2 = _frameBlocks[registrationCon.block2->_uniqueId];
            regFrameConn.con_type = REGISTRATION;
            regFrameConn.Trans1_2 = registrationCon.Trans1_2;
            regFrameConn.information_matrix = registrationCon.information_matrix;
            _pgOptimizer.addEdgeConstraint(regFrameConn);
        }

        _lcPairs.emplace_back(registrationCon.block1->_idInStrip, registrationCon.block2->_idInStrip);
        //std::cout<<"Wrong loop count:"<<nWrongLoop<<std::endl;
       // printf("VGICP and assertion time: %f ms\n", time.toc());
        return true;
    }

   // printf("VGICP and assertion time: %f ms\n", time.toc());
    return false;
}

bool LoopCloser::detectLoopContourContext(const CloudBlockPtr blockPtr)
{
    //static int nWrongLoop=0;
    //TicToc time;
    std::shared_ptr<ContourManager> tgtCMPtr = _ptrEvaluator->getCurrContourManager(blockPtr, _cmConfig);

#if SAVE_MID_FILE
        for (int i = 0; i < _cmConfig.lv_grads_.size(); i++)
        {
          std::string f_name = std::string(ROOT_DIR) + "result/layer_img/contour_" + "lv" + std::to_string(i) + "_" +
                               tgtCMPtr->getStrID() + ".png";   // TODO: what should be the str name of scans?
          tgtCMPtr->saveContourImage(f_name, i);
        }
#endif
    tgtCMPtr->clearImage();  // a must to save memory

    std::vector<std::shared_ptr<const ContourManager>> ptrCands;
    std::vector<double> candCorr;
    std::vector<Eigen::Isometry2d> bevTfs;
    _ptrContourDB->queryRangedKNN(tgtCMPtr, _thresLb, _thresUb, ptrCands, candCorr, bevTfs);


    _ptrContourDB->addScan(tgtCMPtr, 2*blockPtr->_idInStrip);
    _ptrContourDB->pushAndBalance(blockPtr->_idInStrip, 2*blockPtr->_idInStrip);

    //printf("ContourContext detect time: %f ms\n", time.toc());


    if(!ptrCands.empty())
    {
        if(tgtCMPtr->getIntID()- ptrCands[0]->getIntID()<_config._minSubmapIdDiff)
            return false;
#if SAVE_MID_FILE
        // save images of pairs
        std::string f_name = std::string(ROOT_DIR) + "result/match_comp_img/lc_" + tgtCMPtr->getStrID() + "-" + ptrCands[0]->getStrID() +".png";
        ContourManager::saveMatchedPairImg(f_name, *tgtCMPtr, *ptrCands[0]);
        printf("Image saved: %s-%s\n", tgtCMPtr->getStrID().c_str(), ptrCands[0]->getStrID().c_str());
#endif
     //   time.tic();
        Constraint registrationCon;
        registrationCon.block1 = _subMapBlocks[ptrCands[0]->getIntID()]; //history map (target)
        registrationCon.block2 = blockPtr;       //current map (source)
        registrationCon.con_type = REGISTRATION;
        registrationCon.Trans1_2 = registrationCon.block1->_poseLo.inverse() * registrationCon.block2->_poseLo;//T1w*Tw2

        //_voxelFilter.setLeafSize(_config._voxelDownSize,_config._voxelDownSize,_config._voxelDownSize);
//        _voxelFilter.setInputCloud(registrationCon.block1->_pcDown->makeShared());
//        _voxelFilter.filter(*registrationCon.block1->_pcDown);
//        _voxelFilter.setInputCloud(registrationCon.block2->_pcDown->makeShared());
//        _voxelFilter.filter(*registrationCon.block2->_pcDown);

//        Eigen::Matrix4d initT12=Eigen::Matrix4d::Identity();
//        Isometry2d& bestT2d=bevTfs[0];
//        initT12.block<3,1>(0,3)=Eigen::Vector3d (bestT2d.translation()(0),bestT2d.translation()(1),0);
//        Eigen::Matrix2d rot2D=bestT2d.rotation().matrix();
//        double yaw=atan2(rot2D(1,0),rot2D(0,0));
//        initT12.block<3,3>(0,0)=(Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX())*
//                                 Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())*
//                                 Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitZ())).matrix();
//        Eigen::Matrix4d trans12=Eigen::Matrix4d::Identity();

//        initT12=initT12*registrationCon.Trans1_2;
        Eigen::Matrix4d initT12=registrationCon.Trans1_2;

        PointCloudXYZI::Ptr globalDownCloud1(new PointCloudXYZI()), globalDownCloud2(new PointCloudXYZI()), transDownCloud1(new PointCloudXYZI());
        Eigen::Matrix4d Tw2 = registrationCon.block1->_poseLo * initT12;
        pcl::transformPointCloud(*registrationCon.block1->_pcDown, *globalDownCloud1, registrationCon.block1->_poseLo);
        pcl::transformPointCloud(*registrationCon.block2->_pcDown, *globalDownCloud2, registrationCon.block2->_poseLo);
        Tw2 = registrationCon.block1->_poseLo * initT12;
        pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);
#if SAVE_MID_FILE
        _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/trans_block_ori"+to_string(registrationCon.block2->_idInStrip)+".pcd", *transDownCloud1);
#endif
//        {
//            _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/condi_block_ori"+to_string(registrationCon.block1->_idInStrip)+".pcd", *globalDownCloud1);
//            _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/trans_block_ori"+to_string(registrationCon.block2->_idInStrip)+".pcd", *transDownCloud1);
//        }
        Eigen::Matrix4d gravAlignMat = Eigen::Matrix4d::Identity();
        gravAlignMat.block<3,3>(0,0) = Eigen::Quaterniond::FromTwoVectors(registrationCon.block1->_gravity, Eigen::Vector3d {0, 0, -1.0}).toRotationMatrix();
        Eigen::Matrix4d poseAlign1=gravAlignMat*registrationCon.block1->_poseLo;
        Eigen::Matrix4d poseAlign2=gravAlignMat*registrationCon.block2->_poseLo;
        double hDiff=poseAlign1(2,3)-poseAlign2(2,3);
//        std::cout<<"hDiff:"<<hDiff<<std::endl;
        if(abs(hDiff)>1.0)
        {
            initT12(2,3)+=hDiff;
//            std::cout<<"Correct height diff:"<<hDiff<<std::endl;
        }

        Tw2 = registrationCon.block1->_poseLo * initT12;
        pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);

        CFilter<PointType> cf;
        cf.get_cloud_bbx(globalDownCloud1, registrationCon.block1->_bound);
        cf.get_cloud_bbx(transDownCloud1, registrationCon.block2->_bound);
        double iou1,iou2,iouShare1;
        iouShare1=_confinder.calculate_iou(registrationCon.block1->_bound, registrationCon.block2->_bound,iou1,iou2);
//        std::cout<<"IOU1/IOU2/iou before reg:"<<iou1<<"/"<<iou2<<"/"<<iouShare1<<std::endl;
        if(iou1<0.5&&iou2<0.5)
            return false;
        if(iouShare1<0.4)
            return false;

       std::pair<float, float> distScore1 = _confinder.getFitnessScoreDist(globalDownCloud1, transDownCloud1);
        if(distScore1.first<0.1)
        {
//            std::cerr<<"before vgicp ratio/dist:"<<distScore1.first<<"/"<<distScore1.second<<std::endl;
           // return false;
        }

        double dist=(registrationCon.block1->_poseLo.block<3,1>(0,3)-registrationCon.block2->_poseLo.block<3,1>(0,3)).norm();
        //std::cout<<"Loop candidate dist:"<<dist<<std::endl;

#if SAVE_MID_FILE
        {
            _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/trans_block_init"+
                                   to_string(registrationCon.block2->_idInStrip)+
                                   "-"+
                                   to_string(registrationCon.block1->_idInStrip)+".pcd", *transDownCloud1);
            _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/condi_block"+to_string(registrationCon.block1->_idInStrip)+".pcd", *globalDownCloud1);
            _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/condi_block"+to_string(registrationCon.block2->_idInStrip)+".pcd", *globalDownCloud2);
        }
#endif

        fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> fastVGICPReg;
        fastVGICPReg.setResolution(_config._vgicpVoxRes);
        fastVGICPReg.setNumThreads(MP_PROC_NUM);
        fastVGICPReg.setMaxCorrespondenceDistance(dist);
        pcl::PointCloud<pcl::PointXYZ>::Ptr localCloudXYZPtr2(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr localCloudXYZPtr1(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*registrationCon.block1->_pcDown, *localCloudXYZPtr1);
        pcl::copyPointCloud(*registrationCon.block2->_pcDown, *localCloudXYZPtr2);
        fastVGICPReg.setInputSource(localCloudXYZPtr2);
        fastVGICPReg.setInputTarget(localCloudXYZPtr1);
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        fastVGICPReg.align(*aligned, initT12);
        registrationCon.Trans1_2=fastVGICPReg.getFinalTransformation();
        bool isConverged=fastVGICPReg.hasConverged();
        if(!isConverged)
        {
            fastVGICPReg.setMaximumIterations(200);
            std::cout<<"VGICP reg not converged!"<<std::endl;
            //nWrongLoop++;
            //return false;
        }

//        RegistrationPCL_VGICP<PointType, PointType> regVGICP;
//        bool isCoverged = regVGICP.reg(downCloud2,downCloud1,
//                                       initT12, registrationCon.Trans1_2);
//        registrationCon.sigma = sqrt(regVGICP.compute_error(Eigen::Isometry3d(registrationCon.Trans1_2)));//马氏距离
//        std::cout<<"VGICP Error:"<<registrationCon.sigma<<std::endl;

        //Verify loop block clouds after registration
        Tw2 = registrationCon.block1->_poseLo * registrationCon.Trans1_2;
        pcl::transformPointCloud(*registrationCon.block1->_pcDown, *globalDownCloud1, registrationCon.block1->_poseLo);
        pcl::transformPointCloud(*registrationCon.block2->_pcDown, *globalDownCloud2, registrationCon.block2->_poseLo);
        pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);

//        cf.get_cloud_bbx(globalDownCloud1, registrationCon.block1->_bound);
//        cf.get_cloud_bbx(transDownCloud1, registrationCon.block2->_bound);
//        double iouShare2 = _confinder.calculate_iou(registrationCon.block1->_bound, registrationCon.block2->_bound);
//        std::cout<<"IOU after reg:"<<iouShare2<<std::endl;
//        if(iouShare2<0.8*iouShare1)
//        {
//            isConverged=false;
//            //nWrongLoop++;
//            std::cout<<"IOU decreased too much after reg!"<<std::endl;
//        }

        std::pair<float, float> distScore2 = _confinder.getFitnessScoreDist(globalDownCloud1, transDownCloud1);
        //std::cerr<<"after vgicp ratio/dist:"<<distScore2.first<<"/"<<distScore2.second<<std::endl;
        //if (distScore.first < 0.68 || distScore.second > 0.28)
        if (!isConverged||distScore2.first<0.45||distScore2.first < 0.8*distScore1.first||distScore2.second > 0.25)
        {
            //std::cout<<"Increase vgicp voxel resolution!"<<std::endl;
            fastVGICPReg.setResolution(3*_config._vgicpVoxRes);

            fastVGICPReg.align(*aligned, fastVGICPReg.getFinalTransformation());
            registrationCon.Trans1_2=fastVGICPReg.getFinalTransformation();
            isConverged=fastVGICPReg.hasConverged();
            if(!isConverged)
            {
                isConverged=false;
                //std::cout<<"Secondary vgicp reg not converged!"<<std::endl;
            }
            else
            {
                Tw2 = registrationCon.block1->_poseLo * registrationCon.Trans1_2;
                pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);
                distScore2 = _confinder.getFitnessScoreDist(globalDownCloud1, transDownCloud1);
                //std::cerr<<"After secondary vgicp ratio/dist:"<<distScore2.first<<"/"<<distScore2.second<<std::endl;
                //if (distScore.first < 0.68 || distScore.second > 0.28)
                if (distScore2.first<0.45||distScore2.first < 0.8*distScore1.first||distScore2.second > 0.25)
                {
                    isConverged=false;
                    //std::cout << "Coarse register..."<<std::endl;
                }
            }
        }
        if (isConverged)
        {
#if SAVE_MID_FILE
            _pcdWriter.writeBinary(string(ROOT_DIR) + "PCD/trans_block"+
                                   to_string(registrationCon.block2->_idInStrip)+
                                   "-"+
                                   to_string(registrationCon.block1->_idInStrip)+".pcd", *transDownCloud1);
#endif
//            std::cout<<"Trans1_2 after vgicp registration:\n"<<currentCandiEdges[j].Trans1_2<<std::endl;
 //           std::cout<<"VGICP reg done successful!"<<std::endl;
            //registrationCon.sigma = sqrt(fastVGICPReg.compute_error(Eigen::Isometry3d(registrationCon.Trans1_2)));//马氏距离
            registrationCon.sigma = fastVGICPReg.getFitnessScore();
            //std::cout<<"VGICP FitnessScore:"<< registrationCon.sigma<<std::endl;
            registrationCon.information_matrix=100*registrationCon.information_matrix;
            _pgoEdges.push_back(registrationCon);

            //Add frame reg edge to pose graph
            if(config()._isPgoIncremental)
            {
                Constraint regFrameConn;
                regFrameConn.block1 = _frameBlocks[registrationCon.block1->_uniqueId];
                regFrameConn.block2 = _frameBlocks[registrationCon.block2->_uniqueId];
                regFrameConn.con_type = REGISTRATION;
                regFrameConn.Trans1_2 = registrationCon.Trans1_2;
                regFrameConn.information_matrix = registrationCon.information_matrix;
                _pgOptimizer.addEdgeConstraint(regFrameConn);
                //std::cout<<"PGO added a registration edge!"<<std::endl;
            }

            _lcPairs.emplace_back(tgtCMPtr->getIntID(), ptrCands[0]->getIntID());
            //std::cout<<"Wrong loop count:"<<nWrongLoop<<std::endl;
            //printf("VGICP and assertion time: %f ms\n", time.toc());
            return true;
        }
        //printf("VGICP and assertion time: %f ms\n", time.toc());
    }
    return false;
}

void LoopCloser::correctSubmapLoop()
{
    TicToc time;
  //  std::cout<<"PGO on submap..."<<std::endl;
    bool pgo_successful;
    if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "g2o"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_g2o(_subMapBlocks, _pgoEdges);
    else if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "ceres"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_ceres(_subMapBlocks, _pgoEdges, _config._tInterSubmapLimit, _config._RInterSubmapLimit);
    else if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "gtsam"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_gtsam(_subMapBlocks, _pgoEdges); //TODO: you'd better use gtsam instead (just like lego-loam)
    else //default: ceres
        pgo_successful = _pgOptimizer.optimize_pose_graph_ceres(_subMapBlocks, _pgoEdges, _config._tInterSubmapLimit, _config._RInterSubmapLimit);
    if (pgo_successful)
    {
        // _mutexProcess.lock();
        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_subMapBlocks);
            savePgoCloud("Block-Prev"+to_string(_loopCount));
        }

        _loopTrans=_subMapBlocks.back()->_poseOptimized*_subMapBlocks.back()->_poseLo.inverse(); //Tw'w=Tw'1*T1w   (Tw''w=  Tw''w' * Tw'w)

        _pgOptimizer.update_optimized_nodes(_subMapBlocks, true, true);      //let _poseLo = = _poseInit = _poseOptimized && update bbx at the same time
        _curSubMapBlock->_poseLo = _subMapBlocks.back()->_poseLo; //update current local map's pose

        for (int k = 0; k < _subMapBlocks.size(); k++)
            _subMapBlocks[k]->_poseStable = true;

        printf("Optimization time: %f ms\n", time.toc());
        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_subMapBlocks);
            savePgoCloud("Block-After"+to_string(_loopCount));
        }

        if(_config._isCorrectRealTime)
            correctFrontendMap(_subMapBlocks);
        //_mutexProcess.unlock();
    }
    else
        PLOGI.printf("PGO failed!\n");
}

void LoopCloser::correctFrameLoop()
{
    TicToc time;
    std::cout<<"PGO on frames..."<<std::endl;

    constraints framewisePgoEdges;
    constructFrameEdges(framewisePgoEdges);

    bool pgo_successful;
    if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "g2o"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_g2o(_frameBlocks, framewisePgoEdges);
    else if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "ceres"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_ceres(_frameBlocks, framewisePgoEdges, _config._tInterSubmapLimit, _config._RInterSubmapLimit);
    else if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "gtsam"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_gtsam(_frameBlocks, framewisePgoEdges); //TODO: you'd better use gtsam instead (just like lego-loam)
    else //default: ceres
        pgo_successful = _pgOptimizer.optimize_pose_graph_ceres(_frameBlocks, framewisePgoEdges, _config._tInterSubmapLimit, _config._RInterSubmapLimit);
    if (pgo_successful)
    {
        // _mutexProcess.lock();
        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_frameBlocks);
            savePgoCloud("Frame-Prev");
        }

        _loopTrans=_frameBlocks.back()->_poseOptimized*_frameBlocks.back()->_poseLo.inverse();//Tw'w=Tw'1*T1w   (Tw''w=  Tw''w' * Tw'w)
        //std::cout<<"Loop Trans:\n"<<_loopTrans<<std::endl;
        _pgOptimizer.update_optimized_nodes(_frameBlocks, false, false);      //let _poseLo = _poseInit = _poseOptimized && update bbx at the same time

        for (int k = 0; k < _frameBlocks.size(); k++)
            _frameBlocks[k]->_poseStable = true;
        printf("Optimization time: %f ms\n", time.toc());

        // Reconstruct submap
//        time.tic();
        for(auto& submap:_subMapBlocks)
        {
            const int& firstIdx=submap->_uniqueId;
            const int& lastIdx=submap->_lastFrameId;
            submap->_poseLo = _frameBlocks[firstIdx]->_poseLo;
//            submap->_pcDown.reset(new PointCloudXYZI());
//            std::vector<PointWithCov>().swap(submap->_pvList);
//            for(int i=firstIdx;i<=lastIdx;i++)
//            {
//                CloudBlockPtr frameBlock = getFrameBlock(i);
//                _blockMapManager.updateLocalMap(submap, frameBlock);
//            }
//            _voxelFilter.setInputCloud(submap->_pcDown->makeShared());
//            _voxelFilter.filter(*submap->_pcDown);// For registration
//            std::vector<PointWithCov>().swap(submap->_pvList);
//            submap->_pcDown->points.shrink_to_fit();
//            submap->_pvList.shrink_to_fit();
        }
        //_curSubMapBlock->_poseLo = _subMapBlocks.back()->_poseLo; //update current local map's pose
//        printf("Reconstructing submap time: %f ms\n", time.toc());

        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_frameBlocks);
            savePgoCloud("Frame-After");
        }

        if(_config._isCorrectRealTime)
            correctFrontendMap(_frameBlocks);
            // _mutexProcess.unlock();
    }
    else
        std::cerr<<"PGO Failed!"<<std::endl;
}

bool LoopCloser::constructFrameEdges(constraints& framewisePgoEdges)
{
    if(_frameBlocks.empty())
        return false;
    _frameBlocks[0]->_poseFixed = true; //fix the first frame
    for (int i = 0; i < _frameBlocks.size(); i++)
    {
        _frameBlocks[i]->_idInStrip = i;
//        _frameBlocks[i]->_poseInit = _frameBlocks[i]->_poseLo;
        if (i < _frameBlocks.size() - 1)
            _confinder.add_adjacent_constraint(_frameBlocks, framewisePgoEdges, i + 2);
    }
    for (int i = 0; i < _pgoEdges.size(); i++)
    {
        if (_pgoEdges[i].con_type == REGISTRATION)
        {
            Constraint regFrameEdge;
            regFrameEdge.con_type=_pgoEdges[i].con_type;
            regFrameEdge.block1 = _frameBlocks[_pgoEdges[i].block1->_uniqueId];
            regFrameEdge.block2 = _frameBlocks[_pgoEdges[i].block2->_uniqueId];
            regFrameEdge.Trans1_2=_pgoEdges[i].Trans1_2;
            regFrameEdge.information_matrix=_pgoEdges[i].information_matrix;
            regFrameEdge.sigma=_pgoEdges[i].sigma;
            framewisePgoEdges.emplace_back(regFrameEdge);
            std::cout<<"Added registration edges"<<std::endl;
        }
    }

    for(auto idxGNSSPair:_gnssMap)
    {
        const int& idInStrip=getFrameStripID(idxGNSSPair.first);
        _pgOptimizer.addGPSConstraint(idxGNSSPair.second, idInStrip);
    }

    return true;
}

bool LoopCloser::submapGraphOptimize()
{
    TicToc time;
    std::cout<<"PGO on submap incremental"<<std::endl;

    bool pgo_successful=false;
    if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "gtsam"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_gtsam(_keyframeBlocks);
    else
        std::cerr<<"Cannot support PGO incremental without gtsam"<<std::endl;

    if (pgo_successful)
    {
        // _mutexProcess.lock();
        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_frameBlocks);
            savePgoCloud("Frame-Prev");
        }

        _loopTrans=_keyframeBlocks.back()->_poseOptimized*_keyframeBlocks.back()->_poseLo.inverse();//Tw'w=Tw'1*T1w   (Tw''w=  Tw''w' * Tw'w)
        // std::cout<<"Loop Trans:\n"<<_loopTrans<<std::endl;
        _pgOptimizer.update_optimized_nodes(_keyframeBlocks, true, false);      //let _poseLo = = _poseInit = _poseOptimized && update bbx at the same time

        for (int k = 0; k < _keyframeBlocks.size(); k++)
            _keyframeBlocks[k]->_poseStable = true;
        printf("Optimization time: %f ms\n", time.toc());

        auto keyframeItr=_keyframeBlocks.begin();
        int frameSize=_frameBlocks.size();
        for(int i=0;i<frameSize;i++)
        {
            if(_frameBlocks[i]->_uniqueId==(*keyframeItr)->_uniqueId)
            {
                _frameBlocks[i]->_poseLo=(*keyframeItr)->_poseLo;
                if(keyframeItr<_keyframeBlocks.end()-1)
                    keyframeItr++;
            }
            else
                _frameBlocks[i]->_poseLo= _frameBlocks[i-1]->_poseLo*_relPoses[i-1];
        }
        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_frameBlocks);
            savePgoCloud("Frame-After");
        }

        if(_config._isCorrectRealTime)
            correctFrontendMap(_frameBlocks);
        // _mutexProcess.unlock();
    }
    else
    {
        std::cerr<<"PGO incremental Failed!"<<std::endl;
        return false;
    }
    return true;
}

bool LoopCloser::frameGraphOptimize()
{
    TicToc time;
    std::cout<<"PGO on frame incremental"<<std::endl;

    bool pgo_successful=false;
    if (!strcmp(_config._poseGraphOptimizationMethod.c_str(), "gtsam"))
        pgo_successful = _pgOptimizer.optimize_pose_graph_gtsam(_frameBlocks);
    else
        std::cerr<<"Cannot support PGO incremental without gtsam"<<std::endl;

    if (pgo_successful)
    {
        // _mutexProcess.lock();
        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_frameBlocks);
            savePgoCloud("Frame-Prev");
        }

        _loopTrans=_frameBlocks.back()->_poseOptimized*_frameBlocks.back()->_poseLo.inverse();//Tw'w=Tw'1*T1w   (Tw''w=  Tw''w' * Tw'w)
       // std::cout<<"Loop Trans:\n"<<_loopTrans<<std::endl;
        _pgOptimizer.update_optimized_nodes(_frameBlocks, true, false);      //let _poseLo = = _poseInit = _poseOptimized && update bbx at the same time

        //wait for several submap (without loop closure detection)
        for (int k = 0; k < _frameBlocks.size(); k++)
            _frameBlocks[k]->_poseStable = true;
        printf("Optimization time: %f ms\n", time.toc());

        // Reconstruct submap
        time.tic();
        for(auto& submap:_subMapBlocks)
        {
            const int& firstIdx=submap->_uniqueId;
            const int& lastIdx=submap->_lastFrameId;
            submap->_poseLo = _frameBlocks[firstIdx]->_poseLo;
//            submap->_pcDown.reset(new PointCloudXYZI());
//            std::vector<PointWithCov>().swap(submap->_pvList);
//            for(int i=firstIdx;i<=lastIdx;i++)
//            {
//                CloudBlockPtr frameBlock = getFrameBlock(i);
//                _blockMapManager.updateLocalMap(submap, frameBlock);
//            }
//            _voxelFilter.setInputCloud(submap->_pcDown->makeShared());
//            _voxelFilter.filter(*submap->_pcDown);// For registration
//            std::vector<PointWithCov>().swap(submap->_pvList);
//            submap->_pcDown->points.shrink_to_fit();
//            submap->_pvList.shrink_to_fit();
        }
        //_curSubMapBlock->_poseLo = _subMapBlocks.back()->_poseLo; //update current local map's pose
        printf("Reconstructing submap time: %f ms\n", time.toc());

        if(_config._isOutputPCD)
        {
            updatePgoBlockClouds(_frameBlocks);
            savePgoCloud("Frame-After");
        }

        if(_config._isCorrectRealTime)
            correctFrontendMap(_frameBlocks);
        // _mutexProcess.unlock();
    }
    else
    {
        std::cerr<<"PGO incremental Failed!"<<std::endl;
        return false;
    }
    return true;
}

void LoopCloser::updatePgoBlockClouds(CloudBlockPtrs allBlocks)
{
    _pgoTraj->clear();
    _pgoCloud->clear();
    for(auto& block:allBlocks)
    {
        PointType trajPt;
        trajPt.x = block->_poseLo(0, 3);
        trajPt.y = block->_poseLo(1, 3);
        trajPt.z = block->_poseLo(2, 3);
        _pgoTraj->push_back(trajPt) ;
        PointCloudXYZI::Ptr globalDownCloud(new PointCloudXYZI());
        pcl::transformPointCloud(*block->_pcDown, *globalDownCloud, block->_poseLo);//pw=Tw1*p1
        *_pgoCloud += *globalDownCloud;
    }
}

void LoopCloser::savePgoCloud(const std::string& suffix)
{
    _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/pgo"+suffix+".pcd", *_pgoCloud);
    _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/pgoTraj"+suffix+".pcd", *_pgoTraj);
}

//void LoopCloser::correctFrontendMap()
//{
//    //Build voxel map by submaps in pgo
//    TicToc time;
//    std::vector<PointWithCov> totalSurfPvList;
//    _globalCloudDown->clear();
//    for(auto& submap:_subMapBlocks)
//    {
//        std::cout<<"Accumulate submap id:" <<submap->_uniqueId<<", point size:"<<totalSurfPvList.size()<<std::endl;
//        for(auto& pv:submap->_pvList)
//        {
//            pv.pw= submap->_poseOptimized.block<3,3>(0, 0) * pv.pi + submap->_poseOptimized.block<3,1>(0, 3);
//            totalSurfPvList.emplace_back(pv);
//        }
//        PointCloudXYZI::Ptr globalPcDown(new PointCloudXYZI());
//        pcl::transformPointCloud(*submap->_pcDown, *globalPcDown, submap->_poseOptimized);//pw=Tw1*p1
//        *_globalCloudDown+=*globalPcDown;
//    }
//    printf("Rebuilding voxel map...\n");
//    buildVoxelMap(totalSurfPvList, -1, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
//                  _config._maxPointsSize, _config._maxCovPointsSize, _config._minEigenValue,
//                  _voxelSurfMap);
//    printf("Rebuild voxel map point size: %d\n", totalSurfPvList.size());
//    printf("Rebuild voxel map time: %f ms\n", time.toc());
//    if(_frameBuffer.empty())
//    {
//        _isUpdated=true;
//        _lastCorrectBlockID = _subMapBlocks.back()->_lastFrameId;
//        std::cout<<"Loop thread corrected Last frame id: "<<_lastCorrectBlockID<<std::endl;
//        return;
//    }
//
//    _mutexUpdate.lock();
//    //Update voxel map by new blocks in buffer
//    _mutexBuf.lock();//Avoid main thread pushing data in
//	std::cout << "Loop thread correct buffer block size:" << _frameBuffer.size() << std::endl;
//    time.tic() ;
//    totalSurfPvList.clear();
//    for(auto& block: _frameBuffer)
//    {
//        std::cout<<"Fronten suspend: Correct buffer block id:"<<block->_uniqueId<<std::endl;
//        block->_poseLo=_loopTrans*block->_poseLo;//Tw2'=Tw1'*T1w*Tw2=Tw1'*T12
//        block->_poseInit=_loopTrans*block->_poseLo;//Tw2'=Tw1'*T1w*Tw2=Tw1'*T12
//        //
//        for(auto& pv:block->_pvList)
//        {
//            pv.pw= block->_poseLo.block<3,3>(0, 0) * pv.pi + block->_poseLo.block<3,1>(0, 3);
//            totalSurfPvList.emplace_back(pv);
//        }
//        PointCloudXYZI::Ptr globalPcDown(new PointCloudXYZI());
//        pcl::transformPointCloud(*block->_pcDown, *globalPcDown, block->_poseLo);
//        *_globalCloudDown+=*globalPcDown;
//        PointType trajPt;
//        trajPt.x = block->_poseLo(0, 3);
//        trajPt.y = block->_poseLo(1, 3);
//        trajPt.z = block->_poseLo(2, 3);
//        _pgoTraj->push_back(trajPt) ;
//    }
//    std::vector<BoxPointType> boxToDel;
//    updateVoxelMap(totalSurfPvList, -1, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
//                   _config._maxPointsSize, _config._maxCovPointsSize, _config._minEigenValue,
//                   _voxelSurfMap, boxToDel);
//    _lastCorrectBlockID = _frameBuffer.back()->_uniqueId;
//    _isUpdated=true;
//    std::cout<<"Loop thread corrected Last frame id: "<<_lastCorrectBlockID<<std::endl;
//    printf("Loop thread correct buffer block time: %f ms\n", time.toc());
//    _mutexBuf.unlock();
//    _mutexUpdate.unlock();
//}

void LoopCloser::correctFrontendMap(CloudBlockPtrs allBlocks)
{
    //Build voxel map by submaps in pgo
    TicToc time;
    if (!_voxelSurfMap.empty())
        _voxelSurfMap.clear(); // rebuild voxel map

    _globalCloudDown->clear();
    int ptSize=0;
    int totalSize=allBlocks.size();

    int firstBlockIdx = std::max(0, totalSize-config()._frontendWinSize);
    auto firstBlock=allBlocks[firstBlockIdx];
    if(!firstBlock->_pvList.empty())
    {
        for(auto& pv:firstBlock->_pvList)
            pv.pw= firstBlock->_poseOptimized.block<3,3>(0, 0) * pv.pi + firstBlock->_poseOptimized.block<3,1>(0, 3);
        buildVoxelMap(firstBlock->_pvList, -1, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
                      _config._maxPointsSize, _config._maxCovPointsSize, _config._minEigenValue,
                      _voxelSurfMap);
        ptSize+=firstBlock->_pvList.size();
    }

    for(int i=firstBlockIdx;i<totalSize;i++)
    {
        auto block=allBlocks[i];
        if(!block->_pvList.empty())
        {
            for(auto& pv:block->_pvList)
                pv.pw= block->_poseOptimized.block<3,3>(0, 0) * pv.pi + block->_poseOptimized.block<3,1>(0, 3);
            std::vector<BoxPointType> boxToDel;
            updateVoxelMap(block->_pvList, -1, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
                           _config._maxPointsSize, _config._maxCovPointsSize, _config._minEigenValue,
                           _voxelSurfMap, boxToDel);
            ptSize+=block->_pvList.size();
            printf("Block %d updated voxel map, total point size: %d\n", block->_uniqueId, ptSize);
        }
    }
    if(ptSize>0)
    {
        printf("Rebuild voxel map point size: %d\n", ptSize);
        printf("Rebuild voxel map time: %f ms\n", time.toc());
    }

    if(_frameBuffer.empty())
    {
        _isUpdated=true;
        _lastCorrectBlockID = getFrameBlock(allBlocks.back()->_lastFrameId)->_uniqueId;
        std::cout<<"Loop thread corrected Last frame id: "<<_lastCorrectBlockID<<std::endl;
        return;
    }

    //Update voxel map by new blocks in buffer
    std::unique_lock<std::mutex> lock1(_mutexUpdate);
    std::unique_lock<std::mutex> lock2(_mutexBuf);//Avoid main thread pushing data in
    std::cout << "Loop thread correct buffer block size:" << _frameBuffer.size() << std::endl;
    time.tic() ;
    for(auto& block: _frameBuffer)
    {
        std::cout<<"Fronten suspend: Correct buffer block id:"<<block->_uniqueId<<std::endl;
        block->_poseLo=_loopTrans*block->_poseLo;//Tw2'=Tw1'*T1w*Tw2=Tw1'*T12
        block->_poseInit=_loopTrans*block->_poseLo;//Tw2'=Tw1'*T1w*Tw2=Tw1'*T12
        //
        if(!block->_pvList.empty())
        {
            for(auto& pv:block->_pvList)
                pv.pw= block->_poseLo.block<3,3>(0, 0) * pv.pi + block->_poseLo.block<3,1>(0, 3);
            std::vector<BoxPointType> boxToDel;
            updateVoxelMap(block->_pvList, -1, _config._voxelLength, _config._maxLayers, _config._layerPointSizeList,
                           _config._maxPointsSize, _config._maxCovPointsSize, _config._minEigenValue,
                           _voxelSurfMap, boxToDel);
        }
    }

    _lastCorrectBlockID = _frameBuffer.back()->_uniqueId;
    _isUpdated=true;
    std::cout<<"Loop thread corrected Last frame id: "<<_lastCorrectBlockID<<std::endl;
    printf("Loop thread correct buffer block time: %f ms\n", time.toc());
}

//void LoopCloser::forceLoopCorrect()
//{
//    const int blockSize=getSubMapBlocks().size();
//    //Find constraint between back and front blocks
//    //for(int i=0;i<10;i++)
//    for(int i=0;i<1;i++)
//    {
//        TicToc time;
//        Constraint registrationCon;
//        registrationCon.block1 = getSubMapBlocks().front();
//        CloudBlockPtr lastBlockPtr = *(getSubMapBlocks().end() - 1 - i);
//        registrationCon.block2 = lastBlockPtr;
//        std::cout << "Detecting block1-block2:" << registrationCon.block1->_idInStrip << "-" << registrationCon.block2->_idInStrip << std::endl;
//        registrationCon.con_type = REGISTRATION;
//        registrationCon.Trans1_2 = registrationCon.block1->_poseLo.inverse() * registrationCon.block2->_poseLo;
//
//        _voxelFilter.setLeafSize(1, 1, 1);
//        PointCloudXYZI::Ptr downCloud1(new PointCloudXYZI()), downCloud2(new PointCloudXYZI()), downCloud2Trans(new PointCloudXYZI());
//        _voxelFilter.setInputCloud(registrationCon.block1->_pcDown);
//        _voxelFilter.filter(*downCloud1);
//        _voxelFilter.setInputCloud(registrationCon.block2->_pcDown);
//        _voxelFilter.filter(*downCloud2);
//
//        Eigen::Matrix4d regTrans12 = registrationCon.Trans1_2;
//        printf("downsample time: %f ms\n", time.toc());
//
//        _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/downCloud1.pcd", *downCloud1);
//        pcl::transformPointCloud(*downCloud2, *downCloud2Trans, regTrans12);//pw=Tw2*p2
//        _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/downCloud2.pcd", *downCloud2Trans);
//
//        CRegistration<PointType> cReg;
//        time.tic();
//        int regStatus = cReg.coarse_reg_s4pcs(downCloud1, downCloud2, regTrans12,
//                                              1.5 * mutableConfig()._cloudPCANeighRadius,
//                                              mutableConfig()._salientRadiusRatio,
//                                              mutableConfig()._nonMaxRadiusRatio,
//                                              mutableConfig()._voxelDownSize);
//        printf("coarse_reg_s4pcs time: %f ms\n", time.toc());
//
//        if (regStatus != 1)
//        {
//            std::cout << "Coarse reg by s4pcs failed!" << std::endl;
//            regTrans12 = registrationCon.Trans1_2;
//        }
//        else
//        {
//            std::cout << "Coarse reg by s4pcs succeeded!" << std::endl;
//            std::cout << "initTrans12:" << regTrans12 << std::endl;
//            pcl::transformPointCloud(*downCloud2, *downCloud2Trans, regTrans12);//pw=Tw2*p2
//            _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/downCloud2_s4pcs.pcd", *downCloud2Trans);
//        }
//
//        CFilter<PointType> cf;
//        ConstraintFinder confinder;
//        PointCloudXYZI::Ptr globalDownCloud1(new PointCloudXYZI()), globalDownCloud2(new PointCloudXYZI()), transDownCloud1(new PointCloudXYZI());
//        Eigen::Matrix4d Tw2 = registrationCon.block1->_poseLo * regTrans12;
//        pcl::transformPointCloud(*registrationCon.block1->_pcDown, *globalDownCloud1, registrationCon.block1->_poseLo);
//        pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);
//        cf.get_cloud_bbx(globalDownCloud1, registrationCon.block1->_bound);
//        cf.get_cloud_bbx(transDownCloud1, registrationCon.block2->_bound);
//        double iou1 = confinder.calculate_iou(registrationCon.block1->_bound, registrationCon.block2->_bound);
//        std::cout << "IOU before VGICP:" << iou1 << std::endl;
//        if(iou1<0.5)
//        {
//            std::cerr<<"IOU between front-end is too little"<<std::endl;
//            break;
//        }
//
//        time.tic();
//        RegistrationPCL_VGICP<PointType, PointType> regVGICP;
//        bool isCoverged = regVGICP.reg(downCloud2,downCloud1,
//                                       regTrans12,regTrans12,
//                                       1.0);
//        printf("regVGICP time: %f ms\n", time.toc());
//        //assert trans diff
//        isCoverged = _confinder.double_check_tran(regTrans12, registrationCon.Trans1_2, regTrans12,
//                                                  _config._wrongEdgeTranThre,
//                                                  1.5*_config._wrongEdgeRotThreDeg);
//        if(!isCoverged)
//            break;
//        time.tic();
//        Tw2 = registrationCon.block1->_poseLo * regTrans12;
//        pcl::transformPointCloud(*registrationCon.block2->_pcDown, *transDownCloud1, Tw2);
//        cf.get_cloud_bbx(transDownCloud1, registrationCon.block2->_bound);
//        double iou2 = confinder.calculate_iou(registrationCon.block1->_bound, registrationCon.block2->_bound);
//        std::cout << "IOU after VGICP:" << iou2 << std::endl;
//
//        if (iou1 > 1.2 * iou2)
//        {
//            std::cerr << "IOU decreased too much!" << std::endl;
//            break;
//        }
//        else
//        {
//            registrationCon.Trans1_2 = regTrans12;
//            registrationCon.information_matrix=100*registrationCon.information_matrix;
//            std::cout << "PGO add edge: " << registrationCon.block1->_idInStrip << "-" << registrationCon.block2->_idInStrip << std::endl;
//            addPgoEdges(registrationCon);
//
//            pcl::transformPointCloud(*downCloud2, *downCloud2Trans, registrationCon.Trans1_2);//pw=Tw2*p2
//            _pcdWriter.writeBinary(std::string(ROOT_DIR) + "PCD/downCloud2_vgicp.pcd", *downCloud2Trans);
//        }
//        printf("assert time: %f ms\n", time.toc());
//    }
//    //Find constraint between blocks in other segment
////    CloudBlockPtrs subMapBlocks=_subMapBlocks;
////    subMapBlocks.pop_front();//push out first block
////    for(int i=0;i<blockSize-1;i++)
////    {
////        CloudBlockPtr frontBlock=subMapBlocks.front();
////        subMapBlocks.pop_front();
////        constraints currentLoopEdges;
////        detectLoop(frontBlock, subMapBlocks, currentLoopEdges);
////    }
//    TicToc time;
//    correctSubmapLoop();
//    if(_config._isFramewisePGO&&!_frameBlocks.empty())
//        correctFrameLoop();
//    printf("correctLoop time: %f ms\n", time.toc());
//}

bool LoopCloser::processGNSS()
{
    if(config()._isPgoIncremental)
    {//PGO incremental for a existed global pose graph
        if (!config()._isFramewisePGO)
        {
            if(_gnssFrameIndices.find(_curFrameBlock->_uniqueId)!=_gnssFrameIndices.end())
            {
                if(_keyframeBlocks.back()->_uniqueId!=_curFrameBlock->_uniqueId)
                {
                    std::cout<<"Added gnss keyframe:"<<_curFrameBlock->_uniqueId<<std::endl;
                    _keyframeBlocks.emplace_back(_curFrameBlock);
                    constraints adjCons;
                    _confinder.add_adjacent_constraint(_keyframeBlocks, adjCons, _keyframeBlocks.size());
                    _pgOptimizer.addEdgeConstraint(adjCons[0], false);
                }
            }
        }
    }

    bool isGNSSAdded=false;
    auto gnssItr=_gnssMapBuffer.find(_curFrameBlock->_uniqueId);
    if (gnssItr!=_gnssMapBuffer.end())
    {
        for(auto& idxGNSSPair:_gnssMapBuffer)
        {
            const int& idInStrip=getFrameStripID(idxGNSSPair.first);
            auto gnssFrameBlock=getFrameBlock(idInStrip);
            if((idxGNSSPair.second->pos-gnssFrameBlock->_poseLo.block<3,1>(0,3)).norm()<config()._gnssMaxError)
            {
                _pgOptimizer.addGPSConstraint(idxGNSSPair.second, idInStrip);
                isGNSSAdded=true;
            }
        }
        _gnssMapBuffer.erase(_gnssMapBuffer.begin(), gnssItr);
    }
    return isGNSSAdded;
}

void LoopCloser::printGNSSError()
{
    if(_gnssMap.empty())
        return;
    //std::cout<<"gnss map size:"<<_gnssMap.size()<<std::endl;
    for(auto idxGNSSPair:_gnssMap)
    {
        auto gnssData=idxGNSSPair.second;
        int idInStrip=getFrameStripID(idxGNSSPair.first);
        if(idInStrip>=0)
        {
            auto frameBlock=getFrameBlock(idInStrip);
            double error=(gnssData->pos - frameBlock->_poseLo.block<3,1>(0,3)).norm();
            std::cout<<"Align error:"<<error<<std::endl;
        }
    }
}
