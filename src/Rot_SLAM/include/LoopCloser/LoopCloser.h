//
// Created by w on 2022/9/15.
//

#ifndef SRC_LOOPCLOSER_H
#define SRC_LOOPCLOSER_H

#include <list>
#include <mutex>
#include <thread>
#include <stdio.h>
#include <iostream>
#include <unordered_map>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include "BlockManager.h"
#include "CloudBlock.h"
#include "ConstraintFinder.h"
#include "Registrator/CRegistration.hpp"
#include "Optimizer/GraphOptimizer.h"
#include "VoxelMapper.h"
#include "FeatureExtractor.h"
#include "use-ikfom.h"
#include "ReadWriter.h"

#include "cont2/contour_db.h"
#include "eval/evaluator.h"
#include "tools/config_handler.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define LOOP_CLOSER_API __declspec(dllexport)
#else
#define LOOP_CLOSER_API __declspec(dllimport)
#endif
#else
#define LOOP_CLOSER_API
#endif

#define SAVE_MID_FILE 0

class LOOP_CLOSER_API LoopCloser
{
public:
    struct LOOP_CLOSER_API Config
    {
        int                 _detectLoopMethod;
        int                 _numFrameLargeDrift;
        int                 _minSubmapIdDiff;
        int                 _maxUsedRegEdgePerOptimization;
        int                 _featureCorrNum;
        int                 _globalRegMinInlierCount;
        int                 _coolingSubmapNum;
        int                 _maxIterInterSubmap;
        int                 _maxSubMapAccuFrames;
        int                 _coarseRegistrationType;
        int                 _frameStoreInterval;
        bool                _isOverallSearchingOn;
        bool                _isSearchNeighbor2d;
        bool                _isTransferCorrectRegTranOn;
        bool                _isRobustKernelOn;
        bool                _isEqualWeightOn;
        bool                _isDiagonalInformationMatrixOn;
        bool                _isFreeNodeOn;
        bool                _isLoopDetectEn;
        bool                _isCorrectRealTime;
        bool                _issavemap;
        bool                _isFramewisePGO;
        bool                _isPgoIncremental;
        bool                _isStoreFrameBlock;
        bool                _isNeedGravAligned;
        double              _neighborSearchDist;
        double              _minIOUThreGlobalReg;
        double              _minIOUThre;
        double              _cloudPCANeighRadius;
        double              _wrongEdgeTranThre;
        double              _wrongEdgeRotThreDeg;
        double              _tInterSubmapLimit;
        double              _RInterSubmapLimit;
        double              _minConstraintConfidence;
        double              _maxConstraintSigma;
        double              _maxSubMapAccuTran;
        double              _maxSubMapAccuRot;
        double              _normalRadiusRatio;
        double              _fpfhRadiusRatio;
        double              _salientRadiusRatio;
        double              _nonMaxRadiusRatio;
        double              _voxelDownSize;
        double              _vgicpVoxRes;
        std::string         _poseGraphOptimizationMethod;

        double              _voxelLength;
        int                 _maxLayers;
        std::vector<int>    _layerPointSizeList;
        int                 _maxPointsSize;
        int                 _maxCovPointsSize;
        double              _minEigenValue;

        std::string         _detectorConfigPath;
        bool                _isOutputPCD;

        int                 _frontendWinSize;

        double              _gnssMaxError;

        Config()
        {
            _detectLoopMethod = 1; // 0: kNN   1:ContourContext
            _minIOUThre = 0.;//0.4; // min boundingbox iou for candidate registration edge
            _minIOUThreGlobalReg=0.5;
            _neighborSearchDist = 6.0; // max distance for candidate registration edge
            _minSubmapIdDiff = 20;//8; //min submap id difference between two submaps for a putable registration edge
            _numFrameLargeDrift = 1000;
            _featureCorrNum=1000;
            _isOverallSearchingOn = false;
            _isSearchNeighbor2d=false;
            _coarseRegistrationType=2; //0: teaser 1: ransac 2: s4pcs=true;
            _isTransferCorrectRegTranOn=true;//enable the registration tranformation transfer (only do global reg. once for each query submap
            _maxUsedRegEdgePerOptimization=3;
            _cloudPCANeighRadius = 2.0; //0.6;    // pca neighborhood searching radius(m) for target point cloud
            _globalRegMinInlierCount=7; //min inlier correspondence for a successful feature based registration (for teaser or ransac)
            _wrongEdgeTranThre=5.0; //translation threshold for judging if a edge is wrong or not");
            _wrongEdgeRotThreDeg=25.0; //rotation threshold for judging if a edge is wrong or not");
            _poseGraphOptimizationMethod= "gtsam";//"gtsam";//"ceres"; // use which library to do pgo (select from g2o, ceres and gtsam
            _coolingSubmapNum=10;//waiting for several submaps (without loop closure detection) after applying a successful pgo
            _isRobustKernelOn=false; //turn on the robust kernel function in pgo
            _isEqualWeightOn=false;   // using equal weight for the information matrix in pose graph optimization
            _maxIterInterSubmap =100;  //max iteration number for inter submap pgo
            _isDiagonalInformationMatrixOn = false; // use diagonal information matrix in pgo or not
            _isFreeNodeOn = true; //enable the free node module or not
            _isLoopDetectEn = true;
            _isCorrectRealTime = false;
            _isFramewisePGO = false;
            _isPgoIncremental = false;
            _isStoreFrameBlock=false;
            _frameStoreInterval=1;
            _isNeedGravAligned=true;
            _tInterSubmapLimit = 2.0; //the submap node's limit of translation variation, unit:m
            _RInterSubmapLimit = 0.5;//0.05; //the submap node's limit of rotation variation, unit quaternion
            _minConstraintConfidence=0;
            _maxConstraintSigma=0.3;
            _maxSubMapAccuTran=5.;
            _maxSubMapAccuRot=90.;
            _maxSubMapAccuFrames=20;
            _normalRadiusRatio=8;
            _fpfhRadiusRatio=16;
            _salientRadiusRatio=6;
            _nonMaxRadiusRatio=4;
            _voxelDownSize=0.3;
            _vgicpVoxRes=0.5;

            _maxLayers = 2;
            _voxelLength = 1.0;
            _maxPointsSize = 50;
            _maxCovPointsSize = 50;
            _minEigenValue = 0.01;
            _detectorConfigPath="";
            _isOutputPCD=false;
            _frontendWinSize=100;
            _gnssMaxError=0.3;
        }
    };

public:
    LoopCloser(): _bFinished(true),
                  _bStopped(false),
                  _bFinishRequested(false),
                  _bStopRequested(false),
                  _loopClosingThread(nullptr),
                  _lastFrameBlock(nullptr),
                  _curFrameBlock(nullptr),
                  _curSubMapBlock(new CloudBlock()),
                  _coolingIndex(0),
                  _accuTran(0),
                  _accuRotDeg(0),
                  _nAccuFrames(0),
                  _nAccuOptFrames(0),
                  _isNewSubMap(true),
                  _pgoCloud(new PointCloudXYZI()),
                  _pgoTraj(new PointCloudXYZI()),
                  _lastCorrectBlockID(-1),
                  _isUpdated(false),
                  _isDetected(false),
                  _loopTrans(Eigen::Matrix4d::Identity()),
                  _globalCloudDown(new PointCloudXYZI()),
                  _loopCount(0)
    {
        loadDetectorConfig(_config._detectorConfigPath);
#if SAVE_MID_FILE
        ReadWriter::cleanDir(std::string(ROOT_DIR) + "result/layer_img");
        ReadWriter::cleanDir(std::string(ROOT_DIR) + "result/match_comp_img");
#endif
        _pgOptimizer.set_robust_function(_config._isRobustKernelOn);
        _pgOptimizer.set_equal_weight(_config._isEqualWeightOn);
        _pgOptimizer.set_max_iter_num(_config._maxIterInterSubmap);
        _pgOptimizer.set_diagonal_information_matrix(_config._isDiagonalInformationMatrixOn);
        _pgOptimizer.set_free_node(_config._isFreeNodeOn);
        _pgOptimizer.set_problem_size(false);
    }

    ~LoopCloser() {}

    static Config& mutableConfig(){return _config;}
    static const Config& config(){return _config;}

    void loadDetectorConfig(const std::string &configPath);

    std::thread* getLoopThread() { return _loopClosingThread; }

    bool startThread();

    void storeFrameData();

    void inputFrameData(int frameId, const double& timeStamp,
                        const M3D& rot, const V3D& pos,
                        PointCloudXYZI::Ptr localCloud,
                        PointCloudXYZI::Ptr localCloudDown,
                        std::vector<PointWithCov>& pvList,
                        std::vector<IMUData>& imuDataList,
                        bool isNormalizeIntensity=false,
                        V3D gravity=V3D(0,0,-G_m_s2));

    void inputGNSSData(SensorMsgs::GNSSData::Ptr gnssData, const int& frameID);

    bool detectLoop();

    bool processGNSS();

    void run();

    GlobalOptimize& getPGOptimizer(){return _pgOptimizer;}

    std::mutex& getUpdateLock(){return _mutexUpdate;}
    std::mutex& getProcessLock(){return _mutexProcess;}
    std::mutex& getBufLock(){return _mutexBuf;}

    bool isUpdated(){return _isUpdated;}
    void setUpdated(const bool& flag){_isUpdated=flag;}

    bool isDetected(){return _isDetected;}
    void setDetected(const bool& flag){_isDetected=flag;}

    int loopCount(){return _loopCount;}

    Matrix4ds getRelPoses(){return _relPoses;}

    void addPgoEdges(Constraint& cons){_pgoEdges.push_back(cons);}
    constraints& getPgoEdges(){return _pgoEdges;}

    PointCloudXYZI::Ptr getPgoTraj(){return _pgoTraj;}

    std::vector<CloudBlockPtr>& getFrameBuffer(){return _frameBuffer;}

    CloudBlockPtr getSubMapBlock(const int& id){return _subMapBlocks[id];}
    CloudBlockPtrs getSubMapBlocks(){return _subMapBlocks;}
    void addSubMapBlock(CloudBlockPtr subMap){ _subMapBlocks.push_back(subMap); }

    CloudBlockPtr getFrameBlock(const int& id){return _frameBlocks[id];}
    CloudBlockPtrs getFrameBlocks(){return _frameBlocks;}
    void addFrameBlock(CloudBlockPtr block){ _frameBlocks.push_back(block); }

    std::map<int,SensorMsgs::GNSSData::Ptr>& getGNSSMap(){return _gnssMap;}
    int getFrameStripID(const int& frameID)
    {
        if(_frameIDMap.find(frameID)==_frameIDMap.end())
            return -1;
        return _frameIDMap[frameID];
    }

    CloudBlockPtr getCurSubmap(){return _curSubMapBlock;}

    int lastCorrectBlockID(){return _lastCorrectBlockID;}

    Eigen::Matrix4d getLoopTrans(){return _loopTrans;}

    std::vector<std::pair<int, int>> getLCPairs(){return _lcPairs;}

    std::unordered_map<VOXEL_LOC, OctoTree *>& getVoxelSurfMap(){return _voxelSurfMap;}

    PointCloudXYZI::Ptr getGlobalCloudDown(){return _globalCloudDown;}

    void addGNSSFrameIdx(const int& id){_gnssFrameIndices.insert(id);}

    void savePgoCloud(const std::string& suffix);

    void release();

    void requestFinish()
    {
        std::unique_lock<std::mutex> lock(_mutexFinish);
        _bFinishRequested = true;
    }

    bool checkFinish()
    {
        std::unique_lock<std::mutex> lock(_mutexFinish);
        return _bFinishRequested;
    }
    void requestStop()
    {
        std::unique_lock<std::mutex> lock(_mutexStop);
        _bStopRequested = true;
        std::unique_lock<std::mutex> lock2(_mutexSubmapUpdate);
    }
    bool stop()
    {
        std::unique_lock<std::mutex> lock(_mutexStop);
        if (_bStopRequested )
        {
            _bStopped = true;
            std::cout << "Loop closing STOP" << std::endl;
            return true;
        }
        return false;
    }

    bool isStopped()
    {
        std::unique_lock<std::mutex> lock(_mutexStop);
        return _bStopped;
    }

    bool stopRequested()
    {
        std::unique_lock<std::mutex> lock(_mutexStop);
        return _bStopRequested;
    }

    void setFinish()
    {
        std::unique_lock<std::mutex> lock(_mutexFinish);
        _bFinished = true;
        _bStopped = true;
    }

    bool isFinished()
    {
        std::unique_lock<std::mutex> lock(_mutexFinish);
        return _bFinished;
    }

    bool constructFrameEdges(constraints& framewisePgoEdges);

    void correctSubmapLoop();

    void correctFrameLoop();

    bool frameGraphOptimize();

    bool submapGraphOptimize();

    void updatePgoBlockClouds(CloudBlockPtrs allBlocks);

    bool detectLoopNN(const CloudBlockPtr block, CloudBlockPtrs &subBlocks);

    bool detectLoopContourContext(const CloudBlockPtr blockPtr);

    void correctFrontendMap(CloudBlockPtrs allBlocks);

    void printGNSSError();

private:

    static Config                                _config;
    std::mutex				                     _mutexSubmapUpdate;
    std::mutex	                                 _mutexFinish;
    std::mutex	                                 _mutexStop;
    std::mutex                                   _mutexProcess;
    std::mutex                                   _mutexBuf;
    std::mutex                                   _mutexUpdate;

    bool					                     _bFinished;
    bool					                     _bStopped;
    bool				                         _bFinishRequested;
    bool					                     _bStopRequested;

    CloudBlockPtr                                _lastFrameBlock;
    CloudBlockPtr                                _curFrameBlock;
    CloudBlockPtr                                _curSubMapBlock;
    CloudBlockPtrs                               _frameBlocks;
    CloudBlockPtrs                               _keyframeBlocks;
    CloudBlockPtrs                               _subMapBlocks;
    constraints                                  _pgoEdges;

    std::vector<CloudBlockPtr>                   _frameBuffer;
    std::map<int,SensorMsgs::GNSSData::Ptr>      _gnssMap;
    std::map<int,SensorMsgs::GNSSData::Ptr>      _gnssMapBuffer;
    std::map<int,int>                            _frameIDMap;
    std::set<int>                                _gnssFrameIndices;

    ConstraintFinder                             _confinder;
    BlockManager                                 _blockMapManager;
    GlobalOptimize                               _pgOptimizer;

    std::thread*			                     _loopClosingThread;

    Matrix4ds                                    _relPoses;

    float                                        _accuTran ;
    float                                        _accuRotDeg;
    int                                          _nAccuFrames;
    int                                          _nAccuOptFrames;
    int                                          _coolingIndex;

    bool                                         _isNewSubMap;

    std::unordered_map<VOXEL_LOC, OctoTree *>    _voxelSurfMap;
    PointCloudXYZI::Ptr                          _globalCloudDown;
    PointCloudXYZI::Ptr                          _pgoCloud;
    PointCloudXYZI::Ptr                          _pgoTraj;
    int                                          _lastCorrectBlockID;
    Eigen::Matrix4d                              _loopTrans;
    bool                                         _isUpdated;
    bool                                         _isDetected;
    int                                          _loopCount;

    std::vector<std::pair<int, int>>             _lcPairs;
    pcl::VoxelGrid<PointType>                    _voxelFilter;

    pcl::PCDWriter                               _pcdWriter;

    //"contour context" loop detector
    std::unique_ptr<ContourDB>                   _ptrContourDB;
    std::unique_ptr<ContLCDEvaluator>            _ptrEvaluator;

    ContourManagerConfig                         _cmConfig;
    ContourDBConfig                              _dbConfig;

    CandidateScoreEnsemble                       _thresLb;
    CandidateScoreEnsemble                       _thresUb;  // check thresholds
};



#endif //SRC_LOOPCLOSER_H
