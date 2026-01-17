#include "RigelSLAMRawIOTools.h"
#include <stdio.h>

#include <set>
#include <vector>

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

struct LoopPairInfoStru_1_0
{
    float noise;
    int curKfId;              // fromIdx
    int prevKfId;             // ToIdx
    float poseFrom[6] = {0};  // roll pitch yaw x y z
    float poseTo[6] = {0};    // roll pitch yaw x y z
};

struct ControlPointInfoStru_1_0
{
    int kfID;  


    float x;
    float y;
    float z;
};

class CRawFileData::RawFileDataImpl
{
public:
    RawFileDataImpl() {}

    ~RawFileDataImpl() {}

public:
    RawFileHeadStru fileHead;

    std::vector<LidarFrameInfoStru> lidarFramesInfos;     
    std::vector<LoopPairInfoStru> loopPairInfos;          
    std::vector<ControlPointInfoStru> controlPointInfos;  
    std::vector<OdometryInfoStru> odometryInfos;          
    std::vector<IMUInfo> imuInfos;
    ContinueTaskInfo continueTaskData;                   
    void Clear()
    {
        imuInfos.clear();
        loopPairInfos.clear();
        controlPointInfos.clear();
        continueTaskData.Clear();
        odometryInfos.clear();
        for (auto &info : lidarFramesInfos) info.Clear();
    }
};

CRawFileData::CRawFileData() { impl_ = new RawFileDataImpl; }

CRawFileData::~CRawFileData() { delete impl_; }

RawFileHeadStru &CRawFileData::getFileHead() { return impl_->fileHead; }

std::vector<LidarFrameInfoStru> &CRawFileData::getLidarFramesInfos()
{
    return impl_->lidarFramesInfos;
}

std::vector<LoopPairInfoStru> &CRawFileData::getLoopPairInfos()
{
    return impl_->loopPairInfos;
}

std::vector<ControlPointInfoStru> &CRawFileData::getControlPointInfos()
{
    return impl_->controlPointInfos;
}

std::vector<OdometryInfoStru> &CRawFileData::getOdometryInfos()
{
    return impl_->odometryInfos;
}

std::vector<IMUInfo>& CRawFileData::getIMUInfos(){return impl_->imuInfos;}

//std::vector<Eigen::Matrix4d>& CRawFileData::getExtrinsicInfos(){return impl_->TilList;}

ContinueTaskInfo &CRawFileData::getContinueTaskData()
{
    return impl_->continueTaskData;
}

void CRawFileData::Clear() { return impl_->Clear(); }

CRigelSLAMRawIOTools::CRigelSLAMRawIOTools() {}

CRigelSLAMRawIOTools::~CRigelSLAMRawIOTools() {}

READ_RAW_FILE_RETURN CRigelSLAMRawIOTools::ReadRawFile(const std::string &filePath,
                                                       CRawFileData &rawFileData,
                                                       ProgressCallBackPtr pCallbackPtr)
{
    FILE *pFile = fopen(filePath.c_str(), "rb");
    if (pFile == nullptr)
    {
        std::cout << "error: cannot open file[read]: " << filePath << std::endl;;
        return READ_RAW_FILE_RETURN::READ_OPEN_FAILED;
    }

    try
    {
        auto &fileHeader = rawFileData.getFileHead();

        fread(&fileHeader, sizeof(RawFileHeadStru), 1, pFile);

        if (fileHeader.version == RAWFileVersionEnum::VERSION_1_0)
        {
            std::cout << "reading raw 1.0\n";
            READ_RAW_FILE_RETURN ret = readRawFile1_0(filePath, rawFileData, pCallbackPtr);
            return ret;
        }
        if (fileHeader.version == RAWFileVersionEnum::VERSION_2_0)
        {
            std::cout << "reading raw 2.0\n";
            READ_RAW_FILE_RETURN ret = readRawFile2_0(filePath, rawFileData, pCallbackPtr);
            return ret;
        }
        if (fileHeader.version == RAWFileVersionEnum::VERSION_3_0)
        {
            std::cout << "reading raw 3.0\n";
            READ_RAW_FILE_RETURN ret = readRawFile3_0(filePath, rawFileData, pCallbackPtr);
            return ret;
        }
        else
            return READ_RAW_FILE_RETURN::READ_RAW_FAILED;
    }
    catch (const std::exception &)
    {
        return READ_RAW_FILE_RETURN::READ_RAW_FAILED;
    }

    return READ_RAW_FILE_RETURN::READ_RAW_SUCCESSED;
}

WRITE_RAW_FILE_RETURN CRigelSLAMRawIOTools::WriteRawFile(const std::string &filePath,
                                                         CRawFileData &rawFileData,
                                                         ProgressCallBackPtr pCallbackPtr)
{

    auto &fileHead = rawFileData.getFileHead();

    // update file header info
    fileHead.lidarFrameNum = rawFileData.getLidarFramesInfos().size();
    fileHead.odometryNum = rawFileData.getOdometryInfos().size();
    fileHead.loopPairNum = rawFileData.getLoopPairInfos().size();
    fileHead.controlPointsNum = rawFileData.getControlPointInfos().size();
    fileHead.imuNum=rawFileData.getIMUInfos().size();

    if (fileHead.lidarFrameNum == 0)
    {
        std::cout << "error: the target data is empty!\n";
        return WRITE_RAW_FILE_RETURN::EMPTY_RAW_DATA;
    }

    FILE *pFile = fopen(filePath.c_str(), "wb");
    if (pFile == NULL)
    {
        std::cout << "error: cannot open file[write]: " << filePath << std::endl;;
        return WRITE_RAW_FILE_RETURN::WRITE_OPEN_FAILED;
    }

    fwrite(&fileHead, sizeof(RawFileHeadStru), 1, pFile);

    int frameNum = fileHead.lidarFrameNum;

    for (int i = 0; i < frameNum; i++)
    {
        auto &frameInfo = rawFileData.getLidarFramesInfos()[i];

        fwrite(&frameInfo.affiliateRawID, sizeof(int), 1, pFile);

        fwrite(&frameInfo.kfID, sizeof(int), 1, pFile);
        fwrite(&frameInfo.globalKFID, sizeof(int), 1, pFile);

   
        fwrite(&frameInfo.timestamp, sizeof(double), 1, pFile);

 
        frameInfo.q_orientation.normalize();  
        double position[7] = {0.};
        position[0] = frameInfo.q_orientation.w();
        position[1] = frameInfo.q_orientation.x();
        position[2] = frameInfo.q_orientation.y();
        position[3] = frameInfo.q_orientation.z();
        position[4] = frameInfo.x;
        position[5] = frameInfo.y;
        position[6] = frameInfo.z;

        fwrite(position, sizeof(double), 7, pFile);

        fwrite(&frameInfo.Til, sizeof(Eigen::Matrix4d), 1, pFile);

  
        fwrite(&frameInfo.bHaveGPSInfo, sizeof(bool), 1, pFile);
        if (frameInfo.bHaveGPSInfo)
            fwrite(&frameInfo.gpsInfo, sizeof(GPSInfoStru), 1, pFile);

 
        int rawPtSize = frameInfo.fullCloud->points.size();  

        fwrite(&rawPtSize, sizeof(int), 1, pFile);

        std::vector<float> xyziData(rawPtSize * 5);
        for (int j = 0; j < rawPtSize; j++)
        {
            auto &rawPts = frameInfo.fullCloud->points;
            xyziData[j * 5 + 0] = rawPts[j].x;
            xyziData[j * 5 + 1] = rawPts[j].y;
            xyziData[j * 5 + 2] = rawPts[j].z;
            xyziData[j * 5 + 3] = rawPts[j].intensity;
            xyziData[j * 5 + 4] = rawPts[j].curvature;
        }

        if (rawPtSize > 0)
            fwrite(&xyziData[0], sizeof(float), rawPtSize * 5, pFile);


        int cornPtSize = frameInfo.cornCloud->points.size();
        fwrite(&cornPtSize, sizeof(int), 1, pFile);

        xyziData.resize(cornPtSize * 5);
        for (int j = 0; j < cornPtSize; j++)
        {
            auto &cornPts = frameInfo.cornCloud->points;
            xyziData[j * 5 + 0] = cornPts[j].x;
            xyziData[j * 5 + 1] = cornPts[j].y;
            xyziData[j * 5 + 2] = cornPts[j].z;
            xyziData[j * 5 + 3] = cornPts[j].intensity;
            xyziData[j * 5 + 4] = cornPts[j].curvature;
        }

        if (cornPtSize > 0)
            fwrite(&xyziData[0], sizeof(float), cornPtSize * 5, pFile);


        int surfPtSize = frameInfo.surfCloud->points.size();
        fwrite(&surfPtSize, sizeof(int), 1, pFile);

        xyziData.resize(surfPtSize * 5);
        for (int j = 0; j < surfPtSize; j++)
        {
            auto &surfPts = frameInfo.surfCloud->points;
            xyziData[j * 5 + 0] = surfPts[j].x;
            xyziData[j * 5 + 1] = surfPts[j].y;
            xyziData[j * 5 + 2] = surfPts[j].z;
            xyziData[j * 5 + 3] = surfPts[j].intensity;
            xyziData[j * 5 + 4] = surfPts[j].curvature;
        }

        if (surfPtSize > 0)
            fwrite(&xyziData[0], sizeof(float), surfPtSize * 5, pFile);


        int colorPtSize = frameInfo.coloredCloud->points.size();
        fwrite(&colorPtSize, sizeof(int), 1, pFile);

        xyziData.resize(colorPtSize * 6);
        for (int j = 0; j < colorPtSize; j++)
        {
            auto &colorPts = frameInfo.coloredCloud->points;
            xyziData[j * 6 + 0] = colorPts[j].x;
            xyziData[j * 6 + 1] = colorPts[j].y;
            xyziData[j * 6 + 2] = colorPts[j].z;
            xyziData[j * 6 + 3] = colorPts[j].r;
            xyziData[j * 6 + 4] = colorPts[j].g;
            xyziData[j * 6 + 5] = colorPts[j].b;
        }

        if (colorPtSize > 0)
            fwrite(&xyziData[0], sizeof(float), colorPtSize * 6, pFile);

        if (pCallbackPtr)
        {
            int curProgress = static_cast<int>(i * 1.f * 100 / (frameNum + 3));
            std::string backStr = "Save raw data";
            pCallbackPtr(backStr, curProgress);
        }
    }

    auto &loopPairInfos = rawFileData.getLoopPairInfos();
    int loopPairNum = loopPairInfos.size();
    fwrite(&loopPairNum, sizeof(int), 1, pFile);
    if (loopPairNum > 0)
        fwrite(&loopPairInfos[0], sizeof(LoopPairInfoStru), loopPairNum, pFile);

    auto &controlPointInfos = rawFileData.getControlPointInfos();
    int controlMarkerSize = controlPointInfos.size();
    fwrite(&controlMarkerSize, sizeof(int), 1, pFile);
    if (controlMarkerSize > 0)
        fwrite(&controlPointInfos[0], sizeof(ControlPointInfoStru),  controlMarkerSize, pFile);

    auto &continueInfo = rawFileData.getContinueTaskData();
    fwrite(&continueInfo.affiliateRawID, sizeof(int), 1, pFile);
    fwrite(&continueInfo.isAvaliableContinueParams, sizeof(bool), 1, pFile);

    double pose6D[6] = {continueInfo.refX, continueInfo.refY,
                        continueInfo.refZ, continueInfo.refRoll,
                        continueInfo.refPitch, continueInfo.refYaw};

    fwrite(pose6D, sizeof(double), 6, pFile);

    fwrite(&continueInfo.submapKFNum, sizeof(int), 1, pFile);

    //if (continueInfo.submapKFNum > 0)
      //  fwrite(&continueInfo.submapFrameIDs[0], sizeof(int), continueInfo.submapKFNum, pFile);

    int odometry_num = fileHead.odometryNum;

    if (odometry_num > 0)
    {
        auto &odeometry_infos = rawFileData.getOdometryInfos();
        fwrite(odeometry_infos.data(), sizeof(odeometry_infos[0]), odometry_num, pFile);
    }
    if (fileHead.imuNum > 0)
    {
        auto &imuInfos = rawFileData.getIMUInfos();
        fwrite(imuInfos.data(), sizeof(imuInfos[0]), fileHead.imuNum, pFile);
    }
//    auto &TilList = rawFileData.getExtrinsicInfos();
//    if (!TilList.empty())
//        fwrite(TilList.data(), sizeof(TilList[0]), TilList.size(), pFile);

    if (pCallbackPtr)
    {
        int curProgress = 100;
        std::string backStr = "Save raw data";
        pCallbackPtr(backStr, curProgress);
    }

    fclose(pFile);
    return WRITE_RAW_FILE_RETURN::WRITE_RAW_SUCCESSED;
}

bool CRigelSLAMRawIOTools::SetTransform(const Eigen::Matrix3d &rotMatrix,
                                        const Eigen::Vector3d &tPose,
                                        CRawFileData &rawFileData)
{
    auto &lidarFramesInfos = rawFileData.getLidarFramesInfos();
    int frameNum = lidarFramesInfos.size();
    for (int i = 0; i < frameNum; i++)
    {
        auto &frameInfo = lidarFramesInfos[i];
        Eigen::Matrix3d resRot = rotMatrix * frameInfo.q_orientation.matrix();

        frameInfo.q_orientation = Eigen::Quaterniond(resRot);
        frameInfo.q_orientation.normalize();

        Eigen::Vector3d resPose =
                rotMatrix * Eigen::Vector3d(frameInfo.x, frameInfo.y, frameInfo.z) +
                tPose;
        frameInfo.x = resPose(0);
        frameInfo.y = resPose(1);
        frameInfo.z = resPose(2);
    }

    auto &continueTaskData = rawFileData.getContinueTaskData();
    Eigen::Affine3d transFrom = Eigen::Affine3d::Identity();
    pcl::getTransformation(continueTaskData.refX, continueTaskData.refY,
                           continueTaskData.refZ, continueTaskData.refRoll,
                           continueTaskData.refPitch, continueTaskData.refYaw,
                           transFrom);

    Eigen::Matrix4d transToMatrix = Eigen::Matrix4d::Identity();
    transToMatrix.block<3, 3>(0, 0) = rotMatrix;
    transToMatrix.block<3, 1>(0, 3) = tPose;

    Eigen::Affine3d transToAff = Eigen::Affine3d(transToMatrix);

    Eigen::Affine3d finalTransAffd = transToAff * transFrom;

    Eigen::Affine3f finalTransAffine = Eigen::Affine3f::Identity();
    finalTransAffine = finalTransAffd.cast<float>();

    float x = 0.f, y = 0.f, z = 0.f;
    float roll = 0.f, pitch = 0.f, yaw = 0.f;
    pcl::getTranslationAndEulerAngles(finalTransAffine, x, y, z, roll, pitch,
                                      yaw);

    continueTaskData.refX = x * 1.0;
    continueTaskData.refY = y * 1.0;
    continueTaskData.refZ = z * 1.0;

    continueTaskData.refRoll = roll * 1.0;
    continueTaskData.refPitch = pitch * 1.0;
    continueTaskData.refYaw = yaw * 1.0;

    return true;
}

bool CRigelSLAMRawIOTools::MultipleRawMerge(
        std::vector<CRawFileData> &rawFileDatas, CRawFileData &mergedRawFileData,
        ProgressCallBackPtr pCallbackPtr)
{
    if (rawFileDatas.empty()) return false;

    mergedRawFileData = rawFileDatas[0];

    SetAffiliationInfo(0, mergedRawFileData);
    SetGlobalID(0, mergedRawFileData);

    for (int i = 1; i < rawFileDatas.size(); i++)
    {
        auto &curRaw = rawFileDatas[i];

        SetAffiliationInfo(i, curRaw);

        std::vector<int> redunIDs;
        GetRedundantFrameIDs(mergedRawFileData, curRaw, redunIDs);

        int offset =
                mergedRawFileData.getLidarFramesInfos().size(); 

        if (redunIDs.empty())
        {
            SetGlobalID(offset, curRaw);

            mergedRawFileData.getLidarFramesInfos().insert(
                    mergedRawFileData.getLidarFramesInfos().end(),
                    curRaw.getLidarFramesInfos().begin(), curRaw.getLidarFramesInfos().end());
            mergedRawFileData.getLoopPairInfos().insert(
                    mergedRawFileData.getLoopPairInfos().end(), curRaw.getLoopPairInfos().begin(),
                    curRaw.getLoopPairInfos().end());
            mergedRawFileData.getControlPointInfos().insert(
                    mergedRawFileData.getControlPointInfos().end(),
                    curRaw.getControlPointInfos().begin(), curRaw.getControlPointInfos().end());
        } else
        {
            std::sort(redunIDs.begin(), redunIDs.end()); 

            std::vector<LidarFrameInfoStru> tmpLidarFramesInfos(
                    curRaw.getLidarFramesInfos().size() - redunIDs.size());

            std::set<int> redunIDsSet;
            for (auto id : redunIDs) redunIDsSet.insert(id);

            std::vector<std::pair<int, int>>
                    oriAndNowKfIDs;  
            int nowID = 0;
            for (int j = 0; j < curRaw.getLidarFramesInfos().size(); j++)
            {
                if (redunIDsSet.find(j) == redunIDsSet.end())
                {
                    tmpLidarFramesInfos[nowID] = curRaw.getLidarFramesInfos()[j];
                    tmpLidarFramesInfos[nowID].kfID = nowID;
                    tmpLidarFramesInfos[nowID].globalKFID = offset + nowID;
                    oriAndNowKfIDs.push_back(std::make_pair(j, nowID));

                    nowID++;
                }
            }


            for (auto &lpPair : curRaw.getLoopPairInfos())
            {
                bool bfd1 = false, bfd2 = false;
                for (auto &idPair : oriAndNowKfIDs)
                {
                    if (lpPair.prevKFID == idPair.first)
                    {
                        lpPair.globalPrevKFID = offset + idPair.second;
                        bfd1 = true;
                    }

                    if (lpPair.curKFID == idPair.first)
                    {
                        lpPair.globalCurKFID = offset + idPair.second;
                        bfd2 = true;
                    }
                }

                if (!bfd1 || !bfd2)
                {
                    lpPair.bAvaliable = false; 
                    printf("merge raw, inavaliable loop pair\n");
                }
            }

            for (auto &ctrlPtInfo : curRaw.getControlPointInfos())
            {
                bool bfd = false;
                for (auto &idPair : oriAndNowKfIDs)
                {
                    if (ctrlPtInfo.kfID == idPair.first)
                    {
                        ctrlPtInfo.globalKFID = offset + idPair.second;
                        bfd = true;
                    }
                }

                if (!bfd)
                {
                    ctrlPtInfo.bAvaliable =
                            false; 
                    printf("merge raw, inavaliable control point\n");
                }
            }

            mergedRawFileData.getLidarFramesInfos().insert(
                    mergedRawFileData.getLidarFramesInfos().end(), tmpLidarFramesInfos.begin(),
                    tmpLidarFramesInfos.end());
            mergedRawFileData.getLoopPairInfos().insert(
                    mergedRawFileData.getLoopPairInfos().end(), curRaw.getLoopPairInfos().begin(),
                    curRaw.getLoopPairInfos().end());
            mergedRawFileData.getControlPointInfos().insert(
                    mergedRawFileData.getControlPointInfos().end(),
                    curRaw.getControlPointInfos().begin(), curRaw.getControlPointInfos().end());

        }  // end - else

        if (pCallbackPtr)
        {
            int curProgress = static_cast<int>(i * 1.f * 100 / rawFileDatas.size());
            std::string backStr = "Merging raw data";
            pCallbackPtr(backStr, curProgress);
        }
    }

    SetMergedRawKfID(mergedRawFileData);

    SetRawHeadAttribute(mergedRawFileData);


    mergedRawFileData.getFileHead().isEmergedRawFile = true;


    mergedRawFileData.getContinueTaskData().Clear();

    return true;
}

void CRigelSLAMRawIOTools::RawInvert(CRawFileData &rawData)
{
    std::set<int> loopAndCtrlKFIDsSet;
    for (auto &pair : rawData.getLoopPairInfos())
    {
        loopAndCtrlKFIDsSet.insert(pair.curKFID);
        loopAndCtrlKFIDsSet.insert(pair.prevKFID);
    }
    for (auto &ctrl : rawData.getControlPointInfos())
        loopAndCtrlKFIDsSet.insert(ctrl.kfID);

    int newKFID = 0;
    std::vector<std::pair<int, int>> oriAndNowKFIDs;
    std::vector<LidarFrameInfoStru> tmpLidarFrameInfos(
            rawData.getLidarFramesInfos().size());
    for (int oldID = rawData.getLidarFramesInfos().size() - 1; oldID >= 0; oldID--)
    {
        tmpLidarFrameInfos[newKFID] = rawData.getLidarFramesInfos()[oldID];
        tmpLidarFrameInfos[newKFID].kfID = newKFID;
        tmpLidarFrameInfos[newKFID].timestamp =
                rawData.getLidarFramesInfos()[newKFID].timestamp;  

        if (loopAndCtrlKFIDsSet.find(oldID) != loopAndCtrlKFIDsSet.end())
        {
            oriAndNowKFIDs.push_back(std::make_pair(oldID, newKFID));
        }
        newKFID++;
    }

    rawData.getLidarFramesInfos() = tmpLidarFrameInfos;


    for (auto &pair : rawData.getLoopPairInfos())
    {
        for (int ii = 0; ii < oriAndNowKFIDs.size(); ii++)
        {
            if (pair.prevKFID == oriAndNowKFIDs[ii].first)
                pair.prevKFID = oriAndNowKFIDs[ii].second;
            if (pair.curKFID == oriAndNowKFIDs[ii].first)
                pair.curKFID = oriAndNowKFIDs[ii].second;
        }
    }

    std::vector<ControlPointInfoStru> controlPointInfos(
            rawData.getControlPointInfos().size());
    int ctrlIdx = rawData.getControlPointInfos().size() - 1;
    for (auto &ctrl : rawData.getControlPointInfos())
    {
        for (int ii = 0; ii < oriAndNowKFIDs.size(); ii++)
        {
            if (ctrl.kfID == oriAndNowKFIDs[ii].first)
                ctrl.kfID = oriAndNowKFIDs[ii].second;
        }
        controlPointInfos[ctrlIdx] = ctrl;
        ctrlIdx--;
    }
    rawData.getControlPointInfos() = controlPointInfos;
}


void CRigelSLAMRawIOTools::GetRedundantFrameIDs(CRawFileData &baseRawData,
                                                CRawFileData &curRawData,
                                                std::vector<int> &redunIDs)
{
    auto &baseLidarFrameInfos = baseRawData.getLidarFramesInfos();
    auto &curLidarFrameInfos = curRawData.getLidarFramesInfos();

    if (baseLidarFrameInfos.empty() ||
        curLidarFrameInfos.empty())
        return;

    int minAnchorFrameID =
            INT_MAX; 
    for (auto &info : curRawData.getLoopPairInfos())
    {
        if (info.curKFID < minAnchorFrameID) minAnchorFrameID = info.curKFID;
        if (info.prevKFID < minAnchorFrameID) minAnchorFrameID = info.prevKFID;
    }
    for (auto &info : curRawData.getControlPointInfos())
    {
        if (info.kfID < minAnchorFrameID) minAnchorFrameID = info.kfID;
    }

    int maxSpan = 100;  
    int minDistFrameID = 0;
    float minDistSq = FLT_MAX;

    auto &curLidarFramesInfos = curRawData.getLidarFramesInfos();
    auto &baseLidarFramesInfos = baseRawData.getLidarFramesInfos();
    for (int i = 0; i < maxSpan; i++)
    {
        if (i >= curLidarFramesInfos.size()) continue;

        float dx = baseLidarFramesInfos.back().x -
                   curLidarFramesInfos[i].x;
        float dy = baseLidarFramesInfos.back().y -
                   curLidarFramesInfos[i].y;
        float dz = baseLidarFramesInfos.back().z -
                   curLidarFramesInfos[i].z;
        float distSq = dx * dx + dy * dy + dz * dz;
        if (distSq < minDistSq)
        {
            minDistFrameID = i;
            minDistSq = distSq;
        }
    }

    redunIDs.clear();

    const float minDistThre = 1.f;
    if (minDistFrameID == 0 ||
        minDistFrameID > minAnchorFrameID)  
        return;

    if (sqrt(minDistSq) < minDistThre)
    {
      
        for (int i = 0; i < minDistFrameID; i++) redunIDs.push_back(i);
    }
}

void CRigelSLAMRawIOTools::SetAffiliationInfo(const int AffiliID,
                                              CRawFileData &rawFileData)
{
    for (auto &frameInfo : rawFileData.getLidarFramesInfos())
    {
        frameInfo.affiliateRawID = AffiliID;
    }

    for (auto &pairInfo : rawFileData.getLoopPairInfos())
    {
        pairInfo.affiliateRawID = AffiliID;
    }

    for (auto &ctrlInfo : rawFileData.getControlPointInfos())
    {
        ctrlInfo.affiliateRawID = AffiliID;
    }

    rawFileData.getContinueTaskData().affiliateRawID = AffiliID;
}

void CRigelSLAMRawIOTools::SetGlobalID(const int offet,
                                       CRawFileData &rawFileData)
{
    for (auto &frameInfo : rawFileData.getLidarFramesInfos())
    {
        frameInfo.globalKFID = frameInfo.kfID + offet;
    }

    for (auto &pairInfo : rawFileData.getLoopPairInfos())
    {
        pairInfo.globalCurKFID = pairInfo.curKFID + offet;
        pairInfo.globalPrevKFID = pairInfo.prevKFID + offet;
    }

    for (auto &ctrlInfo : rawFileData.getControlPointInfos())
    {
        ctrlInfo.globalKFID = ctrlInfo.kfID + offet;
    }
}

void CRigelSLAMRawIOTools::SetMergedRawKfID(CRawFileData &mergedRaw)
{
    for (auto &frameInfo : mergedRaw.getLidarFramesInfos())
    {
        frameInfo.kfID = frameInfo.globalKFID;
    }

    for (auto &pairInfo : mergedRaw.getLoopPairInfos())
    {
        pairInfo.curKFID = pairInfo.globalCurKFID;
        pairInfo.prevKFID = pairInfo.globalPrevKFID;
    }

    for (auto &ctrlInfo : mergedRaw.getControlPointInfos())
    {
        ctrlInfo.kfID = ctrlInfo.globalKFID;
    }
}


READ_RAW_FILE_RETURN CRigelSLAMRawIOTools::readRawFile1_0(const std::string& filePath,
    CRawFileData& rawFileData,
    ProgressCallBackPtr pCallbackPtr)
{
    FILE* pFile = fopen(filePath.c_str(), "rb");
    if (pFile == NULL)
    {
        std::cout << "error: cannot open file[read]: " << filePath << std::endl;;
        return READ_RAW_FILE_RETURN::READ_OPEN_FAILED;
    }

    try
    {
        unsigned long long frameSize = 0;
        fread(&frameSize, sizeof(unsigned long long), 1, pFile);
        if (frameSize == 0)
        {
            std::cout << "error: the number of KF is 0\n";
            return READ_RAW_FILE_RETURN::EMPTY_RAW_FILE;
        }

        float pathLength = 0.f;
        for (unsigned long long i = 0; i < frameSize; i++)
        {
            LidarFrameInfoStru aKFInfo;
            aKFInfo.kfID = static_cast<int>(i);

            // read timestamp
            fread(&aKFInfo.timestamp, sizeof(double), 1, pFile);

            // read odometry
            fread(&aKFInfo.q_orientation.w(), sizeof(double), 1, pFile);
            fread(&aKFInfo.q_orientation.x(), sizeof(double), 1, pFile);
            fread(&aKFInfo.q_orientation.y(), sizeof(double), 1, pFile);
            fread(&aKFInfo.q_orientation.z(), sizeof(double), 1, pFile);
            aKFInfo.q_orientation.normalize();

            fread(&aKFInfo.x, sizeof(double), 1, pFile);
            fread(&aKFInfo.y, sizeof(double), 1, pFile);
            fread(&aKFInfo.z, sizeof(double), 1, pFile);

            // read full cloud
            unsigned long long fullCloudSize = 0;
            fread(&fullCloudSize, sizeof(unsigned long long), 1, pFile);
            aKFInfo.fullCloud->resize(fullCloudSize);
            for (unsigned long long j = 0; j < fullCloudSize; j++)
            {
                fread(&aKFInfo.fullCloud->points[j].x, sizeof(float), 1, pFile);
                fread(&aKFInfo.fullCloud->points[j].y, sizeof(float), 1, pFile);
                fread(&aKFInfo.fullCloud->points[j].z, sizeof(float), 1, pFile);
                fread(&aKFInfo.fullCloud->points[j].intensity, sizeof(float), 1, pFile);
            }

            // read corner cloud
            unsigned long long cornCloudSize = 0;
            fread(&cornCloudSize, sizeof(unsigned long long), 1, pFile);
            aKFInfo.cornCloud->resize(cornCloudSize);
            for (unsigned long long j = 0; j < cornCloudSize; j++)
            {
                fread(&aKFInfo.cornCloud->points[j].x, sizeof(float), 1, pFile);
                fread(&aKFInfo.cornCloud->points[j].y, sizeof(float), 1, pFile);
                fread(&aKFInfo.cornCloud->points[j].z, sizeof(float), 1, pFile);
                fread(&aKFInfo.cornCloud->points[j].intensity, sizeof(float), 1, pFile);
            }

            // read surface cloud
            unsigned long long surfCloudSize = 0;
            fread(&surfCloudSize, sizeof(unsigned long long), 1, pFile);
            aKFInfo.surfCloud->resize(surfCloudSize);
            for (unsigned long long j = 0; j < surfCloudSize; j++)
            {
                fread(&aKFInfo.surfCloud->points[j].x, sizeof(float), 1, pFile);
                fread(&aKFInfo.surfCloud->points[j].y, sizeof(float), 1, pFile);
                fread(&aKFInfo.surfCloud->points[j].z, sizeof(float), 1, pFile);
                fread(&aKFInfo.surfCloud->points[j].intensity, sizeof(float), 1, pFile);
            }

            auto& lidarFramesInfos = rawFileData.getLidarFramesInfos();
            lidarFramesInfos.push_back(aKFInfo);

            if (i > 0)
            {
                float dx = lidarFramesInfos[i].x -
                    lidarFramesInfos[i - 1].x;
                float dy = lidarFramesInfos[i].y -
                    lidarFramesInfos[i - 1].y;
                float dz = lidarFramesInfos[i].z -
                    lidarFramesInfos[i - 1].z;
                pathLength += sqrt(dx * dx + dy * dy + dz * dz);
            }

            if (pCallbackPtr)
            {
                int curProgress = static_cast<int>(i * 1.f * 100 / (frameSize + 5));
                std::string backStr = "Convert raw data";
                pCallbackPtr(backStr, curProgress);
            }
        }

        //      LoopPairInfoStru_1_0
        unsigned long long ctrlPtSize = 0;
        fread(&ctrlPtSize, sizeof(unsigned long long), 1, pFile);

        auto& controlPointInfos = rawFileData.getControlPointInfos();
        controlPointInfos.resize(ctrlPtSize);
        std::vector<ControlPointInfoStru_1_0> ctrlPtsInfos_1_0(ctrlPtSize);
        if (ctrlPtSize > 0)
        {
            fread(&ctrlPtsInfos_1_0[0], sizeof(ControlPointInfoStru_1_0), ctrlPtSize,
                pFile);

            for (unsigned long long ii = 0; ii < ctrlPtSize; ii++)
            {
                controlPointInfos[ii].affiliateRawID = 0;
                controlPointInfos[ii].kfID = ctrlPtsInfos_1_0[ii].kfID;
                controlPointInfos[ii].x = ctrlPtsInfos_1_0[ii].x;
                controlPointInfos[ii].y = ctrlPtsInfos_1_0[ii].y;
                controlPointInfos[ii].z = ctrlPtsInfos_1_0[ii].z;
            }
        }

        int loopPairNum = 0;
        fread(&loopPairNum, sizeof(int), 1, pFile);
        auto& loopPairInfos = rawFileData.getLoopPairInfos();
        loopPairInfos.resize(loopPairNum);

        for (int i = 0; i < loopPairNum; i++)
        {
            LoopPairInfoStru_1_0 loopInfo;
            fread(&loopInfo, sizeof(LoopPairInfoStru_1_0), 1, pFile);

            loopPairInfos[i].affiliateRawID = 0;
            loopPairInfos[i].curKFID = loopInfo.curKfId;
            loopPairInfos[i].prevKFID = loopInfo.prevKfId;
            loopPairInfos[i].noise = loopInfo.noise;
            memcpy(loopPairInfos[i].poseFrom, loopInfo.poseFrom,
                sizeof(float) * 6);
            memcpy(loopPairInfos[i].poseTo, loopInfo.poseTo,
                sizeof(float) * 6);
        }


        auto& fileHead = rawFileData.getFileHead();
        fileHead.version = RAWFileVersionEnum::VERSION_2_0;
        fileHead.lidarFrameNum = rawFileData.getLidarFramesInfos().size();
        fileHead.loopPairNum = loopPairNum;
        fileHead.controlPointsNum =
            rawFileData.getControlPointInfos().size();
        fileHead.scanDuration =
            rawFileData.getLidarFramesInfos().back().timestamp -
            rawFileData.getLidarFramesInfos().front().timestamp;

        fileHead.scanPathLength = pathLength;

        char chs[128] = { "Windows\0\0" };
        memcpy(fileHead.writeFileSystem, chs, sizeof(char) * 128);

        fclose(pFile);
        pFile = nullptr;

        if (pCallbackPtr)
        {
            int curProgress = 100;
            std::string backStr = "Convert raw data";
            pCallbackPtr(backStr, curProgress);
        }
    }
    catch (const std::exception&)
    {
        return READ_RAW_FILE_RETURN::READ_RAW_FAILED;
    }

    return READ_RAW_FILE_RETURN::READ_RAW_SUCCESSED;
}


READ_RAW_FILE_RETURN CRigelSLAMRawIOTools::readRawFile2_0(const std::string& filePath,
    CRawFileData& rawFileData,
    ProgressCallBackPtr pCallbackPtr)
{
    FILE* pFile = fopen(filePath.c_str(), "rb");
    if (pFile == nullptr)
    {
        std::cout << "error: cannot open file[read]: " << filePath << std::endl;
        return READ_RAW_FILE_RETURN::READ_OPEN_FAILED;
    }

    try {
        // 读文件头
        auto& fileHeader = rawFileData.getFileHead();

        fread(&fileHeader, sizeof(RawFileHeadStru), 1, pFile);

        // 读取2.0版本的raw
        if (fileHeader.lidarFrameNum == 0)
            return READ_RAW_FILE_RETURN::EMPTY_RAW_FILE;

        int frameNum = fileHeader.lidarFrameNum;
        for (int i = 0; i < frameNum; i++) {
            LidarFrameInfoStru aFrame;

            fread(&aFrame.affiliateRawID, sizeof(int), 1, pFile);

            fread(&aFrame.kfID, sizeof(int), 1, pFile);
            fread(&aFrame.globalKFID, sizeof(int), 1, pFile);

            fread(&aFrame.timestamp, sizeof(double), 1, pFile);

            double pose[7];
            fread(&pose, sizeof(double), 7, pFile);

            aFrame.q_orientation.w() = pose[0];
            aFrame.q_orientation.x() = pose[1];
            aFrame.q_orientation.y() = pose[2];
            aFrame.q_orientation.z() = pose[3];
            aFrame.x = pose[4];
            aFrame.y = pose[5];
            aFrame.z = pose[6];

            aFrame.q_orientation.normalize();

            // 读取GPS信息
            fread(&aFrame.bHaveGPSInfo, sizeof(bool), 1, pFile);
            if (aFrame.bHaveGPSInfo)
                fread(&aFrame.gpsInfo, sizeof(GPSInfoStru), 1, pFile);

            // 读取KF的raw cloud
            int fullCloudSize = 0;
            fread(&fullCloudSize, sizeof(int), 1, pFile);

            std::vector<float> tmpPts(fullCloudSize * 4);
            if (fullCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), fullCloudSize * 4, pFile);

            aFrame.fullCloud->resize(fullCloudSize);
            for (int j = 0; j < fullCloudSize; j++) {
                aFrame.fullCloud->points[j].x = tmpPts[j * 4 + 0];
                aFrame.fullCloud->points[j].y = tmpPts[j * 4 + 1];
                aFrame.fullCloud->points[j].z = tmpPts[j * 4 + 2];
                aFrame.fullCloud->points[j].intensity = tmpPts[j * 4 + 3];
            }

            // 读取KF的corner cloud
            int cornerCloudSize = 0;
            fread(&cornerCloudSize, sizeof(int), 1, pFile);

            tmpPts.resize(cornerCloudSize * 4);
            if (cornerCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), cornerCloudSize * 4, pFile);

            aFrame.cornCloud->resize(cornerCloudSize);
            for (int j = 0; j < cornerCloudSize; j++) {
                aFrame.cornCloud->points[j].x = tmpPts[j * 4 + 0];
                aFrame.cornCloud->points[j].y = tmpPts[j * 4 + 1];
                aFrame.cornCloud->points[j].z = tmpPts[j * 4 + 2];
                aFrame.cornCloud->points[j].intensity = tmpPts[j * 4 + 3];
            }

            // 读取KF的surface cloud
            int surfCloudSize = 0;
            fread(&surfCloudSize, sizeof(int), 1, pFile);

            tmpPts.resize(surfCloudSize * 4);
            if (surfCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), surfCloudSize * 4, pFile);

            aFrame.surfCloud->resize(surfCloudSize);
            for (int j = 0; j < surfCloudSize; j++) {
                aFrame.surfCloud->points[j].x = tmpPts[j * 4 + 0];
                aFrame.surfCloud->points[j].y = tmpPts[j * 4 + 1];
                aFrame.surfCloud->points[j].z = tmpPts[j * 4 + 2];
                aFrame.surfCloud->points[j].intensity = tmpPts[j * 4 + 3];
            }

            // 读取KF的colors cloud
            int colorCloudSize = 0;
            fread(&colorCloudSize, sizeof(int), 1, pFile);

            tmpPts.resize(colorCloudSize * 6);
            if (colorCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), colorCloudSize * 6, pFile);

            aFrame.coloredCloud->resize(colorCloudSize);
            for (unsigned long j = 0; j < colorCloudSize; j++) {
                aFrame.coloredCloud->points[j].x = tmpPts[j * 6 + 0];
                aFrame.coloredCloud->points[j].y = tmpPts[j * 6 + 1];
                aFrame.coloredCloud->points[j].z = tmpPts[j * 6 + 2];
                aFrame.coloredCloud->points[j].r = tmpPts[j * 6 + 3];
                aFrame.coloredCloud->points[j].g = tmpPts[j * 6 + 4];
                aFrame.coloredCloud->points[j].b = tmpPts[j * 6 + 5];
            }
            auto& lidarFramesInfos = rawFileData.getLidarFramesInfos();
            lidarFramesInfos.push_back(aFrame);

            if (pCallbackPtr) {
                int curProgress = static_cast<int>(i * 1.f * 100 / (frameNum + 3));
                std::string backStr = "Read raw data";
                pCallbackPtr(backStr, curProgress);
            }
        }  // end - for i

        // 读取闭环信息
        auto& loopPairInfos = rawFileData.getLoopPairInfos();
        int loopPairNum = 0;
        fread(&loopPairNum, sizeof(int), 1, pFile);
        if (loopPairNum > 0) {
            loopPairInfos.resize(loopPairNum);
            fread(&loopPairInfos[0], sizeof(LoopPairInfoStru), loopPairNum, pFile);
        }

        // 读取控制点信息
        auto& controlPointInfos = rawFileData.getControlPointInfos();
        int controlMarkerNum = controlPointInfos.size();
        fread(&controlMarkerNum, sizeof(int), 1, pFile);
        if (controlMarkerNum > 0) {
            controlPointInfos.resize(controlMarkerNum);
            fread(&controlPointInfos[0], sizeof(ControlPointInfoStru),
                controlMarkerNum, pFile);
        }

        // 读取断点续扫信息
        auto& continueTaskData = rawFileData.getContinueTaskData();
        fread(&continueTaskData.affiliateRawID, sizeof(int), 1, pFile);
        fread(&continueTaskData.isAvaliableContinueParams, sizeof(bool), 1, pFile);

        double tmpPose6D[6] = { 0. };
        fread(tmpPose6D, sizeof(double), 6, pFile);
        continueTaskData.refX = tmpPose6D[0];
        continueTaskData.refY = tmpPose6D[1];
        continueTaskData.refZ = tmpPose6D[2];

        continueTaskData.refRoll = tmpPose6D[3];
        continueTaskData.refPitch = tmpPose6D[4];
        continueTaskData.refYaw = tmpPose6D[5];

        // 读取submap的KF id
        std::vector<int> ids;
        int kfNum = 0;
        fread(&kfNum, sizeof(int), 1, pFile);
        if (kfNum > 0)
            fread(&ids[0], sizeof(int), kfNum, pFile);

        // 读取Odometry
        int odometry_num = fileHeader.odometryNum;
        if (odometry_num > 0) {
            auto& odeometry_infos = rawFileData.getOdometryInfos();
            odeometry_infos.resize(odometry_num);
            fread(odeometry_infos.data(), sizeof(odeometry_infos[0]), odometry_num, pFile);
        }

        if (pCallbackPtr) {
            int curProgress = 100;
            std::string backStr = "Read raw data";
            pCallbackPtr(backStr, curProgress);
        }

        fclose(pFile);
    }
    catch (const std::exception&) 
    {
        return READ_RAW_FILE_RETURN::READ_RAW_FAILED;
    }

    return READ_RAW_FILE_RETURN::READ_RAW_SUCCESSED;
}


READ_RAW_FILE_RETURN CRigelSLAMRawIOTools::readRawFile3_0(const std::string& filePath,
    CRawFileData& rawFileData,
    ProgressCallBackPtr pCallbackPtr)
{

    FILE* pFile = fopen(filePath.c_str(), "rb");
    if (pFile == nullptr)
    {
        std::cout << "error: cannot open file[read]: " << filePath << std::endl;
        return READ_RAW_FILE_RETURN::READ_OPEN_FAILED;
    }

    try
    {
        // 读文件头
        auto& fileHeader = rawFileData.getFileHead();

        fread(&fileHeader, sizeof(RawFileHeadStru), 1, pFile);

        if (fileHeader.lidarFrameNum == 0)
            return READ_RAW_FILE_RETURN::EMPTY_RAW_FILE;

        int frameNum = fileHeader.lidarFrameNum;
        for (int i = 0; i < frameNum; i++)
        {
            LidarFrameInfoStru aFrame;

            fread(&aFrame.affiliateRawID, sizeof(int), 1, pFile);

            fread(&aFrame.kfID, sizeof(int), 1, pFile);
            fread(&aFrame.globalKFID, sizeof(int), 1, pFile);

            fread(&aFrame.timestamp, sizeof(double), 1, pFile);

            double pose[7];
            fread(&pose, sizeof(double), 7, pFile);

            aFrame.q_orientation.w() = pose[0];
            aFrame.q_orientation.x() = pose[1];
            aFrame.q_orientation.y() = pose[2];
            aFrame.q_orientation.z() = pose[3];
            aFrame.x = pose[4];
            aFrame.y = pose[5];
            aFrame.z = pose[6];

            aFrame.q_orientation.normalize();

            fread(&aFrame.Til, sizeof(Eigen::Matrix4d), 1, pFile);

            fread(&aFrame.bHaveGPSInfo, sizeof(bool), 1, pFile);
            if (aFrame.bHaveGPSInfo)
                fread(&aFrame.gpsInfo, sizeof(GPSInfoStru), 1, pFile);

            int fullCloudSize = 0;
            fread(&fullCloudSize, sizeof(int), 1, pFile);

            std::vector<float> tmpPts(fullCloudSize * 5);
            if (fullCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), fullCloudSize * 5, pFile);

            aFrame.fullCloud->resize(fullCloudSize);
            for (int j = 0; j < fullCloudSize; j++)
            {
                aFrame.fullCloud->points[j].x = tmpPts[j * 5 + 0];
                aFrame.fullCloud->points[j].y = tmpPts[j * 5 + 1];
                aFrame.fullCloud->points[j].z = tmpPts[j * 5 + 2];
                aFrame.fullCloud->points[j].intensity = tmpPts[j * 5 + 3];
                aFrame.fullCloud->points[j].curvature = tmpPts[j * 5 + 4];
            }


            int cornerCloudSize = 0;
            fread(&cornerCloudSize, sizeof(int), 1, pFile);

            tmpPts.resize(cornerCloudSize * 5);
            if (cornerCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), cornerCloudSize * 5, pFile);

            aFrame.cornCloud->resize(cornerCloudSize);
            for (int j = 0; j < cornerCloudSize; j++)
            {
                aFrame.cornCloud->points[j].x = tmpPts[j * 5 + 0];
                aFrame.cornCloud->points[j].y = tmpPts[j * 5 + 1];
                aFrame.cornCloud->points[j].z = tmpPts[j * 5 + 2];
                aFrame.cornCloud->points[j].intensity = tmpPts[j * 5 + 3];
                aFrame.cornCloud->points[j].curvature = tmpPts[j * 5 + 4];
            }


            int surfCloudSize = 0;
            fread(&surfCloudSize, sizeof(int), 1, pFile);

            tmpPts.resize(surfCloudSize * 5);
            if (surfCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), surfCloudSize * 5, pFile);

            aFrame.surfCloud->resize(surfCloudSize);
            for (int j = 0; j < surfCloudSize; j++)
            {
                aFrame.surfCloud->points[j].x = tmpPts[j * 5 + 0];
                aFrame.surfCloud->points[j].y = tmpPts[j * 5 + 1];
                aFrame.surfCloud->points[j].z = tmpPts[j * 5 + 2];
                aFrame.surfCloud->points[j].intensity = tmpPts[j * 5 + 3];
                aFrame.surfCloud->points[j].curvature = tmpPts[j * 5 + 4];
            }


            int colorCloudSize = 0;
            fread(&colorCloudSize, sizeof(int), 1, pFile);

            tmpPts.resize(colorCloudSize * 6);
            if (colorCloudSize > 0)
                fread(&tmpPts[0], sizeof(float), colorCloudSize * 6, pFile);

            aFrame.coloredCloud->resize(colorCloudSize);
            for (unsigned long j = 0; j < colorCloudSize; j++)
            {
                aFrame.coloredCloud->points[j].x = tmpPts[j * 6 + 0];
                aFrame.coloredCloud->points[j].y = tmpPts[j * 6 + 1];
                aFrame.coloredCloud->points[j].z = tmpPts[j * 6 + 2];
                aFrame.coloredCloud->points[j].r = tmpPts[j * 6 + 3];
                aFrame.coloredCloud->points[j].g = tmpPts[j * 6 + 4];
                aFrame.coloredCloud->points[j].b = tmpPts[j * 6 + 5];
            }
            auto& lidarFramesInfos = rawFileData.getLidarFramesInfos();
            lidarFramesInfos.push_back(aFrame);

            if (pCallbackPtr)
            {
                int curProgress = static_cast<int>(i * 1.f * 100 / (frameNum + 3));
                std::string backStr = "Read raw data";
                pCallbackPtr(backStr, curProgress);
            }
        }  // end - for i


        auto& loopPairInfos = rawFileData.getLoopPairInfos();
        int loopPairNum = 0;
        fread(&loopPairNum, sizeof(int), 1, pFile);
        if (loopPairNum > 0)
        {
            loopPairInfos.resize(loopPairNum);
            fread(&loopPairInfos[0], sizeof(LoopPairInfoStru), loopPairNum, pFile);
        }


        auto& controlPointInfos = rawFileData.getControlPointInfos();
        int controlMarkerNum = controlPointInfos.size();
        fread(&controlMarkerNum, sizeof(int), 1, pFile);
        if (controlMarkerNum > 0)
        {
            controlPointInfos.resize(controlMarkerNum);
            fread(&controlPointInfos[0], sizeof(ControlPointInfoStru), controlMarkerNum, pFile);
        }


        auto& continueTaskData = rawFileData.getContinueTaskData();
        fread(&continueTaskData.affiliateRawID, sizeof(int), 1, pFile);
        fread(&continueTaskData.isAvaliableContinueParams, sizeof(bool), 1, pFile);

        double tmpPose6D[6] = { 0. };
        fread(tmpPose6D, sizeof(double), 6, pFile);
        continueTaskData.refX = tmpPose6D[0];
        continueTaskData.refY = tmpPose6D[1];
        continueTaskData.refZ = tmpPose6D[2];

        continueTaskData.refRoll = tmpPose6D[3];
        continueTaskData.refPitch = tmpPose6D[4];
        continueTaskData.refYaw = tmpPose6D[5];


        int kfNum = 0;
        fread(&kfNum, sizeof(int), 1, pFile);
        // if (kfNum > 0)
        //     fread(&continueTaskData.submapFrameIDs[0], sizeof(int), kfNum, pFile);


        int odometry_num = fileHeader.odometryNum;
        if (odometry_num > 0)
        {
            auto& odeometry_infos = rawFileData.getOdometryInfos();
            odeometry_infos.resize(odometry_num);
            fread(odeometry_infos.data(), sizeof(odeometry_infos[0]), odometry_num, pFile);
        }
        int imuNum = fileHeader.imuNum;
        if (imuNum > 0)
        {
            auto& imuInfos = rawFileData.getIMUInfos();
            imuInfos.resize(imuNum);
            fread(imuInfos.data(), sizeof(imuInfos[0]), imuNum, pFile);
        }
        //        auto &TilList = rawFileData.getExtrinsicInfos();
        //        if (frameNum!=0)
        //        {
        //            TilList.resize(frameNum);
        //            fread(TilList.data(), sizeof(TilList[0]), frameNum, pFile);
        //        }
        if (pCallbackPtr)
        {
            int curProgress = 100;
            std::string backStr = "Read raw data";
            pCallbackPtr(backStr, curProgress);
        }

        fclose(pFile);
    }
    catch (const std::exception&)
    {
        return READ_RAW_FILE_RETURN::READ_RAW_FAILED;
    }

    return READ_RAW_FILE_RETURN::READ_RAW_SUCCESSED;
}


RAW_SPLIT_RETURN CRigelSLAMRawIOTools::RawSplit(
        CRawFileData &mergedRawFileData, std::vector<CRawFileData> &rawFileDatas)
{
    if (mergedRawFileData.getLidarFramesInfos().empty())
        return RAW_SPLIT_RETURN::INPUT_RAW_IS_EMPTY;

    rawFileDatas.clear();


    CRawFileData aRaw;
    int lastRawID = mergedRawFileData.getLidarFramesInfos()[0].affiliateRawID;
    for (auto &frameInfo : mergedRawFileData.getLidarFramesInfos())
    {
        if (frameInfo.affiliateRawID != lastRawID &&
            !aRaw.getLidarFramesInfos().empty())
        {
            rawFileDatas.push_back(aRaw);

            aRaw.Clear();
            lastRawID = frameInfo.affiliateRawID;
        }
        aRaw.getLidarFramesInfos().push_back(frameInfo);
    }


    if (!aRaw.getLidarFramesInfos().empty())
    {
        rawFileDatas.push_back(aRaw);
        aRaw.Clear();
    }

    for (auto &loopInfo : mergedRawFileData.getLoopPairInfos())
    {
        int rawID = loopInfo.affiliateRawID;
        if (rawID < 0 || rawID >= rawFileDatas.size())
            return RAW_SPLIT_RETURN::SPLIT_FAILED;

        rawFileDatas[rawID].getLoopPairInfos().push_back(loopInfo);
    }

    for (auto &ctrlPtInfo : mergedRawFileData.getControlPointInfos())
    {
        int rawID = ctrlPtInfo.affiliateRawID;
        if (rawID < 0 || rawID >= rawFileDatas.size())
            return RAW_SPLIT_RETURN::SPLIT_FAILED;

        rawFileDatas[rawID].getControlPointInfos().push_back(ctrlPtInfo);
    }

    for (auto &raw : rawFileDatas) SetRawHeadAttribute(raw);

    return RAW_SPLIT_RETURN::SPLIT_SUCCESSED;
}

void CRigelSLAMRawIOTools::SetRawHeadAttribute(CRawFileData &rawData)
{
    auto &fileHead = rawData.getFileHead();
    auto &lidarFramesInfos = rawData.getLidarFramesInfos();

    fileHead.version = RAWFileVersionEnum::VERSION_2_0;
    fileHead.lidarFrameNum = rawData.getLidarFramesInfos().size();
    fileHead.loopPairNum = rawData.getLoopPairInfos().size();
    fileHead.controlPointsNum = rawData.getControlPointInfos().size();
    fileHead.scanDuration = lidarFramesInfos.back().timestamp -
                            lidarFramesInfos.front().timestamp;
    fileHead.odometryNum = rawData.getOdometryInfos().size();

    char chs[128] = {"Windows\0\0"};
    memcpy(fileHead.writeFileSystem, chs, sizeof(char) * 128);

    float pathLength = 0.f;

    for (int i = 1; i < lidarFramesInfos.size(); i++)
    {
        float dx =
                lidarFramesInfos[i].x - lidarFramesInfos[i - 1].x;
        float dy =
                lidarFramesInfos[i].y - lidarFramesInfos[i - 1].y;
        float dz =
                lidarFramesInfos[i].z - lidarFramesInfos[i - 1].z;
        pathLength += sqrt(dx * dx + dy * dy + dz * dz);
    }

    fileHead.scanPathLength = pathLength;
    fileHead.isContinueTask = false;
    fileHead.isPlaybackTask = false;
    fileHead.isEmergedRawFile = false;
}