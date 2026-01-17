#ifndef RIGEL_SLAM_RAW_IO_TOOLS_H
#define RIGEL_SLAM_RAW_IO_TOOLS_H

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <set>
#include <vector>


#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define AFX_RIGELSEE_OPTIMISE_CLASS __declspec(dllexport)
#else
#define AFX_RIGELSEE_OPTIMISE_CLASS __declspec(dllimport)
#endif
#else
#define AFX_RIGELSEE_OPTIMISE_CLASS
#endif


enum RAWFileVersionEnum
{
    VERSION_1_0 = 1,  // 1.0
    VERSION_2_0 = 2,   // 2.0
    VERSION_3_0 = 3   // 3.0
};


typedef void (*ProgressCallBackPtr)(const std::string &, int);


enum READ_RAW_FILE_RETURN
{
    READ_OPEN_FAILED,  
    READ_RAW_FAILED,  
    EMPTY_RAW_FILE,  
    READ_RAW_SUCCESSED 
};


enum WRITE_RAW_FILE_RETURN
{
    WRITE_OPEN_FAILED,  
    EMPTY_RAW_DATA,      
    WRITE_RAW_SUCCESSED 
};


enum RAW_CONVERT_RETURN
{
    IS_LATEST_VERSION,   
    OLD_RAW_OPEN_FAILED,  
    OLD_RAW_EMPTY,        
    CONVERT_FAILED,      
    
    CONVERT_SUCCESSED  
};


enum RAW_SPLIT_RETURN
{
    INPUT_RAW_IS_EMPTY,
    DO_NOT_SPLIT,       
    SPLIT_FAILED,       
    SPLIT_SUCCESSED      
};

typedef struct RawFileHeadStru
{
   
    RAWFileVersionEnum version;

    char deviceType[128];  // or RigelSLAM, RigelSLAM-Pano
    char writeFileSystem[128];  
    char scanSceneName[128];  

    char scanTime[128];    
    int lidarFrameNum = 0;  

    float scanDuration = 0.f;    
    float scanPathLength = 0.f;  

    bool isContinueTask = false;   
    bool isPlaybackTask = false;    
    bool isEmergedRawFile = false;  

    int loopPairNum = 0;      
    int controlPointsNum = 0; 

    // char freeSpace[512];  /
    int odometryNum = 0;     

    int imuNum = 0;    

    char freeSpace[504];     

    RawFileHeadStru()
    {
        version = RAWFileVersionEnum::VERSION_2_0;

        memset(writeFileSystem, '\0', sizeof(writeFileSystem));
        strcpy(writeFileSystem, "Linux");

        memset(scanSceneName, '\0', sizeof(scanSceneName));
        strcpy(scanSceneName, "Street");

        memset(scanTime, '\0', sizeof(scanTime));
        strcpy(scanTime, "2023-04-17 14:27");

        memset(freeSpace, '\0', sizeof(freeSpace));
    }
} RawFileHead;

typedef struct IMUInfoStru
{
    double timestamp;
    Eigen::Vector3d angularVelocity;
    Eigen::Vector3d linearAcceleration;

    IMUInfoStru()
    {
        timestamp = 0.f;
        angularVelocity=Eigen::Vector3d::Zero();
        linearAcceleration=Eigen::Vector3d::Zero();
    }

    IMUInfoStru(const double& time, const Eigen::Vector3d& angVel, const Eigen::Vector3d& linearAcc)
    {
        timestamp = time;
        angularVelocity=angVel;
        linearAcceleration=linearAcc;
    }
} IMUInfo;

typedef struct GPSInfoStru
{
    bool bAvaliable;
    double timestamp;
    int solutionType;  
    double gpsX, gpsY, gpsZ;       
    double longitude, latitude;    
    float noiseX, noiseY, noiseZ; 
    double gpsVelocity;           
    double pseudorange;            
    GPSInfoStru()
    {
        bAvaliable = false;
        timestamp = -1.0;
        solutionType = 0;
        gpsX = gpsY = gpsZ = 0.;
        longitude = latitude = 0.;
        noiseX = noiseY = noiseZ = 1e-3;
        gpsVelocity = 0;
        pseudorange = 0.;
    }

    GPSInfoStru(double time, double x, double y, double z, double noiX,
                double noiY, double noiZ)
            : timestamp(time),
              gpsX(x),
              gpsY(y),
              gpsZ(z),
              noiseX(noiX),
              noiseY(noiY),
              noiseZ(noiZ) {}
} GPSInfo;

#ifndef ODOMETRY_TYPE
#define ODOMETRY_TYPE
typedef struct OdometryInfoStru
{
    double second_;
    Eigen::Vector3d position_;
    Eigen::Quaterniond orientation_;
    Eigen::Vector3d velocity_;
    Eigen::Vector3d accelerate_;
    Eigen::Vector3d angular_velocity_;
    Eigen::Vector3d ba_;
    Eigen::Vector3d bg_;
} OdometryInfo;
#endif 

typedef struct LidarFrameInfoStru
{
    int affiliateRawID;  

    int kfID;        
    int globalKFID;  

    double timestamp; 

    Eigen::Quaterniond q_orientation;  
    double x, y, z;                  
 
    bool bHaveGPSInfo;  
    GPSInfoStru gpsInfo;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr fullCloud;          
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cornCloud;          
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr surfCloud;         
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud; 

    Eigen::Matrix4d Til;

    LidarFrameInfoStru()
    {
        affiliateRawID = 0;
        timestamp = -1;

        q_orientation = Eigen::Quaterniond::Identity();
        x = y = z = 0.0;
        Til=Eigen::Matrix4d::Identity();
        bHaveGPSInfo = false;

        cornCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        surfCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        fullCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        coloredCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    }

    void Clear()
    {
        pcl::PointCloud<pcl::PointXYZINormal>().swap(*cornCloud);
        pcl::PointCloud<pcl::PointXYZINormal>().swap(*surfCloud);
        pcl::PointCloud<pcl::PointXYZINormal>().swap(*fullCloud);
        pcl::PointCloud<pcl::PointXYZRGB>().swap(*coloredCloud);
    }
} LidarFrameInfo;

typedef struct LoopPairInfoStru
{
    int affiliateRawID = 0;  

    bool bAvaliable = true; 
    double timestamp;        

    int curKFID = -1;   
    int prevKFID = -1;  

    int globalCurKFID = -1;   
    int globalPrevKFID = -1;  

    float poseFrom[6] = {0.f};  
    float poseTo[6] = {0.f};

    float noise = 1e-5;  
    void SetNoise(const float noi) { noise = noi; }
} LoopPairInfo;

typedef struct ControlPointInfoStru
{
    int affiliateRawID = 0;  

    bool bAvaliable = true; 

    double collectTimestamp = 0.; 
    int kfID = -1;                 
    int globalKFID = -1;           

    float x, y, z;                 
    float noiseX, noiseY, noiseZ; 

    ControlPointInfoStru()
    {
        x = y = z = 0.f;
        noiseX = noiseY = noiseZ = 8e-3;
    }

    ControlPointInfoStru(double time, int id, float aX, float aY, float aZ)
            : collectTimestamp(time), kfID(id), x(aX), y(aY), z(aZ)
    {
        noiseX = noiseY = noiseZ = 8e-3;
    }

    void SetNoise(const float noiX, const float noiY, const float noiZ)
    {
        noiseX = noiX;
        noiseY = noiY;
        noiseZ = noiZ;
    }
} ControlPointInfo;

typedef struct ContinueTaskInfoStru
{
    int affiliateRawID; 

    bool isAvaliableContinueParams;
    double refX, refY, refZ;
    double refRoll, refPitch, refYaw;
  

    int submapKFNum;                
    //std::vector<int> submapFrameIDs; 

    ContinueTaskInfoStru()
    {
        affiliateRawID = 0;
        isAvaliableContinueParams = true;
        refX = refY = refZ = 0;
        refRoll = refPitch = refYaw = 0;
        submapKFNum = 0;
    }

    void Clear()
    {
        affiliateRawID = 0;
        isAvaliableContinueParams = true;
        refX = refY = refZ = 0;
        refRoll = refPitch = refYaw = 0;
        submapKFNum = 0;
        //submapFrameIDs.clear();
    }

} ContinueTaskInfo;

class AFX_RIGELSEE_OPTIMISE_CLASS CRawFileData
{
public:
    CRawFileData();

    ~CRawFileData();

public:

    RawFileHeadStru &getFileHead();

    std::vector<LidarFrameInfoStru> &getLidarFramesInfos(); 
    std::vector<LoopPairInfoStru> &getLoopPairInfos();
    std::vector<ControlPointInfoStru> &getControlPointInfos(); 
    std::vector<OdometryInfoStru> &getOdometryInfos();   
    std::vector<IMUInfo>& getIMUInfos();
//    std::vector<Eigen::Matrix4d>& getExtrinsicInfos();

    ContinueTaskInfo &getContinueTaskData(); 
    void Clear();

private:
    class RawFileDataImpl;

    RawFileDataImpl *impl_;
};

class AFX_RIGELSEE_OPTIMISE_CLASS CRigelSLAMRawIOTools
{
public:
    CRigelSLAMRawIOTools();

    ~CRigelSLAMRawIOTools();

public:
    static READ_RAW_FILE_RETURN ReadRawFile(
            const std::string &filePath, CRawFileData &rawFileData,
            ProgressCallBackPtr pCallbackPtr = nullptr);

    static WRITE_RAW_FILE_RETURN WriteRawFile(
            const std::string &filePath, CRawFileData &rawFileData,
            ProgressCallBackPtr pCallbackPtr = nullptr);

    static bool SetTransform(const Eigen::Matrix3d &rotMatrix,
                             const Eigen::Vector3d &tPose,
                             CRawFileData &rawFileData);


    static bool MultipleRawMerge(std::vector<CRawFileData> &rawFileDatas,
                                 CRawFileData &mergedRawFileData,
                                 ProgressCallBackPtr pCallbackPtr = nullptr);


    static RAW_SPLIT_RETURN RawSplit(CRawFileData &mergedRawFileData,
                                     std::vector<CRawFileData> &rawFileDatas);


    static void RawInvert(CRawFileData &rawData);

private:

    static READ_RAW_FILE_RETURN readRawFile1_0(
            const std::string &filePath, CRawFileData &rawFileData,
            ProgressCallBackPtr pCallbackPtr = nullptr);

    static READ_RAW_FILE_RETURN readRawFile2_0(
        const std::string& filePath, CRawFileData& rawFileData,
        ProgressCallBackPtr pCallbackPtr = nullptr);

    static READ_RAW_FILE_RETURN readRawFile3_0(
        const std::string& filePath, CRawFileData& rawFileData,
        ProgressCallBackPtr pCallbackPtr = nullptr);

    static void SetAffiliationInfo(const int AffiliID, CRawFileData &rawFileData);

    static void SetGlobalID(const int offet, CRawFileData &rawFileData);


    static void SetMergedRawKfID(CRawFileData &mergedRaw);

    static void GetRedundantFrameIDs(CRawFileData &baseRawData,
                                     CRawFileData &curRawData,
                                     std::vector<int> &redunIDs);

    static void SetRawHeadAttribute(CRawFileData &rawData);
};

#endif  // RIGEL_SLAM_RAW_IO_TOOLS_H
