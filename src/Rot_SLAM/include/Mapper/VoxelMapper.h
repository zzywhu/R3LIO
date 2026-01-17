#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP

#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <pcl/common/io.h>
#include <pcl/search/kdtree.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <float.h>

#include "Misc/Common.h"
#include "IkdTree.h"
#include "TicToc.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define VOXEL_MAP_UTIL_API __declspec(dllexport)
#else
#define VOXEL_MAP_UTIL_API __declspec(dllimport)
#endif
#else
#define VOXEL_MAP_UTIL_API
#endif

#define HASH_P 116101
#define MAX_N 10000000000

static int _planeId = 0;

// a point to plane matching structure
class OctoTree;

typedef struct MatchOctoTreeInfo
{
    PointWithCov pv;
    Eigen::Vector3d point;
    Eigen::Vector3d normal;
    Eigen::Vector3d center;
    Eigen::Matrix<double, 6, 6> plane_cov;
    double d;
    int layer;
    OctoTree *voxel_correspondence;
    double R;
} ptpl;


enum class NeighborSearchMethod
{
    DIRECT27, DIRECT7, DIRECT1,
};

class VOXEL_MAP_UTIL_API VOXEL_LOC
{
public:
    int64_t x, y, z;

    VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
            : x(vx), y(vy), z(vz)
    {}

    bool operator==(const VOXEL_LOC &other) const
    { return (x == other.x && y == other.y && z == other.z); }

    const VOXEL_LOC operator+=(const Eigen::Vector3i &add) const
    {
        return VOXEL_LOC(x + (int64_t) add[0], y + (int64_t) add[0], z + (int64_t) add[0]);
    }
};

typedef struct Voxel
{
    float size;
    Eigen::Vector3d voxel_origin;
    Eigen::Vector3d voxel_color;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    Voxel(float _size) : size(_size)
    {
        voxel_origin << 0, 0, 0;
        cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    };
} Voxel;

struct M_POINT
{
    float xyz[3];
    int count = 0;
};

// Hash value
namespace std
{
    template<>
    struct hash<VOXEL_LOC>
    {
        int64_t operator()(const VOXEL_LOC &s) const
        {
            using std::hash;
            using std::size_t;
            return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
        }
    };
} // namespace std

// P_fix in the paper
// Summation of P_fix
class VOXEL_MAP_UTIL_API SigVecClass
{
public:
    Eigen::Matrix3d _sigmaCov;
    Eigen::Vector3d _sigmaCenter;
    int _sigmaSize;

    SigVecClass()
    {
        _sigmaCov.setZero();
        _sigmaCenter.setZero();
        _sigmaSize = 0;
    }

    void tozero()
    {
        _sigmaCov.setZero();
        _sigmaCenter.setZero();
        _sigmaSize = 0;
    }
};

#ifdef BA_ENABLE

class LM_SLWD_VOXEL;

typedef struct Plane
{
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Vector3d yNormal;
    Eigen::Vector3d xNormal;
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 6, 6> planeCov;
    float radius = 0;
    float minEigenValue = 1;
    float midEigenValue = 1;
    float maxEigenValue = 1;
    float d = 0;
    int pointsSize = 0;

    bool isPlane = false;
    bool isInit = false;
    int id;
} Plane;

class VOXEL_MAP_UTIL_API OctoTree
{
public:
    OctoTree(const int& layer,
             const int& maxLayers,
             const std::vector<int>& layerPointSize,
             const float& planerThreshold,
             const int& capacity);

    void transPoints(PvList& cloud, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

    // check is plane , calc plane parameters including plane covariance
    void calcPlaneParams(const std::vector<PvList*>& pointsVec, int minWinIdx, int maxWinIdx);

    void cutOctoTree(uint frameHead);

    void marginalize(const int& margiSize,
                     const int& windowBase,
                     const vector<Eigen::Quaterniond>& qPoses,
                     const vector<Eigen::Vector3d>& tPoses);

    int getWinCloudSize();

    void traversalOpt(LM_SLWD_VOXEL* optLsv);

public:
    static int              _voxelWindowSize;
    std::vector<PvList*>     _winCloudVec; // window points in an octo tree
    PvList                   _marginCloud;
    int                     _capacity;
    float                   _planerThreshold;
    int                     _planePointsThres;
    int                     _updateSizeThres;

    int                     _maxLayers;
    int                     _layer;
    int                     _octoState; // 0 is end of tree, 1 is not
    bool                    _isObs;

    PvList                   _marginPointsVec;
    SigVecClass             _marginParams;

    double                  _voxelCenter[3]; // x, y, z
    std::vector<int>        _layerPointSizeList;
    float                   _quaterLength;

    Plane*                  _planePtr;
    OctoTree*               _leaves[8];

    int                     _kCorrespondences;
    Eigen::Matrix4d         _cloudCov;
    Eigen::Vector4d         _mean;
};

void VOXEL_MAP_UTIL_API updateVoxelMap(unordered_map<VOXEL_LOC, OctoTree*>& featMap,
                    const std::vector<PointWithCov>& pvList,
                    const int& winIdx,
                    const int& capacity,
                    const float& voxelSize,
                    const int& maxLayer,
                    const std::vector<int>& layerPointSizeList,
                    const float& planerThreshold);

void VOXEL_MAP_UTIL_API marginalizeVoxelMap(unordered_map<VOXEL_LOC, OctoTree*>& featMap,
                         const int& margiSize,
                         const int& windowBase,
                         vector<Eigen::Quaterniond> &qPoses,
                         vector<Eigen::Vector3d> &tPoses);

#else
typedef struct PlaneParams
{
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Vector3d yNormal;
    Eigen::Vector3d xNormal;
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 6, 6> planeCov;
    float radius = 0;
    Eigen::Matrix3cd evecs;
    Eigen::Vector3cd evals;
    float minEigenValue = 1;
    float midEigenValue = 1;
    float maxEigenValue = 1;
    float d = 0;
    int pointsSize = 0;

    bool isPlane = false;
    bool isInit = false;
    int id;
    // isUpdate and lastUpdatePointsSize are only for publish plane
    bool isUpdate = false;
    int lastUpdatePointsSize = 0;
    bool updateEnable = true;
} PlaneParams;

typedef struct LineParams
{
    Eigen::Vector3d center;
    Eigen::Vector3d direct;
    Eigen::Matrix3d covariance;
    float minEigenValue = 1;
    float midEigenValue = 1;
    float maxEigenValue = 1;
    int pointsSize = 0;

    bool isLine = false;
    bool isInit = false;
    int id;
    // isUpdate and lastUpdatePointsSize are only for publish plane
    bool isUpdate = false;
    int lastUpdatePointsSize = 0;
    bool updateEnable = true;
} LineParams;

class VOXEL_MAP_UTIL_API OctoTree
{
public:
    OctoTree(const int refFrameID,
             const FeatType& featType,
             const int& maxLayer,
             const int& layer,
             const std::vector<int>& layerPointSize,
             const int& maxPointSize,
             const int& maxCovPointsSize,
             const float& featThreshold) : _refFrameID(refFrameID),
                                           _featType(featType),
                                           _maxLayers(maxLayer),
                                           _layer(layer),
                                           _layerPointSizeList(layerPointSize),
                                           _maxPointsSize(maxPointSize),
                                           _maxCovPointsSize(maxCovPointsSize),
                                           _featThreshold(featThreshold),
                                           _linePtr(nullptr),
                                           _planePtr(nullptr)
    {
        _tempPoints.clear();
        _octoState = 0;
        _newPointsNum = 0;
        _allPointsNum = 0;
        // when new points num > 5, do a update
        _updateSizeThreshold = 5;
        _initOcto = false;
        _isUpdateEnable = true;
        _isUpdateCovEnable = true;
        _minFeatUpdateThreshold = _layerPointSizeList[_layer];

        for (int i = 0; i < 8; i++)
            _leaves[i] = nullptr;
        if(_featType==Line)
            _linePtr = new LineParams;
        else if(_featType==Plane)
            _planePtr = new PlaneParams;

        _kCorrespondences = 20;
        _cloudCov = Eigen::Matrix4d::Zero();
        _nVisible++;
    }

    ~OctoTree()
    {
        //std::vector<PointWithCov>().swap(_tempPoints);
        //std::vector<PointWithCov>().swap(_newPoints);
        if(_planePtr)
        {
            delete _planePtr;
            _planePtr= nullptr;
        }
        if(_linePtr)
        {
            delete _linePtr;
            _linePtr= nullptr;
        }
        
        deleteLeaves();
    }

    void deleteLeaves()
    {
//#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
        for(int i=0;i<8;i++)
        {
            if(_leaves[i])
            {
                delete _leaves[i];
                _leaves[i]= nullptr;
            }
        }
    }

    void updatePlaneCov(std::vector<PointWithCov>& pvList, PlaneParams* plane);

    void calcNeighCov(std::vector<PointWithCov>& pvList, PointVector& cloud);

    void calcNeighCov(std::vector<PointWithCov>& pvList);

    void calcNeighCov(std::vector<PointWithCov>& pvList , KD_TREE* ikdTree);

    // check is plane , calc plane parameters including plane covariance
    void initPlane(std::vector<PointWithCov> &points, PlaneParams *plane);

    // check is line , calc line parameters including line covariance
    void initLine(std::vector<PointWithCov> &pvList, LineParams *line);

    // only update plane normal, center and radius with new points
    void updatePlane(std::vector<PointWithCov> &points, PlaneParams *plane);

    void updateLine(std::vector<PointWithCov> &pvList, LineParams *line);

    void initOctoTree();

    void cutOctoTree();

    void updateOctoTree(const PointWithCov &pv);

    int getVoxelObsCount(const int& nObs);

public:
    FeatType _featType;
    std::vector<PointWithCov> _tempPoints; // all points in an octo tree
    std::vector<PointWithCov> _newPoints;  // new points in an octo tree
    PlaneParams* _planePtr;
    LineParams* _linePtr;
    int _maxLayers;
    int _layer;
    int _octoState; // 0 is end of tree, 1 is not
    OctoTree *_leaves[8];
    double _voxelCenter[3]; // x, y, z
    std::vector<int> _layerPointSizeList;
    float _quaterLength;
    float _featThreshold;
    int _minFeatUpdateThreshold;
    int _updateSizeThreshold;
    int _allPointsNum;
    int _newPointsNum;
    int _maxPointsSize;
    int _maxCovPointsSize;
    bool _initOcto;
    bool _isUpdateCovEnable;
    bool _isUpdateEnable;
    int  _nVisible=0;
    int  _refFrameID=-1;

    int _kCorrespondences;
    Eigen::Matrix4d _cloudCov;
    Eigen::Vector4d _mean;
};

void VOXEL_MAP_UTIL_API buildVoxelMap(const std::vector<PointWithCov> &inputPoints, const int& frameId,
                                      const float voxelSize, const int maxLayer,
                                      const std::vector<int> &layerPointSize,
                                      const int maxPointsSize, const int maxCovPointsSize,
                                      const float featThreshold,
                                      std::unordered_map<VOXEL_LOC, OctoTree *> &featMap);

void VOXEL_MAP_UTIL_API updateVoxelMap(const std::vector<PointWithCov> &inputPoints, const int& frameId,
                                       const float voxelSize, const int maxLayer,
                                       const std::vector<int> &layerPointSize,
                                       const int maxPointsSize, const int maxCovPointsSize,
                                       const float featThreshold,
                                       std::unordered_map<VOXEL_LOC, OctoTree *> &featMap,
                                       std::vector<BoxPointType> &octoBoxToDel);

#endif //BA_ENABLE

void buildSingleResidual(const PointWithCov &pv, OctoTree *currentOcto,
                         const int currentLayer, const int maxLayers,
                         const double sigmaNum, bool &isSucess,
                         double &prob, MatchOctoTreeInfo &singlePtpl, bool isStrict = true,double radius_k=3);


void VOXEL_MAP_UTIL_API buildResidualListOmp(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                                             const double voxel_size,
                                             const double sigma_num,
                                             const int max_layers,
                                             const std::vector<PointWithCov> &pv_list,
                                             std::vector<MatchOctoTreeInfo> &ptpl_list,
                                             std::vector<Eigen::Vector3d> &non_match,
                                             bool isStrict = true,
                                             double radius_k=3,
                                             NeighborSearchMethod searchMethod = NeighborSearchMethod::DIRECT1
                                           );

void VOXEL_MAP_UTIL_API buildResidualListNormal(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                                                const double voxel_size, const double sigma_num, const int max_layers,
                                                const std::vector<PointWithCov> &pv_list, std::vector<MatchOctoTreeInfo> &ptpl_list,
                                                std::vector<Eigen::Vector3d> &non_match,
                                                bool isStrict = true);


void VOXEL_MAP_UTIL_API getUpdatePlane(const OctoTree *currentOcto, const int pubMaxVoxelLayer, std::vector<PlaneParams> &planeList);

void VOXEL_MAP_UTIL_API mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);

void VOXEL_MAP_UTIL_API downSamplingVoxel(PvList &cloud, double voxel_size);

void VOXEL_MAP_UTIL_API downSamplingVoxel(const PvList &cloudIn, PvList &cloudOut, double voxel_size);

#endif