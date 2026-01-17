//
// Created by w on 2022/5/26.
//

#ifndef SRC_OPTIMIZER_H
#define SRC_OPTIMIZER_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "list"
#include "mutex"
#include "thread"
#include "Misc/So3.h"
#include "Mapper/VoxelMapper.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define OPTIMIZER_API __declspec(dllexport)
#else
#define OPTIMIZER_API __declspec(dllimport)
#endif
#else
#define OPTIMIZER_API
#endif

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

void eigToDouble(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double *pose);

void doubleToEig(const double *pose, Eigen::Matrix3d &R, Eigen::Vector3d &t);

template<typename T>
void QuaternionInverse(const T q[4], T q_inverse[4])
{
    q_inverse[0] = q[0];
    q_inverse[1] = -q[1];
    q_inverse[2] = -q[2];
    q_inverse[3] = -q[3];
};

template<typename T>
T computeResidual(const T *const exT, T *pabcd, T *point3d, T theta)
{
    T pw[3], po[3];
    //po=Tol*pl
    ceres::AngleAxisRotatePoint(exT, point3d, po);
    po[0] += exT[3];
    po[1] += exT[4];
    po[2] += exT[5];
    //pw=Rwo*po
    T rot[3] = {(T) 0, (T) 0, theta};//Rwo
    ceres::AngleAxisRotatePoint(rot, po, pw);

    return pabcd[0] * pw[0] + pabcd[1] * pw[1] + pabcd[2] * pw[2] + pabcd[3];
}

template<typename T>
T computeInitResidual(const T *const exT, T *pabcd, T *point3d, T theta)
{
    T pw[3], po[3], pww[3];
    //po=Tol*pl
    ceres::AngleAxisRotatePoint(exT, point3d, po);
    po[0] += exT[3];
    po[1] += exT[4];
    po[2] += exT[5];
    //pw=Rwo*po
    T rot[3] = {(T) 0, (T) 0, theta};
    ceres::AngleAxisRotatePoint(rot, po, pw);

    Eigen::Matrix<T, 3, 1> Rvec(exT[0], exT[1], exT[2]);
    Eigen::Matrix<T, 3, 1> tvec(exT[3], exT[4], exT[5]);
    Eigen::Matrix<T, 3, 3> Rlo = AngleAxisToRotationMatrix(Rvec).inverse();
    Eigen::Matrix<T, 3, 1> tlo = -Rlo * tvec;
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> poInit(pw);
    Eigen::Matrix<T, 3, 1> plInit = Rlo * poInit.template cast<T>() + tlo;
//    ceres::AngleAxisRotatePoint(exT, pw, pww);
//    pww[0] += exT[3];
//    pww[1] += exT[4];
//    pww[2] += exT[5];

    return pabcd[0] * plInit[0] + pabcd[1] * plInit[1] + pabcd[2] * plInit[2] + pabcd[3];
}

template<typename T>
bool computeResidual(T *residuals,
                     const T *const tPtr, const T *const qPtr,
                     const T *const toRelPtr, const T *const qoRelPtr,
                     const T *const tlRelPtr, const T *const qlRelPtr,
                     const T &tVar, const T &qVar)
{
    T qoRel[4];
    qoRel[0] = T(qoRelPtr[3]);//w
    qoRel[1] = T(qoRelPtr[0]);//x
    qoRel[2] = T(qoRelPtr[1]);//y
    qoRel[3] = T(qoRelPtr[2]);//z
    T toRel[3];
    toRel[0] = T(toRelPtr[0]);
    toRel[1] = T(toRelPtr[1]);
    toRel[2] = T(toRelPtr[2]);

    T qlRel[4];
    qlRel[0] = T(qlRelPtr[3]);
    qlRel[1] = T(qlRelPtr[0]);
    qlRel[2] = T(qlRelPtr[1]);
    qlRel[3] = T(qlRelPtr[2]);
    T tlRel[3];
    tlRel[0] = T(tlRelPtr[0]);
    tlRel[1] = T(tlRelPtr[1]);
    tlRel[2] = T(tlRelPtr[2]);

    T t[3];
    //to1l2=Qo1o2*tol + to1o2
    ceres::QuaternionRotatePoint(qoRel, tPtr, t);
    T relTol1[3];
    relTol1[0] = t[0] + toRel[0];
    relTol1[1] = t[1] + toRel[1];
    relTol1[2] = t[2] + toRel[2];

    T qol[4];
    qol[0] = T(qPtr[3]);
    qol[1] = T(qPtr[0]);
    qol[2] = T(qPtr[1]);
    qol[3] = T(qPtr[2]);

    //to1l2'=Qol*tl1l2 + tol
    ceres::QuaternionRotatePoint(qol, tlRel, t);
    T relTol2[3];
    relTol2[0] = t[0] + tPtr[0];
    relTol2[1] = t[1] + tPtr[1];
    relTol2[2] = t[2] + tPtr[2];

    residuals[0] = (relTol1[0] - relTol2[0]) / T(tVar);
    residuals[1] = (relTol1[1] - relTol2[1]) / T(tVar);
    residuals[2] = (relTol1[2] - relTol2[2]) / T(tVar);

    T relQol1[4], relQol2[4];
    ceres::QuaternionProduct(qoRel, qol, relQol1);//Qo1l2=Qo1o2*Qol
    ceres::QuaternionProduct(qol, qlRel, relQol2);//Qo1l2'=Qol*Ql1l2

    T relQol2Inv[4];
    QuaternionInverse(relQol2, relQol2Inv);
    T errorQ[4];
    ceres::QuaternionProduct(relQol2Inv, relQol1, errorQ);

    residuals[3] = 2. * errorQ[1] / qVar;
    residuals[4] = 2. * errorQ[2] / qVar;
    residuals[5] = 2. * errorQ[3] / qVar;

    return true;
}

struct RegErrorOnlyExPose
{
    RegErrorOnlyExPose(const double *const pabcd, const Eigen::Vector3d &point3d, double theta) : _pabcd(pabcd),
                                                                                                  _point3d(point3d),
                                                                                                  _rotationAngle(theta)
    {}

    template<typename T>
    bool operator()(const T *const exT, T *residuals) const
    {
        T point[3];
        point[0] = T(_point3d[0]);
        point[1] = T(_point3d[1]);
        point[2] = T(_point3d[2]);
        T pabcd[4];
        pabcd[0] = T(_pabcd[0]);
        pabcd[1] = T(_pabcd[1]);
        pabcd[2] = T(_pabcd[2]);
        pabcd[3] = T(_pabcd[3]);
        residuals[0] = computeResidual(exT, pabcd, point, T(_rotationAngle));
        return true;
    }

    static ceres::CostFunction *Create(const double *const pabcd, const Eigen::Vector3d &point3d, double theta)
    {
        return (new ceres::AutoDiffCostFunction<RegErrorOnlyExPose, 1, 6>(
                new RegErrorOnlyExPose(pabcd, point3d, theta)));
    }

    const double *const _pabcd;
    Eigen::Vector3d _point3d;
    double _rotationAngle;
};

struct InitRegErrorExPose
{
    InitRegErrorExPose(const double *const pabcd, const Eigen::Vector3d &point3d, double theta) : _pabcd(pabcd),
                                                                                                  _point3d(point3d),
                                                                                                  _rotationAngle(theta)
    {}

    template<typename T>
    bool operator()(const T *const exT, T *residuals) const
    {
        T point[3];
        point[0] = T(_point3d[0]);
        point[1] = T(_point3d[1]);
        point[2] = T(_point3d[2]);
        T pabcd[4];
        pabcd[0] = T(_pabcd[0]);
        pabcd[1] = T(_pabcd[1]);
        pabcd[2] = T(_pabcd[2]);
        pabcd[3] = T(_pabcd[3]);
        residuals[0] = computeInitResidual(exT, pabcd, point, T(_rotationAngle));
        return true;
    }

    static ceres::CostFunction *Create(const double *const pabcd, const Eigen::Vector3d &point3d, double theta)
    {
        return (new ceres::AutoDiffCostFunction<InitRegErrorExPose, 1, 6>(
                new InitRegErrorExPose(pabcd, point3d, theta)));
    }

    const double *const _pabcd;
    Eigen::Vector3d _point3d;
    double _rotationAngle;
};

struct RelativeRTError
{
    RelativeRTError(const double *const toRelPtr, const double *const qoRelPtr,
                    const double *const tlRelPtr, const double *const qlRelPtr,
                    const double &tVar, const double &qVar) : _toRelPtr(toRelPtr),
                                                              _qoRelPtr(qoRelPtr),
                                                              _tlRelPtr(tlRelPtr),
                                                              _qlRelPtr(qlRelPtr),
                                                              _tVar(tVar),
                                                              _qVar(qVar)
    {}

    template<typename T>
    bool operator()(const T *const tPtr, const T *const qPtr, T *residuals) const
    {
        T qoRel[4];
        qoRel[0] = T(_qoRelPtr[0]);//w
        qoRel[1] = T(_qoRelPtr[1]);//x
        qoRel[2] = T(_qoRelPtr[2]);//y
        qoRel[3] = T(_qoRelPtr[3]);//z
        T toRel[3];
        toRel[0] = T(_toRelPtr[0]);
        toRel[1] = T(_toRelPtr[1]);
        toRel[2] = T(_toRelPtr[2]);

        T qlRel[4];
        qlRel[0] = T(_qlRelPtr[0]);
        qlRel[1] = T(_qlRelPtr[1]);
        qlRel[2] = T(_qlRelPtr[2]);
        qlRel[3] = T(_qlRelPtr[3]);
        T tlRel[3];
        tlRel[0] = T(_tlRelPtr[0]);
        tlRel[1] = T(_tlRelPtr[1]);
        tlRel[2] = T(_tlRelPtr[2]);
        return computeResidual(residuals,
                               tPtr, qPtr,
                               toRel, qoRel,
                               tlRel, qlRel,
                               (T) _tVar, (T) _qVar);
    }

    static ceres::CostFunction *Create(const double *const toRelPtr, const double *const qoRelPtr,
                                       const double *const tlRelPtr, const double *const qlRelPtr,
                                       const double tVar, const double qVar)
    {
        return (new ceres::AutoDiffCostFunction<
                RelativeRTError, 6, 3, 4>(new
                                                  RelativeRTError(toRelPtr, qoRelPtr, tlRelPtr, qlRelPtr, tVar, qVar)));
    }

    const double *_qoRelPtr;
    const double *_qlRelPtr;
    const double *_toRelPtr;
    const double *_tlRelPtr;
    const double _tVar;
    const double _qVar;
};

void OPTIMIZER_API poseOptimize(MatchedInfoList &matchedInfoList, const double &rotationAngle,
                  Eigen::Matrix3d &exR, Eigen::Vector3d &ext,
                  double maxOulierErr = 0, bool isVerbose = false);

void initOptimize(const MatchedInfoList &matchedInfoList, const double &rotationAngle,
                  Eigen::Matrix3d &exR, Eigen::Vector3d &ext);

void OPTIMIZER_API calcHandeyeExtrinsic(const Eigen::Matrix4d &relativeTo, const Eigen::Matrix4d &relativeTl,
                          Eigen::Matrix3d &exR, Eigen::Vector3d &ext);

void OPTIMIZER_API calcHandeyeExtrinsic(const std::vector<Eigen::Matrix4d> &relativeTos, const std::vector<Eigen::Matrix4d> &relativeTls,
                          Eigen::Matrix3d &exR, Eigen::Vector3d &ext);


#endif //SRC_OPTIMIZER_H
