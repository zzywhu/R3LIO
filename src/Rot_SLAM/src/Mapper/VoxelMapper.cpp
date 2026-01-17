#include "Mapper/VoxelMapper.h"
#include "Optimizer/Optimizer.h"

#ifdef BA_ENABLE

int OctoTree::_voxelWindowSize = 0;

OctoTree::OctoTree(const int& layer,
                   const int& maxLayers,
                   const std::vector<int>& layerPointSize,
                   const float& planerThreshold,
                   const int& capacity): _layer(layer),
                                   _maxLayers(maxLayers),
                                   _layerPointSizeList(layerPointSize),
                                   _planerThreshold(planerThreshold),
                                   _capacity(capacity)
{
    _octoState = 0;
    _isObs = true;
    // when new points num > 5, do a update
    _updateSizeThres = 5;
    _planePointsThres = _layerPointSizeList[_layer];

    for (int i = 0; i < 8; i++)
        _leaves[i] = nullptr;
    for(int i=0; i < _capacity; i++)
        _winCloudVec.push_back(new PvList());

    _planePtr = new Plane;
}

void OctoTree::transPoints(PvList& cloud, const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
    uint cloudSize = cloud.size();
    for(uint i=0; i<cloudSize; i++)
        cloud[i].pw = R*cloud[i].point + t;
}

// check is plane , calc plane parameters including plane covariance
void OctoTree::calcPlaneParams(const std::vector<PvList*>& pointsVec, int minWinIdx, int maxWinIdx)
{
    if(minWinIdx < 0 || maxWinIdx < 0)
        return;
    _planePtr->planeCov = Eigen::Matrix<double, 6, 6>::Zero();
    _planePtr->covariance = Eigen::Matrix3d::Zero();
    _planePtr->center = Eigen::Vector3d::Zero();
    _planePtr->normal = Eigen::Vector3d::Zero();
    _planePtr->radius = 0;
    _planePtr->pointsSize=0;
    for(int i=minWinIdx; i <= maxWinIdx; i++)
    {
        auto& points=pointsVec[i];
        for (auto pv : *points)
        {
            _planePtr->covariance += pv.pw * pv.pw.transpose();
            _planePtr->center += pv.pw;
        }
        _planePtr->pointsSize += points->size();
    }
    _planePtr->pointsSize += _marginParams._sigmaSize;
    _planePtr->covariance += _marginParams._sigmaCov;
    _planePtr->center += _marginParams._sigmaCenter;

    _planePtr->center /= _planePtr->pointsSize;
    _planePtr->covariance =  _planePtr->covariance/_planePtr->pointsSize -  _planePtr->center* _planePtr->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(_planePtr->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / _planePtr->pointsSize, 0, 0, 0, 1.0 / _planePtr->pointsSize, 0, 0, 0, 1.0 / _planePtr->pointsSize;
    if (evalsReal(evalsMin) < _planerThreshold)
    {
        for(int i=minWinIdx; i <= maxWinIdx; i++)
        {
            auto &points = *pointsVec[i];
            const int size = points.size();
            for (int j = 0; j < size; j++)
            {
                Eigen::Matrix<double, 6, 3> J;
                Eigen::Matrix3d F;
                for (int m = 0; m < 3; m++)
                {
                    if (m != (int)evalsMin)
                        F.row(m) =(points[j].pw - _planePtr->center).transpose() /
                                  ((_planePtr->pointsSize) * (evalsReal[evalsMin] - evalsReal[m])) *
                                  (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                                   evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
                    else
                        F.row(m) << 0, 0, 0;
                }
                J.block<3, 3>(0, 0) = evecs.real() * F;
                J.block<3, 3>(3, 0) = J_Q;
                _planePtr->planeCov += J * points[j].obsCov * J.transpose();
            }
        }

        _planePtr->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),evecs.real()(2, evalsMin);
        _planePtr->yNormal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),evecs.real()(2, evalsMid);
        _planePtr->xNormal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),evecs.real()(2, evalsMax);
        _planePtr->minEigenValue = evalsReal(evalsMin);
        _planePtr->midEigenValue = evalsReal(evalsMid);
        _planePtr->maxEigenValue = evalsReal(evalsMax);
        _planePtr->radius = sqrt(evalsReal(evalsMax));
        _planePtr->d = -(_planePtr->normal(0) * _planePtr->center(0) + _planePtr->normal(1) * _planePtr->center(1) + _planePtr->normal(2) * _planePtr->center(2));
        _planePtr->isPlane = true;

        if (!_planePtr->isInit)
        {
            _planePtr->id = _planeId++;
            _planePtr->isInit = true;
        }
    }
    else
    {
        if (!_planePtr->isInit)
        {
            _planePtr->id = _planeId++;
            _planePtr->isInit = true;
        }
        _planePtr->isPlane = false;
        _planePtr->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        _planePtr->yNormal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
        _planePtr->xNormal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
        _planePtr->minEigenValue = evalsReal(evalsMin);
        _planePtr->midEigenValue = evalsReal(evalsMid);
        _planePtr->maxEigenValue = evalsReal(evalsMax);
        _planePtr->radius = sqrt(evalsReal(evalsMax));
        _planePtr->d = -(_planePtr->normal(0) * _planePtr->center(0) + _planePtr->normal(1) * _planePtr->center(1) + _planePtr->normal(2) * _planePtr->center(2));
    }
}

void OctoTree::cutOctoTree(uint frameHead)
{
    if(_octoState == 0)
    {
        int pointsSize = 0;
        for(int i=0; i < _voxelWindowSize; i++)
            pointsSize += _winCloudVec[i]->size();
        pointsSize += _marginParams._sigmaSize;
        if(pointsSize < _planePointsThres)
        {
            _planePtr->isPlane= false;
            return;
        }

        calcPlaneParams(_winCloudVec, 0, _voxelWindowSize-1);

        if(_planePtr->isPlane)
            return;

        if(_layer == _maxLayers - 1)
            return;

//        _marginParams.tozero();

        _octoState = 1;
        // All points in slidingwindow should be put into subvoxel
        frameHead = 0;
    }

    for(int i=frameHead; i<_voxelWindowSize; i++)
    {
        const int size = _winCloudVec[i]->size();
        for(uint j=0; j<size; j++)
        {
            int xyz[3] = {0, 0, 0};
            for(uint k=0; k<3; k++)
                if((*_winCloudVec[i])[j].pw[k] > _voxelCenter[k])
                    xyz[k] = 1;

            int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
            if(_leaves[leafnum] == nullptr)
            {
                _leaves[leafnum] = new OctoTree(_layer+1, _maxLayers, _layerPointSizeList, _planerThreshold, _capacity);
                _leaves[leafnum]->_voxelCenter[0] = _voxelCenter[0] + (2*xyz[0]-1)*_quaterLength;
                _leaves[leafnum]->_voxelCenter[1] = _voxelCenter[1] + (2*xyz[1]-1)*_quaterLength;
                _leaves[leafnum]->_voxelCenter[2] = _voxelCenter[2] + (2*xyz[2]-1)*_quaterLength;
                _leaves[leafnum]->_quaterLength = _quaterLength / 2;
            }
            _leaves[leafnum]->_winCloudVec[i]->push_back((*_winCloudVec[i])[j]);
        }
    }

    if(_layer != 0)
        for(int i=frameHead; i<_voxelWindowSize; i++)
            if(_winCloudVec[i]->size() != 0)
                PvList().swap(*_winCloudVec[i]);


    for(uint i=0; i<8; i++)
        if(_leaves[i] != nullptr)
            _leaves[i]->cutOctoTree(frameHead);
}

void OctoTree::marginalize(const int& margiSize,
                           const int& windowBase,
                           const vector<Eigen::Quaterniond>& qPoses,
                           const vector<Eigen::Vector3d>& tPoses)
{
    if(_octoState!=1 || _layer==0)
    {//leaf node exists or root node new established
        if(_octoState != 1)
            for(int i=0; i<_voxelWindowSize; i++)// Update points by new poses
                transPoints(*_winCloudVec[i], qPoses[i + windowBase].matrix(), tPoses[i + windowBase]);

        // If is a plane, push front 5 scans into P_fix
        if(_planePtr->isPlane)
        {
            _marginCloud.clear();
            for(int i=0; i < margiSize; i++)
                _marginCloud.insert(_marginCloud.end(), _winCloudVec[i]->begin(), _winCloudVec[i]->end());
            //downSamplingVoxel(_marginCloud, _quaterLength);
            //_marginParams.tozero();
            _marginParams._sigmaSize +=_marginCloud.size();
            for(uint i=0; i<_marginCloud.size(); i++)
            {
                _marginParams._sigmaCov += _marginCloud[i].pw * _marginCloud[i].pw.transpose();
                _marginParams._sigmaCenter  += _marginCloud[i].pw;
            }
        }

        // Clear front 5 scans
        for(int i=0; i < margiSize; i++)
            PvList().swap(*_winCloudVec[i]);
        // push foward 5 scans
        for(int i=margiSize; i < _voxelWindowSize; i++)
            _winCloudVec[i]->swap(*_winCloudVec[i - margiSize]);

        if(_layer == 0)
        {
            int leftSize = 0;
            for(int i = 0; i < _voxelWindowSize-margiSize; i++)
                leftSize += _winCloudVec[i]->size();
            if(leftSize == 0)// Root voxel has no points in slidingwindow
                _isObs = false;
        }
        if(_octoState != 1)
        {
            int ptSize = 0;
            for(int i=0; i< _voxelWindowSize - margiSize; i++)
                ptSize += _winCloudVec[i]->size();

            ptSize += _marginParams._sigmaSize;
            if(ptSize < _planePointsThres)
            {
                _planePtr->isPlane= false;
                return;
            }

            calcPlaneParams(_winCloudVec, 0, _voxelWindowSize-1);

            if(isnan(_planePtr->maxEigenValue/_planePtr->minEigenValue))
            {
                _planePtr->isPlane= false;
                return;
            }
            if(!_planePtr->isPlane)  // if a plane voxel changed, recut it
                cutOctoTree(0);
        }
    }

    if(_octoState == 1)
    {
        for(int i=0; i<8; i++)
            if(_leaves[i] != nullptr)
                _leaves[i]->marginalize(margiSize, windowBase, qPoses, tPoses);
    }
}

void OctoTree::traversalOpt(LM_SLWD_VOXEL* optLsv)
{
    if(_octoState != 1)
    {
        int pointsSize = 0;
        for(int i=0; i < _voxelWindowSize; i++)
            pointsSize += _winCloudVec[i]->size();
        pointsSize += _marginParams._sigmaSize;

        if(pointsSize < _planePointsThres)
            return;

        calcPlaneParams(_winCloudVec, 0, _voxelWindowSize-1);
        if(!_planePtr->isPlane)
            return;
        optLsv->push_voxel(_winCloudVec, _marginParams, 0);
    }
    else
    {
        for(int i=0; i<8; i++)
            if(_leaves[i] != nullptr)
                _leaves[i]->traversalOpt(optLsv);
    }
}

int OctoTree::getWinCloudSize()
{
    int ptSize = 0;
    for(int i=0; i< _voxelWindowSize; i++)
        ptSize += _winCloudVec[i]->size();
    return ptSize;
}

void updateVoxelMap(unordered_map<VOXEL_LOC, OctoTree*>& featMap,
                    const std::vector<PointWithCov>& pvList,
                    const int& winIdx,
                    const int& capacity,
                    const float& voxelSize,
                    const int& maxLayer,
                    const std::vector<int>& layerPointSizeList,
                    const float& planerThreshold)
{
    std::unordered_map<VOXEL_LOC, OctoTree *> updateOctTrees;
    for(auto& pv: pvList)
    {
        // Determine the key of hash table
        float locXYZ[3];
        for(int j=0; j<3; j++)
        {
            locXYZ[j] = pv.pw[j] / voxelSize;
            if(locXYZ[j] < 0)
                locXYZ[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t)locXYZ[0], (int64_t)locXYZ[1], (int64_t)locXYZ[2]);

        // Find corresponding voxel
        auto itr = featMap.find(position);
        if(itr != featMap.end())
        {
            itr->second->_winCloudVec[winIdx]->push_back(pv);
            itr->second->_isObs = true;
            updateOctTrees[position]=itr->second;
        }
        else
        {// If not finding, build a new voxel
            OctoTree *oct = new OctoTree(0, maxLayer, layerPointSizeList, planerThreshold, capacity);
            oct->_winCloudVec[winIdx]->push_back(pv);
            // Voxel center coordinate
            oct->_voxelCenter[0] = (0.5+position.x) * voxelSize;
            oct->_voxelCenter[1] = (0.5+position.y) * voxelSize;
            oct->_voxelCenter[2] = (0.5+position.z) * voxelSize;
            oct->_quaterLength = voxelSize / 4.0; // A quater of side length
            featMap[position] = oct;
            updateOctTrees[position]=oct;
        }
    }

    for(auto& locOct : updateOctTrees)
    {
        locOct.second->cutOctoTree(winIdx);
//        if(locOct.second->_planePtr->isPlane)
//            std::cout<<locOct.first.x<<","<<locOct.first.y<<","<<locOct.first.z<<": "<<locOct.second->_planePtr->center.transpose()<<std::endl;
    }

//    for(auto iter=featMap.begin(); iter!=featMap.end(); ++iter)
//        if(iter->second->_isObs)
//            iter->second->cutOctoTree(winIdx);
}

void marginalizeVoxelMap(unordered_map<VOXEL_LOC, OctoTree*>& featMap,
                         const int& margiSize,
                         const int& windowBase,
                         vector<Eigen::Quaterniond> &qPoses,
                         vector<Eigen::Vector3d> &tPoses)
{
    for(auto iter=featMap.begin(); iter!=featMap.end(); ++iter)
        if(iter->second->_isObs)
            iter->second->marginalize(margiSize, windowBase, qPoses, tPoses);
}

#else

// check is plane , calc plane parameters including plane covariance
void OctoTree::initPlane(std::vector<PointWithCov>& pvList, PlaneParams* plane)
{
    plane->planeCov = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance = Eigen::Matrix3d::Zero();
    plane->center = Eigen::Vector3d::Zero();
    plane->normal = Eigen::Vector3d::Zero();
    plane->pointsSize = pvList.size();
    plane->radius = 0;
    for (auto pv : pvList)
    {
        plane->covariance += pv.pw * pv.pw.transpose();
        plane->center += pv.pw;
    }
    plane->center = plane->center / plane->pointsSize;
    plane->covariance = plane->covariance / plane->pointsSize - plane->center * plane->center.transpose();
    //TicToc time;
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    plane->evecs = es.eigenvectors();
    plane->evals = es.eigenvalues();
   // printf("EigenSolver time %f ms\n", time.toc());

    Eigen::Vector3d evalsReal;
    evalsReal = plane->evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->pointsSize, 0, 0, 0, 1.0 / plane->pointsSize, 0, 0, 0, 1.0 / plane->pointsSize;
    if (evalsReal(evalsMin) < _featThreshold)
    {
        std::vector<int> index(pvList.size());
        for (int i = 0; i < pvList.size(); i++)
        {
            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F;
            for (int m = 0; m < 3; m++)
            {
                if (m != (int) evalsMin)
                {
                    Eigen::Matrix<double, 1, 3> F_m =
                            (pvList[i].pw - plane->center).transpose() /
                            ((plane->pointsSize) * (evalsReal[evalsMin] - evalsReal[m])) *
                            (plane->evecs.real().col(m) * plane->evecs.real().col(evalsMin).transpose() +
                             plane->evecs.real().col(evalsMin) * plane->evecs.real().col(m).transpose());
                    F.row(m) = F_m;
                }
                else
                {
                    Eigen::Matrix<double, 1, 3> F_m;
                    F_m << 0, 0, 0;
                    F.row(m) = F_m;
                }
            }
            J.block<3, 3>(0, 0) = plane->evecs.real() * F;
            J.block<3, 3>(3, 0) = J_Q;
            //plane->planeCov += J * pvList[i].neighCov.block<3, 3>(0, 0) * J.transpose();
            plane->planeCov += J * pvList[i].obsCov * J.transpose();
        }

        plane->normal << plane->evecs.real()(0, evalsMin), plane->evecs.real()(1, evalsMin), plane->evecs.real()(2, evalsMin);
        plane->yNormal << plane->evecs.real()(0, evalsMid), plane->evecs.real()(1, evalsMid), plane->evecs.real()(2, evalsMid);
        plane->xNormal << plane->evecs.real()(0, evalsMax), plane->evecs.real()(1, evalsMax), plane->evecs.real()(2, evalsMax);
        plane->minEigenValue = evalsReal(evalsMin);
        plane->midEigenValue = evalsReal(evalsMid);
        plane->maxEigenValue = evalsReal(evalsMax);
        plane->radius = sqrt(evalsReal(evalsMax));
        plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) + plane->normal(2) * plane->center(2));
        plane->isPlane = true;
        if (plane->lastUpdatePointsSize == 0)
        {
            plane->lastUpdatePointsSize = plane->pointsSize;
            plane->isUpdate = true;
        }
        else if (plane->pointsSize - plane->lastUpdatePointsSize > 100)
        {
            plane->lastUpdatePointsSize = plane->pointsSize;
            plane->isUpdate = true;
        }

        if (!plane->isInit)
        {
            plane->id = _planeId;
            _planeId++;
            plane->isInit = true;
        }
    }
    else
    {
        if (!plane->isInit)
        {
            plane->id = _planeId;
            _planeId++;
            plane->isInit = true;
        }
        if (plane->lastUpdatePointsSize == 0)
        {
            plane->lastUpdatePointsSize = plane->pointsSize;
            plane->isUpdate = true;
        }
        else if (plane->pointsSize - plane->lastUpdatePointsSize > 100)
        {
            plane->lastUpdatePointsSize = plane->pointsSize;
            plane->isUpdate = true;
        }
        plane->isPlane = false;
        plane->normal << plane->evecs.real()(0, evalsMin), plane->evecs.real()(1, evalsMin), plane->evecs.real()(2, evalsMin);
        plane->yNormal << plane->evecs.real()(0, evalsMid), plane->evecs.real()(1, evalsMid), plane->evecs.real()(2, evalsMid);
        plane->xNormal << plane->evecs.real()(0, evalsMax), plane->evecs.real()(1, evalsMax), plane->evecs.real()(2, evalsMax);
        plane->minEigenValue = evalsReal(evalsMin);
        plane->midEigenValue = evalsReal(evalsMid);
        plane->maxEigenValue = evalsReal(evalsMax);
        plane->radius = sqrt(evalsReal(evalsMax));
        plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) + plane->normal(2) * plane->center(2));
    }
}

void OctoTree::initLine(std::vector<PointWithCov> &pvList, LineParams *line)
{
    line->covariance = Eigen::Matrix3d::Zero();
    line->center = Eigen::Vector3d::Zero();
    line->direct = Eigen::Vector3d::Zero();
    line->pointsSize = pvList.size();
    for (auto pv : pvList)
    {
        line->covariance += pv.pw * pv.pw.transpose();
        line->center += pv.pw;
    }
    line->center = line->center / line->pointsSize;
    line->covariance = line->covariance / line->pointsSize - line->center * line->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(line->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / line->pointsSize, 0, 0, 0, 1.0 / line->pointsSize, 0, 0, 0, 1.0 / line->pointsSize;
    if (evalsReal(evalsMin) < _featThreshold)
    {
        std::vector<int> index(pvList.size());
        line->direct << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
        line->minEigenValue = evalsReal(evalsMin);
        line->midEigenValue = evalsReal(evalsMid);
        line->maxEigenValue = evalsReal(evalsMax);
        line->isLine = true;
        if (line->lastUpdatePointsSize == 0)
        {
            line->lastUpdatePointsSize = line->pointsSize;
            line->isUpdate = true;
        }
        else if (line->pointsSize - line->lastUpdatePointsSize > 100)
        {
            line->lastUpdatePointsSize = line->pointsSize;
            line->isUpdate = true;
        }

        if (!line->isInit)
        {
            line->id = _planeId;
            _planeId++;
            line->isInit = true;
        }
    }
    else
    {
        if (!line->isInit)
        {
            line->id = _planeId;
            _planeId++;
            line->isInit = true;
        }
        if (line->lastUpdatePointsSize == 0)
        {
            line->lastUpdatePointsSize = line->pointsSize;
            line->isUpdate = true;
        }
        else if (line->pointsSize - line->lastUpdatePointsSize > 100)
        {
            line->lastUpdatePointsSize = line->pointsSize;
            line->isUpdate = true;
        }
        line->isLine = false;
        line->direct << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
        line->minEigenValue = evalsReal(evalsMin);
        line->midEigenValue = evalsReal(evalsMid);
        line->maxEigenValue = evalsReal(evalsMax);
    }
}

// only update plane normal, center and radius with new points
void OctoTree::updatePlane(std::vector<PointWithCov> &pvList, PlaneParams *plane)
{
    Eigen::Matrix3d old_covariance = plane->covariance;
    Eigen::Vector3d old_center = plane->center;
    Eigen::Matrix3d sum_ppt = (plane->covariance + plane->center * plane->center.transpose()) * plane->pointsSize;
    Eigen::Vector3d sum_p = plane->center * plane->pointsSize;
    for (size_t i = 0; i < pvList.size(); i++)
    {
        Eigen::Vector3d pv = pvList[i].pw;
        sum_ppt += pv * pv.transpose();
        sum_p += pv;
    }
    plane->pointsSize = plane->pointsSize + pvList.size();
    plane->center = sum_p / plane->pointsSize;
    plane->covariance = sum_ppt / plane->pointsSize - plane->center * plane->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    plane->evecs = es.eigenvectors();
    plane->evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = plane->evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = plane->evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = plane->evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = plane->evecs.real().col(evalsMax);
    if (evalsReal(evalsMin) < _featThreshold)
    {
        plane->normal << plane->evecs.real()(0, evalsMin), plane->evecs.real()(1, evalsMin), plane->evecs.real()(2, evalsMin);
        plane->yNormal << plane->evecs.real()(0, evalsMid), plane->evecs.real()(1, evalsMid), plane->evecs.real()(2, evalsMid);
        plane->xNormal << plane->evecs.real()(0, evalsMax), plane->evecs.real()(1, evalsMax), plane->evecs.real()(2, evalsMax);
        plane->minEigenValue = evalsReal(evalsMin);
        plane->midEigenValue = evalsReal(evalsMid);
        plane->maxEigenValue = evalsReal(evalsMax);
        plane->radius = sqrt(evalsReal(evalsMax));
        plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) + plane->normal(2) * plane->center(2));

        plane->isPlane = true;
        plane->isUpdate = true;
    }
    else
    {
        plane->normal << plane->evecs.real()(0, evalsMin), plane->evecs.real()(1, evalsMin), plane->evecs.real()(2, evalsMin);
        plane->yNormal << plane->evecs.real()(0, evalsMid), plane->evecs.real()(1, evalsMid), plane->evecs.real()(2, evalsMid);
        plane->xNormal << plane->evecs.real()(0, evalsMax), plane->evecs.real()(1, evalsMax), plane->evecs.real()(2, evalsMax);
        plane->minEigenValue = evalsReal(evalsMin);
        plane->midEigenValue = evalsReal(evalsMid);
        plane->maxEigenValue = evalsReal(evalsMax);
        plane->radius = sqrt(evalsReal(evalsMax));
        plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) + plane->normal(2) * plane->center(2));
        plane->isPlane = false;
        plane->isUpdate = true;
    }
}

void OctoTree::updateLine(std::vector<PointWithCov> &pvList, LineParams *line)
{
    Eigen::Matrix3d old_covariance = line->covariance;
    Eigen::Vector3d old_center = line->center;
    Eigen::Matrix3d sum_ppt = (line->covariance + line->center * line->center.transpose()) * line->pointsSize;
    Eigen::Vector3d sum_p = line->center * line->pointsSize;
    for (size_t i = 0; i < pvList.size(); i++)
    {
        Eigen::Vector3d pv = pvList[i].pw;
        sum_ppt += pv * pv.transpose();
        sum_p += pv;
    }
    line->pointsSize = line->pointsSize + pvList.size();
    line->center = sum_p / line->pointsSize;
    line->covariance = sum_ppt / line->pointsSize - line->center * line->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(line->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
    if (evalsReal(evalsMin) < _featThreshold)
    {
        line->direct << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);

        line->minEigenValue = evalsReal(evalsMin);
        line->midEigenValue = evalsReal(evalsMid);
        line->maxEigenValue = evalsReal(evalsMax);

        line->isLine = true;
        line->isUpdate = true;
    }
    else
    {
        line->direct << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
        line->minEigenValue = evalsReal(evalsMin);
        line->midEigenValue = evalsReal(evalsMid);
        line->maxEigenValue = evalsReal(evalsMax);

        line->isLine = false;
        line->isUpdate = true;
    }
}

void OctoTree::initOctoTree()
{//根节点计算面特征,如为面特征则设置为叶节点且不再分割,非面特征则继续分割

    if (_tempPoints.size() > _minFeatUpdateThreshold)
    {
        if(_featType==Plane)
            initPlane(_tempPoints, _planePtr);
        else if(_featType==Line)
            initLine(_tempPoints, _linePtr);
        if ((_featType==Plane&&_planePtr->isPlane == true)||
            (_featType==Line&&_linePtr->isLine == true))
        {
            _octoState = 0;
            if (_tempPoints.size() > _maxCovPointsSize)
                _isUpdateCovEnable = false;
            if (_tempPoints.size() > _maxPointsSize)
                _isUpdateEnable = false;
        }
        else
        {
            _octoState = 1;
            cutOctoTree();
        }
        _initOcto = true;
        _newPointsNum = 0;
        //      _tempPoints.clear();
    }

}

void OctoTree::cutOctoTree()
{
    if (_layer >= _maxLayers)
    {
        _octoState = 0;
        return;
    }

    for (int i = 0; i < _tempPoints.size(); i++)
    {
        int xyz[3] = {0, 0, 0};
        if (_tempPoints[i].pw[0] > _voxelCenter[0])
            xyz[0] = 1;
        if (_tempPoints[i].pw[1] > _voxelCenter[1])
            xyz[1] = 1;
        if (_tempPoints[i].pw[2] > _voxelCenter[2])
            xyz[2] = 1;
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (_leaves[leafnum] == nullptr)
        {
            _leaves[leafnum] = new OctoTree(_refFrameID, _featType, _maxLayers, _layer + 1, _layerPointSizeList, _maxPointsSize,
                                            _maxCovPointsSize, _featThreshold);
            _leaves[leafnum]->_voxelCenter[0] = _voxelCenter[0] + (2 * xyz[0] - 1) * _quaterLength;
            _leaves[leafnum]->_voxelCenter[1] = _voxelCenter[1] + (2 * xyz[1] - 1) * _quaterLength;
            _leaves[leafnum]->_voxelCenter[2] = _voxelCenter[2] + (2 * xyz[2] - 1) * _quaterLength;
            _leaves[leafnum]->_quaterLength = _quaterLength / 2;
        }
        _leaves[leafnum]->_tempPoints.push_back(_tempPoints[i]);
        _leaves[leafnum]->_newPointsNum++;
    }
    if (_tempPoints.size() != 0)
        std::vector<PointWithCov>().swap(_tempPoints);
    for (int i = 0; i < 8; i++)
    {
        if (_leaves[i] != nullptr)
        {
            if (_leaves[i]->_tempPoints.size() > _leaves[i]->_minFeatUpdateThreshold)
            {
                if(_featType==Plane)
                    initPlane(_leaves[i]->_tempPoints, _leaves[i]->_planePtr);
                else if(_featType==Line)
                    initLine(_leaves[i]->_tempPoints, _leaves[i]->_linePtr);
                if ((_featType==Plane&&_leaves[i]->_planePtr->isPlane)||
                    (_featType==Line&&_leaves[i]->_linePtr->isLine))
                    _leaves[i]->_octoState = 0;
                else
                {
                    _leaves[i]->_octoState = 1;
                    _leaves[i]->cutOctoTree();
                }
                _leaves[i]->_initOcto = true;
                _leaves[i]->_newPointsNum = 0;
            }
        }
    }
}

void OctoTree::updateOctoTree(const PointWithCov &pv)
{
    if(pv.featType!=_featType)
        throw "pv type not match the voxel!";
    _nVisible++;
    if (!_initOcto)
    {
        _newPointsNum++;
        _allPointsNum++;
        _tempPoints.push_back(pv);

        if (_tempPoints.size() > _minFeatUpdateThreshold)
            initOctoTree();
    }
    else
    {
        if ((_featType==Plane&&_planePtr->isPlane)||
            (_featType==Line&&_linePtr->isLine))
        {
            if (_isUpdateEnable)
            {
                _newPointsNum++;
                _allPointsNum++;
                if (_isUpdateCovEnable)
                    _tempPoints.push_back(pv);
                else
                    _newPoints.push_back(pv);
                if (_newPointsNum > _updateSizeThreshold)
                {
                    if (_isUpdateCovEnable)
                        if(_featType==Plane)
                            initPlane(_tempPoints, _planePtr);
                        else if(_featType==Line)
                            initLine(_tempPoints, _linePtr);
                    _newPointsNum = 0;
                }
                if (_allPointsNum >= _maxCovPointsSize)
                {
                    _isUpdateCovEnable = false;
                    std::vector<PointWithCov>().swap(_tempPoints);
                }
                if (_allPointsNum >= _maxPointsSize)
                {
                    _isUpdateEnable = false;
                    if(_featType==Plane)
                        _planePtr->updateEnable = false;
                    else if(_featType==Line)
                        _linePtr->updateEnable = false;
                    std::vector<PointWithCov>().swap(_newPoints);
                }
            }
            else
                return;
        }
        else
        {
            if (_layer < _maxLayers)
            {
                if (_tempPoints.size() != 0)
                    std::vector<PointWithCov>().swap(_tempPoints);

                if (_newPoints.size() != 0)
                    std::vector<PointWithCov>().swap(_newPoints);

                int xyz[3] = {0, 0, 0};
                if (pv.pw[0] > _voxelCenter[0])
                    xyz[0] = 1;

                if (pv.pw[1] > _voxelCenter[1])
                    xyz[1] = 1;

                if (pv.pw[2] > _voxelCenter[2])
                    xyz[2] = 1;

                int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
                if (_leaves[leafnum] != nullptr)
                    _leaves[leafnum]->updateOctoTree(pv);
                else
                {
                    _leaves[leafnum] = new OctoTree(_refFrameID, _featType, _maxLayers, _layer + 1, _layerPointSizeList, _maxPointsSize,
                                                    _maxCovPointsSize, _featThreshold);
                    _leaves[leafnum]->_layerPointSizeList = _layerPointSizeList;
                    _leaves[leafnum]->_voxelCenter[0] = _voxelCenter[0] + (2 * xyz[0] - 1) * _quaterLength;
                    _leaves[leafnum]->_voxelCenter[1] = _voxelCenter[1] + (2 * xyz[1] - 1) * _quaterLength;
                    _leaves[leafnum]->_voxelCenter[2] = _voxelCenter[2] + (2 * xyz[2] - 1) * _quaterLength;
                    _leaves[leafnum]->_quaterLength = _quaterLength / 2;
                    _leaves[leafnum]->updateOctoTree(pv);
                }
            }
            else
            {//_layer == _maxLayers
                if (_isUpdateEnable)
                {
                    _newPointsNum++;
                    _allPointsNum++;
                    if (_isUpdateCovEnable)
                        _tempPoints.push_back(pv);
                    else
                        _newPoints.push_back(pv);

                    if (_newPointsNum > _updateSizeThreshold)
                    {
                        if (_isUpdateCovEnable)
                        {
                            if(_featType==Plane)
                                initPlane(_tempPoints, _planePtr);
                            else if(_featType==Line)
                                initLine(_tempPoints,_linePtr);
                        }
                        else
                        {
                            if(_featType==Plane)
                                updatePlane(_newPoints, _planePtr);
                            else if(_featType==Line)
                                updateLine(_newPoints, _linePtr);
                            _newPoints.clear();
                        }
                        _newPointsNum = 0;
                    }
                    if (_allPointsNum >= _maxCovPointsSize)
                    {
                        _isUpdateCovEnable = false;
                        std::vector<PointWithCov>().swap(_tempPoints);
                    }
                    if (_allPointsNum >= _maxPointsSize)
                    {
                        _isUpdateEnable = false;
                        if(_featType==Plane)
                            _planePtr->updateEnable = false;
                        else if(_featType==Line)
                            _linePtr->updateEnable= false;
                        std::vector<PointWithCov>().swap(_newPoints);
                    }
                }
            }
        }
    }
}

void OctoTree::updatePlaneCov(std::vector<PointWithCov>& pvList, PlaneParams* plane)
{
    plane->planeCov = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Vector3d evalsReal;
    evalsReal = plane->evals.real();

    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->pointsSize, 0, 0, 0, 1.0 / plane->pointsSize, 0, 0, 0, 1.0 / plane->pointsSize;
    if (evalsReal(int(plane->minEigenValue)) < _featThreshold)
    {
        std::vector<int> index(pvList.size());
        for (int i = 0; i < pvList.size(); i++)
        {
            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F;
            for (int m = 0; m < 3; m++)
            {
                if (m != (int) plane->minEigenValue)
                {
                    Eigen::Matrix<double, 1, 3> F_m =
                            (pvList[i].pw - plane->center).transpose() /
                            ((plane->pointsSize) * (evalsReal[plane->minEigenValue] - evalsReal[m])) *
                            (plane->evecs.real().col(m) * plane->evecs.real().col(plane->minEigenValue).transpose() +
                             plane->evecs.real().col(plane->minEigenValue) * plane->evecs.real().col(m).transpose());
                    F.row(m) = F_m;
                }
                else
                {
                    Eigen::Matrix<double, 1, 3> F_m;
                    F_m << 0, 0, 0;
                    F.row(m) = F_m;
                }
            }
            J.block<3, 3>(0, 0) = plane->evecs.real() * F;
            J.block<3, 3>(3, 0) = J_Q;
            //plane->planeCov += J * pvList[i].neighCov.block<3, 3>(0, 0) * J.transpose();
            plane->planeCov += J * pvList[i].obsCov * J.transpose();
        }
    }

}

void OctoTree::calcNeighCov(std::vector<PointWithCov>& pvList, PointVector& cloud)
{
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(auto& p: cloud)
        pclCloud->push_back(pcl::PointXYZ(p.x,p.y,p.z));

    kdtree->setInputCloud(pclCloud);

    _cloudCov.setZero();
    _mean.setZero();
    const int pvSize = pvList.size();
    const int nearSize = std::min(pvSize, _kCorrespondences);
    Eigen::Matrix4d* covariances=new Eigen::Matrix4d[pvSize];
    Eigen::Vector4d* means=new Eigen::Vector4d[pvSize];

#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
    for (int i = 0; i < pvSize; i++)
    {
        pcl::PointXYZ pt;
        pt.x=pvList[i].pw.x();
        pt.y=pvList[i].pw.y();
        pt.z=pvList[i].pw.z();
        std::vector<int> kIndices;
        std::vector<float> kSqDistances;
        kdtree->nearestKSearch(pt, nearSize, kIndices, kSqDistances);

        Eigen::Matrix<double, 4, -1> neighbors(4, nearSize);
        for (int j = 0; j < kIndices.size(); j++)
            neighbors.col(j) = pclCloud->at(kIndices[j]).getVector4fMap().template cast<double>();

        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        Eigen::Matrix4d cov = neighbors * neighbors.transpose() / nearSize;

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        covariances[i] = Eigen::Matrix4d::Zero();
        covariances[i].block<3, 3>(0, 0) = svd.matrixU() * Eigen::Vector3d(1, 1, 1e-3).asDiagonal() * svd.matrixV().transpose();
        pvList[i].neighCov =  covariances[i];
        means[i] = Eigen::Vector4d(pt.x, pt.y, pt.z, 1.);
    }
    for(int i=0;i<pvSize;i++)
    {
        _cloudCov+= covariances[i];
        _mean+=means[i];
    }
    _cloudCov /= pvSize;
    _mean /= pvSize;
    delete[] covariances, means;
}

void OctoTree::calcNeighCov(std::vector<PointWithCov>& pvList)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pclCloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (auto &pv:pvList)
        pclCloud->push_back(pcl::PointXYZ(pv.pw.x(), pv.pw.y(), pv.pw.z()));
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    kdtree->setInputCloud(pclCloud);

    _cloudCov.setZero();
    _mean.setZero();
    const int cloudSize = pvList.size();
    const int nearSize = std::min(cloudSize, _kCorrespondences);
    Eigen::Matrix4d* covariances=new Eigen::Matrix4d[cloudSize];
    Eigen::Vector4d* means=new Eigen::Vector4d[cloudSize];

#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
    for (int i = 0; i < cloudSize; i++)
    {
        std::vector<int> kIndices;
        std::vector<float> kSqDistances;
        kdtree->nearestKSearch(pclCloud->at(i), nearSize, kIndices, kSqDistances);

        Eigen::Matrix<double, 4, -1> neighbors(4, nearSize);
        for (int j = 0; j < kIndices.size(); j++)
            neighbors.col(j) = pclCloud->at(kIndices[j]).getVector4fMap().template cast<double>();

        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        Eigen::Matrix4d cov = neighbors * neighbors.transpose() / nearSize;

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        covariances[i] = Eigen::Matrix4d::Zero();
        covariances[i].block<3, 3>(0, 0) = svd.matrixU() * Eigen::Vector3d(1, 1, 1e-3).asDiagonal() * svd.matrixV().transpose();
        pvList[i].neighCov =  covariances[i];
        means[i] = Eigen::Vector4d(pclCloud->at(i).x, pclCloud->at(i).y, pclCloud->at(i).z, 1.);
    }
    for(int i=0;i<cloudSize;i++)
    {
        _cloudCov+= covariances[i];
        _mean+=means[i];
    }
    _cloudCov /= cloudSize;
    _mean /= cloudSize;
	delete[] covariances, means;
}

void OctoTree::calcNeighCov(std::vector<PointWithCov>& pvList, KD_TREE* ikdTree)
{
    _cloudCov.setZero();
    _mean.setZero();
    //downSamplingVoxel(pvList, downPvList, _quaterLength);

    const int cloudSize = pvList.size();
	Eigen::Matrix4d* covariances = new Eigen::Matrix4d[cloudSize];
	Eigen::Vector4d* means = new Eigen::Vector4d[cloudSize];
    //std::cout<<"cloudSize: "<<cloudSize<<std::endl;
#pragma omp parallel for num_threads(MP_PROC_NUM) schedule(guided, 8)
    for (int i = 0; i < cloudSize; i++)
    {
        std::vector<float> kSqDistances;
        PointType pt;
        pt.x=pvList[i].pw.x();
        pt.y=pvList[i].pw.y();
        pt.z=pvList[i].pw.z();
        PointVector pointsNear;
        ikdTree->Nearest_Search(pt, _kCorrespondences, pointsNear, kSqDistances);

        Eigen::Matrix<double, 4, -1> neighbors(4, _kCorrespondences);
        for (int j = 0; j < pointsNear.size(); j++)
            neighbors.col(j) = Eigen::Vector4d(pointsNear[j].x, pointsNear[j].y, pointsNear[j].z, 1.);

        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        Eigen::Matrix4d cov = neighbors * neighbors.transpose() / _kCorrespondences;

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        covariances[i] = Eigen::Matrix4d::Zero();
        covariances[i].block<3, 3>(0, 0) = svd.matrixU() * Eigen::Vector3d(1, 1, 1e-3).asDiagonal() * svd.matrixV().transpose();
        pvList[i].neighCov =  covariances[i];
        means[i] = Eigen::Vector4d(pt.x, pt.y, pt.z, 1.);
    }
    for(int i=0;i<cloudSize;i++)
    {
        _cloudCov+= covariances[i];
        _mean+=means[i];
    }
    _cloudCov /= cloudSize;
    _mean /= cloudSize;

	delete[] covariances, means;
}

int OctoTree::getVoxelObsCount(const int& nObs)
{
    int nMaxObs= std::max(nObs,_nVisible);

    if (_layer < _maxLayers)
        for (size_t i = 0; i < 8; i++)
            if (_leaves[i] != nullptr)
                nMaxObs=_leaves[i]->getVoxelObsCount(nMaxObs);

    return nMaxObs;
}

void buildVoxelMap(const std::vector<PointWithCov> &inputPoints,
                   const int& frameId,
                   const float voxelSize, const int maxLayer,
                   const std::vector<int> &layerPointSize,
                   const int maxPointsSize, const int maxCovPointsSize,
                   const float featThreshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &featMap)
{
    uint plsize = inputPoints.size();
    for (uint i = 0; i < plsize; i++)
    {
        const PointWithCov pv = inputPoints[i];
        float locXYZ[3];
        for (int j = 0; j < 3; j++)
        {
            locXYZ[j] = pv.pw[j] / voxelSize;
            if (locXYZ[j] < 0)
                locXYZ[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t) locXYZ[0], (int64_t) locXYZ[1], (int64_t) locXYZ[2]);
        auto iter = featMap.find(position);
        if (iter != featMap.end())
        {
            featMap[position]->_tempPoints.push_back(pv);
            featMap[position]->_newPointsNum++;
            featMap[position]->_allPointsNum++;
        }
        else //new octo
        {
            OctoTree *octoTree = new OctoTree(frameId, pv.featType, maxLayer, 0, layerPointSize, maxPointsSize, maxCovPointsSize, featThreshold);
            featMap[position] = octoTree;
            featMap[position]->_quaterLength = voxelSize / 4;
            featMap[position]->_voxelCenter[0] = (0.5 + position.x) * voxelSize;
            featMap[position]->_voxelCenter[1] = (0.5 + position.y) * voxelSize;
            featMap[position]->_voxelCenter[2] = (0.5 + position.z) * voxelSize;
            featMap[position]->_tempPoints.push_back(pv);
            featMap[position]->_newPointsNum++;
            featMap[position]->_allPointsNum++;
            featMap[position]->_layerPointSizeList = layerPointSize;
        }
    }
    for (auto iter = featMap.begin(); iter != featMap.end(); ++iter)
        iter->second->initOctoTree();
}

void updateVoxelMap(const std::vector<PointWithCov> &inputPoints,
                    const int& frameId,
                    const float voxelSize, const int maxLayer,
                    const std::vector<int> &layerPointSize,
                    const int maxPointsSize, const int maxCovPointsSize,
                    const float featThreshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &featMap,
                    std::vector<BoxPointType> &octoBoxToDel)
{
    uint plsize = inputPoints.size();
    for (uint i = 0; i < plsize; i++)
    {
        const PointWithCov pv = inputPoints[i];
        float locXYZ[3];
        for (int j = 0; j < 3; j++)
        {
            locXYZ[j] = pv.pw[j] / voxelSize;
            if (locXYZ[j] < 0)
                locXYZ[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t) locXYZ[0], (int64_t) locXYZ[1], (int64_t) locXYZ[2]);
        auto iter = featMap.find(position);
        if (iter != featMap.end())
            featMap[position]->updateOctoTree(pv);
        else
        {
            OctoTree *octo_tree = new OctoTree(frameId, pv.featType, maxLayer, 0, layerPointSize, maxPointsSize, maxCovPointsSize, featThreshold);
            featMap[position] = octo_tree;
            featMap[position]->_quaterLength = voxelSize / 4;
            featMap[position]->_voxelCenter[0] = (0.5 + position.x) * voxelSize;
            featMap[position]->_voxelCenter[1] = (0.5 + position.y) * voxelSize;
            featMap[position]->_voxelCenter[2] = (0.5 + position.z) * voxelSize;
            featMap[position]->updateOctoTree(pv);

        }
    }
//    for(auto locOct:featMap)
//        if(locOct.second->_planePtr->isPlane)
//            std::cout<<locOct.second->_planePtr->normal.transpose()<<std::endl;

//    std::vector<VOXEL_LOC> octoToErase;
//    for(auto& locOct:featMap)
//    {
//        if(locOct.second->_refFrameID>=0)
//            if(frameId-locOct.second->_refFrameID>40)
//                if(locOct.second->getVoxelObsCount(locOct.second->_nVisible)<4)
//                {
//                    //octoToErase.push_back(locOct.first);
//                }
//    }
//    for(auto& loc:octoToErase)
//    {
//        OctoTree* oct=featMap[loc];
//
//        BoxPointType box;
//        box.vertex_min[0]=oct->_voxelCenter[0] - voxelSize / 2.;
//        box.vertex_min[1]=oct->_voxelCenter[1] - voxelSize / 2.;
//        box.vertex_min[2]=oct->_voxelCenter[2] - voxelSize / 2.;
//        box.vertex_max[0]=oct->_voxelCenter[0] + voxelSize / 2.;
//        box.vertex_max[1]=oct->_voxelCenter[1] + voxelSize / 2.;
//        box.vertex_max[2]=oct->_voxelCenter[2] + voxelSize / 2.;
//        octoBoxToDel.push_back(box);
//        featMap.erase(loc);
//    }


}

#endif //BA_ENABLE


static std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
neighborOffsets(NeighborSearchMethod search_method)
{
    switch (search_method)
    {
        // clang-format off
        default:
            std::cerr << "unsupported neighbor search method" << std::endl;
            abort();
        case NeighborSearchMethod::DIRECT1:
            return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>{
                Eigen::Vector3i(0, 0, 0)
            };
        case NeighborSearchMethod::DIRECT7:
            return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
                    {
                            Eigen::Vector3i(0, 0, 0),
                            Eigen::Vector3i(1, 0, 0),
                            Eigen::Vector3i(-1, 0, 0),
                            Eigen::Vector3i(0, 1, 0),
                            Eigen::Vector3i(0, -1, 0),
                            Eigen::Vector3i(0, 0, 1),
                            Eigen::Vector3i(0, 0, -1)
                    };
        case NeighborSearchMethod::DIRECT27:
            break;
            // clang-format on
    }

    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> offsets27;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                offsets27.push_back(Eigen::Vector3i(i - 1, j - 1, k - 1));
            }
        }
    }
    return offsets27;
}

void buildSingleResidual(const PointWithCov &pv, OctoTree *currentOcto,
                         const int currentLayer, const int maxLayers,
                         const double sigmaNum, bool &isSucess,
                         double &prob, MatchOctoTreeInfo &singlePtpl, bool isStrict,double radius_k)
{
    //radius_k = 1; //1 for stair building 3 for normal scene
    Eigen::Vector3d p_w = pv.pw;
    //std::cout<<radius_k<<std::endl;
    if (currentOcto->_planePtr->isPlane)
    {
        PlaneParams &plane = *currentOcto->_planePtr;
        Eigen::Vector3d p_world_to_center = p_w - plane.center;
        double proj_x = p_world_to_center.dot(plane.xNormal);
        double proj_y = p_world_to_center.dot(plane.yNormal);
        float dis_to_plane = fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) + plane.normal(2) * p_w(2) + plane.d);
        float dis_to_center = (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
                              (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
                              (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= radius_k * plane.radius)
        {
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_w - plane.center;
            J_nq.block<1, 3>(0, 3) = -plane.normal;
            double sigma_l = J_nq * plane.planeCov * J_nq.transpose();
            sigma_l += plane.normal.transpose() * pv.obsCov * plane.normal;
            double sigmaThres=FLT_MAX;
            if(isStrict)
                sigmaThres = sigmaNum * sqrt(sigma_l);

            if (dis_to_plane < sigmaThres)
            {
                isSucess = true;
                double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
                if (this_prob > prob)
                {
                    prob = this_prob;
                    singlePtpl.pv = pv;
                    singlePtpl.point = pv.pl;
                    singlePtpl.plane_cov = plane.planeCov;
                    singlePtpl.normal = plane.normal;
                    singlePtpl.center = plane.center;
                    singlePtpl.d = plane.d;
                    singlePtpl.layer = currentLayer;

                    singlePtpl.voxel_correspondence = currentOcto;
                }
                return;
            }
            else
            {
                // isSucess = false;
                return;
            }
        }
        else
        {
            // isSucess = false;
            return;
        }
    }
    else
    {
        if (currentLayer < maxLayers)
        {
            for (size_t leafnum = 0; leafnum < 8; leafnum++)
            {
                if (currentOcto->_leaves[leafnum] != nullptr)
                {
                    OctoTree *leaf_octo = currentOcto->_leaves[leafnum];
                    buildSingleResidual(pv, leaf_octo, currentLayer + 1, maxLayers,
                                        sigmaNum, isSucess, prob, singlePtpl, isStrict,radius_k);
                }
            }
            return;
        }
        else
        {
            // isSucess = false;
            return;
        }
    }


}


void buildResidualListOmp(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const double voxel_size,
                          const double sigma_num,
                          const int max_layers,
                          const std::vector<PointWithCov> &pv_list,
                          std::vector<MatchOctoTreeInfo> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match,
                          bool isStrict,
                          double radius_k,
                          NeighborSearchMethod searchMethod
                        )
{
    std::mutex mylock;
    ptpl_list.clear();
    std::vector<MatchOctoTreeInfo> all_ptpl_list(pv_list.size());
    std::vector<bool> useful_ptpl(pv_list.size());
    std::vector<size_t> index(pv_list.size());
    for (size_t i = 0; i < index.size(); ++i)
    {
        index[i] = i;
        useful_ptpl[i] = false;
    }
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < index.size(); i++)
    {
        PointWithCov pv = pv_list[i];
        std::vector<OctoTree *> voxelList;
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = pv.pw[j] / voxel_size;
            if (loc_xyz[j] < 0)
                loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
        auto offsets = neighborOffsets(searchMethod);
        for (const auto &offset : offsets)
        {
            VOXEL_LOC nearPosition = position;
            nearPosition.x += offset[0];
            nearPosition.y += offset[1];
            nearPosition.z += offset[2];
            auto voxelItr = voxel_map.find(nearPosition);
            if (voxelItr != voxel_map.end())
                voxelList.emplace_back(voxelItr->second);
        }

        for (auto &voxel:voxelList)
        {
            MatchOctoTreeInfo single_ptpl;
            bool is_sucess = false;
            double prob = 0;
            buildSingleResidual(pv, voxel, 0, max_layers, sigma_num, is_sucess, prob, single_ptpl, isStrict,radius_k);
            if (!is_sucess)
            {
                VOXEL_LOC near_position = position;
                if (loc_xyz[0] > (voxel->_voxelCenter[0] + voxel->_quaterLength))
                    near_position.x = near_position.x + 1;
                else if (loc_xyz[0] < (voxel->_voxelCenter[0] - voxel->_quaterLength))
                    near_position.x = near_position.x - 1;
                if (loc_xyz[1] > (voxel->_voxelCenter[1] + voxel->_quaterLength))
                    near_position.y = near_position.y + 1;
                else if (loc_xyz[1] < (voxel->_voxelCenter[1] - voxel->_quaterLength))
                    near_position.y = near_position.y - 1;
                if (loc_xyz[2] > (voxel->_voxelCenter[2] + voxel->_quaterLength))
                    near_position.z = near_position.z + 1;
                else if (loc_xyz[2] < (voxel->_voxelCenter[2] - voxel->_quaterLength))
                    near_position.z = near_position.z - 1;
                auto iter_near = voxel_map.find(near_position);
                if (iter_near != voxel_map.end())
                    buildSingleResidual(pv, iter_near->second, 0, max_layers, sigma_num, is_sucess, prob, single_ptpl, isStrict,radius_k);
            }
            if (is_sucess)
            {
                mylock.lock();
                useful_ptpl[i] = true;
                all_ptpl_list[i] = single_ptpl;
                mylock.unlock();
            }
            else
            {
                mylock.lock();
                useful_ptpl[i] = false;
                mylock.unlock();
            }
        }
    }
    for (size_t i = 0; i < useful_ptpl.size(); i++)
        if (useful_ptpl[i])
            ptpl_list.push_back(all_ptpl_list[i]);
}

void buildResidualListNormal(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                             const double voxel_size, const double sigma_num, const int max_layers,
                             const std::vector<PointWithCov> &pv_list, std::vector<MatchOctoTreeInfo> &ptpl_list,
                             std::vector<Eigen::Vector3d> &non_match)
{
    ptpl_list.clear();
    std::vector<size_t> index(pv_list.size());
    for (size_t i = 0; i < pv_list.size(); ++i)
    {
        PointWithCov pv = pv_list[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = pv.pw[j] / voxel_size;
            if (loc_xyz[j] < 0)
                loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1],
                           (int64_t) loc_xyz[2]);
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end())
        {
            OctoTree *current_octo = iter->second;
            MatchOctoTreeInfo single_ptpl;
            bool is_sucess = false;
            double prob = 0;
            buildSingleResidual(pv, current_octo, 0, max_layers, sigma_num,
                                is_sucess, prob, single_ptpl,3);

            if (!is_sucess)
            {
                VOXEL_LOC near_position = position;
                if (loc_xyz[0] > (current_octo->_voxelCenter[0] + current_octo->_quaterLength))
                    near_position.x = near_position.x + 1;
                else if (loc_xyz[0] < (current_octo->_voxelCenter[0] - current_octo->_quaterLength))
                    near_position.x = near_position.x - 1;
                if (loc_xyz[1] > (current_octo->_voxelCenter[1] + current_octo->_quaterLength))
                    near_position.y = near_position.y + 1;
                else if (loc_xyz[1] < (current_octo->_voxelCenter[1] - current_octo->_quaterLength))
                    near_position.y = near_position.y - 1;
                if (loc_xyz[2] > (current_octo->_voxelCenter[2] + current_octo->_quaterLength))
                    near_position.z = near_position.z + 1;
                else if (loc_xyz[2] < (current_octo->_voxelCenter[2] - current_octo->_quaterLength))
                    near_position.z = near_position.z - 1;

                auto iter_near = voxel_map.find(near_position);
                if (iter_near != voxel_map.end())
                    buildSingleResidual(pv, iter_near->second, 0, max_layers, sigma_num, is_sucess, prob, single_ptpl,3);
            }
            if (is_sucess)
                ptpl_list.push_back(single_ptpl);
            else
                non_match.push_back(pv.pw);
        }
    }
}

void getUpdatePlane(const OctoTree *currentOcto, const int pubMaxVoxelLayer, std::vector<PlaneParams> &planeList)
{
    if (currentOcto->_layer > pubMaxVoxelLayer)
        return;

    if (currentOcto->_planePtr->isPlane)
        planeList.push_back(*currentOcto->_planePtr);

    if (currentOcto->_layer < currentOcto->_maxLayers)
        if (!currentOcto->_planePtr->isPlane)
            for (size_t i = 0; i < 8; i++)
                if (currentOcto->_leaves[i] != nullptr)
                    getUpdatePlane(currentOcto->_leaves[i], pubMaxVoxelLayer, planeList);
    return;
}


void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
    r = 255;
    g = 255;
    b = 255;

    if (v < vmin)
    {
        v = vmin;
    }

    if (v > vmax)
    {
        v = vmax;
    }

    double dr, dg, db;

    if (v < 0.1242)
    {
        db = 0.504 + ((1. - 0.504) / 0.1242) * v;
        dg = dr = 0.;
    }
    else if (v < 0.3747)
    {
        db = 1.;
        dr = 0.;
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
    }
    else if (v < 0.6253)
    {
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
        dg = 1.;
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
    }
    else if (v < 0.8758)
    {
        db = 0.;
        dr = 1.;
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
    }
    else
    {
        db = 0.;
        dg = 0.;
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
    }

    r = (uint8_t) (255 * dr);
    g = (uint8_t) (255 * dg);
    b = (uint8_t) (255 * db);
}

void downSamplingVoxel(PvList &cloud, double voxel_size)
{
    std::unordered_map<VOXEL_LOC, M_POINT> featMap;
    uint ptSize = cloud.size();
    for (uint i = 0; i < ptSize; i++)
    {
        PointWithCov &pv = cloud[i];
        double locXYZ[3];
        for (int j = 0; j < 3; j++)
        {
            locXYZ[j] = pv.pw[j] / voxel_size;
            if (locXYZ[j] < 0)
                locXYZ[j] -= 1.0;
        }

        VOXEL_LOC position((int64_t) locXYZ[0], (int64_t) locXYZ[1], (int64_t) locXYZ[2]);
        auto iter = featMap.find(position);
        if (iter != featMap.end())
        {
            iter->second.xyz[0] += pv.pw[0];
            iter->second.xyz[1] += pv.pw[1];
            iter->second.xyz[2] += pv.pw[2];
            iter->second.count++;
        }
        else
        {
            M_POINT anp;
            anp.xyz[0] = pv.pw[0];
            anp.xyz[1] = pv.pw[1];
            anp.xyz[2] = pv.pw[2];
            anp.count = 1;
            featMap[position] = anp;
        }
    }

    cloud.resize(featMap.size());
    uint i = 0;
    for (auto iter = featMap.begin(); iter != featMap.end(); ++iter)
    {
        cloud[i].pw.x() = iter->second.xyz[0] / iter->second.count;
        cloud[i].pw.y() = iter->second.xyz[1] / iter->second.count;
        cloud[i].pw.z() = iter->second.xyz[2] / iter->second.count;
        i++;
    }
}

void downSamplingVoxel(const PvList &cloudIn, PvList &cloudOut, double voxel_size)
{
    std::unordered_map<VOXEL_LOC, M_POINT> featMap;
    uint ptSize = cloudIn.size();
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < ptSize; i++)
    {
        const PointWithCov &pv = cloudIn[i];
        double locXYZ[3];
        for (int j = 0; j < 3; j++)
        {
            locXYZ[j] = pv.pw[j] / voxel_size;
            if (locXYZ[j] < 0)
                locXYZ[j] -= 1.0;
        }

        VOXEL_LOC position((int64_t) locXYZ[0], (int64_t) locXYZ[1], (int64_t) locXYZ[2]);
        auto iter = featMap.find(position);
        if (iter != featMap.end())
        {
            iter->second.xyz[0] += pv.pw[0];
            iter->second.xyz[1] += pv.pw[1];
            iter->second.xyz[2] += pv.pw[2];
            iter->second.count++;
        }
        else
        {
            M_POINT anp;
            anp.xyz[0] = pv.pw[0];
            anp.xyz[1] = pv.pw[1];
            anp.xyz[2] = pv.pw[2];
            anp.count = 1;
            featMap[position] = anp;
        }
    }

    cloudOut.resize(featMap.size());
    uint i = 0;
    for (auto iter = featMap.begin(); iter != featMap.end(); ++iter)
    {
        cloudOut[i].pw.x() = iter->second.xyz[0] / iter->second.count;
        cloudOut[i].pw.y() = iter->second.xyz[1] / iter->second.count;
        cloudOut[i].pw.z() = iter->second.xyz[2] / iter->second.count;
        i++;
    }
}