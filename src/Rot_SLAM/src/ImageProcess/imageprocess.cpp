#include "ImageProcess/imageprocess.h"
#include "ImageProcess/depth_completion.hpp"
#include <limits>
pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::Boundary>::Ptr boundaries(
    new pcl::PointCloud<pcl::Boundary>); //声明一个boundary类指针，作为返回值
pcl::PointCloud<pcl::PointXYZI> next_cloud;
pcl::PointCloud<pcl::PointXYZI>::Ptr
    planner_cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
pcl::PointCloud<pcl::PointXYZI>::Ptr
    cloud_filter(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    color_planner_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    mergedPts(new pcl::PointCloud<pcl::PointXYZRGB>);
std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
std::vector<imgProcesser::Plane> plane_list;

// inliers????????????????????????
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
bool imgProcesser::startThread() {
    _imgThread = new std::thread(&imgProcesser::run, this);
    if (!_imgThread)
        return false;
    return true;
}
void imgProcesser::inputframedata(
    pcl::PointCloud<pcl::PointXYZI>::Ptr sparsecloud, cv::Mat curdenseimg,
    cv::Mat curdenseimg_left, cv::Mat curdenseimg_right) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr tempsparsecloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    tempsparsecloud->width = sparsecloud->width;
    tempsparsecloud->height = sparsecloud->height;
    tempsparsecloud->resize(sparsecloud->size());
    for (int i = 0; i < sparsecloud->size(); i++) {
        tempsparsecloud->points[i].x = sparsecloud->points[i].x;
        tempsparsecloud->points[i].y = sparsecloud->points[i].y;
        tempsparsecloud->points[i].z = sparsecloud->points[i].z;
        tempsparsecloud->points[i].intensity = sparsecloud->points[i].intensity;
    }
    _sparsecloudbuffer.push_back(tempsparsecloud);
    _denseimgbuffer.push_back(curdenseimg);
    _denseimgbuffer_left.push_back(curdenseimg_left);
    _denseimgbuffer_right.push_back(curdenseimg_right);
}

void imgProcesser::run() {
    bool hasdata = false;
    // 添加总计时变量
    auto total_start_time = std::chrono::high_resolution_clock::now();
    int iteration_count = 0;
    double total_processing_time = 0.0;

    while (1) {
        // 每次循环的计时
        auto loop_start_time = std::chrono::high_resolution_clock::now();
        _mutexBuf.lock();
        std::vector<std::vector<int>> linelist;
        std::vector<std::vector<int>> linelist_left;
        std::vector<std::vector<int>> linelist_right;
        std::vector<Matchlinelist> _matchlinelist;
        std::vector<Matchlinelist> _matchlinelist_left;
        std::vector<Matchlinelist> _matchlinelist_right;

        if (_sparsecloudbuffer.size() > 0) {
            std::cout << "get img framedata" << std::endl;
            cursparsecloud = _sparsecloudbuffer.front();
            // intensityMapdense = _densecloudbuffer.front();
            // transCloud3(intensityMapdense, intensityMapdense_left, Rleft,
            // Eigen::Vector3d(0,0,0)); transCloud3(intensityMapdense,
            // intensityMapdense_right, Rright, Eigen::Vector3d(0,0,0)); curdenseimg =
            // projectPinhole(intensityMapdense, false); curdenseimg_left =
            // projectPinhole(intensityMapdense_left, false); curdenseimg_right =
            // projectPinhole(intensityMapdense_right, false);
            curdenseimg = _denseimgbuffer.front();
            curdenseimg_left = _denseimgbuffer_left.front();
            curdenseimg_right = _denseimgbuffer_right.front();
            _sparsecloudbuffer.erase(_sparsecloudbuffer.begin());
            _denseimgbuffer.erase(_denseimgbuffer.begin());
            _denseimgbuffer_left.erase(_denseimgbuffer_left.begin());
            _denseimgbuffer_right.erase(_denseimgbuffer_right.begin());
            //_densecloudbuffer.erase(_densecloudbuffer.begin());
            hasdata = true;
        }
        _mutexBuf.unlock();

        if (hasdata) {
            _mutexBuf.lock();

            // 特征提取计时
            auto feature_start = std::chrono::high_resolution_clock::now();
            extractIntensityEdgesOptimized(cursparsecloud, sparselinecloud);
            auto feature_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> feature_time =
                feature_end - feature_start;

            // 图像处理计时
            auto fill_start = std::chrono::high_resolution_clock::now();
            curdenseimg = fillHolesFast(curdenseimg);
            curdenseimg_left = fillHolesFast(curdenseimg_left);
            curdenseimg_right = fillHolesFast(curdenseimg_right);
            cv::imwrite("/home/zzy/SLAM/mri-mms/src/Rot_intensity_augm_LIVO/image/final.png",
                        curdenseimg);

            // cv::imwrite(std::string(ROOT_DIR)+"image/medium/"+std::to_string(frame)+".png",
            // curdenseimg);
            // cv::imwrite(std::string(ROOT_DIR)+"image/right/"+std::to_string(frame)+".png",curdenseimg_left);
            // cv::imwrite(std::string(ROOT_DIR)+"image/left/"+std::to_string(frame)+".png",curdenseimg_right);

            auto fill_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> fill_time =
                fill_end - fill_start;

            cv::Mat colorintensityImg;
            cv::cvtColor(curdenseimg, colorintensityImg, cv::COLOR_GRAY2BGR);
            cv::Mat colorintensityImg_left;
            cv::cvtColor(curdenseimg_left, colorintensityImg_left,
                         cv::COLOR_GRAY2BGR);
            cv::Mat colorintensityImg_right;
            cv::cvtColor(curdenseimg_right, colorintensityImg_right,
                         cv::COLOR_GRAY2BGR);

            // colorintensityImg=ultraFastGrayToRGB(curdenseimg);
            // colorintensityImg_left=ultraFastGrayToRGB(curdenseimg_left);
            // colorintensityImg_right=ultraFastGrayToRGB(curdenseimg_right);

            // colorintensityImg=enhanceSubtleChangesFast(curdenseimg);
            // colorintensityImg_left=enhanceSubtleChangesFast(curdenseimg_left);
            // colorintensityImg_right=enhanceSubtleChangesFast(curdenseimg_right);

            // 线段检测计时
            auto line_detect_start = std::chrono::high_resolution_clock::now();
            cannyimg = _lsd.detect(colorintensityImg, linelist);
            cannyimg_left = _lsd.detect(colorintensityImg_left, linelist_left);
            cannyimg_right = _lsd.detect(colorintensityImg_right, linelist_right);
            auto line_detect_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> line_detect_time =
                line_detect_end - line_detect_start;

            // 匹配列表建立计时
            auto match_build_start = std::chrono::high_resolution_clock::now();
            buildmatchlinelist(_matchlinelist, sparselinecloud, linelist);
            buildmatchlinelist_left(_matchlinelist_left, sparselinecloud,
                                    linelist_left);
            buildmatchlinelist_right(_matchlinelist_right, sparselinecloud,
                                     linelist_right);
            auto match_build_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> match_build_time =
                match_build_end - match_build_start;

            // 可视化计时
            auto vis_start = std::chrono::high_resolution_clock::now();
            _matchImg = visualimg(colorintensityImg, _matchlinelist);
            _matchImg_left = visualimg(colorintensityImg_left, _matchlinelist_left);
            _matchImg_right =
                visualimg(colorintensityImg_right, _matchlinelist_right);
            auto vis_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> vis_time = vis_end - vis_start;
            // cv::imwrite(std::string(ROOT_DIR)+"image/match/"+std::to_string(frame)+".png",
            // colorintensityImg);
            // cv::imwrite(std::string(ROOT_DIR)+"image/match_right/"+std::to_string(frame)+".png",_matchImg_left);
            // cv::imwrite(std::string(ROOT_DIR)+"image/match_left/"+std::to_string(frame)+".png",_matchImg_right);
            // cv::imwrite(std::string(ROOT_DIR)+"image/medium/"+std::to_string(frame)+".png",
            // _matchImg);
            // cv::imwrite(std::string(ROOT_DIR)+"image/right/"+std::to_string(frame)+".png",_matchImg_left);
            // cv::imwrite(std::string(ROOT_DIR)+"image/left/"+std::to_string(frame)+".png",_matchImg_right);
            // 保存数据
            _matchimgbuffer.push_back(_matchImg);
            _matchimgbuffer_left.push_back(_matchImg_left);
            _matchimgbuffer_right.push_back(_matchImg_right);
            _matchlinelistbuffer.push_back(_matchlinelist);
            _matchlinelistbuffer_left.push_back(_matchlinelist_left);
            _matchlinelistbuffer_right.push_back(_matchlinelist_right);
            _sparselinecloudbuffer.push_back(sparselinecloud);
            _linelistbuffer.push_back(linelist);
            _linelistbuffer_left.push_back(linelist_left);
            _linelistbuffer_right.push_back(linelist_right);

            hasdata = false;

            _mutexBuf.unlock();

            // 打印各阶段计时
            std::cout << "填充空洞时间: " << fill_time.count() << " ms" << std::endl;
            std::cout << "线段检测时间: " << line_detect_time.count() << " ms"
                      << std::endl;
            frame++;
        }
    }
}
void imgProcesser::transCloud3(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudIn,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudOut, const Eigen::Matrix3d &Rol,
    const Eigen::Vector3d &tol) {
    const int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    for (int i = 0; i < cloudSize; i++) {
        const auto &pb = cloudIn->points[i];
        V3D pl(pb.x, pb.y, pb.z); // pl
        V3D po(Rol * pl + tol);   // po=(Tol*pl)

        cloudOut->points[i].x = po[0];
        cloudOut->points[i].y = po[1];
        cloudOut->points[i].z = po[2];
        cloudOut->points[i].intensity = pb.intensity;
    }
}
cv::Mat imgProcesser::visualimg(cv::Mat img,
                                std::vector<Matchlinelist> &matchlinelist) {
    cv::Mat visualImage = img.clone();

    // 设置颜色和大小参数
    const cv::Scalar LINE_COLOR(255, 0, 0);  // 蓝色线 (BGR)
    const cv::Scalar POINT_COLOR(0, 0, 255); // 红色点 (BGR)
    const cv::Scalar ARROW_COLOR(0, 255, 0); // 绿色箭头 (BGR)
    const int POINT_SIZE = 5;                // 点大小
    const int LINE_THICKNESS = 2;            // 线粗细
    const int ARROW_THICKNESS = 2;           // 箭头粗细
    const double ARROW_LENGTH = 30.0;        // 箭头长度

    // 处理每一个匹配线
    for (const auto &match : matchlinelist) {
        // 绘制直线
        if (match.linestart.size() >= 2 && match.lineend.size() >= 2) {
            cv::Point lineStart(match.linestart[0], match.linestart[1]);
            cv::Point lineEnd(match.lineend[0], match.lineend[1]);
            cv::line(visualImage, lineStart, lineEnd, LINE_COLOR, LINE_THICKNESS);
        }

        // 绘制点
        if (match.p2d.size() >= 2) {
            cv::Point p2dPoint(match.p2d[0], match.p2d[1]);
            cv::circle(visualImage, p2dPoint, POINT_SIZE, POINT_COLOR,
                       -1); // -1表示填充圆

            // 使用直线参数计算垂直方向
            // 直线方程：ax + by + c = 0
            // 垂直向量：(a, b)

            // 归一化垂直向量
            double norm = std::sqrt(match.a * match.a + match.b * match.b);
            if (norm > 1e-6) { // 避免除以零
                double nx = match.a / norm;
                double ny = match.b / norm;

                // 计算箭头终点
                cv::Point arrowEnd;

                // 计算点到直线的距离
                double dist =
                    std::abs(match.a * p2dPoint.x + match.b * p2dPoint.y + match.c) / norm;

                // 确定箭头方向（指向直线）
                double sign =
                    (match.a * p2dPoint.x + match.b * p2dPoint.y + match.c) > 0 ? -1.0 : 1.0;

                arrowEnd.x = p2dPoint.x + sign * nx * ARROW_LENGTH;
                arrowEnd.y = p2dPoint.y + sign * ny * ARROW_LENGTH;

                // 绘制箭头
                cv::arrowedLine(visualImage, p2dPoint, arrowEnd, ARROW_COLOR,
                                ARROW_THICKNESS);

                // 可选：绘制距离标签
                // std::string distText = cv::format("%.2f", dist);
                // cv::putText(visualImage, distText,
                //             cv::Point((p2dPoint.x + arrowEnd.x) / 2, (p2dPoint.y +
                //             arrowEnd.y) / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                //             ARROW_COLOR, 1);
            }
        }
    }

    return visualImage;
}
void imgProcesser::buildmatchlinelist(
    std::vector<Matchlinelist> &matchlinelist,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &_sparselinecloud,
    std::vector<std::vector<int>> segments_list) {
    matchlinelist.clear();
    for (int i = 0; i < _sparselinecloud->points.size(); i++) {
        Matchlinelist matchline;
        pcl::PointXYZI p3d = _sparselinecloud->points[i];
        p3d.intensity = _sparselinecloud->points[i].intensity;
        Eigen::Vector2d p2d = projectTosingle(_sparselinecloud->points[i]);
        if (p2d[0] < 0) {
            continue;
        }
        for (int j = 0; j < segments_list.size(); j++) {
            Eigen::Vector2d start2d =
                Eigen::Vector2d(segments_list[j][0], segments_list[j][1]);
            Eigen::Vector2d end2d =
                Eigen::Vector2d(segments_list[j][2], segments_list[j][3]);
            double a = segments_list[j][4];
            double b = segments_list[j][5];
            double c = segments_list[j][6];
            double aabb = a * a + b * b;
            double dist = (a * p2d[0] + b * p2d[1] + c) / sqrt(aabb);
            double pdist = (p2d - start2d).dot(p2d - end2d);
            if (abs(dist) < 5 && pdist < 0) {
                matchline.a = a / sqrt(aabb);
                matchline.b = b / sqrt(aabb);
                matchline.c = c / sqrt(aabb);
                matchline.p2d.push_back(p2d[0]);
                matchline.p2d.push_back(p2d[1]);
                matchline.p3d = p3d;
                matchline.linestart.push_back(segments_list[j][0]);
                matchline.linestart.push_back(segments_list[j][1]);
                matchline.lineend.push_back(segments_list[j][2]);
                matchline.lineend.push_back(segments_list[j][3]);
                matchlinelist.push_back(matchline);
                matchline.dist = dist;
                break;
            }
        }
    }
}
void imgProcesser::buildmatchlinelist_left(
    std::vector<Matchlinelist> &matchlinelist,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &_sparselinecloud,
    std::vector<std::vector<int>> segments_list) {
    matchlinelist.clear();
    pcl::PointCloud<pcl::PointXYZI>::Ptr _sparselinecloud_left(
        new pcl::PointCloud<pcl::PointXYZI>);
    transCloud3(_sparselinecloud, _sparselinecloud_left, Rleft,
                Eigen::Vector3d(0, 0, 0));
    for (int i = 0; i < _sparselinecloud_left->points.size(); i++) {
        Matchlinelist matchline;
        pcl::PointXYZI p3d = _sparselinecloud->points[i];
        p3d.intensity = 1;
        Eigen::Vector2d p2d = projectTosingle(_sparselinecloud_left->points[i]);
        if (p2d[0] < 0) {
            continue;
        }
        for (int j = 0; j < segments_list.size(); j++) {
            Eigen::Vector2d start2d =
                Eigen::Vector2d(segments_list[j][0], segments_list[j][1]);
            Eigen::Vector2d end2d =
                Eigen::Vector2d(segments_list[j][2], segments_list[j][3]);
            double a = segments_list[j][4];
            double b = segments_list[j][5];
            double c = segments_list[j][6];
            double aabb = a * a + b * b;
            double dist = (a * p2d[0] + b * p2d[1] + c) / sqrt(aabb);
            double pdist = (p2d - start2d).dot(p2d - end2d);
            if (abs(dist) < 5 && pdist < 0) {
                matchline.a = a / sqrt(aabb);
                matchline.b = b / sqrt(aabb);
                matchline.c = c / sqrt(aabb);
                matchline.p2d.push_back(p2d[0]);
                matchline.p2d.push_back(p2d[1]);
                matchline.p3d = p3d;
                matchline.linestart.push_back(segments_list[j][0]);
                matchline.linestart.push_back(segments_list[j][1]);
                matchline.lineend.push_back(segments_list[j][2]);
                matchline.lineend.push_back(segments_list[j][3]);
                matchlinelist.push_back(matchline);
                matchline.dist = dist;
                break;
            }
        }
    }
    _sparselinecloud_left->clear();
}
void imgProcesser::buildmatchlinelist_right(
    std::vector<Matchlinelist> &matchlinelist,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &_sparselinecloud,
    std::vector<std::vector<int>> segments_list) {
    matchlinelist.clear();
    pcl::PointCloud<pcl::PointXYZI>::Ptr _sparselinecloud_right(
        new pcl::PointCloud<pcl::PointXYZI>);
    transCloud3(_sparselinecloud, _sparselinecloud_right, Rright,
                Eigen::Vector3d(0, 0, 0));
    for (int i = 0; i < _sparselinecloud_right->points.size(); i++) {
        Matchlinelist matchline;
        pcl::PointXYZI p3d = _sparselinecloud->points[i];
        p3d.intensity = 2;
        Eigen::Vector2d p2d = projectTosingle(_sparselinecloud_right->points[i]);
        if (p2d[0] < 0) {
            continue;
        }
        for (int j = 0; j < segments_list.size(); j++) {
            Eigen::Vector2d start2d =
                Eigen::Vector2d(segments_list[j][0], segments_list[j][1]);
            Eigen::Vector2d end2d =
                Eigen::Vector2d(segments_list[j][2], segments_list[j][3]);
            double a = segments_list[j][4];
            double b = segments_list[j][5];
            double c = segments_list[j][6];
            double aabb = a * a + b * b;
            double dist = (a * p2d[0] + b * p2d[1] + c) / sqrt(aabb);
            double pdist = (p2d - start2d).dot(p2d - end2d);
            if (abs(dist) < 5 && pdist < 0) {
                matchline.a = a / sqrt(aabb);
                matchline.b = b / sqrt(aabb);
                matchline.c = c / sqrt(aabb);
                matchline.p2d.push_back(p2d[0]);
                matchline.p2d.push_back(p2d[1]);
                matchline.p3d = p3d;
                matchline.linestart.push_back(segments_list[j][0]);
                matchline.linestart.push_back(segments_list[j][1]);
                matchline.lineend.push_back(segments_list[j][2]);
                matchline.lineend.push_back(segments_list[j][3]);
                matchlinelist.push_back(matchline);
                matchline.dist = dist;
                break;
            }
        }
    }
    _sparselinecloud_right->clear();
}
cv::Mat imgProcesser::fillHolesFast(const cv::Mat &input) {
    // 检查输入图像
    if (input.empty()) {
        return input.clone();
    }

    // 确保输入图像为8位单通道
    cv::Mat processImage;
    if (input.type() != CV_8U) {
        input.convertTo(processImage, CV_8U);
    } else {
        processImage = input.clone();
    }

    // 创建输出图像
    cv::Mat result = processImage.clone();

    // 标识空洞位置
    cv::Mat mask = (processImage == 0);

    // 获取空洞位置的坐标
    std::vector<cv::Point> holePoints;
    cv::findNonZero(mask, holePoints);

    // 如果没有空洞，直接返回
    if (holePoints.empty()) {
        return result;
    }

    // 固定搜索半径
    const int radius = 2;

// 对每个空洞点进行一次插值
#pragma omp parallel for // 可选：启用OpenMP并行处理
    for (int idx = 0; idx < holePoints.size(); idx++) {
        const cv::Point &point = holePoints[idx];
        int x = point.x;
        int y = point.y;

        // 收集周围非空洞点的ss值
        int totalValue = 0;
        int count = 0;

        // 在周围搜索有效点
        for (int j = -radius; j <= radius; j++) {
            for (int i = -radius; i <= radius; i++) {
                int nx = x + i;
                int ny = y + j;

                // 检查边界
                if (nx >= 0 && nx < processImage.cols && ny >= 0 && ny < processImage.rows) {
                    uchar value = processImage.at<uchar>(ny, nx);
                    if (value > 0) { // 非空洞点
                        totalValue += value;
                        count++;
                    }
                }
            }
        }
        // 如果找到有效点，使用平均值填充
        if (count > 0) {
            result.at<uchar>(y, x) = totalValue / count;
        } else {
            // 如果没找到有效点，使用默认值
            result.at<uchar>(y, x) = 0; // 中间灰度值
        }
    }

    return result;
}

void imgProcesser::cannyEdgeDetection(const cv::Mat &src, cv::Mat &dst) {
    // 检查输入图像
    if (src.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return;
    }
    double lowThreshold = 150.0;
    double highThreshold = 300.0;
    int kernelSize = 7;
    bool useL2gradient = false;
    // 转换为灰度图像
    cv::Mat grayImage;
    if (src.channels() > 1) {
        cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = src.clone();
    }

    // 可选：使用高斯滤波去噪
    // cv::Mat blurred;
    // cv::GaussianBlur(grayImage, blurred, cv::Size(kernelSize, kernelSize), 0);

    // 应用Canny边缘检测
    cv::Canny(grayImage, dst, lowThreshold, highThreshold, kernelSize,
              useL2gradient);
}

/**
 * @brief 提取点云中强度变化显著的边缘点（优化版）
 * @param input_cloud 输入点云
 * @param intensity_edge 输出的强度边缘点
 * @param radius 搜索半径
 * @param threshold 强度协方差阈值基准值
 * @param voxel_size 体素大小，用于空间划分
 * @param points_per_voxel 每个体素中保留的最大边缘点数量
 */
void imgProcesser::extractIntensityEdgesOptimized(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &intensity_edge) {
    // 清空输出点云
    intensity_edge->clear();
    float radius = 0.2;
    float threshold = 5;
    float voxel_size = 1;
    int points_per_voxel = 5;

    // 如果输入为空则返回
    if (input_cloud->empty()) {
        return;
    }

    // 计算点云边界
    pcl::PointXYZI min_pt, max_pt;
    pcl::getMinMax3D(*input_cloud, min_pt, max_pt);

    // 创建KdTree用于近邻搜索
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(
        new pcl::search::KdTree<pcl::PointXYZI>());
    kdtree->setInputCloud(input_cloud);

    // 使用哈希表存储体素及其点信息
    // 键：体素索引（编码为单个整数）
    // 值：pair<协方差值, 点索引>的优先队列
    typedef std::pair<float, size_t> CovariancePointPair;
    typedef std::priority_queue<CovariancePointPair> VoxelPointQueue;
    std::unordered_map<size_t, VoxelPointQueue> voxel_map;

    // 计算网格尺寸
    int grid_size_x = static_cast<int>(ceil((max_pt.x - min_pt.x) / voxel_size));
    int grid_size_y = static_cast<int>(ceil((max_pt.y - min_pt.y) / voxel_size));
    int grid_size_z = static_cast<int>(ceil((max_pt.z - min_pt.z) / voxel_size));

    // 确保网格至少为1
    grid_size_x = std::max(1, grid_size_x);
    grid_size_y = std::max(1, grid_size_y);
    grid_size_z = std::max(1, grid_size_z);

    // 哈希函数，将3D体素索引转换为单个整数
    auto voxelHash = [grid_size_y, grid_size_z](int x, int y, int z) -> size_t {
        return x * grid_size_y * grid_size_z + y * grid_size_z + z;
    };

// 第一步：计算所有点的强度协方差并按体素分组
#pragma omp parallel
    {
        // 线程局部哈希表，减少线程间竞争
        std::unordered_map<size_t, VoxelPointQueue> local_voxel_map;

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < static_cast<int>(input_cloud->size()); ++i) {
            const pcl::PointXYZI &point = input_cloud->points[i];

            // 计算点所属的体素索引
            int voxel_x = static_cast<int>((point.x - min_pt.x) / voxel_size);
            int voxel_y = static_cast<int>((point.y - min_pt.y) / voxel_size);
            int voxel_z = static_cast<int>((point.z - min_pt.z) / voxel_size);
            size_t voxel_idx = voxelHash(voxel_x, voxel_y, voxel_z);

            // 搜索半径内的邻居点
            std::vector<int> neighbor_indices;
            std::vector<float> neighbor_distances;
            kdtree->radiusSearch(point, radius, neighbor_indices, neighbor_distances);

            // 如果邻居点太少则跳过
            if (neighbor_indices.size() < 5) {
                continue;
            }

            // 计算邻域内强度的均值和方差（一次遍历）
            float sum_intensity = 0.0;
            float sum_squared = 0.0;

            for (size_t j = 0; j < neighbor_indices.size(); ++j) {
                float intensity = input_cloud->points[neighbor_indices[j]].intensity;
                sum_intensity += intensity;
                sum_squared += intensity * intensity;
            }

            float n = static_cast<float>(neighbor_indices.size());
            float mean_intensity = sum_intensity / n;
            float variance = (sum_squared / n) - (mean_intensity * mean_intensity);

            // 如果协方差大于阈值，则考虑这个点
            if (variance > threshold) {
                // 添加到对应体素的优先队列
                if (local_voxel_map[voxel_idx].size() < points_per_voxel) {
                    local_voxel_map[voxel_idx].push(std::make_pair(variance, i));
                } else if (variance > local_voxel_map[voxel_idx].top().first) {
                    // 如果协方差比队列中最小的还大，则替换
                    local_voxel_map[voxel_idx].pop();
                    local_voxel_map[voxel_idx].push(std::make_pair(variance, i));
                }
            }
        }

// 合并线程局部结果到全局哈希表
#pragma omp critical
        {
            for (const auto &voxel_entry : local_voxel_map) {
                size_t voxel_idx = voxel_entry.first;
                VoxelPointQueue point_queue = voxel_entry.second;

                // 合并优先队列
                while (!point_queue.empty()) {
                    CovariancePointPair pair = point_queue.top();
                    point_queue.pop();

                    if (voxel_map[voxel_idx].size() < points_per_voxel) {
                        voxel_map[voxel_idx].push(pair);
                    } else if (pair.first > voxel_map[voxel_idx].top().first) {
                        voxel_map[voxel_idx].pop();
                        voxel_map[voxel_idx].push(pair);
                    }
                }
            }
        }
    }

    // 第二步：从每个体素中提取最佳点
    intensity_edge->reserve(voxel_map.size() * points_per_voxel); // 预分配内存

    for (const auto &voxel_entry : voxel_map) {
        VoxelPointQueue point_queue = voxel_entry.second;

        // 从优先队列中提取所有点
        while (!point_queue.empty()) {
            size_t point_idx = point_queue.top().second;
            point_queue.pop();

            intensity_edge->push_back(input_cloud->points[point_idx]);
        }
    }

    // 设置输出点云属性
    intensity_edge->width = intensity_edge->size();
    intensity_edge->height = 1;
    intensity_edge->is_dense = true;

    std::cout << "找到 " << intensity_edge->size() << " 个强度边缘点（来自 "
              << voxel_map.size() << " 个体素）" << std::endl;
}

void imgProcesser::initVoxel(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map) {
    // for voxel test
    // srand((unsigned)time(NULL));
    // pcl::PointCloud<pcl::PointXYZRGB> test_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr VoxeledPts(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    for (size_t i = 0; i < input_cloud->size(); i++) {
        const pcl::PointXYZI &p_c = input_cloud->points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = (p_c.data[j] + voxel_size / 2) / voxel_size;
            // if (loc_xyz[j] < 0)
            // {
            //   loc_xyz[j] = loc_xyz[j] - 1.0;//maybe no +0 and -0
            // }
        }
        // cout<<voxel_size<<"voxel_size"<<endl;
        //??????voxel?????, +???????, -?????????
        VOXEL_LOC position((int64_t)ceil(loc_xyz[0]), (int64_t)ceil(loc_xyz[1]),
                           (int64_t)ceil(loc_xyz[2]));
        //??????int????????????????????voxel
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end()) {
            voxel_map[position]->cloud->push_back(p_c);
            pcl::PointXYZRGB p_rgb;
            p_rgb.x = p_c.x;
            p_rgb.y = p_c.y;
            p_rgb.z = p_c.z;
            p_rgb.r = voxel_map[position]->voxel_color(0);
            p_rgb.g = voxel_map[position]->voxel_color(1);
            p_rgb.b = voxel_map[position]->voxel_color(2);
            VoxeledPts->push_back(p_rgb);
        } else //??????voxel?????????????????????????
        {
            Voxel *voxel = new Voxel(voxel_size);
            voxel_map[position] = voxel;
            voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
            voxel_map[position]->voxel_origin[1] = (position.y - 0.5) * voxel_size;
            voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
            voxel_map[position]->cloud->push_back(p_c);
            int r = rand() % 256;
            int g = rand() % 256;
            int b = rand() % 256;
            voxel_map[position]->voxel_color << r, g, b;
        }
    }

    // pcl::io::savePCDFileASCII("/home/zzy/SLAM/my_slam_work/src/Rot_intensity_augm_LIVO/image/check/voxel.pcd",
    // *VoxeledPts);

    // sensor_msgs::PointCloud2 pub_cloud;
    // pcl::toROSMsg(test_cloud, pub_cloud);
    // pub_cloud.header.frame_id = "livox";
    // rgb_cloud_pub_.publish(pub_cloud);
    // for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
    // {
    //   if (iter->second->cloud->size() > 20)
    //   {
    //     //down_sampling_voxel(*(iter->second->cloud), 0.02);
    //     //?????????С??0.01??????
    //     //????????????????????????????
    //   }
    // }
}

void imgProcesser::calcLineLin(
    std::vector<Plane> &plane_list,
    std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list) {
    if (plane_list.size() >= 2 && plane_list.size() <= 10) {
        // pcl::PointCloud<pcl::PointXYZI> temp_line_cloud;
        for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1;
             plane_index1++) {
            for (size_t plane_index2 = plane_index1 + 1;
                 plane_index2 < plane_list.size(); plane_index2++) {
                float a1 = plane_list[plane_index1].normal[0];
                float b1 = plane_list[plane_index1].normal[1];
                float c1 = plane_list[plane_index1].normal[2];
                float x1 = plane_list[plane_index1].p_center.x;
                float y1 = plane_list[plane_index1].p_center.y;
                float z1 = plane_list[plane_index1].p_center.z;
                float minusd1 = a1 * x1 + b1 * y1 + c1 * z1;
                float a2 = plane_list[plane_index2].normal[0];
                float b2 = plane_list[plane_index2].normal[1];
                float c2 = plane_list[plane_index2].normal[2];
                float x2 = plane_list[plane_index2].p_center.x;
                float y2 = plane_list[plane_index2].p_center.y;
                float z2 = plane_list[plane_index2].p_center.z;
                float minusd2 = a2 * x2 + b2 * y2 + c2 * z2;
                float matrix[4][5];
                matrix[1][1] = a1;
                matrix[1][2] = b1;
                matrix[1][3] = c1;
                matrix[1][4] = minusd1;
                matrix[2][1] = a2;
                matrix[2][2] = b2;
                matrix[2][3] = c2;
                matrix[2][4] = minusd2;

                std::vector<Eigen::Vector3d> points;
                Eigen::Vector3d pointA;
                Eigen::Vector3d pointB;

                float theta = a1 * a2 + b1 * b2 + c1 * c2;
                //
                float point_dis_threshold = 0.00;
                if (theta > cos(DEG2RAD(150)) && theta < cos(DEG2RAD(30))) // cos(30) -- cos(150)
                {
                    ////////////////////////??????????
                    pcl::PointCloud<pcl::PointXYZ>::Ptr GlobalCloud(
                        new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::copyPointCloud(plane_list[plane_index1].cloud + plane_list[plane_index2].cloud,
                                        *GlobalCloud);
                    // pcl::PointCloud<pcl::PointXYZI> GlobalCloud;
                    // GlobalCloud = plane_list[plane_index1].cloud +
                    // plane_list[plane_index2].cloud;
                    ////////////////////////???????Χ???????????
                    pcl::PointXYZ minXYZ, maxXYZ;
                    minXYZ.x = minXYZ.y = minXYZ.z = std::numeric_limits<float>::max();
                    maxXYZ.x = maxXYZ.y = maxXYZ.z = -std::numeric_limits<float>::max();

                    for (const auto &point : GlobalCloud->points) {
                        minXYZ.x = std::min(minXYZ.x, point.x);
                        minXYZ.y = std::min(minXYZ.y, point.y);
                        minXYZ.z = std::min(minXYZ.z, point.z);

                        maxXYZ.x = std::max(maxXYZ.x, point.x);
                        maxXYZ.y = std::max(maxXYZ.y, point.y);
                        maxXYZ.z = std::max(maxXYZ.z, point.z);
                    }

                    ////////////////////////??????????????????????????
                    Eigen::Vector3d LineDirection = plane_list[plane_index1].normal.cross(
                        plane_list[plane_index2].normal);
                    // std::cout<<endl<<"plane_list[plane_index1].normal"<<endl;
                    // std::cout<<plane_list[plane_index1].normal<<endl;
                    // std::cout<<plane_list[plane_index2].normal<<endl;
                    // std::cout<<LineDirection<<endl;
                    int IntscFlag = -1; // 0yoz//1xoz//2xoy
                    if (abs(LineDirection[0]) > abs(LineDirection[1]) && abs(LineDirection[0]) > abs(LineDirection[2]))
                        IntscFlag = 0;
                    if (abs(LineDirection[1]) > abs(LineDirection[0]) && abs(LineDirection[1]) > abs(LineDirection[2]))
                        IntscFlag = 1;
                    if (abs(LineDirection[2]) > abs(LineDirection[0]) && abs(LineDirection[2]) > abs(LineDirection[1]))
                        IntscFlag = 2;
                    ////////////////////////??????????????????
                    if (IntscFlag == -1)
                        break;
                    if (IntscFlag == 0) {
                        matrix[3][1] = 1;
                        matrix[3][2] = 0;
                        matrix[3][3] = 0;
                        matrix[3][4] = minXYZ.x;
                        calc<float>(matrix, pointA);
                        points.push_back(pointA);
                        matrix[3][4] = maxXYZ.x;
                        calc<float>(matrix, pointB);
                        points.push_back(pointB);
                    }
                    if (IntscFlag == 1) {
                        matrix[3][1] = 0;
                        matrix[3][2] = 1;
                        matrix[3][3] = 0;
                        matrix[3][4] = minXYZ.y;
                        calc<float>(matrix, pointA);
                        points.push_back(pointA);
                        matrix[3][4] = maxXYZ.y;
                        calc<float>(matrix, pointB);
                        points.push_back(pointB);
                    }
                    if (IntscFlag == 2) {
                        matrix[3][1] = 0;
                        matrix[3][2] = 0;
                        matrix[3][3] = 1;
                        matrix[3][4] = minXYZ.z;
                        calc<float>(matrix, pointA);
                        points.push_back(pointA);
                        matrix[3][4] = maxXYZ.z;
                        calc<float>(matrix, pointB);
                        points.push_back(pointB);
                    }
                    // std::cout<<endl<<"AB"<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<pointA<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<pointB<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    ////////////////////////???????????????????????????????????????????????,????????
                    // pcl::PointCloud<pcl::PointXYZI> line_cloud;
                    // Eigen::Vector3d EndPointA = pointA;
                    // Eigen::Vector3d EndPointB = pointB;

                    Eigen::Vector3d EndPointA;
                    Eigen::Vector3d EndPointB;

                    // std::cout << "points size:" << points.size() << std::endl;

                    pcl::PointCloud<pcl::PointXYZI> line_cloud;
                    pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]); // let it
                                                                                // A
                    pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]); // let it
                                                                                // B
                    float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                    // ??????????
                    int K = 1;
                    int EndPointFlag = 0;
                    // ????????????????????????????????????????
                    std::vector<int> pointIdxRSearch1(K);
                    std::vector<float> pointRSquaredDistance1(K);
                    std::vector<int> pointIdxRSearch2(K);
                    std::vector<float> pointRSquaredDistance2(K);
                    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(
                        new pcl::search::KdTree<pcl::PointXYZI>());
                    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(
                        new pcl::search::KdTree<pcl::PointXYZI>());
                    kdtree1->setInputCloud(plane_list[plane_index1].cloud.makeShared());
                    kdtree2->setInputCloud(plane_list[plane_index2].cloud.makeShared());

                    for (float inc = 0; inc <= length;
                         inc += 0.01) // From A to B searching
                    {
                        pcl::PointXYZI p;
                        p.x = p1.x + (p2.x - p1.x) * inc / length;
                        p.y = p1.y + (p2.y - p1.y) * inc / length;
                        p.z = p1.z + (p2.z - p1.z) * inc / length;
                        p.intensity = 100;
                        // number of neighbors found in radius
                        if (kdtree1->radiusSearch(p, 0.03, pointIdxRSearch1,
                                                  pointRSquaredDistance1)
                                > 0
                            || kdtree2->radiusSearch(p, 0.03, pointIdxRSearch2,
                                                     pointRSquaredDistance2)
                                   > 0) {
                            // EndPointA = p.getVector3fMap();
                            EndPointA[0] = p.x;
                            EndPointA[1] = p.y;
                            EndPointA[2] = p.z;
                            EndPointFlag++;
                            break;
                        }
                    }
                    for (float inc = 0; inc <= length;
                         inc += 0.01) // From B to A searching
                    {
                        pcl::PointXYZI p;
                        p.x = p2.x + (p1.x - p2.x) * inc / length;
                        p.y = p2.y + (p1.y - p2.y) * inc / length;
                        p.z = p2.z + (p1.z - p2.z) * inc / length;
                        p.intensity = 100;
                        // number of neighbors found in radius
                        if (kdtree1->radiusSearch(p, 0.03, pointIdxRSearch1,
                                                  pointRSquaredDistance1)
                                > 0
                            || kdtree2->radiusSearch(p, 0.03, pointIdxRSearch2,
                                                     pointRSquaredDistance2)
                                   > 0) {
                            // EndPointB = p.getVector3fMap();
                            EndPointB[0] = p.x;
                            EndPointB[1] = p.y;
                            EndPointB[2] = p.z;
                            EndPointFlag++;
                            break;
                        }
                    }
                    // std::cout<<endl<<"ENDAB"<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<EndPointA<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    // std::cout<<EndPointB<<endl;
                    // std::cout<<"/////////////////////////////////////"<<endl;
                    ////////////////////////??????????????????????????Σ???????????0.01
                    if (EndPointFlag == 2) {
                        float SegmentLength = sqrt(pow(EndPointA[0] - EndPointB[0], 2) + pow(EndPointA[1] - EndPointB[1], 2) + pow(EndPointA[2] - EndPointB[2], 2));

                        if (SegmentLength > 0.4) {
                            Eigen::Vector3d NewA, NewB;
                            NewA = EndPointA + (EndPointB - EndPointA) * 0.04 / SegmentLength;
                            NewB = EndPointB + (EndPointA - EndPointB) * 0.04 / SegmentLength;
                            EndPointA = NewA;
                            EndPointB = NewB;
                        }

                        for (float inc = 0; inc <= SegmentLength; inc += 0.01) {
                            pcl::PointXYZI p;
                            p.x = EndPointA[0] + (EndPointB[0] - EndPointA[0]) * inc / SegmentLength;
                            p.y = EndPointA[1] + (EndPointB[1] - EndPointA[1]) * inc / SegmentLength;
                            p.z = EndPointA[2] + (EndPointB[2] - EndPointA[2]) * inc / SegmentLength;
                            p.intensity = 100;
                            line_cloud.push_back(p);
                        }
                    }
                    // cout<<"now we have"<<line_cloud.size()<<endl;
                    if (line_cloud.size() > 10) {
                        line_cloud_list.push_back(line_cloud);
                    }
                }
            }
        }
    }
}

void imgProcesser::extractLineFeatures(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &_sparselinecloud) {
    _sparselinecloud->clear();
    float plane_distance_threshold = 0.05; // 平面分割时点到平面的距离阈值
    int min_plane_points = 50;             // 平面最小点数
    float intensity_threshold = 20.0;      // 同一平面内点的强度差异阈值
    float min_line_length = 0.3;           // 最小线段长度
    float plane_angle_threshold = 15.0;    // 平面交线角度阈值(度)
    // 结果点云
    // pcl::PointCloud<pcl::PointXYZI>::Ptr line_cloud(new
    // pcl::PointCloud<pcl::PointXYZI>);

    // 复制输入点云以防修改原始数据
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZI>(*input_cloud));

    // std::cout<<"cloud_filtered->points.size()"<<cloud_filtered->points.size()<<std::endl;

    // 平面分割器
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(plane_distance_threshold);
    seg.setMaxIterations(100);

    // 提取器
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    // 存储平面信息的结构
    struct PlaneInfo {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        Eigen::Vector4f coefficients; // ax + by + cz + d = 0
        Eigen::Vector3f center;
        float mean_intensity;
    };

    // 存储所有检测到的平面
    std::vector<PlaneInfo> planes;

    // 持续提取平面，直到剩余点数小于阈值
    // 持续提取平面，直到剩余点数小于阈值
    // std::mutex mylock;
    // mylock.lock();
    while (cloud_filtered->points.size() > min_plane_points) {
        std::cout << "剩余点云数量: " << cloud_filtered->points.size() << std::endl;

        // 分割模型和内点索引 - 每次迭代都创建新的
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        // 执行分割
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        // 如果没有找到平面，退出循环
        if (inliers->indices.size() < min_plane_points) {
            std::cout << "未找到足够大的平面，退出循环" << std::endl;
            break;
        }

        // 提取平面点云 - 每次迭代都创建新的提取器
        pcl::ExtractIndices<pcl::PointXYZI> extract_plane;
        extract_plane.setInputCloud(cloud_filtered);
        extract_plane.setIndices(inliers);
        extract_plane.setNegative(false);

        pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cloud(
            new pcl::PointCloud<pcl::PointXYZI>);
        extract_plane.filter(*plane_cloud);

        // 计算平面点云的强度统计信息
        float intensity_sum = 0.0f;
        float intensity_max = -std::numeric_limits<float>::max();
        float intensity_min = std::numeric_limits<float>::max();

        for (const auto &point : plane_cloud->points) {
            intensity_sum += point.intensity;
            intensity_max = std::max(intensity_max, point.intensity);
            intensity_min = std::min(intensity_min, point.intensity);
        }

        float mean_intensity = intensity_sum / plane_cloud->points.size();

        // 创建一个全新的提取器用于移除处理过的点
        pcl::ExtractIndices<pcl::PointXYZI> extract_remainder;
        extract_remainder.setInputCloud(cloud_filtered);
        extract_remainder.setIndices(inliers);
        extract_remainder.setNegative(true);

        pcl::PointCloud<pcl::PointXYZI>::Ptr remainder_cloud(
            new pcl::PointCloud<pcl::PointXYZI>);
        extract_remainder.filter(*remainder_cloud);

        // 检查平面内点的强度差异
        if ((intensity_max - intensity_min) > intensity_threshold) {
            // std::cout << "平面强度差异过大，继续下一个迭代" << std::endl;
            // 不保存这个平面，但更新剩余点云
            cloud_filtered = remainder_cloud;
            continue;
        }

        // 计算平面中心
        Eigen::Vector3f center(0, 0, 0);
        for (const auto &point : plane_cloud->points) {
            center[0] += point.x;
            center[1] += point.y;
            center[2] += point.z;
        }
        center /= static_cast<float>(plane_cloud->points.size());

        // 保存有效平面信息
        PlaneInfo plane_info;
        plane_info.cloud = plane_cloud;
        plane_info.coefficients =
            Eigen::Vector4f(coefficients->values[0], coefficients->values[1],
                            coefficients->values[2], coefficients->values[3]);
        plane_info.center = center;
        plane_info.mean_intensity = mean_intensity;

        planes.push_back(plane_info);

        // 更新剩余点云
        cloud_filtered = remainder_cloud;

        std::cout << "已保存平面，当前平面数: " << planes.size() << std::endl;
    }
    // mylock.unlock();
    // 如果检测到的平面少于2个，无法形成交线
    if (planes.size() < 2) {
        return;
    }

    // 计算平面交线
    for (size_t i = 0; i < planes.size(); ++i) {
        for (size_t j = i + 1; j < planes.size(); ++j) {
            // 获取平面法向量
            Eigen::Vector3f normal_i(planes[i].coefficients[0],
                                     planes[i].coefficients[1],
                                     planes[i].coefficients[2]);
            Eigen::Vector3f normal_j(planes[j].coefficients[0],
                                     planes[j].coefficients[1],
                                     planes[j].coefficients[2]);

            // 归一化法向量
            normal_i.normalize();
            normal_j.normalize();

            // 计算平面夹角
            float cos_angle = normal_i.dot(normal_j);
            float angle = std::acos(std::abs(cos_angle));

            // 如果平面几乎平行，跳过
            float angle_threshold_rad = plane_angle_threshold * M_PI / 180.0f;
            if (angle < angle_threshold_rad || (M_PI - angle) < angle_threshold_rad) {
                continue;
            }

            // 计算交线方向向量 (两平面法向量的叉积)
            Eigen::Vector3f line_direction = normal_i.cross(normal_j);
            line_direction.normalize();

            // 计算交线上的一点
            // 解方程组 ax + by + cz + d = 0 (两个平面方程)
            Eigen::Matrix3f A;
            Eigen::Vector3f b;

            // 设置第一个平面方程
            A.row(0) = normal_i;
            b(0) = -planes[i].coefficients[3];

            // 设置第二个平面方程
            A.row(1) = normal_j;
            b(1) = -planes[j].coefficients[3];

            // 设置约束方程使解唯一 (点到两平面中心的中点最近)
            Eigen::Vector3f midpoint = (planes[i].center + planes[j].center) / 2.0f;
            A.row(2) = line_direction;
            b(2) = line_direction.dot(midpoint);

            // 求解方程获取交线上的点
            Eigen::Vector3f line_point;
            bool solved = false;

            // 尝试求解线性方程组
            try {
                line_point = A.colPivHouseholderQr().solve(b);
                solved = true;
            } catch (...) {
                solved = false;
            }

            if (!solved)
                continue;

            // 计算线段在两个平面上的投影边界
            std::vector<Eigen::Vector3f> boundary_points_i, boundary_points_j;

            for (const auto &point : planes[i].cloud->points) {
                Eigen::Vector3f p(point.x, point.y, point.z);
                // 计算点到直线的投影
                float t = line_direction.dot(p - line_point);
                boundary_points_i.push_back(line_point + t * line_direction);
            }

            for (const auto &point : planes[j].cloud->points) {
                Eigen::Vector3f p(point.x, point.y, point.z);
                float t = line_direction.dot(p - line_point);
                boundary_points_j.push_back(line_point + t * line_direction);
            }

            // 查找两个平面上投影的最大范围
            float min_t = std::numeric_limits<float>::max();
            float max_t = -std::numeric_limits<float>::max();

            for (const auto &p : boundary_points_i) {
                float t = line_direction.dot(p - line_point);
                min_t = std::min(min_t, t);
                max_t = std::max(max_t, t);
            }

            for (const auto &p : boundary_points_j) {
                float t = line_direction.dot(p - line_point);
                min_t = std::min(min_t, t);
                max_t = std::max(max_t, t);
            }

            // 计算线段长度
            float line_length = max_t - min_t;

            // 如果线段太短，跳过
            if (line_length < min_line_length) {
                continue;
            }

            // 计算线段起点和终点
            Eigen::Vector3f start_point = line_point + min_t * line_direction;
            Eigen::Vector3f end_point = line_point + max_t * line_direction;

            // 计算线段的平均强度 (使用两个平面强度的平均值)
            float line_intensity =
                (planes[i].mean_intensity + planes[j].mean_intensity) / 2.0f;

            // 在线段上采样点
            int num_points =
                std::max(10, static_cast<int>(line_length / 0.05)); // 粗略每5cm一个点，至少10个点

            for (int k = 0; k < num_points; ++k) {
                float t = min_t + k * (max_t - min_t) / (num_points - 1);
                Eigen::Vector3f p = line_point + t * line_direction;

                pcl::PointXYZI point;
                point.x = p[0];
                point.y = p[1];
                point.z = p[2];
                point.intensity = line_intensity;

                _sparselinecloud->points.push_back(point);
            }
        }
    }

    // 设置点云属性
    _sparselinecloud->width = _sparselinecloud->points.size();
    _sparselinecloud->height = 1;
    _sparselinecloud->is_dense = true;

    return;
}
void imgProcesser::LiDAREdgeExtraction(
    std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map_edge,
    const float ransac_dis_thre, const int plane_size_threshold,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d) {
    lidar_line_cloud_3d->clear();
    // lidar_line_cloud_3d = pcl::PointCloud<pcl::PointXYZI>::Ptr(new
    // pcl::PointCloud<pcl::PointXYZI>); std::cout<<voxel_map.size()<<std::endl;
    for (auto iter = voxel_map_edge.begin(); iter != voxel_map_edge.end();
         iter++ ++++) {
        if (iter->second->cloud->size() > 30) {
            plane_list.clear();
            // ????????????????

            pcl::copyPointCloud(*iter->second->cloud, *cloud_filter);
            //??????????????????????????

            //????????????
            pcl::SACSegmentation<pcl::PointXYZI> seg;
            // Optional,???????????????????????????????
            seg.setOptimizeCoefficients(true);
            // Mandatory-????????????
            seg.setModelType(pcl::SACMODEL_PLANE);
            //????????????????
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(ransac_dis_thre); // 0.01
            seg.setMaxIterations(50);
            // pcl::PointCloud<pcl::PointXYZRGB> color_planner_cloud;
            color_planner_cloud->clear();
            mergedPts->clear();
            int plane_index = 0;
            // std::cout<<"cloud_filter size"<<cloud_filter->size()<<std::endl;
            pcl::ExtractIndices<pcl::PointXYZI> extract;
            while (cloud_filter->size() > 10) {
                // std::cout<<"finish"<<std::endl;
                //???????
                seg.setInputCloud(cloud_filter);
                //??????
                seg.segment(*inliers, *coefficients);

                if (inliers->indices.size() == 0) {
                    break;
                }
                extract.setIndices(inliers);
                extract.setInputCloud(cloud_filter);
                extract.filter(*planner_cloud);

                if (planner_cloud->size() > plane_size_threshold) {
                    pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
                    std::vector<unsigned int> colors;
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    pcl::PointXYZRGB p_center;
                    p_center.x = 0;
                    p_center.y = 0;
                    p_center.z = 0;
                    for (size_t i = 0; i < planner_cloud->points.size(); i++) {
                        pcl::PointXYZRGB p;
                        p.x = planner_cloud->points[i].x;
                        p.y = planner_cloud->points[i].y;
                        p.z = planner_cloud->points[i].z;
                        p_center.x += p.x;
                        p_center.y += p.y;
                        p_center.z += p.z;
                        p.r = colors[0];
                        p.g = colors[1];
                        p.b = colors[2];
                        color_cloud.push_back(p);
                        color_planner_cloud->push_back(p);
                    }
                    p_center.x = p_center.x / planner_cloud->size();
                    p_center.y = p_center.y / planner_cloud->size();
                    p_center.z = p_center.z / planner_cloud->size();
                    Plane single_plane;
                    single_plane.cloud = *planner_cloud;
                    single_plane.p_center = p_center;
                    single_plane.normal << coefficients->values[0],
                        coefficients->values[1], coefficients->values[2];
                    single_plane.index = plane_index;
                    plane_list.push_back(single_plane);
                    plane_index++;
                }
                // std::cout<<"finish"<<std::endl;
                extract.setNegative(true);

                extract.filter(next_cloud);
                *cloud_filter = next_cloud; // 直接替换指针
                // planner_cloud->clear();
                next_cloud.clear();
                // std::cout<<"cloud_filter size"<<cloud_filter->size()<<std::endl;
                // std::cout<<"finish"<<std::endl;
                // std::cout << "提取到的平面数量: " << plane_list.size() << std::endl
            }

            // //std::cout<<"finish"<<std::endl;
            mergePlanes(plane_list);

            savePlanes(plane_list, mergedPts);

            // pcl::io::savePCDFileASCII("/home/zzy/SLAM/my_slam_work/src/Rot_intensity_augm_LIVO/image/check/plane.pcd",
            // *color_planner_cloud); pcl::io::savePCDFileASCII("check/" +
            // std::to_string(dataProcessingNum) + "_LaserMergedPlanes.pcd",
            // *mergedPts);

            line_cloud_list.clear();
            // calcLine(plane_list, voxel_size_, iter->second->voxel_origin,
            // line_cloud_list);
            calcLineLin(plane_list, line_cloud_list);

            // ouster 5,normal 3
            // std::cout<<line_cloud_list.size()<<std::endl;
            // std::cout<<plane_list.size()<<std::endl;
            if (line_cloud_list.size() > 0 && line_cloud_list.size() <= 10) {
                for (size_t cloud_index = 0; cloud_index < line_cloud_list.size();
                     cloud_index++) {
                    for (size_t i = 0; i < line_cloud_list[cloud_index].size(); i++) {
                        pcl::PointXYZI p = line_cloud_list[cloud_index].points[i];
                        lidar_line_cloud_3d->points.push_back(p);
                        // plane_line_number_.push_back(line_number_);
                    }
                    // line_number_++;
                }
            }
        }
    }

    lidar_line_cloud_3d->height = 1;
    lidar_line_cloud_3d->width = lidar_line_cloud_3d->size();
}
cv::Mat imgProcesser::densifyLidarPointCloud(const cv::Mat &inputImage,
                                             const cv::Mat &templateImage) {
    try {
        // 检查输入
        if (inputImage.empty() || templateImage.empty()) {
            return inputImage.clone();
        }

        // 确保输入为单通道图像
        CV_Assert(inputImage.type() == CV_8UC1 && templateImage.type() == CV_8UC1);

        // 保留原始像素信息的副本
        cv::Mat originalValues = inputImage.clone();

        // 创建二值化版本用于结构分析
        cv::Mat binaryInput, binaryTemplate;
        cv::threshold(inputImage, binaryInput, 1, 255, cv::THRESH_BINARY);
        cv::threshold(templateImage, binaryTemplate, 1, 255, cv::THRESH_BINARY);

        // 计算参考密度图
        cv::Mat densityMap;
        cv::GaussianBlur(binaryTemplate, densityMap, cv::Size(11, 11), 0);

        // 扩展输入图像结构
        cv::Mat dilated;
        cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::dilate(binaryInput, dilated, kernel, cv::Point(-1, -1), 2);

        // 创建距离变换，用于查找最近点
        cv::Mat distMap;
        cv::distanceTransform(~binaryInput, distMap, cv::DIST_L2, 3);

        // 创建结果图像，初始化为dilated的结构，保留原始点的像素值
        cv::Mat result = cv::Mat::zeros(inputImage.size(), CV_8UC1);

        // 首先保留原始点
        for (int y = 0; y < inputImage.rows; y++) {
            for (int x = 0; x < inputImage.cols; x++) {
                if (inputImage.at<uchar>(y, x) > 0) {
                    result.at<uchar>(y, x) = originalValues.at<uchar>(y, x);
                }
            }
        }

        // 收集所有原始非零点的位置和值
        std::vector<cv::Point> nonZeroPoints;
        std::vector<uchar> nonZeroValues;

        for (int y = 0; y < inputImage.rows; y++) {
            for (int x = 0; x < inputImage.cols; x++) {
                if (inputImage.at<uchar>(y, x) > 0) {
                    nonZeroPoints.push_back(cv::Point(x, y));
                    nonZeroValues.push_back(inputImage.at<uchar>(y, x));
                }
            }
        }

        // 如果没有点，直接返回
        if (nonZeroPoints.empty()) {
            return result;
        }

        // 对dilated中的新点进行插值填充
        for (int y = 0; y < result.rows; y++) {
            for (int x = 0; x < result.cols; x++) {
                // 如果这个点在dilated中是非零的，但在原始图像中是零，需要插值
                if (dilated.at<uchar>(y, x) > 0 && inputImage.at<uchar>(y, x) == 0) {
                    // 检查密度图的值，确保只在合理的区域插值
                    if (densityMap.at<uchar>(y, x) > 30) { // 提高阈值减少杂点
                        // 寻找k个最近的原始点进行插值
                        const int k = 5; // 使用5个最近点
                        std::vector<std::pair<float, int>> distances;

                        // 计算到所有原始点的距离
                        for (int i = 0; i < nonZeroPoints.size(); i++) {
                            float dx = nonZeroPoints[i].x - x;
                            float dy = nonZeroPoints[i].y - y;
                            float distance = std::sqrt(dx * dx + dy * dy);
                            distances.push_back(std::make_pair(distance, i));
                        }

                        // 排序找到k个最近点
                        std::partial_sort(distances.begin(),
                                          distances.begin() + std::min(k, (int)distances.size()),
                                          distances.end());

                        // 基于距离的加权平均插值
                        float sumWeights = 0.0f;
                        float sumValues = 0.0f;

                        for (int i = 0; i < std::min(k, (int)distances.size()); i++) {
                            float dist = distances[i].first;
                            int idx = distances[i].second;

                            // 避免除零，同时使权重随距离增加而迅速减小
                            float weight = 1.0f / (dist + 0.1f);
                            sumWeights += weight;
                            sumValues += weight * nonZeroValues[idx];
                        }

                        // 如果找到了有效的权重，设置插值结果
                        if (sumWeights > 0) {
                            uchar interpolatedValue =
                                static_cast<uchar>(sumValues / sumWeights);
                            result.at<uchar>(y, x) = interpolatedValue;
                        }
                    }
                }
            }
        }

        return result;
    } catch (const std::exception &e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return inputImage.clone();
    }
}

cv::Mat imgProcesser::interpolateLidarImage(const cv::Mat &sparse_image) {
    int max_distance = 5;
    int method = 0;
    // CV_Assert(!sparse_image.empty() && (sparse_image.type() == CV_32FC1 ||
    // sparse_image.type() == CV_64FC1));

    cv::Mat dense_image = sparse_image.clone();

    // 找出所有有效点（非零点）
    std::vector<cv::Point> valid_points;
    std::vector<float> valid_values;

    for (int y = 0; y < sparse_image.rows; ++y) {
        for (int x = 0; x < sparse_image.cols; ++x) {
            float value = sparse_image.at<float>(y, x);
            if (std::abs(value) > 1e-6) { // 非零点
                valid_points.push_back(cv::Point(x, y));
                valid_values.push_back(value);
            }
        }
    }

    // 对每个空点进行插值
    for (int y = 0; y < dense_image.rows; ++y) {
        for (int x = 0; x < dense_image.cols; ++x) {
            // 如果是有效点则跳过
            if (std::abs(sparse_image.at<float>(y, x)) > 1e-6) {
                continue;
            }

            // 当前空点
            cv::Point current_point(x, y);

            if (method == 0) {
                // 最近邻插值
                float min_distance = std::numeric_limits<float>::max();
                float interpolated_value = 0.0f;
                bool found = false;

                for (size_t i = 0; i < valid_points.size(); ++i) {
                    float distance = cv::norm(current_point - valid_points[i]);
                    if (distance < min_distance && distance <= max_distance) {
                        min_distance = distance;
                        interpolated_value = valid_values[i];
                        found = true;
                    }
                }

                if (found) {
                    dense_image.at<float>(y, x) = interpolated_value;
                }
            } else if (method == 1) {
                // 加权双线性插值
                float sum_weights = 0.0f;
                float sum_weighted_values = 0.0f;
                bool found = false;

                for (size_t i = 0; i < valid_points.size(); ++i) {
                    float distance = cv::norm(current_point - valid_points[i]);
                    if (distance <= max_distance) {
                        // 使用距离的倒数作为权重
                        float weight = 1.0f / (distance + 1e-6);
                        sum_weights += weight;
                        sum_weighted_values += weight * valid_values[i];
                        found = true;
                    }
                }

                if (found) {
                    dense_image.at<float>(y, x) = sum_weighted_values / sum_weights;
                }
            } else if (method == 2) {
                // 径向基函数插值（RBF）
                float sum_weights = 0.0f;
                float sum_weighted_values = 0.0f;
                bool found = false;

                for (size_t i = 0; i < valid_points.size(); ++i) {
                    float distance = cv::norm(current_point - valid_points[i]);
                    if (distance <= max_distance) {
                        // 使用高斯径向基函数
                        float sigma = max_distance / 3.0f; // 控制径向基函数的宽度
                        float weight =
                            std::exp(-(distance * distance) / (2 * sigma * sigma));
                        sum_weights += weight;
                        sum_weighted_values += weight * valid_values[i];
                        found = true;
                    }
                }

                if (found) {
                    dense_image.at<float>(y, x) = sum_weighted_values / sum_weights;
                }
            }
        }
    }

    return dense_image;
}

void imgProcesser::removeLines(cv::Mat &img) {
    // Perform highpass vertically

    // cv::Mat im_hpf;
    // cv::filter2D(img, im_hpf, CV_32F , high_pass_fir_);
    // // Perform lowpass horizontally
    // cv::Mat im_lpf;
    // cv::filter2D(im_hpf, im_lpf, CV_32F , low_pass_fir_.t());
    // // Remove filtered signal from original image
    // img -= im_lpf;
    // img.setTo(0, img < 0);
}

void imgProcesser::filterBrightness(cv::Mat &img) {
    // Create brightness map
    cv::Mat brightness;
    cv::Size window_size_ = cv::Size(41, 7);
    cv::blur(img, brightness, window_size_);
    brightness += 1;
    // Normalize and scale image
    cv::Mat normalized_img = (140. * img / brightness);
    img = normalized_img;
}

Eigen::Vector2d imgProcesser::projectTosingle(pcl::PointXYZI p) {
    const int IMG_HEIGHT = 800; // 图像高度
    const int IMG_WIDTH = 800;  // 图像宽度

    // 设置相机内参
    const float fx = 400.0f;            // 焦距x
    const float fy = 400.0f;            // 焦距y
    const float cx = IMG_WIDTH / 2.0f;  // 光心x坐标
    const float cy = IMG_HEIGHT / 2.0f; // 光心y坐标
    if (p.x > 0) {
        return Eigen::Vector2d(-1, -1);
    }

    // 计算投影点坐标
    float x = -p.x; // 转换为相机坐标系（假设-x是前方）
    float y = p.y;
    float z = p.z;

    // 针孔投影
    float depth = x; // 深度值为x轴距离

    // 如果深度为0或负数，跳过
    if (depth <= 0) {
        return Eigen::Vector2d(-1, -1);
    }

    // 计算像素坐标
    float u = (fx * y / x) + cx;
    float v = (fy * -z / x) + cy;

    // 检查像素坐标是否在图像范围内
    if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
        // 强制转换为整数坐标
        // int ui = static_cast<int>(u);
        // int vi = static_cast<int>(v);
        return Eigen::Vector2d(u, v);
    } else {
        return Eigen::Vector2d(-1, -1);
    }
}

cv::Mat
imgProcesser::projectDepth(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
    // 创建空深度图
    cv::Mat depth_map = cv::Mat::zeros(800, 800, CV_32F);
    cv::Mat point_count = cv::Mat::zeros(800, 800, CV_32F);

    // 将3D点投影到2D图像平面
    for (const auto &point : cloud->points) {
        // 跳过摄像机后方的点（z为负）
        if (point.z <= 0 || point.z > 80)
            continue;

        // 投影点到图像平面
        int u = static_cast<int>((400 * point.x) / point.z + 400 + 0.5f);
        int v = static_cast<int>((400 * point.y) / point.z + 400 + 0.5f);

        // 检查投影点是否在图像范围内
        if (u >= 0 && u < 800 && v >= 0 && v < 800) {
            // 累积相同像素的深度值（用于后续平均）
            depth_map.at<float>(v, u) = point.z;
            // point_count.at<float>(v, u) += 1.0f;
        }
    }

    // // 对有多个点的像素取平均深度值
    // for (int y = 0; y < 800 ; y++) {
    //     for (int x = 0; x < 800 ; x++) {
    //         if (point_count.at<float>(y, x) > 0) {
    //             depth_map.at<float>(y, x) /= point_count.at<float>(y, x);
    //         }
    //     }
    // }

    // 应用深度补全算法
    cv::Mat completed_depth = depth_completion(depth_map, 800, 800);

    return completed_depth;
}

cv::Mat
imgProcesser::projectPinhole(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                             bool isinter) {
    // 设置图像尺寸
    const int IMG_HEIGHT = 800; // 图像高度
    const int IMG_WIDTH = 800;  // 图像宽度

    // 设置相机内参
    const float fx = 400.0f;            // 焦距x
    const float fy = 400.0f;            // 焦距y
    const float cx = IMG_WIDTH / 2.0f;  // 光心x坐标
    const float cy = IMG_HEIGHT / 2.0f; // 光心y坐标

    // 创建强度图，初始化为0
    cv::Mat intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);

    // 如果点云为空，返回空图像
    if (cloud->size() == 0) {
        intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        return intensityImage;
    }

    // 创建深度图，用于处理遮挡
    cv::Mat depthMap = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);
    depthMap.setTo(std::numeric_limits<float>::max()); // 初始化为最大值

    // 投影点云到图像平面
    for (const auto &p : cloud->points) {
        // 只处理相机前方的点（假设相机朝向-x方向）
        if (p.x > 0) {
            continue;
        }

        // 计算投影点坐标
        float x = -p.x; // 转换为相机坐标系（假设-x是前方）
        float y = p.y;
        float z = p.z;

        // 针孔投影
        float depth = x; // 深度值为x轴距离

        // 如果深度为0或负数，跳过
        if (depth <= 0) {
            continue;
        }

        // 计算像素坐标
        float u = (fx * y / x) + cx;
        float v = (fy * -z / x) + cy;

        // 检查像素坐标是否在图像范围内
        if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
            // 强制转换为整数坐标
            int ui = static_cast<int>(u);
            int vi = static_cast<int>(v);

            // 处理遮挡（近处的点覆盖远处的点）
            if (depth < depthMap.at<float>(vi, ui)) {
                depthMap.at<float>(vi, ui) = depth;
                intensityImage.at<float>(vi, ui) = p.intensity;
            }
        }
    }

    // 归一化到 [0, 255]
    double minVal, maxVal;
    cv::minMaxLoc(intensityImage, &minVal, &maxVal);

    // 避免除以零
    if (maxVal > minVal) {
        intensityImage.convertTo(intensityImage, CV_8U, 255.0 / (maxVal - minVal),
                                 -minVal * 255.0 / (maxVal - minVal));
    } else {
        intensityImage.convertTo(intensityImage, CV_8U);
    }

    // 直方图均衡化
    cv::equalizeHist(intensityImage, intensityImage);

    // 伽马校正增强对比度
    cv::Mat gammaImage;
    intensityImage.convertTo(gammaImage, CV_32F, 1.0 / 255.0);
    cv::pow(gammaImage, 0.5, gammaImage); // γ = 0.5
    gammaImage.convertTo(intensityImage, CV_8U, 255);

    return intensityImage;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat>
imgProcesser::projectPinholeall(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                const Eigen::Matrix3d &Rleft,
                                const Eigen::Matrix3d &Rright, bool isinter) {
    // 设置图像尺寸和相机参数 - 使用constexpr提高编译时优化
    constexpr int IMG_HEIGHT = 800;
    constexpr int IMG_WIDTH = 800;
    constexpr float fx = 400.0f;
    constexpr float fy = 400.0f;
    constexpr float cx = IMG_WIDTH / 2.0f;
    constexpr float cy = IMG_HEIGHT / 2.0f;

    // 提前获取点云大小以避免多次调用size()函数
    const size_t cloud_size = cloud->size();

    // 创建强度图和深度图
    cv::Mat intensityAccum = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);
    cv::Mat intensityAccum_left = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);
    cv::Mat intensityAccum_right = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);

    cv::Mat depthMap = cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_32F,
                               cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat depthMap_left =
        cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_32F,
                cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat depthMap_right =
        cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_32F,
                cv::Scalar(std::numeric_limits<float>::max()));

    // 如果点云为空，直接返回空图像
    if (cloud_size == 0) {
        cv::Mat empty = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        return {empty, empty, empty};
    }

    // 预计算旋转矩阵，避免在循环中重复计算
    const float r00 = Rleft(0, 0), r01 = Rleft(0, 1), r02 = Rleft(0, 2);
    const float r10 = Rleft(1, 0), r11 = Rleft(1, 1), r12 = Rleft(1, 2);
    const float r20 = Rleft(2, 0), r21 = Rleft(2, 1), r22 = Rleft(2, 2);

    const float rr00 = Rright(0, 0), rr01 = Rright(0, 1), rr02 = Rright(0, 2);
    const float rr10 = Rright(1, 0), rr11 = Rright(1, 1), rr12 = Rright(1, 2);
    const float rr20 = Rright(2, 0), rr21 = Rright(2, 1), rr22 = Rright(2, 2);

    // 使用直接访问以提高速度
    float *depthMapPtr = (float *)depthMap.data;
    float *depthMapLeftPtr = (float *)depthMap_left.data;
    float *depthMapRightPtr = (float *)depthMap_right.data;

    float *intensityPtr = (float *)intensityAccum.data;
    float *intensityLeftPtr = (float *)intensityAccum_left.data;
    float *intensityRightPtr = (float *)intensityAccum_right.data;

    const int stride = IMG_WIDTH; // 每行的元素数

    // 投影点云到图像平面 - 使用数组索引代替at<>操作以提高速度
    for (size_t i = 0; i < cloud_size; ++i) {
        const auto &p = cloud->points[i];
        const float px = p.x, py = p.y, pz = p.z, intensity = p.intensity;

        // 原始视角处理
        {
            // 计算投影点坐标
            const float x = -px; // 相机朝向-x方向

            // 如果深度为0或负数，跳过
            if (x <= 0)
                goto left_view;

            // 计算像素坐标 - 使用乘法代替除法
            const float inv_x = 1.0f / x;
            const float u = (fx * py * inv_x) + cx;
            const float v = (fy * -pz * inv_x) + cy;

            // 检查像素坐标是否在图像范围内
            if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
                const int ui = static_cast<int>(u);
                const int vi = static_cast<int>(v);
                const int idx = vi * stride + ui;

                // 处理遮挡（近处的点覆盖远处的点）
                if (x < depthMapPtr[idx]) {
                    depthMapPtr[idx] = x;
                    intensityPtr[idx] = intensity;
                }
            }
        }

    left_view:
        // 左视角处理
        {
            // 内联矩阵乘法，避免创建Eigen对象
            const float rotx = r00 * px + r01 * py + r02 * pz;
            const float roty = r10 * px + r11 * py + r12 * pz;
            const float rotz = r20 * px + r21 * py + r22 * pz;

            const float x = -rotx;

            // 如果深度为0或负数，跳过
            if (x <= 0)
                goto right_view;

            // 计算像素坐标
            const float inv_x = 1.0f / x;
            const float u = (fx * roty * inv_x) + cx;
            const float v = (fy * -rotz * inv_x) + cy;

            // 检查像素坐标是否在图像范围内
            if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
                const int ui = static_cast<int>(u);
                const int vi = static_cast<int>(v);
                const int idx = vi * stride + ui;

                // 处理遮挡
                if (x < depthMapLeftPtr[idx]) {
                    depthMapLeftPtr[idx] = x;
                    intensityLeftPtr[idx] = intensity;
                }
            }
        }

    right_view:
        // 右视角处理
        {
            // 内联矩阵乘法
            const float rotx = rr00 * px + rr01 * py + rr02 * pz;
            const float roty = rr10 * px + rr11 * py + rr12 * pz;
            const float rotz = rr20 * px + rr21 * py + rr22 * pz;

            const float x = -rotx;

            // 如果深度为0或负数，跳过
            if (x <= 0)
                continue;

            // 计算像素坐标
            const float inv_x = 1.0f / x;
            const float u = (fx * roty * inv_x) + cx;
            const float v = (fy * -rotz * inv_x) + cy;

            // 检查像素坐标是否在图像范围内
            if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
                const int ui = static_cast<int>(u);
                const int vi = static_cast<int>(v);
                const int idx = vi * stride + ui;

                // 处理遮挡
                if (x < depthMapRightPtr[idx]) {
                    depthMapRightPtr[idx] = x;
                    intensityRightPtr[idx] = intensity;
                }
            }
        }
    }

    // 创建最终输出图像
    cv::Mat intensityImage, intensityImage_left, intensityImage_right;

    // 处理原始视角图像
    {
        // 找到强度范围
        double minVal, maxVal;
        cv::minMaxLoc(intensityAccum, &minVal, &maxVal);

        // 归一化到[0, 255]
        if (maxVal > minVal) {
            intensityAccum.convertTo(intensityImage, CV_8U, 255.0 / (maxVal - minVal),
                                     -minVal * 255.0 / (maxVal - minVal));
        } else {
            intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        }
        // intensityImage =ultraFastGrayToRGB(intensityImage);
        cv::imwrite(
            "/home/zzy/SLAM/mri-mms/src/Rot_intensity_augm_LIVO/image/original.png",
            intensityImage);
        // 直方图均衡化
        cv::equalizeHist(intensityImage, intensityImage);
        cv::imwrite(
            "/home/zzy/SLAM/mri-mms/src/Rot_intensity_augm_LIVO/image/equal.png",
            intensityImage);

        // // 优化伽马校正，使用查找表
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar *p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, 0.5) * 255.0);
        }
        cv::LUT(intensityImage, lookUpTable, intensityImage);
        cv::imwrite(
            "/home/zzy/SLAM/mri-mms/src/Rot_intensity_augm_LIVO/image/gamma.png",
            intensityImage);
    }

    // 处理左视角图像
    {
        double minVal, maxVal;
        cv::minMaxLoc(intensityAccum_left, &minVal, &maxVal);

        if (maxVal > minVal) {
            intensityAccum_left.convertTo(intensityImage_left, CV_8U,
                                          255.0 / (maxVal - minVal),
                                          -minVal * 255.0 / (maxVal - minVal));
        } else {
            intensityImage_left = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        }

        // intensityImage_left =ultraFastGrayToRGB(intensityImage_left);

        cv::equalizeHist(intensityImage_left, intensityImage_left);

        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar *p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, 0.5) * 255.0);
        }
        cv::LUT(intensityImage_left, lookUpTable, intensityImage_left);
    }

    // 处理右视角图像
    {
        double minVal, maxVal;
        cv::minMaxLoc(intensityAccum_right, &minVal, &maxVal);

        if (maxVal > minVal) {
            intensityAccum_right.convertTo(intensityImage_right, CV_8U,
                                           255.0 / (maxVal - minVal),
                                           -minVal * 255.0 / (maxVal - minVal));
        } else {
            intensityImage_right = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        }
        // intensityImage_right =ultraFastGrayToRGB(intensityImage_right);

        cv::equalizeHist(intensityImage_right, intensityImage_right);

        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar *p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, 0.5) * 255.0);
        }
        cv::LUT(intensityImage_right, lookUpTable, intensityImage_right);
    }

    return {intensityImage, intensityImage_left, intensityImage_right};
}
cv::Mat imgProcesser::ultraFastGrayToRGB(const cv::Mat &grayImage) {
    // 确保输入是灰度图
    CV_Assert(grayImage.channels() == 1);

    // 预先计算查找表 - 使用静态以避免重复计算
    static cv::Mat lookupTable(1, 256, CV_8UC3);
    static bool tableInitialized = false;

    if (!tableInitialized) {
        for (int i = 0; i < 256; i++) {
            // 改进的JET颜色映射算法，增强对比度
            float r = 0, g = 0, b = 0;
            float normalized = i / 255.0f;

            // 使用正弦函数产生更强烈的色彩变化
            if (i < 64) {
                // 蓝色到青色区域
                b = 255;
                g = i * 4;
                // 增加微小的红色分量，增强对比度
                r = i;
            } else if (i < 128) {
                // 青色到绿色区域
                b = 255 - (i - 64) * 4;
                g = 255;
                // 降低蓝色，增强与邻近颜色的对比
                b = std::max(0.0f, b - 20.0f);
            } else if (i < 192) {
                // 绿色到黄色区域
                g = 255;
                r = (i - 128) * 4;
                // 增加微小的蓝色分量，增强对比度
                b = (i - 128);
            } else {
                // 黄色到红色区域
                g = 255 - (i - 192) * 4;
                r = 255;
                // 降低绿色，增强与邻近颜色的对比
                g = std::max(0.0f, g - 20.0f);
            }

            // 增加饱和度 - 让颜色更鲜艳
            float maxChannel = std::max(r, std::max(g, b));
            if (maxChannel > 0) {
                // 计算当前饱和度
                float minChannel = std::min(r, std::min(g, b));
                float saturation = (maxChannel - minChannel) / maxChannel;

                // 增加饱和度（降低最小通道的值）
                float saturationFactor = 1.3f; // 增强饱和度因子
                if (saturation < 1.0f && saturation > 0.0f) {
                    float newSaturation = std::min(1.0f, saturation * saturationFactor);
                    float saturationRatio = newSaturation / saturation;

                    // 重新计算RGB值以增加饱和度
                    float adjustment =
                        (maxChannel - minChannel) * (saturationRatio - 1.0f);

                    if (r < maxChannel)
                        r = std::max(0.0f, r - adjustment);
                    if (g < maxChannel)
                        g = std::max(0.0f, g - adjustment);
                    if (b < maxChannel)
                        b = std::max(0.0f, b - adjustment);
                }
            }

            // 增加对比度 - 拉伸色彩差异
            float contrastFactor = 1.2f; // 对比度增强因子
            r = std::min(255.0f, 128.0f + (r - 128.0f) * contrastFactor);
            g = std::min(255.0f, 128.0f + (g - 128.0f) * contrastFactor);
            b = std::min(255.0f, 128.0f + (b - 128.0f) * contrastFactor);

            // 设置BGR顺序的值
            lookupTable.at<cv::Vec3b>(0, i) =
                cv::Vec3b(cv::saturate_cast<uchar>(b), cv::saturate_cast<uchar>(g),
                          cv::saturate_cast<uchar>(r));
        }
        tableInitialized = true;
    }

    // 为输出图像分配内存
    cv::Mat colorImage(grayImage.size(), CV_8UC3);

    // 如果内存连续，使用更快的算法
    if (grayImage.isContinuous() && colorImage.isContinuous()) {
        const int totalPixels = grayImage.total();
        const uchar *grayData = grayImage.ptr<uchar>(0);
        cv::Vec3b *colorData = colorImage.ptr<cv::Vec3b>(0);

        // 一次性处理所有像素
        for (int i = 0; i < totalPixels; i++) {
            colorData[i] = lookupTable.at<cv::Vec3b>(0, grayData[i]);
        }
    } else {
        // 如果不连续，逐行处理
        for (int y = 0; y < grayImage.rows; y++) {
            const uchar *grayRow = grayImage.ptr<uchar>(y);
            cv::Vec3b *colorRow = colorImage.ptr<cv::Vec3b>(y);

            for (int x = 0; x < grayImage.cols; x++) {
                colorRow[x] = lookupTable.at<cv::Vec3b>(0, grayRow[x]);
            }
        }
    }

    return colorImage;
}

cv::Mat imgProcesser::enhanceSubtleChangesFast(const cv::Mat &grayImage) {
    // 确保输入是灰度图
    CV_Assert(grayImage.channels() == 1);

    // 步骤1: 局部归一化 - 比CLAHE更快但仍能增强局部对比度
    cv::Mat enhancedGray;

    // 应用小半径高斯模糊获取局部平均值
    cv::Mat localMean;
    cv::GaussianBlur(grayImage, localMean, cv::Size(7, 7), 0);

    // 计算本地方差
    cv::Mat localVariance;
    cv::GaussianBlur(grayImage.mul(grayImage), localVariance, cv::Size(7, 7), 0);
    localVariance -= localMean.mul(localMean);

    // 计算标准差
    cv::Mat localStdDev;
    cv::sqrt(cv::max(localVariance, 0), localStdDev);

    // 归一化图像
    cv::Mat normalizedLocal = (grayImage - localMean) / (localStdDev + 10.0);

    // 将归一化的值映射回[0,255]范围
    normalizedLocal = normalizedLocal * 40.0 + 128.0;
    normalizedLocal.convertTo(enhancedGray, CV_8U);

    // 步骤2: 使用伪彩色映射使微小变化可见
    // 选择Turbo色彩映射，它特别适合显示微小变化
    cv::Mat colorImage;
    cv::applyColorMap(enhancedGray, colorImage, cv::COLORMAP_JET);

    return colorImage;
}
cv::Mat imgProcesser::projectToXZ(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                  cv::Mat scanlineIdMap, bool isinter) {
    const int IMG_HEIGHT = 800; // 俯仰角分辨率（Z 方向）
    const int IMG_WIDTH = 800;  // 水平角度分辨率（X 方向）

    cv::Mat intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);

    if (cloud->size() == 0) {
        intensityImage = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_8U);
        return intensityImage;
    }

    if (isinter) {
        // scanlineIdMap = cv::Mat::zeros(IMG_HEIGHT, IMG_WIDTH, CV_32F);
        for (const auto &p : cloud->points) {
            if (p.x > 0) {
                continue;
            }
            float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            // 计算角度
            float theta = std::atan2(-p.x, p.y); // 前方（-x）方向的角度
            float phi = std::asin(p.z / r);      // 俯仰角（z 方向）
            // if(abs(phi)>(M_PI / 3)){continue;}
            int u;
            // 归一化到图像尺寸
            if (theta < 0) {
                u = static_cast<int>((-theta) / (M_PI)*IMG_WIDTH);
            }
            if (theta >= 0) {
                u = static_cast<int>((-theta + M_PI) / (M_PI)*IMG_WIDTH);
            }
            // int u = static_cast<int>((theta + (M_PI/2)) / (M_PI) * IMG_WIDTH);  //
            // 归一化到 [0, IMG_WIDTH]
            int v = static_cast<int>((-phi + M_PI / 2) / (M_PI)*IMG_HEIGHT); // 归一化到 [0, IMG_HEIGHT]

            if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
                intensityImage.at<float>(v, u) = p.intensity;
            }
        }
    }

    if (!isinter) {
        for (const auto &p : cloud->points) {
            if (p.x > 0) {
                continue;
            }
            float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            // 计算角度
            float theta = std::atan2(-p.x, p.y); // 前方（-x）方向的角度
            float phi = std::asin(p.z / r);      // 俯仰角（z 方向）
            // if(abs(phi)>(M_PI / 3)){continue;}
            int u;
            // 归一化到图像尺寸
            if (theta < 0) {
                u = static_cast<int>((-theta) / (M_PI)*IMG_WIDTH);
            }
            if (theta >= 0) {
                u = static_cast<int>((-theta + M_PI) / (M_PI)*IMG_WIDTH);
            }
            // int u = static_cast<int>((theta + (M_PI/2)) / (M_PI) * IMG_WIDTH);  //
            // 归一化到 [0, IMG_WIDTH]
            int v = static_cast<int>((-phi + M_PI / 2) / (M_PI)*IMG_HEIGHT); // 归一化到 [0, IMG_HEIGHT]

            if (u >= 0 && u < IMG_WIDTH && v >= 0 && v < IMG_HEIGHT) {
                intensityImage.at<float>(v, u) = p.intensity;
            }
        }
    }

    // 归一化到 [0, 255]
    double minVal, maxVal;
    cv::minMaxLoc(intensityImage, &minVal, &maxVal);
    intensityImage.convertTo(intensityImage, CV_8U, 255.0 / (maxVal - minVal),
                             -minVal * 255.0 / (maxVal - minVal));

    // 直方图均衡化
    cv::equalizeHist(intensityImage, intensityImage);

    //可选：伽马校正增强对比度
    cv::Mat gammaImage;
    intensityImage.convertTo(gammaImage, CV_32F, 1.0 / 255.0);
    pow(gammaImage, 0.5, gammaImage); // γ = 0.5
    gammaImage.convertTo(intensityImage, CV_8U, 255);
    cv::Mat intensityresult;
    if (isinter) {
        // intensityresult=interpolateLidarImage(intensityImage);
    }
    // removeLines(intensityImage);
    // filterBrightness(intensityImage);

    return intensityImage;
}

cv::Mat imgProcesser::overlayRedOnGrayscale(const cv::Mat &gray1,
                                            const cv::Mat &gray2) {
    // CV_Assert(gray1.size() == gray2.size() && gray1.type() == CV_8UC1 &&
    // gray2.type() == CV_8UC1);

    // 1. 将 gray1 映射到红色通道（创建彩色图像）
    cv::Mat redImage(gray1.size(), CV_8UC3, cv::Scalar(0, 0, 0)); // 初始化为黑色
    std::vector<cv::Mat> channels(3);
    channels[0] = cv::Mat::zeros(gray1.size(), CV_8UC1); // B 通道
    channels[1] = cv::Mat::zeros(gray1.size(), CV_8UC1); // G 通道
    channels[2] = gray1.clone();                         // R 通道
    cv::merge(channels, redImage);
    cv::Mat colorImage;
    // 2. 将 gray2 转换为 BGR 彩色图像
    if (gray2.type() == CV_8UC1) {
        cv::cvtColor(gray2, colorImage, cv::COLOR_GRAY2BGR);
    }
    if (gray2.type() == CV_8UC3) {
        colorImage = gray2.clone();
    }

    // 3. 在 gray2（BGR 图像）上绘制红色实心圆
    int circleRadius = 3; // 圆的半径

    // 遍历 gray1 找到非零红色点
    for (int y = 0; y < gray1.rows; y++) {
        for (int x = 0; x < gray1.cols; x++) {
            int redIntensity = gray1.at<uchar>(y, x);
            if (redIntensity > 0) { // 只有非零点才画圆
                cv::circle(colorImage, cv::Point(x, y), circleRadius,
                           cv::Scalar(0, 0, redIntensity), -1);
            }
        }
    }

    return colorImage;
}

cv::Mat imgProcesser::stackImagesVertical(const cv::Mat &gray1,
                                          const cv::Mat &gray2) {
    CV_Assert(gray1.cols == gray2.cols && gray1.type() == CV_8UC1 && gray2.type() == CV_8UC1);
    cv::Mat stackedImage;
    cv::vconcat(gray1, gray2, stackedImage);

    return stackedImage;
}
cv::Mat imgProcesser::draw_matches(cv::Mat &ref_points, cv::Mat &dst_points,
                                   cv::Mat &img1, cv::Mat &img2) {
    // prepare keypoints and matches for drawMatches function
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < ref_points.rows; ++i) {
        keypoints1.emplace_back(ref_points.at<cv::Point2f>(i, 0), 5);
        keypoints2.emplace_back(dst_points.at<cv::Point2f>(i, 0), 5);
        matches.emplace_back(i, i, 0);
    }

    // Draw inlier matches
    cv::Mat img_matches;
    if (!keypoints1.empty() && !keypoints2.empty() && !matches.empty()) {
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    } else {
        std::cerr << "Keypoints or matches are empty, cannot draw matches"
                  << std::endl;
    }

    return img_matches;
}