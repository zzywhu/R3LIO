#include "BEVprojector.h"
#include <stdexcept>
#define DUPLICATE_POINTS 10

BEVProjector::BEVProjector(ros::NodeHandle nh) {
    try {
        loadParameters(nh);
    } catch (const std::runtime_error &e) {
        ROS_ERROR_STREAM(e.what());
        exit(1);
    }
}

void BEVProjector::loadParameters(ros::NodeHandle nh) {
    nh.param("bev/resolution", resolution_, 0.01);             // 分辨率，单位为米
    nh.param("bev/max_x_", max_x_, 500.0);                     // 最大 x 坐标
    nh.param("bev/min_x_", min_x_, -500.0);                    // 最小 x 坐标
    nh.param("bev/max_y_", max_y_, 500.0);                     // 最大 y 坐标
    nh.param("bev/min_y_", min_y_, -500.0);                    // 最小 y 坐标
    nh.param("bev/voxel_size", voxel_size_, 0.4);              // 体素大小
    nh.param("bev/downsample", downsample_, true);             // 图像垂直方向上栅格数
    nh.param("bev/normalize_to_255", normalize_to_255_, true); // 归一到255
    nh.param("bev/use_dense", use_dense_, true);               // 是否使用密度图

    x_min_ind_ = static_cast<int>(std::floor(min_x_ / resolution_));
    x_max_ind_ = static_cast<int>(std::floor(max_x_ / resolution_));
    y_min_ind_ = static_cast<int>(std::floor(min_y_ / resolution_));
    y_max_ind_ = static_cast<int>(std::floor(max_y_ / resolution_));
    x_num_ = x_max_ind_ - x_min_ind_ + 1;
    y_num_ = y_max_ind_ - y_min_ind_ + 1;
}

// void BEVProjector::voxelDownSample(BEVFrame& frame) {
//     pcl::VoxelGrid<PointType> sor;
//     sor.setInputCloud(frame.points);
//     sor.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
//     sor.filter(*frame.points);
// }

V3D BEVProjector::turn_pixel_to_point(const V3D &pixel_uv) {
    float x = (x_max_ind_ - pixel_uv.x()) * resolution_;
    float y = (y_max_ind_ - pixel_uv.y()) * resolution_;
    return V3D(y, x, 0.0);
}

void BEVProjector::getBEV(BEVFrame &frame) {
    // if (downsample_) {
    //     voxelDownSample(frame);
    // }

    cv::Mat mat_global_image = cv::Mat::zeros(y_num_, x_num_, CV_8UC1);
    frame.img_dense = cv::Mat::zeros(y_num_, x_num_, CV_32FC1);

    for (size_t i = 0; i < frame.points->points.size(); ++i) {
        const auto &point = frame.points->points[i];
        float x = point.x;
        float y = point.y;

        float x_float = (x / resolution_);
        float y_float = (y / resolution_);
        int x_ind = static_cast<int>(std::floor(x_float)) - x_min_ind_;
        int y_ind = static_cast<int>(std::floor(y_float)) - y_min_ind_;

        if (x_ind >= x_num_ || y_ind >= y_num_ || x_ind < 0 || y_ind < 0) {
            continue;
        }

        if (mat_global_image.at<uchar>(y_ind, x_ind) < 15) {
            mat_global_image.at<uchar>(y_ind, x_ind) += 15;
        }
    }

    if (normalize_to_255_) {
        mat_global_image.setTo(0, mat_global_image <= 1);
        mat_global_image *= 15;
        mat_global_image.setTo(255, mat_global_image > 255);
    }

    frame.img_dense = mat_global_image.clone();
    frame.img_dense.convertTo(frame.img_photo_u8, CV_8UC1, 1);
}

void BEVProjector::getBEVOutdoor(BEVFrame &frame) {
    // 按栅格统计密度并用密度映射灰度
    cv::Mat count_mat = cv::Mat::zeros(y_num_, x_num_, CV_32SC1);
    for (size_t i = 0; i < frame.points->points.size(); ++i) {
        const auto &point = frame.points->points[i];
        float x = point.x;
        float y = point.y;
        float z = point.z;
        // 可选过滤低于一定高度点
        //if (z < 2.0f) continue;
        int x_ind = static_cast<int>(std::floor(x / resolution_)) - x_min_ind_;
        int y_ind = static_cast<int>(std::floor(y / resolution_)) - y_min_ind_;
        if (x_ind < 0 || x_ind >= x_num_ || y_ind < 0 || y_ind >= y_num_) continue;
        count_mat.at<int>(y_ind, x_ind)++;
    }

    // 计算最大密度用于归一化
    double min_val, max_val;
    cv::minMaxLoc(count_mat, &min_val, &max_val);
    if (max_val < 1.0) max_val = 1.0; // 防止除零

    cv::Mat density_u8 = cv::Mat::zeros(y_num_, x_num_, CV_8UC1);
    // 使用对数归一，增强稀疏区域对比
    for (int r = 0; r < count_mat.rows; ++r) {
        const int *row_ptr = count_mat.ptr<int>(r);
        uchar *out_ptr = density_u8.ptr<uchar>(r);
        for (int c = 0; c < count_mat.cols; ++c) {
            int cnt = row_ptr[c];
            if (cnt <= 0) {
                out_ptr[c] = 0;
                continue;
            }
            double val = std::log(1.0 + cnt) / std::log(1.0 + max_val); // 0~1
            out_ptr[c] = static_cast<uchar>(std::round(val * 255.0));
        }
    }

    //如果需要再做一次形态学膨胀，增强可视效果（可选）
    if (normalize_to_255_) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(density_u8, density_u8, kernel);
    }

    //增加直方图均衡化与gama纠正
    cv::equalizeHist(density_u8, density_u8);
    //伽马校正
    cv::Mat gammaImage;
    density_u8.convertTo(gammaImage, CV_32F, 1.0 / 255.0);
    cv::pow(gammaImage, 0.5, gammaImage); // γ = 1.2
    gammaImage.convertTo(density_u8, CV_8U, 255.0);

    frame.img_dense = density_u8.clone();
    frame.img_dense.convertTo(frame.img_photo_u8, CV_8UC1, 1.0);
}

void BEVProjector::getMapBEV(BEVFrame &frame) {
    // if (downsample_) {
    //     voxelDownSample(frame);
    // }
    // 先初始化为极值
    min_x_ = std::numeric_limits<float>::max();
    max_x_ = std::numeric_limits<float>::lowest();
    min_y_ = std::numeric_limits<float>::max();
    max_y_ = std::numeric_limits<float>::lowest();

    // 遍历点云，获取实际范围
    for (size_t i = 0; i < frame.points->points.size(); ++i) {
        const auto &point = frame.points->points[i];
        if (point.x < min_x_) min_x_ = point.x;
        if (point.x > max_x_) max_x_ = point.x;
        if (point.y < min_y_) min_y_ = point.y;
        if (point.y > max_y_) max_y_ = point.y;
    }

    x_min_ind_ = static_cast<int>(std::floor(min_x_ / resolution_));
    x_max_ind_ = static_cast<int>(std::floor(max_x_ / resolution_));
    y_min_ind_ = static_cast<int>(std::floor(min_y_ / resolution_));
    y_max_ind_ = static_cast<int>(std::floor(max_y_ / resolution_));
    x_num_ = x_max_ind_ - x_min_ind_ + 1;
    y_num_ = y_max_ind_ - y_min_ind_ + 1;

    // 检查尺寸合法性
    if (x_num_ <= 0 || y_num_ <= 0) {
        std::cerr << "Error: BEV image size invalid! x_num_=" << x_num_ << ", y_num_=" << y_num_ << std::endl;
        return;
    }

    cv::Mat mat_global_image = cv::Mat::zeros(y_num_, x_num_, CV_8UC1);
    frame.img_dense = cv::Mat::zeros(y_num_, x_num_, CV_32FC1);
    cv::Mat count_mat = cv::Mat::zeros(y_num_, x_num_, CV_32SC1);

    for (size_t i = 0; i < frame.points->points.size(); ++i) {
        const auto &point = frame.points->points[i];
        float x = point.x;
        float y = point.y;
        float z = point.z;
        // 可选过滤低于一定高度点
        //if (z < 2.0f) continue;
        int x_ind = static_cast<int>(std::floor(x / resolution_)) - x_min_ind_;
        int y_ind = static_cast<int>(std::floor(y / resolution_)) - y_min_ind_;
        if (x_ind < 0 || x_ind >= x_num_ || y_ind < 0 || y_ind >= y_num_) continue;
        count_mat.at<int>(y_ind, x_ind)++;
    }

    // 计算最大密度用于归一化
    double min_val, max_val;
    cv::minMaxLoc(count_mat, &min_val, &max_val);
    if (max_val < 1.0) max_val = 1.0; // 防止除零

    cv::Mat density_u8 = cv::Mat::zeros(y_num_, x_num_, CV_8UC1);
    // 使用对数归一，增强稀疏区域对比
    for (int r = 0; r < count_mat.rows; ++r) {
        const int *row_ptr = count_mat.ptr<int>(r);
        uchar *out_ptr = density_u8.ptr<uchar>(r);
        for (int c = 0; c < count_mat.cols; ++c) {
            int cnt = row_ptr[c];
            if (cnt <= 0) {
                out_ptr[c] = 0;
                continue;
            }
            double val = std::log(1.0 + cnt) / std::log(1.0 + max_val); // 0~1
            out_ptr[c] = static_cast<uchar>(std::round(val * 255.0));
        }
    }

    //如果需要再做一次形态学膨胀，增强可视效果（可选）
    if (normalize_to_255_) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(density_u8, density_u8, kernel);
    }

    //增加直方图均衡化与gama纠正
    cv::equalizeHist(density_u8, density_u8);
    //伽马校正
    cv::Mat gammaImage;
    density_u8.convertTo(gammaImage, CV_32F, 1.0 / 255.0);
    cv::pow(gammaImage, 0.5, gammaImage); // γ = 1.2
    gammaImage.convertTo(density_u8, CV_8U, 255.0);

    frame.img_dense = density_u8.clone();
    frame.img_dense.convertTo(frame.img_photo_u8, CV_8UC1, 1.0);
}
