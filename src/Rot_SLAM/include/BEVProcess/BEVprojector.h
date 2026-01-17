#include <Common.h>
#include "ros/ros.h"

#pragma once

class BEVProjector {
public:
    BEVProjector(ros::NodeHandle nh);

    //void voxelDownSample(BEVFrame &frame);

    double resolution() const {
        return resolution_;
    }

    V3D turn_pixel_to_point(const V3D &pixel_uv);

    void getBEV(BEVFrame &frame);
    void getMapBEV(BEVFrame &frame);
    void getBEVOutdoor(BEVFrame &frame);

    int rows() const {
        return y_num_;
    }
    int cols() const {
        return x_num_;
    }

    inline int xMinIndex() const {
        return x_min_ind_;
    }
    inline int yMinIndex() const {
        return y_min_ind_;
    }
    // 可选：若需要做边界检查，可再暴露尺寸
    inline int xSize() const {
        return x_num_;
    }
    inline int ySize() const {
        return y_num_;
    }

    inline cv::Point2f backProjectBEVPixelToXY(const cv::Point2f &px) {
        // px.x 为列索引(x_ind)，px.y 为行索引(y_ind)
        const float res = resolution_;
        const int x_min = x_min_ind_;
        const int y_min = y_min_ind_;

        // 栅格中心：(+0.5f)；若需要左下角坐标可去掉 0.5f
        const float x = (px.x + x_min + 0.5f) * res;
        const float y = (px.y + y_min + 0.5f) * res;
        return cv::Point2f(x, y);
    }

private:
    void loadParameters(ros::NodeHandle nh);

    double resolution_; // 分辨率，单位为米
    double max_x_;      // 最大 x 坐标
    double min_x_;      // 最小 x 坐标
    double max_y_;      // 最大 y 坐标
    double min_y_;      // 最小 y 坐标

    double voxel_size_; // 体素大小
    bool downsample_;
    bool normalize_to_255_;
    bool use_dense_;
    int x_min_ind_;
    int x_max_ind_;
    int y_min_ind_;
    int y_max_ind_;

    int x_num_;
    int y_num_;
};