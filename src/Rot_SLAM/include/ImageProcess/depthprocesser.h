#ifndef DEPTH_COMPLETION_HPP
#define DEPTH_COMPLETION_HPP

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <algorithm>
#include <iostream>

/**
 * @class Kernels
 * @brief 提供形态学操作所需的各种卷积核
 */
class Kernels {
public:
    cv::Mat FULL_KERNEL_3;
    cv::Mat FULL_KERNEL_5;
    cv::Mat FULL_KERNEL_7;
    cv::Mat FULL_KERNEL_9;
    cv::Mat FULL_KERNEL_31;

    /**
     * @brief 构造函数，初始化基本的卷积核
     */
    Kernels() {
        FULL_KERNEL_3 = cv::Mat::ones(3, 3, CV_8U);
        FULL_KERNEL_5 = cv::Mat::ones(5, 5, CV_8U);
        FULL_KERNEL_7 = cv::Mat::ones(7, 7, CV_8U);
        FULL_KERNEL_9 = cv::Mat::ones(9, 9, CV_8U);
        FULL_KERNEL_31 = cv::Mat::ones(31, 31, CV_8U);
    }

    /**
     * @brief 生成3x3十字形卷积核
     * @return 3x3十字形卷积核
     */
    cv::Mat cross_kernel_3() {
        cv::Mat kernel = (cv::Mat_<uint8_t>(3, 3) <<
            0, 1, 0,
            1, 1, 1,
            0, 1, 0);
        return kernel;
    }

    /**
     * @brief 生成5x5十字形卷积核
     * @return 5x5十字形卷积核
     */
    cv::Mat cross_kernel_5() {
        cv::Mat kernel = (cv::Mat_<uint8_t>(5, 5) <<
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0);
        return kernel;
    }

    /**
     * @brief 生成5x5菱形卷积核
     * @return 5x5菱形卷积核
     */
    cv::Mat diamond_kernel_5() {
        cv::Mat kernel = (cv::Mat_<uint8_t>(5, 5) <<
            0, 0, 1, 0, 0,
            0, 1, 1, 1, 0,
            1, 1, 1, 1, 1,
            0, 1, 1, 1, 0,
            0, 0, 1, 0, 0);
        return kernel;
    }

    /**
     * @brief 生成7x7十字形卷积核
     * @return 7x7十字形卷积核
     */
    cv::Mat cross_kernel_7() {
        cv::Mat kernel = (cv::Mat_<uint8_t>(7, 7) <<
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0);
        return kernel;
    }

    /**
     * @brief 生成7x7菱形卷积核
     * @return 7x7菱形卷积核
     */
    cv::Mat diamond_kernel_7() {
        cv::Mat kernel = (cv::Mat_<uint8_t>(7, 7) <<
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 0, 0, 0);
        return kernel;
    }
};

/**
 * @class DepthMapProcessor
 * @brief 深度图补全的核心处理类
 */
class DepthMapProcessor {
private:
    Kernels kernels;
    float max_depth;

public:
    /**
     * @brief 构造函数
     * @param max_depth 最大深度值（用于深度反转）
     */
    DepthMapProcessor(float max_depth = 100.0f) : max_depth(max_depth) {}

    /**
     * @brief 深度图补全的主要算法
     * @param main_image 原始RGB图像（在此实现中未使用，保留以保持API一致性）
     * @param depth_map 原始深度图
     * @return 补全后的深度图
     */
    cv::Mat create_map(const cv::Mat& main_image, const cv::Mat& depth_map) {
        cv::Mat depths_in = depth_map.clone();
        depths_in.convertTo(depths_in, CV_32F);
        
        // 根据深度区间计算掩码 - 保留原始深度值的区间划分
        cv::Mat valid_pixels_near, valid_pixels_med, valid_pixels_far;
        cv::inRange(depths_in, 0.1f, 15.0f, valid_pixels_near);
        cv::inRange(depths_in, 15.0f, 30.0f, valid_pixels_med);
        cv::threshold(depths_in, valid_pixels_far, 30.0f, 255, cv::THRESH_BINARY);

        // 将布尔掩码转换为浮点掩码（0.0或1.0）
        valid_pixels_near.convertTo(valid_pixels_near, CV_32F, 1.0/255.0);
        valid_pixels_med.convertTo(valid_pixels_med, CV_32F, 1.0/255.0);
        valid_pixels_far.convertTo(valid_pixels_far, CV_32F, 1.0/255.0);

        // 移除步骤1中的深度反转，直接使用原始深度
        cv::Mat s1_depths = depths_in.clone();
        cv::Mat valid_pixels;
        cv::threshold(depths_in, valid_pixels, 0.1f, 255, cv::THRESH_BINARY);
        valid_pixels.convertTo(valid_pixels, CV_32F, 1.0/255.0);
        
        // 不再需要深度反转循环
        // for (int y = 0; y < s1_inverted_depths.rows; y++) {
        //     for (int x = 0; x < s1_inverted_depths.cols; x++) { 
        //         if (valid_pixels.at<float>(y, x) > 0.5f) {
        //             s1_inverted_depths.at<float>(y, x) = max_depth - s1_inverted_depths.at<float>(y, x);
        //         }
        //     }
        // }

        // 步骤2：根据不同深度范围应用不同的形态学扩张
        cv::Mat dilated_far, dilated_med, dilated_near;
        cv::Mat far_depths = s1_depths.mul(valid_pixels_far);
        cv::Mat med_depths = s1_depths.mul(valid_pixels_med);
        cv::Mat near_depths = s1_depths.mul(valid_pixels_near);
        
        cv::dilate(far_depths, dilated_far, kernels.cross_kernel_3());
        cv::dilate(med_depths, dilated_med, kernels.diamond_kernel_5());
        cv::dilate(near_depths, dilated_near, kernels.diamond_kernel_7());

        // 更新扩张后的掩码
        cv::threshold(dilated_near, valid_pixels_near, 0.1f, 1.0f, cv::THRESH_BINARY);
        cv::threshold(dilated_med, valid_pixels_med, 0.1f, 1.0f, cv::THRESH_BINARY);
        cv::threshold(dilated_far, valid_pixels_far, 0.1f, 1.0f, cv::THRESH_BINARY);

        // 合并不同深度区间的扩张结果
        cv::Mat s2_dilated_depths = s1_depths.clone();
        for (int y = 0; y < s2_dilated_depths.rows; y++) {
            for (int x = 0; x < s2_dilated_depths.cols; x++) {
                if (valid_pixels_far.at<float>(y, x) > 0.5f) {
                    s2_dilated_depths.at<float>(y, x) = dilated_far.at<float>(y, x);
                }
                if (valid_pixels_med.at<float>(y, x) > 0.5f) {
                    s2_dilated_depths.at<float>(y, x) = dilated_med.at<float>(y, x);
                }
                if (valid_pixels_near.at<float>(y, x) > 0.5f) {
                    s2_dilated_depths.at<float>(y, x) = dilated_near.at<float>(y, x);
                }
            }
        }

        // 步骤3：形态学闭操作填充小空洞
        cv::Mat s3_closed_depths;
        cv::morphologyEx(s2_dilated_depths, s3_closed_depths, cv::MORPH_CLOSE, kernels.FULL_KERNEL_5);

        // 步骤4：中值滤波降噪
        cv::Mat s4_blurred_depths = s3_closed_depths.clone();
        cv::Mat blurred;
        cv::medianBlur(s3_closed_depths, blurred, 5);

        cv::threshold(s3_closed_depths, valid_pixels, 0.1f, 1.0f, cv::THRESH_BINARY);
        for (int y = 0; y < s4_blurred_depths.rows; y++) {
            for (int x = 0; x < s4_blurred_depths.cols; x++) {
                if (valid_pixels.at<float>(y, x) > 0.5f) {
                    s4_blurred_depths.at<float>(y, x) = blurred.at<float>(y, x);
                }
            }
        }

        // 步骤5：创建顶部掩码并扩张填充空白区域
        cv::Mat top_mask = cv::Mat::ones(depths_in.size(), CV_8U);
        
        for (int x = 0; x < s4_blurred_depths.cols; x++) {
            int top_row = 0;
            while (top_row < s4_blurred_depths.rows && s4_blurred_depths.at<float>(top_row, x) <= 0.1f) {
                top_row++;
            }
            
            for (int y = 0; y < top_row; y++) {
                top_mask.at<uchar>(y, x) = 0;
            }
        }

        cv::Mat valid_pixels_float;
        cv::threshold(s4_blurred_depths, valid_pixels_float, 0.1f, 1.0f, cv::THRESH_BINARY);
        valid_pixels_float.convertTo(valid_pixels, CV_8U, 255);
        
        cv::Mat empty_pixels;
        cv::bitwise_not(valid_pixels, empty_pixels);
        cv::bitwise_and(empty_pixels, top_mask, empty_pixels);

        cv::Mat dilated;
        cv::dilate(s4_blurred_depths, dilated, kernels.FULL_KERNEL_7);
        
        cv::Mat s5_dilated_depths = s4_blurred_depths.clone();
        for (int y = 0; y < s5_dilated_depths.rows; y++) {
            for (int x = 0; x < s5_dilated_depths.cols; x++) {
                if (empty_pixels.at<uchar>(y, x) > 0) {
                    s5_dilated_depths.at<float>(y, x) = dilated.at<float>(y, x);
                }
            }
        }

        // 步骤6：扩展深度到顶部像素
        cv::Mat s6_extended_depths = s5_dilated_depths.clone();
        
        for (int x = 0; x < s5_dilated_depths.cols; x++) {
            int top_row = 0;
            while (top_row < s5_dilated_depths.rows && s5_dilated_depths.at<float>(top_row, x) <= 0.1f) {
                top_row++;
            }
            
            if (top_row < s5_dilated_depths.rows) {
                float top_value = s5_dilated_depths.at<float>(top_row, x);
                for (int y = 0; y < top_row; y++) {
                    s6_extended_depths.at<float>(y, x) = top_value;
                }
            }
        }

        // 步骤7：多次扩张和平滑处理
        cv::Mat s7_blurred_depths = s6_extended_depths.clone();
        
        for (int i = 0; i < 6; i++) {
            // 创建空白像素掩码
            cv::threshold(s7_blurred_depths, valid_pixels_float, 0.1f, 1.0f, cv::THRESH_BINARY);
            valid_pixels_float.convertTo(valid_pixels, CV_8U, 255);
            cv::bitwise_not(valid_pixels, empty_pixels);
            cv::bitwise_and(empty_pixels, top_mask, empty_pixels);
            
            // 扩张
            cv::dilate(s7_blurred_depths, dilated, kernels.FULL_KERNEL_31);
            
            // 用扩张值填充空白像素
            for (int y = 0; y < s7_blurred_depths.rows; y++) {
                for (int x = 0; x < s7_blurred_depths.cols; x++) {
                    if (empty_pixels.at<uchar>(y, x) > 0) {
                        s7_blurred_depths.at<float>(y, x) = dilated.at<float>(y, x);
                    }
                }
            }
        }
        
        // 应用中值滤波
        cv::threshold(s7_blurred_depths, valid_pixels_float, 0.1f, 1.0f, cv::THRESH_BINARY);
        valid_pixels_float.convertTo(valid_pixels, CV_8U, 255);
        cv::bitwise_and(valid_pixels, top_mask, valid_pixels);
        
        cv::medianBlur(s7_blurred_depths, blurred, 5);
        for (int y = 0; y < s7_blurred_depths.rows; y++) {
            for (int x = 0; x < s7_blurred_depths.cols; x++) {
                if (valid_pixels.at<uchar>(y, x) > 0) {
                    s7_blurred_depths.at<float>(y, x) = blurred.at<float>(y, x);
                }
            }
        }
        
        // 应用高斯滤波
        cv::GaussianBlur(s7_blurred_depths, blurred, cv::Size(5, 5), 0);
        for (int y = 0; y < s7_blurred_depths.rows; y++) {
            for (int x = 0; x < s7_blurred_depths.cols; x++) {
                if (valid_pixels.at<uchar>(y, x) > 0) {
                    s7_blurred_depths.at<float>(y, x) = blurred.at<float>(y, x);
                }
            }
        }

        // 步骤8：反转深度回原始形式
        cv::Mat s8_inverted_depths = s7_blurred_depths.clone();
        cv::threshold(s8_inverted_depths, valid_pixels_float, 0.1f, 1.0f, cv::THRESH_BINARY);
        
        // for (int y = 0; y < s8_inverted_depths.rows; y++) {
        //     for (int x = 0; x < s8_inverted_depths.cols; x++) {
        //         if (valid_pixels_float.at<float>(y, x) > 0.5f) {
        //             s8_inverted_depths.at<float>(y, x) = max_depth - s8_inverted_depths.at<float>(y, x);
        //         }
        //     }
        // }

        // 直接返回s7_blurred_depths作为最终结果
        return s7_blurred_depths;
    }
};
#endif // DEPTH_COMPLETION_HPP