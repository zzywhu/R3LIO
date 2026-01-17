#include <opencv2/opencv.hpp>
#include <vector>

//*******************full kernels********************//
// 保持最小内核不变
cv::Mat full_kernel3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
// 减小其他所有矩形内核
cv::Mat full_kernel5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // 从5x5减小到3x3
cv::Mat full_kernel7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // 从5x5减小到3x3
cv::Mat full_kernel9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)); // 从7x7减小到5x5
// 极大减小最大内核，减少过度平滑
cv::Mat full_kernel31x31 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)); // 从15x15减小到9x9

//*******************cross kernel********************//
// 减小交叉内核
cv::Mat cross_kernel3x3 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
cv::Mat cross_kernel5x5 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)); // 从5x5减小到3x3
cv::Mat cross_kernel7x7 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5)); // 从7x7减小到5x5

//*******************diamond kernel********************//
// 线性内核保持不变
cv::Mat one_kernel3x3 = (cv::Mat_<uint8_t>(3,3) << 0, 1, 0,
    0, 1, 0,
    0, 1, 0);

// 设计更小的菱形内核，减少填充强度
cv::Mat diamond_kernel5x5 = (cv::Mat_<uint8_t>(3,3) << 0, 1, 0,
                                                       1, 1, 1,
                                                       0, 1, 0); // 从5x5减小到3x3

// 减少7x7菱形内核为5x5
cv::Mat diamond_kernel7x7 = (cv::Mat_<uint8_t>(5,5) << 0, 0, 1, 0, 0,
                                                        0, 1, 1, 1, 0,
                                                        1, 1, 1, 1, 1,
                                                        0, 1, 1, 1, 0, 
                                                        0, 0, 1, 0, 0); // 从7x7减小到5x5

//*******************sharpen kernel********************//
// 增强锐化内核的效果，增加中心权重
cv::Mat sharpen_kernel3x3 = (cv::Mat_<char>(3,3) << 0, -1, 0,
    -1, 7, -1,  // 从6增强到7，进一步增加中心像素权重
    0, -1, 0);

//*******************depth completion********************//

cv::Mat depth_completion(const cv::Mat& lidar_depth, int img_width, int img_height)
{
    // double time_start = clock();
    cv::Mat depth_ = lidar_depth.clone();

    cv::Mat mask;
    cv::Mat dilated_map;
    // //depth inversion
    mask = (depth_ > 0.1);
    cv::subtract(100.0, depth_, depth_, mask);
    
    // 使用最小的菱形内核进行初始扩张
    cv::dilate(depth_, depth_, diamond_kernel5x5); // 现在是3x3
    
    // 使用最小的内核进行孔洞闭合
    cv::morphologyEx(depth_, depth_, cv::MORPH_CLOSE, full_kernel3x3);

    // 小孔填充 - 使用最小的内核
    mask = (depth_ < 0.1);
    cv::dilate(depth_, dilated_map, full_kernel3x3); // 从5x5减小到3x3  
    dilated_map.copyTo(depth_, mask);
    
    // 大孔填充 - 使用中等大小的内核
    mask = (depth_ < 0.1);
    cv::dilate(depth_, dilated_map, full_kernel9x9); // 从15x15减小到9x9
    dilated_map.copyTo(depth_, mask);
    
    // 使用最小的中值滤波核
    cv::medianBlur(depth_, depth_, 3);
    
    // 使用最小的高斯模糊核
    mask = (depth_ > 0.1);
    cv::GaussianBlur(depth_, dilated_map, cv::Size(3, 3), 0);
    dilated_map.copyTo(depth_, mask);
    
    // //depth inversion
    mask = (depth_ > 0.1);
    cv::subtract(100.0, depth_, depth_, mask);
    
    return depth_;
}