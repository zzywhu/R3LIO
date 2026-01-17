#ifndef M_LSD_H
#define M_LSD_H

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/point_types.h>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>





/**
 * @brief M-LSD 线段检测类
 * 
 * 这个类实现了基于ONNX Runtime的M-LSD线段检测算法。
 */
class M_LSD
{
public:
    /**
     * @brief 构造函数
     * @param modelpath ONNX模型文件路径
     */
    M_LSD(std::string modelpath);

    /**
     * @brief 检测图像中的线段
     * @param cv_image 输入图像
     * @return 标记了检测线段的图像
     */
    cv::Mat detect(cv::Mat cv_image,std::vector<std::vector<int>> &segments_list);

private:
    /**
     * @brief 图像预处理函数
     * @param srcimg 输入图像
     */
    void preprocess(cv::Mat srcimg);

    int inpWidth;                        // 模型输入宽度
    int inpHeight;                       // 模型输入高度
    int num_lines;                       // 最大线段数量
    int map_h;                           // 特征图高度
    int map_w;                           // 特征图宽度
    std::vector<float> input_image_;     // 预处理后的输入图像数据
    const float conf_threshold = 0.2;    // 置信度阈值
    const float dist_threshold = 5;   // 距离阈值

    // ONNX Runtime相关变量
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "M-LSD");
    Ort::Session *ort_session = nullptr;
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    std::vector<char*> input_names;
    std::vector<char*> output_names;
    std::vector<std::vector<int64_t>> input_node_dims;  // 输入节点维度
    std::vector<std::vector<int64_t>> output_node_dims; // 输出节点维度
};

#endif // M_LSD_H