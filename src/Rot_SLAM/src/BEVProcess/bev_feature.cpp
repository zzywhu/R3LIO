#include "bev_feature.h"

BEVFeatureManager::BEVFeatureManager(ros::NodeHandle &nh, std::shared_ptr<BEVProjector> projector) :
    projector_(projector), stream_(c10::cuda::getStreamFromPool()) {
    loadParameters(nh);

    fast_detector_ = cv::FastFeatureDetector::create(fast_threshold_, true, cv::FastFeatureDetector::TYPE_9_16);
    if (matcher_type_ == "BF") {
        matcher_ = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);
    } else if (matcher_type_ == "BruteForce-Hamming") {
        matcher_ = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING);
    } else {
        ROS_ERROR("Invalid matcher type, setting to BF");
        matcher_ = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);
    }
    // initialize feature extractors
    orb_extractor_ = cv::ORB::create(
        500,                   // 最大特征点数量 (nfeatures)
        1.00f,                 // 尺度因子 (scale_factor)
        1,                     // 金字塔层数 (n_levels)
        31,                    // 边缘阈值 (edge_threshold)
        0,                     // 初始层级 (first_level)
        2,                     // WTA_K
        cv::ORB::HARRIS_SCORE, // 使用 Harris 角点评分 (score_type)
        31,                    // 补丁大小 (patch_size)
        35                     // 快速阈值 (fast_threshold)
    );

    loadmodel();
}

void BEVFeatureManager::loadParameters(ros::NodeHandle &nh) {
    nh.param<int>("visualize_en_", visualize_en_, 0);
    nh.param<int>("draw_keypoints_", draw_keypoints_, 1);
    nh.param<float>("downsample_ratio_", downsample_ratio_, 0.7f);
    nh.param<int>("down_sample_matches_", down_sample_matches_, 1);
    nh.param<double>("image/ransac_threshold", ransac_threshold_, 4.0);
    nh.param<float>("image/ratio_thresh", ratio_thresh_, 0.90f);
    nh.param<int>("image/fast_threshold", fast_threshold_, 10);
    // change it to your path here
    nh.param<std::string>("model_path", model_path_, "/home/chx/bev-lio-lc/src/BEV_LIO/models/gpu.pstst");
    nh.param<std::string>("image/matcher", matcher_type_, "BF");
    nh.param<std::string>("image/match_mode", match_mode_, "knn");
}

void BEVFeatureManager::loadmodel() {
    try {
        std::cout << "=> loading GPU checkpoint from '" << model_path_ << "'" << std::endl;

        int device_count = torch::cuda::device_count();
        std::cout << "CUDA devices available: " << device_count << std::endl;
        // load model
        model_ = torch::jit::load(model_path_);
        model_.to(torch::Device(torch::kCUDA, 0));
        model_.eval();

        stream_ = c10::cuda::getStreamFromPool();
        c10::cuda::setCurrentCUDAStream(stream_);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the checkpoint: " << e.what() << std::endl;
        ROS_ERROR("Error loading the model");
    }
}

torch::Tensor BEVFeatureManager::cvMatToTensor(const cv::Mat &img) {
    try {
        // 检查输入图像
        if (img.empty()) {
            std::cerr << "Error: Input image is empty in cvMatToTensor!" << std::endl;
            return torch::Tensor();
        }

        std::cout << "Converting image: " << img.rows << "x" << img.cols << " channels=" << img.channels() << std::endl;

        cv::Mat img_float;
        img.convertTo(img_float, CV_32F, 1.0 / 255.0); // 修正：应该是255而不是256

        if (img_float.channels() == 1) {
            cv::Mat img_rgb;
            cv::cvtColor(img_float, img_rgb, cv::COLOR_GRAY2RGB);
            img_float = img_rgb;
        } else if (img_float.channels() == 3) {
            cv::cvtColor(img_float, img_float, cv::COLOR_BGR2RGB);
        } else {
            std::cerr << "Error: Unsupported number of channels: " << img_float.channels() << std::endl;
            return torch::Tensor();
        }

        // 检查转换后的图像
        if (img_float.empty() || img_float.channels() != 3) {
            std::cerr << "Error: Failed to convert to 3-channel image!" << std::endl;
            return torch::Tensor();
        }

        // 确保数据是连续的
        if (!img_float.isContinuous()) {
            img_float = img_float.clone();
        }

        std::cout << "Image prepared for tensor conversion: " << img_float.rows << "x" << img_float.cols << std::endl;

        // 使用更安全的方式创建张量，避免from_blob的内存问题
        std::vector<float> img_data;
        img_data.resize(img_float.rows * img_float.cols * 3);

        // 手动复制数据
        std::memcpy(img_data.data(), img_float.data, img_data.size() * sizeof(float));

        // 创建张量
        torch::Tensor tensor_image = torch::from_blob(
                                         img_data.data(),
                                         {1, img_float.rows, img_float.cols, 3},
                                         torch::kFloat)
                                         .clone(); // 使用clone()确保数据独立

        std::cout << "Tensor created successfully" << std::endl;
        return tensor_image.permute({0, 3, 1, 2}); // 转换为 {1, C, H, W}

    } catch (const std::exception &e) {
        std::cerr << "Exception in cvMatToTensor: " << e.what() << std::endl;
        return torch::Tensor();
    }
}

void BEVFeatureManager::detectBEVFeatures(BEVFrame &frame) {
    try {
        // 1. 检查输入图像
        if (frame.img_photo_u8.empty()) {
            std::cerr << "Error: Input image is empty!" << std::endl;
            return;
        }

        std::cout << "Input image size: " << frame.img_photo_u8.rows << "x" << frame.img_photo_u8.cols << std::endl;

        // 2. 检查CUDA设备可用性
        if (!torch::cuda::is_available()) {
            std::cerr << "Error: CUDA not available!" << std::endl;
            return;
        }

        int device_count = torch::cuda::device_count();
        if (device_count == 0) {
            std::cerr << "Error: No CUDA devices available!" << std::endl;
            return;
        }

        c10::cuda::setCurrentCUDAStream(stream_);

        // 3. 安全的图像转换
        torch::Tensor img_tensor;
        try {
            img_tensor = cvMatToTensor(frame.img_photo_u8);

            // 检查tensor是否有效
            if (!img_tensor.defined() || img_tensor.numel() == 0) {
                std::cerr << "Error: Failed to create valid tensor from image!" << std::endl;
                return;
            }

            std::cout << "Image tensor shape: " << img_tensor.sizes() << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error converting image to tensor: " << e.what() << std::endl;
            return;
        }

        torch::NoGradGuard no_grad;

        // 4. 安全的张量移动到GPU
        std::vector<c10::IValue> ivalue_inputs;
        try {
            // 直接移动到GPU并包装为IValue
            torch::Tensor gpu_tensor = img_tensor.to(torch::Device(torch::kCUDA, 0));
            ivalue_inputs.push_back(gpu_tensor);

            std::cout << "Tensors moved to GPU successfully" << std::endl;

            // 5. 安全的模型推理
            c10::IValue model_output;
            try {
                model_output = model_.forward(ivalue_inputs);
                std::cout << "Model forward pass completed" << std::endl;
            } catch (const c10::Error &e) {
                std::cerr << "CUDA Error during model forward: " << e.what() << std::endl;
                return;
            } catch (const std::exception &e) {
                std::cerr << "Error during model forward: " << e.what() << std::endl;
                return;
            }

            // 6. 安全的输出解析
            if (!model_output.isTuple()) {
                std::cerr << "Error: Model output is not a tuple!" << std::endl;
                return;
            }

            auto tuple_output = model_output.toTuple();
            if (!tuple_output) {
                std::cerr << "Error: Failed to convert output to tuple!" << std::endl;
                return;
            }

            // 7. 检查输出元素数量
            size_t num_elements = tuple_output->elements().size();
            std::cout << "Model output tuple has " << num_elements << " elements" << std::endl;

            if (num_elements < 3) {
                std::cerr << "Error: Model output tuple has insufficient elements: " << num_elements << std::endl;
                return;
            }

            // 8. 安全的张量提取
            torch::Tensor output2, output3;
            try {
                // 检查每个元素是否是张量
                if (!tuple_output->elements()[1].isTensor()) {
                    std::cerr << "Error: Element 1 is not a tensor!" << std::endl;
                    return;
                }
                if (!tuple_output->elements()[2].isTensor()) {
                    std::cerr << "Error: Element 2 is not a tensor!" << std::endl;
                    return;
                }

                output2 = tuple_output->elements()[1].toTensor(); // local_feats
                output3 = tuple_output->elements()[2].toTensor(); // global_desc

                // 检查张量是否有效
                if (!output2.defined() || !output3.defined()) {
                    std::cerr << "Error: Output tensors are not defined!" << std::endl;
                    return;
                }

                if (output2.numel() == 0 || output3.numel() == 0) {
                    std::cerr << "Error: Output tensors are empty!" << std::endl;
                    return;
                }

                std::cout << "Output2 (local_feats) shape: " << output2.sizes() << std::endl;
                std::cout << "Output3 (global_desc) shape: " << output3.sizes() << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "Error extracting tensors from output: " << e.what() << std::endl;
                return;
            }

            // 9. 安全的CPU复制
            try {
                // Asynchronously copy local features to CPU on the non-default stream
                {
                    c10::cuda::CUDAStreamGuard guard(stream_);
                    frame.local_feats = output2.to(at::kCPU, false); // async copy
                }

                // Synchronize the non-default stream with the default stream
                at::cuda::CUDAEvent event;
                event.record(stream_);
                event.synchronize();

                // 验证local_feats是否有效
                if (!frame.local_feats.defined() || frame.local_feats.numel() == 0) {
                    std::cerr << "Error: Failed to copy local features to CPU!" << std::endl;
                    return;
                }

                std::cout << "Local features copied to CPU successfully" << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "Error copying local features to CPU: " << e.what() << std::endl;
                return;
            }

            // 10. 安全的全局描述符处理
            try {
                // Flatten the global descriptor and copy it to the CPU
                output3 = output3.view(-1);                 // convert to flattened
                auto output3_data = output3.detach().cpu(); // global_desc

                if (output3_data.numel() <= 0) {
                    std::cerr << "Error: Global descriptor has no elements!" << std::endl;
                    return;
                }

                // 检查数据指针是否有效
                float *data_ptr = output3_data.data_ptr<float>();
                if (!data_ptr) {
                    std::cerr << "Error: Global descriptor data pointer is null!" << std::endl;
                    return;
                }

                frame.global_desc.resize(output3_data.numel());
                std::memcpy(frame.global_desc.data(), data_ptr,
                            output3_data.numel() * sizeof(float));

                std::cout << "Global descriptor processed successfully, size: "
                          << frame.global_desc.size() << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "Error processing global descriptor: " << e.what() << std::endl;
                return;
            }

        } catch (const c10::Error &e) {
            std::cerr << "CUDA Error in detectBEVFeatures: " << e.what() << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Standard Exception in detectBEVFeatures: " << e.what() << std::endl;
        }

    } catch (...) {
        std::cerr << "Unknown error in detectBEVFeatures" << std::endl;
    }
}
void BEVFeatureManager::getFAST(BEVFrame &frame, bool draw_feature) {
    fast_detector_->detect(frame.img_photo_u8, frame.keypoints);

    std::vector<torch::Tensor> descriptors;

    for (const auto &kp : frame.keypoints) {
        int u = static_cast<int>(kp.pt.x);
        int v = static_cast<int>(kp.pt.y);

        // from CWH to WHC
        torch::Tensor descriptor = frame.local_feats.index({0, torch::indexing::Slice(), v, u});
        descriptors.push_back(descriptor);
    }

    //  concat descriptors into a single tensor
    if (!descriptors.empty()) {
        int descriptor_size = descriptors[0].size(0);
        cv::Mat concatenated_descriptors(descriptors.size(), descriptor_size, CV_32F);
        for (size_t i = 0; i < descriptors.size(); ++i) {
            memcpy(concatenated_descriptors.ptr<float>(i), descriptors[i].data_ptr<float>(), descriptor_size * sizeof(float));
        }
        frame.query_descriptors = concatenated_descriptors;
    }

    if (draw_feature) {
        cv::Mat inverted_img;
        cv::bitwise_not(frame.img_photo_u8, inverted_img);
        cv::cvtColor(inverted_img, frame.img_with_keypoints, cv::COLOR_GRAY2BGR); // gray to BGR
        // red markers
        for (const auto &kp : frame.keypoints) {
            cv::drawMarker(frame.img_with_keypoints, kp.pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 5, 1);
        }
    }
}

void BEVFeatureManager::getORB(BEVFrame &frame, bool draw_feature) {
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;

    orb_extractor_->detectAndCompute(frame.img_photo_u8, cv::noArray(), orb_keypoints, orb_descriptors);

    frame.keypoints = orb_keypoints;
    frame.query_descriptors = orb_descriptors;

    if (draw_feature) {
        cv::cvtColor(frame.img_photo_u8, frame.img_with_keypoints, cv::COLOR_GRAY2BGR);
        for (const auto &kp : frame.keypoints) {
            cv::drawMarker(frame.img_with_keypoints, kp.pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 5, 1);
        }
    }
}

std::vector<cv::DMatch> BEVFeatureManager::matchFeatures(const BEVFrame &frame, const BEVFrame &frame_prev, cv::Mat &img_matches) {
    thread_local cv::Ptr<cv::BFMatcher> local_matcher;
    local_matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_L2);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::DMatch> knn_matches_one;

    if (frame.keypoints.empty() || frame_prev.keypoints.empty()) {
        std::cerr << "No keypoints to match" << std::endl;
        std::cout << frame.keypoints.size() << std::endl;
        std::cout << frame_prev.keypoints.size() << std::endl;
        return good_matches;
    }

    if (frame.query_descriptors.type() != frame_prev.query_descriptors.type()) {
        frame.query_descriptors.convertTo(frame.query_descriptors, frame_prev.query_descriptors.type());
        std::cout << "not same type" << std::endl;
    }

    cv::Mat descriptors1 = frame.query_descriptors.clone();
    cv::Mat descriptors2 = frame_prev.query_descriptors.clone();
    local_matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    // local_matcher->knnMatch(frame.query_descriptors, frame_prev.query_descriptors,knn_matches, 2);

    // #pragma omp parallel for
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh_ * knn_matches[i][1].distance) {
            // #pragma omp critical
            good_matches.push_back(knn_matches[i][0]);
        }
        // #pragma omp critical
        knn_matches_one.push_back(knn_matches[i][0]);
    }

    // RANSAC to find inliers
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for (size_t i = 0; i < good_matches.size(); i++) {
        obj.push_back(frame.keypoints[good_matches[i].queryIdx].pt);
        scene.push_back(frame_prev.keypoints[good_matches[i].trainIdx].pt);
    }
    std::vector<uchar> inliers;
    if (obj.size() < 4 || scene.size() < 4) {
        std::cout << "Not enough points for RANSAC" << std::endl;
    } else {
        cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC, ransac_threshold_, inliers);
    }

    std::vector<cv::DMatch> ransac_matches;
    for (size_t i = 0; i < inliers.size(); i++) {
        if (inliers[i]) {
            ransac_matches.push_back(good_matches[i]);
        }
    }

    std::vector<cv::DMatch> sampled_matches;

    // downsample
    if (down_sample_matches_ == 1) {
        int downsample_step = static_cast<int>(1.0f / downsample_ratio_);
        downsample_step = std::max(downsample_step, 1);
        for (size_t i = 0; i < ransac_matches.size(); i += downsample_step) {
            sampled_matches.push_back(ransac_matches[i]);
        }
        std::cout << "ransac_matches.size(): " << ransac_matches.size() << std::endl;
        std::cout << "sampled_matches.size(): " << sampled_matches.size() << std::endl;
    } else {
        sampled_matches = ransac_matches;
    }

    if (visualize_en_ == 1) {
        cv::Mat inverted_img1;
        cv::bitwise_not(frame.img_photo_u8, inverted_img1);
        cv::Mat inverted_img2;
        cv::bitwise_not(frame_prev.img_photo_u8, inverted_img2);
        cv::Mat img_bgr_1;
        cv::Mat img_bgr_2;
        cv::cvtColor(inverted_img1, img_bgr_1, CV_GRAY2RGB);
        cv::cvtColor(inverted_img2, img_bgr_2, CV_GRAY2RGB);
        cv::drawMatches(img_bgr_1, frame.keypoints,
                        img_bgr_2, frame_prev.keypoints,
                        sampled_matches, img_matches,
                        cv::Scalar::all(-1),
                        // cv::Scalar(0, 255, 0),
                        cv::Scalar(0, 0, 255),
                        //  cv::Scalar::all(-1),
                        std::vector<char>() // use std::vector<std::vector<char>>
                                            // cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
        );

        if (draw_keypoints_ == 1) {
            cv::Mat inverted_img1;
            cv::bitwise_not(frame.img_photo_u8, inverted_img1);
            cv::Mat inverted_img2;
            cv::bitwise_not(frame_prev.img_photo_u8, inverted_img2);
            cv::Mat img_bgr_1;
            cv::Mat img_bgr_2;
            cv::cvtColor(inverted_img1, img_bgr_1, CV_GRAY2RGB);
            cv::cvtColor(inverted_img2, img_bgr_2, CV_GRAY2RGB);
            cv::drawMatches(img_bgr_1, frame.keypoints,
                            img_bgr_2, frame_prev.keypoints,
                            sampled_matches, img_matches,
                            cv::Scalar::all(-1),
                            // cv::Scalar(0, 255, 0),
                            cv::Scalar(0, 0, 255),
                            //  cv::Scalar::all(-1),
                            std::vector<char>(), // use std::vector<std::vector<char>>
                            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }
    } else {
        cv::Mat img_bgr_1;
        cv::Mat img_bgr_2;
        cv::cvtColor(frame.img_photo_u8, img_bgr_1, CV_GRAY2RGB);
        cv::cvtColor(frame_prev.img_photo_u8, img_bgr_2, CV_GRAY2RGB);
        cv::drawMatches(img_bgr_1, frame.keypoints,
                        img_bgr_2, frame_prev.keypoints,
                        sampled_matches, img_matches,
                        cv::Scalar::all(-1),
                        // cv::Scalar(0, 255, 0),
                        cv::Scalar(0, 0, 255),
                        //  cv::Scalar::all(-1),
                        std::vector<char>() // use std::vector<std::vector<char>>
                                            // cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
        );
    }

    if (match_mode_ == "knn") {
        matches_ = knn_matches_one;
        return knn_matches_one;
    } else if (match_mode_ == "ransac") {
        matches_ = ransac_matches;
        return ransac_matches;
    } else if (match_mode_ == "good") {
        matches_ = good_matches;
        return good_matches;
    }
}
