#include "MLSD/mlsd.h"
using namespace cv;
using namespace std;
using namespace Ort;


M_LSD::M_LSD(string model_path)
{
    // 初始化ONNX Runtime环境
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "M-LSD");
    
    // 设置图优化级别
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    
    // 使用新的CUDA加速方法
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    api->CreateCUDAProviderOptions(&cuda_options);
    
    // 设置CUDA设备ID为0 - 修复参数类型问题
    const char* key = "device_id";
    const char* value = "0";
    OrtStatus* status = api->UpdateCUDAProviderOptions(cuda_options, &key, &value, 1);
    if (status != nullptr) {
        const char* error_message = api->GetErrorMessage(status);
        std::cerr << "Error setting CUDA options: " << error_message << std::endl;
        api->ReleaseStatus(status);
    }
    
    // 将CUDA Provider添加到会话选项
    status = api->SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cuda_options);
    if (status != nullptr) {
        const char* error_message = api->GetErrorMessage(status);
        std::cerr << "Error adding CUDA provider: " << error_message << std::endl;
        api->ReleaseStatus(status);
    }
    
    // 释放CUDA选项
    api->ReleaseCUDAProviderOptions(cuda_options);
    
    // 创建会话
    ort_session = new Session(env, model_path.c_str(), sessionOptions);
    
    // 获取输入和输出节点信息
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    
    // 修复循环变量类型和GetInputName方法
    for (size_t i = 0; i < numInputNodes; i++)
    {
        // 使用GetInputNameAllocated代替GetInputName，返回类型是AllocatedStringPtr
        Ort::AllocatedStringPtr input_name_ptr = ort_session->GetInputNameAllocated(i, allocator);
        // 使用input_name_ptr.get()获取底层char*指针，并拷贝字符串内容
        input_names.push_back(strdup(input_name_ptr.get()));
        
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    
    // 修复循环变量类型和GetOutputName方法
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        // 使用GetOutputNameAllocated代替GetOutputName，返回类型是AllocatedStringPtr
        Ort::AllocatedStringPtr output_name_ptr = ort_session->GetOutputNameAllocated(i, allocator);
        // 使用output_name_ptr.get()获取底层char*指针，并拷贝字符串内容
        output_names.push_back(strdup(output_name_ptr.get()));
        
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    
    this->inpHeight = input_node_dims[0][1]; /// n, h, w, c
    this->inpWidth = input_node_dims[0][2];
    num_lines = this->output_node_dims[0][1];
    map_h = this->output_node_dims[2][1];
    map_w = this->output_node_dims[2][2];
}

void M_LSD::preprocess(Mat srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_AREA);

	int row = dstimg.rows;
	int col = dstimg.cols;
	this->input_image_.resize(row * col * 4);
	int k = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int c = 0; c < 3; c++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + c];
				///this->input_image_[i * col * 4 + j * 4 + c] = pix;  /// n, h, w, c
				this->input_image_[k] = pix;
				k++;
			}
			///this->input_image_[i * col * 4 + j * 4 + 3] = 1;
			this->input_image_[k] = 1;
			k++;
		}
	}
}

Mat M_LSD::detect(Mat srcimg,vector<vector<int>> &segments_list)
{
	segments_list.clear();
	if (srcimg.channels() != 3)	
	{
		cout << "Input image is not 3 channel" << endl;
		return srcimg;
	}

	const float h_ratio = float(srcimg.rows) / this->inpHeight;
	const float w_ratio = float(srcimg.cols) / this->inpWidth;
	this->preprocess(srcimg);
	array<int64_t, 4> input_shape_{ 1, this->inpHeight, this->inpWidth, 4 };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
    
																																							// post process.	
	int *pts = ort_outputs[0].GetTensorMutableData<int>();
	float *pts_score = ort_outputs[1].GetTensorMutableData<float>();
	float *vmap = ort_outputs[2].GetTensorMutableData<float>();
	
	for (int i = 0; i < num_lines; i++)
	{
		const int y = pts[i * 2];
		const int x = pts[i * 2 + 1];
		
		const float* pdata = vmap + y * map_w * 4 + x * 4;
		const float disp_x_start = pdata[0];
		const float disp_y_start = pdata[1];
		const float disp_x_end = pdata[2];
		const float disp_y_end = pdata[3];
		const float distance = sqrt(powf(disp_x_start - disp_x_end, 2) + powf(disp_y_start - disp_y_end, 2));
		if (pts_score[i] > this->conf_threshold && distance > this->dist_threshold)
		{
			const float x_start = (x + disp_x_start) * 2 * w_ratio;
			const float y_start = (y + disp_y_start) * 2 * h_ratio;
			const float x_end = (x + disp_x_end) * 2 * w_ratio;
			const float y_end = (y + disp_y_end) * 2 * h_ratio;
			double k= (y_end - y_start) / (x_end - x_start);
			double a;double b;double c;
			if(x_start==x_end){a=1;b=0;c=-x_start;}
			else{a=y_end-y_start;b=x_start-x_end;
				c=x_end*y_start-x_start*y_end;}
			vector<int> line = { int(x_start), int(y_start), int(x_end), int(y_end),a,b,c};
			segments_list.push_back(line);
		}
	}

	Mat dstimg = srcimg.clone();
	for (int i = 0; i < segments_list.size(); i++)
	{
		line(dstimg, Point(segments_list[i][0], segments_list[i][1]), Point(segments_list[i][2], segments_list[i][3]), Scalar(255, 0, 0), 2);

	}
	return dstimg;
}
