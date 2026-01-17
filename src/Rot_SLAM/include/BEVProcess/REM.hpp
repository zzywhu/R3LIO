#include <torch/script.h>  // 包含TorchScript的头文件
#include <torch/torch.h>

#include <iostream>
// #inclue <math.h>
class NetVLAD : public torch::nn::Module {
 public:
  // 构造函数，初始化聚类数目和维度，并创建卷积层和质心
  NetVLAD(int num_clusters, int dim) : num_clusters(num_clusters), dim(dim) {
    // 定义一个1x1卷积，用于计算 soft assign 的分配权重
    conv = register_module(
        "conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(dim, num_clusters, 1).bias(false)));
    // 初始化句聚类质心(centroids)
    centroids =
        register_parameter("centroids", torch::rand({num_clusters, dim}));
  }

  // 初始化聚类质心参数,类似于python的init_params
  void init_params(const torch::Tensor &clsts,
                   const torch::Tensor &traindescs) {
    // 将聚类质心进行L2归一化
    auto clstsAssign = clsts / clsts.norm(2, 1, true);
    // 计算质心和描述符点的内积，用于估算alpha参数
    auto dots = torch::matmul(clstsAssign, traindescs.t());
    dots = std::get<0>(dots.sort(0, /*descending=*/true));

    // 计算alpha参数，确保较好的聚类分离
    auto alpha = -std::log(0.01) / (dots[0] - dots[1]).mean();
    this->alpha = alpha.item<float>();

    // 将质心和卷积权重注册到模型
    centroids = register_parameter("centroids", clsts);
    conv->weight = register_parameter(
        "conv.weight", this->alpha * clstsAssign.unsqueeze(2).unsqueeze(3));
  }

  // 前向传播函数，计算VLAD向量
  torch::Tensor forward(torch::Tensor x) {
    auto N = x.size(0);                   // batch size
    auto C = x.size(1);                   // 通道数
    auto x_flatten = x.view({N, C, -1});  // 将特征图展平

    // 计算soft assign权重(卷积操作),得到每个点的权重分配
    auto soft_assign = conv->forward(x).view({N, num_clusters, -1});
    soft_assign = torch::softmax(soft_assign, 1);

    // 初始化 VLAD 向量
    torch::Tensor vlad = torch::zeros(
        {N, num_clusters, C}, torch::dtype(x.dtype()).device(x.device()));

    // 计算每个聚类的 VLAD 描述符
    for (int i = 0; i < num_clusters; ++i) {
      // 计算残差向量 (x_flatten - 质心)
      auto residual = x_flatten.unsqueeze(0).permute({1, 0, 2, 3}) -
                      centroids[i].expand_as(x_flatten).unsqueeze(0);
      // 残差乘以 soft assign 权重
      residual *= soft_assign.select(1, i).unsqueeze(2);
      // 对每个聚类求和，得到 VLAD 向量
      vlad.select(1, i) = residual.sum(-1);
    }

    // intra-normalization 归一化每个聚类的描述符
    // vlad = torch::nn::functional::normalize(vlad, 2, 2);
    vlad = torch::nn::functional::normalize(
        vlad, torch::nn::functional::NormalizeFuncOptions().p(2).dim(2));
    // 将 VLAD 展平为一维向量
    vlad = vlad.view({N, -1});
    // 再次进行 L2 归一化
    // vlad = torch::nn::functional::normalize(vlad, 2, 1);
    vlad = torch::nn::functional::normalize(
        vlad, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    return vlad;
  }

 private:
  int num_clusters;                 // 聚类数
  int dim;                          // 输入数据的维度
  torch::nn::Conv2d conv{nullptr};  // 1x1卷积层
  torch::Tensor centroids;          // 聚类质心
  float alpha;                      // 聚类中心的权重缩放因子
};

// REM 实现，主要用于图像的旋转和特征提取
class REM : public torch::nn::Module {
 public:
  // 构造函数，from_scratch 用于是否从头训练，rotations 指旋转的角度数
  REM(bool from_scratch, int rotations) : rotations(rotations) {
    // 使用预训练的 ResNet34 作为特征提取器
    // change to your path
    torch::jit::script::Module encoder;
    encoder = torch::jit::load(
        "/home/zzy/SLAM/mri-mms/src/Rot_intensity_augm_LIVO/weights/"
        "resnet34_weights.pth");
    // auto encoder =
    // torchvision::models::resnet34(torchvision::models::ResNet34_Weights::IMAGENET1K_V1);
    // auto encoder = torch::vision::models::resnet34(pretrained=!from_scratch);
    // 仅保留ResNet的前几层作为编码器
    // for (int i = 0; i < encoder->children().size() - 4; ++i) {
    //     this->encoder->push_back(encoder->children()[i]);
    // }
    std::vector<torch::jit::IValue> inputs;
    auto children = encoder.named_children();

    int num_layers_to_keep = children.size() - 4;
    int count = 0;
    // 创建一个Sequential容器，用于存储需要保留的层
    // torch::nn::Sequential encoder_sequential;

    // // 将处理后的层注册为模块的子模块
    // register_module("encoder", encoder);
    for (const auto &child : children) {
      if (count < num_layers_to_keep) {
        auto string1 = child.name;
        encoder.register_module(string1, child.value);
      }
      count++;
    }

    // 计算每个旋转的角度 (弧度表示)
    angles = -torch::arange(0, 359.00001, 360.0 / rotations) / 180.0 * M_PI;
  }

  // 前向传播，输入为x，输出为旋转特征
  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
    std::vector<torch::Tensor> equ_features;  // 用于存储每个旋转的特征
    auto batch_size = x.size(0);              // 获取批量大小
                                              // 初始化im1_init_size notice:
                                              // auto options =
    // torch::nn::functional::GridSampleFuncOptions().align_corners(true).mode(torch::kBicubic);
    auto options =
        torch::nn::functional::GridSampleFuncOptions().align_corners(true).mode(
            torch::kBilinear);
    torch::IntArrayRef im1_init_size;
    for (int i = 0; i < angles.size(0); ++i) {
      // 创建旋转矩阵（使用负角度进行输入图像的旋转）
      auto aff = torch::zeros({batch_size, 2, 3}).to(x.device());
      aff.select(2, 0).select(1, 0) = torch::cos(-angles[i]);
      aff.select(2, 0).select(1, 1) = torch::sin(-angles[i]);
      aff.select(2, 1).select(1, 0) = -torch::sin(-angles[i]);
      aff.select(2, 1).select(1, 1) = torch::cos(-angles[i]);

      // 根据旋转矩阵生成新的采样网格
      auto grid = torch::nn::functional::affine_grid(aff, x.sizes(),
                                                     /*align_corners=*/true);
      // 对图像进行旋转采样
      // auto warped_im = torch::nn::functional::grid_sample(x, grid,
      // /*align_corners=*/true, /*mode=*/torch::kBicubic);
      // 确保使用正确定义的选项
      // auto options =
      // torch::nn::functional::GridSampleFuncOptions().align_corners(true); //
      // 使用枚举类型

      auto warped_im = torch::nn::functional::grid_sample(x, grid, options);

      // 提取特征
      auto out = encoder_sequential->forward(warped_im);
      // 第一次计算时，记录输出的初始尺寸
      if (i == 0) {
        auto im1_init_size = out.sizes();
      }

      // 创建逆旋转矩阵（使用正角度）
      auto aff_inv = torch::zeros({batch_size, 2, 3}).to(x.device());
      aff_inv.select(2, 0).select(1, 0) = torch::cos(angles[i]);
      aff_inv.select(2, 0).select(1, 1) = torch::sin(angles[i]);
      aff_inv.select(2, 1).select(1, 0) = -torch::sin(angles[i]);
      aff_inv.select(2, 1).select(1, 1) = torch::cos(angles[i]);

      // 根据逆旋转矩阵生成新的采样网格
      auto grid_inv = torch::nn::functional::affine_grid(
          aff_inv, im1_init_size, /*align_corners=*/true);
      // 将特征回旋到原始位置
      // auto warped_out = torch::nn::functional::grid_sample(out, grid_inv,
      // /*align_corners=*/true, /*mode=*/torch::kBicubic);
      auto warped_out =
          torch::nn::functional::grid_sample(out, grid_inv, options);
      // 将当前旋转后的特征添加到列表中
      equ_features.push_back(warped_out.unsqueeze(-1));
    }

    // 将所有旋转后的特征拼接起来
    auto equ_features_tensor = torch::cat(equ_features, -1);  // B C H W R
    // 对拼接后的特征在最后一个维度取最大值
    // auto equ_max = torch::max(equ_features_tensor, -1,
    // /*keepdim=*/false).values;
    auto equ_max =
        std::get<0>(torch::max(equ_features_tensor, -1, /*keepdim=*/false));

    // 创建单位矩阵以用于上采样
    auto aff_identity = torch::zeros({batch_size, 2, 3}).to(x.device());
    aff_identity.select(2, 0).select(1, 0) = 1;
    aff_identity.select(2, 0).select(1, 1) = 0;
    aff_identity.select(2, 1).select(1, 0) = 0;
    aff_identity.select(2, 1).select(1, 1) = 1;

    // 对 equ_max 进行上采样以得到 out1
    // auto grid1 = torch::nn::functional::affine_grid(aff_identity,
    // {batch_size, equ_max.size(1), x.size(2) / 4, x.size(3) / 4},
    // /*align_corners=*/true);
    auto grid1 = torch::nn::functional::affine_grid(
        aff_identity,
        torch::IntArrayRef(
            {batch_size, equ_max.size(1), x.size(2) / 4, x.size(3) / 4}),
        /*align_corners=*/true);
    // auto out1 = torch::nn::functional::grid_sample(equ_max, grid1,
    // /*align_corners=*/true, /*mode=*/torch::kBicubic);
    auto out1 = torch::nn::functional::grid_sample(equ_max, grid1, options);
    out1 = torch::nn::functional::normalize(
        out1, /*dim=*/torch::nn::functional::NormalizeFuncOptions().dim(
            1));  // 归一化 out1

    // 对 equ_max 进行上采样以得到 out2
    auto grid2 = torch::nn::functional::affine_grid(
        aff_identity, {batch_size, equ_max.size(1), x.size(2), x.size(3)},
        /*align_corners=*/true);
    // auto out2 = torch::nn::functional::grid_sample(equ_max, grid2,
    // /*align_corners=*/true, /*mode=*/torch::kBicubic);
    auto out2 = torch::nn::functional::grid_sample(equ_max, grid2, options);
    out2 = torch::nn::functional::normalize(
        out2, /*dim=*/torch::nn::functional::NormalizeFuncOptions().dim(
            1));  // 归一化 out2

    return std::make_tuple(out1, out2);
  }

 private:
  int rotations;         // 旋转的次数
  torch::Tensor angles;  // 存储所有旋转角度
  torch::nn::Sequential encoder_sequential{
      nullptr};  // 编码器模型 (ResNet34的部分)
};

// REIN 模型实现
class REIN : public torch::nn::Module {
 public:
  // 构造函数
  REIN() {
    // 初始化 REM 和 NetVLAD 模块
    rem = register_module("rem", std::make_shared<REM>(false, 8));
    pooling = register_module("pooling", std::make_shared<NetVLAD>(64, 128));
  }

  // 前向传播
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor x) {
    // 通过 REM 获取旋转不变特征和局部特征
    auto rem_out = rem->forward(x);  // REM 返回两个张量
    torch::Tensor out1 = std::get<0>(rem_out);
    torch::Tensor local_feats = std::get<1>(rem_out);

    // 通过 NetVLAD 进行全局描述符聚类池化
    torch::Tensor global_desc = pooling->forward(out1);

    // 返回局部特征、全局特征和旋转不变特征
    return std::make_tuple(out1, local_feats, global_desc);
    // todo 如果张量是在 GPU 上，你需要先将其复制到 CPU
    // 上，然后再进行转换。可以使用 .to(torch::kCPU) 方法来完成这一操作。
  }

 private:
  // REM 模块，提取旋转不变特征
  std::shared_ptr<REM> rem;

  // NetVLAD 模块，进行 VLAD 聚类池化
  std::shared_ptr<NetVLAD> pooling;
};

// TORCH_MODULE(NetVLAD);  // 用于注册模块
// TORCH_MODULE(REM);  // 用于注册模块
// TORCH_MODULE(REIN);  // 用于注册模块