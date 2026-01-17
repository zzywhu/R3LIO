// created by:  zhiyu zhou 2025/3/2
#ifndef IMG_H
#define IMG_H
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <opencv2/ximgproc.hpp>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Mapper/VoxelMapper.h"
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <tuple>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "Common.h"
#include <mutex>
#include <pcl/features/boundary.h>
#include <thread>
#include "MLSD/mlsd.h"
#include <boost/circular_buffer.hpp>
#include <ImageProcess/depthprocesser.h>


struct Matchlinelist
    {
      double k;
      pcl::PointXYZI p3d;
      std::vector<int> p2d;
      std::vector<int> linestart;
      std::vector<int> lineend;
      double a,b,c;
      double dist;
    };
class imgProcesser
    {
    public:
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _sparsecloudbuffer;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _sparsecloudbuffer_left;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _sparsecloudbuffer_right;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _densecloudbuffer;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _densecloudbuffer_left;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> _densecloudbuffer_right;
    std::vector<cv::Mat> _denseimgbuffer;
    std::vector<cv::Mat> _denseimgbuffer_left;
    std::vector<cv::Mat> _denseimgbuffer_right;
    boost::circular_buffer<std::vector<Matchlinelist>> _matchlinelistbuffer;
    boost::circular_buffer<std::vector<Matchlinelist>> _matchlinelistbuffer_left;
    boost::circular_buffer<std::vector<Matchlinelist>> _matchlinelistbuffer_right;
    boost::circular_buffer<pcl::PointCloud<pcl::PointXYZI>::Ptr> _sparselinecloudbuffer;
    boost::circular_buffer<pcl::PointCloud<pcl::PointXYZI>::Ptr> _sparselinecloudbuffer_left;
    boost::circular_buffer<pcl::PointCloud<pcl::PointXYZI>::Ptr> _sparselinecloudbuffer_right;
    boost::circular_buffer<std::vector<std::vector<int>>>_linelistbuffer;
    boost::circular_buffer<std::vector<std::vector<int>>>_linelistbuffer_left;
    boost::circular_buffer<std::vector<std::vector<int>>>_linelistbuffer_right;
    boost::circular_buffer<cv::Mat> _matchimgbuffer;
    boost::circular_buffer<cv::Mat> _matchimgbuffer_left;
    boost::circular_buffer<cv::Mat> _matchimgbuffer_right;
    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense;
    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense_left;
    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityMapdense_right;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cursparsecloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr sparselinecloud;
    cv::Mat _matchImg;
    cv::Mat _matchImg_left;
    cv::Mat _matchImg_right;
    cv::Mat curdenseimg;
    cv::Mat curdenseimg_left;
    cv::Mat curdenseimg_right;
    cv::Mat cannyimg;
    cv::Mat cannyimg_left;
    cv::Mat cannyimg_right;
    int frame;
    double _sita;
    std::mutex _mutexBuf;
    Eigen::Matrix3d Rleft;
    Eigen::Matrix3d Rright;
    M_LSD _lsd;


    std::thread* _imgThread;
    std::thread* _imgThread_left;
    std::thread* _imgThread_right;
    struct Plane
    {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointXYZRGB p_center;
    Eigen::Vector3d normal;
    int index;
    };
    imgProcesser():_lsd(std::string(ROOT_DIR)+"weights/model_320x320_tiny.onnx"),
    intensityMapdense(new pcl::PointCloud<pcl::PointXYZI>),
    intensityMapdense_left(new pcl::PointCloud<pcl::PointXYZI>),
    intensityMapdense_right(new pcl::PointCloud<pcl::PointXYZI>),
    cursparsecloud(new pcl::PointCloud<pcl::PointXYZI>),
    sparselinecloud(new pcl::PointCloud<pcl::PointXYZI>),
    _imgThread(nullptr),
    _imgThread_left(nullptr),
    _imgThread_right(nullptr),
    _matchlinelistbuffer(1),
    _matchlinelistbuffer_left(1),
    _matchlinelistbuffer_right(1),
    _matchimgbuffer(1),
    _matchimgbuffer_left(1),
    _matchimgbuffer_right(1),
    _linelistbuffer(1),
    _linelistbuffer_left(1),
    _linelistbuffer_right(1),
    _sparselinecloudbuffer(1)
    {
        frame=0;
        _sita=70*M_PI/180,
        Rleft<<cos(_sita),-sin(_sita),0,sin(_sita),cos(_sita),0,0,0,1;
        Rright<<cos(-_sita),-sin(-_sita),0,sin(-_sita),cos(-_sita),0,0,0,1;
    }
    bool startThread();
    bool startThread_left();
    bool startThread_right();
    void run();
    void run_left();
    void run_right();
    void inputframedata(pcl::PointCloud<pcl::PointXYZI>::Ptr sparsecloud,cv::Mat curdenseimg,cv::Mat curdenseimg_left,cv::Mat curdenseimg_right);
    cv::Mat visualimg(cv::Mat img,std::vector<Matchlinelist>& matchlinelist);
    void transCloud3(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudIn, 
        pcl::PointCloud<pcl::PointXYZI>::Ptr &cloudOut,
        const Eigen::Matrix3d &Rol, 
        const Eigen::Vector3d &tol);
    void buildmatchlinelist(std::vector<Matchlinelist>& matchlinelist, pcl::PointCloud<pcl::PointXYZI>::Ptr& _sparselinecloud, std::vector<std::vector<int>> segments_list);
    void buildmatchlinelist_left(std::vector<Matchlinelist>& matchlinelist, pcl::PointCloud<pcl::PointXYZI>::Ptr& _sparselinecloud, std::vector<std::vector<int>> segments_list);
    void buildmatchlinelist_right(std::vector<Matchlinelist>& matchlinelist, pcl::PointCloud<pcl::PointXYZI>::Ptr& _sparselinecloud, std::vector<std::vector<int>> segments_list);
    cv::Mat projectPinhole(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, bool isinter);
    cv::Mat projectDepth(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    cv::Mat fillHolesFast(const cv::Mat& input);
    cv::Mat ultraFastGrayToRGB(const cv::Mat& grayImage);
    cv::Mat enhanceSubtleChangesFast(const cv::Mat& grayImage);
    std::tuple<cv::Mat, cv::Mat, cv::Mat> projectPinholeall(
        pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        const Eigen::Matrix3d& Rleft,
        const Eigen::Matrix3d& Rright,
        bool isinter);
    void cannyEdgeDetection(const cv::Mat& src, cv::Mat& dst);
    void extractIntensityEdgesOptimized(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, 
        pcl::PointCloud<pcl::PointXYZI>::Ptr& intensity_edge);
    cv::Mat densifyLidarPointCloud(const cv::Mat& inputImage, const cv::Mat& templateImage);
    cv::Mat interpolateLidarImage(const cv::Mat& sparse_image);
    void removeLines(cv::Mat& img);
    void filterBrightness(cv::Mat& img);
    cv::Mat projectToXZ(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,cv::Mat scanlineIdMap,bool isinter) ;
    cv::Mat overlayRedOnGrayscale(const cv::Mat& gray1, const cv::Mat& gray2);
    cv::Mat stackImagesVertical(const cv::Mat& gray1, const cv::Mat& gray2) ;
    cv::Mat draw_matches(cv::Mat &ref_points, cv::Mat &dst_points, cv::Mat &img1, cv::Mat &img2);
    Eigen::Vector2d projectTosingle(pcl::PointXYZI p);
    void calcLineLin(std::vector<Plane>& plane_list, std::vector<pcl::PointCloud<pcl::PointXYZI>>& line_cloud_list);
    void initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        const float voxel_size,
        std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map);
    void extractLineFeatures(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,pcl::PointCloud<pcl::PointXYZI>::Ptr& _sparselinecloud
        );
    void LiDAREdgeExtraction(std::unordered_map<VOXEL_LOC, Voxel*>& voxel_map, const float ransac_dis_thre, const int plane_size_threshold,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& lidar_line_cloud_3d);
        void FitPlaneSVD(Plane& single_plane)//????????????????棬????????????????
        {
            // 1??????????	// 2???????
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_buf(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(single_plane.cloud, *cloud_buf);
            Eigen::MatrixXf cloudMat = cloud_buf->getMatrixXfMap(3, 4, 0);
            Eigen::VectorXf centroid(cloudMat.rows());
            for (int i = 0; i < cloudMat.rows(); i++)
            {
                centroid(i) = cloudMat.row(i).mean();
                cloudMat.row(i) = cloudMat.row(i) - (Eigen::VectorXf::Ones(cloudMat.cols()).transpose() * centroid(i));
            }
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(cloudMat, Eigen::ComputeFullU);
            Eigen::MatrixXf U = svd.matrixU();
            Eigen::VectorXf S = svd.singularValues();
        
            // std::cout<<std::endl<<U.col(2)<<std::endl;//model.coefficients.topRows(3).dot(centroid);
            // std::cout<<U.col(2).dot(centroid)<<std::endl;
            // std::cout<<single_plane.p_center<<std::endl;
            // std::cout<<centroid<<std::endl;
        
            single_plane.normal[0] = U.col(2)[0];
            single_plane.normal[1] = U.col(2)[1];
            single_plane.normal[2] = U.col(2)[2];
            single_plane.p_center.getVector3fMap() = centroid;
        }

        void mergePlanes(std::vector<Plane>& PlaneList)
        {
            int CurrentLength = PlaneList.size();
            int BaseIndex = 0;
        
            while (1)
            {
                CurrentLength = PlaneList.size();
                if (BaseIndex >= CurrentLength)
                {
                    break;
                }
                Eigen::Vector3d normal = PlaneList[BaseIndex].normal;
                pcl::PointXYZRGB center_PCL = PlaneList[BaseIndex].p_center;
                Eigen::Vector3d center(center_PCL.x, center_PCL.y, center_PCL.z);
                PlaneList[BaseIndex].index = BaseIndex;
        
                int key = 0;
                while (1)
                {
                    if (key + BaseIndex + 1 >= CurrentLength)
                        break;
                    Eigen::Vector3d normal_t = PlaneList[key + BaseIndex + 1].normal;
                    pcl::PointXYZRGB center_tPCL = PlaneList[key + BaseIndex + 1].p_center;
                    Eigen::Vector3d center_t(center_tPCL.x, center_tPCL.y, center_tPCL.z);
        
                    float D_t = -(normal_t.dot(center_t));//second plane ABCD
                    float cosAngle = abs(normal.dot(normal_t));
                    float Distance = abs(normal_t.dot(center) + D_t);
        
                    if ((cosAngle > 0.7) && (Distance < 0.05))//0.1
                    {
                        pcl::copyPointCloud(PlaneList[BaseIndex].cloud + PlaneList[key + BaseIndex + 1].cloud, PlaneList[BaseIndex].cloud);
                        FitPlaneSVD(PlaneList[BaseIndex]);
                        PlaneList.erase(PlaneList.begin() + key + BaseIndex + 1);
                        // cout<<"fit"<<BaseIndex<<" "<<key + BaseIndex + 1;
                        CurrentLength = PlaneList.size();
                        // PlaneList[index]
                    }
                    else
                    {
                        // cout<<"pass"<<BaseIndex<<" "<<key + BaseIndex + 1;
                        key++;
                    }
                }
                PlaneList[BaseIndex].index = BaseIndex;
                BaseIndex++;
            }
        }
        
        
        void savePlanes(std::vector<Plane> plane_list, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc)
        {
            for (int i = 0; i < plane_list.size(); i++)
            {
                std::vector<unsigned int> colors;
                colors.push_back(static_cast<unsigned int>(rand() % 256));
                colors.push_back(static_cast<unsigned int>(rand() % 256));
                colors.push_back(static_cast<unsigned int>(rand() % 256));
                for (size_t j = 0; j < plane_list[i].cloud.points.size(); j++)
                {
                    //plane_list[i].cloud.points[j].g
                    pcl::PointXYZRGB p;
                    p.getArray3fMap() = plane_list[i].cloud.points[j].getArray3fMap();
                    p.r = colors[0];
                    p.g = colors[1];
                    p.b = colors[2];
                    pc->push_back(p);
                }
            }
        }
    
    private:
        int size_x;
        int size_y;
        DepthMapProcessor _depth_processor;
        int img_width;
        int img_height;
        float fx, fy, cx, cy; // 相机内参

    };

     #endif