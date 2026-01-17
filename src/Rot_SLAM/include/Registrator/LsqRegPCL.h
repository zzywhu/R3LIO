//
// Created by w on 2022/9/15.
//

#ifndef SRC_LSQREGPCL_H
#define SRC_LSQREGPCL_H


enum class LSQ_OPTIMIZER_TYPE
{
    GaussNewton,
    LevenbergMarquardt
};

template<typename PointSource, typename PointTarget>
class RegistrationPCL : public pcl::Registration<PointSource, PointTarget, float>
{
public:
    using Scalar = float;
    using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

    using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
    using PointCloudSourcePtr = typename PointCloudSource::Ptr;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

    using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
    using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
    using Ptr = pcl::shared_ptr<RegistrationPCL<PointSource, PointTarget>>;
        using ConstPtr = pcl::shared_ptr<const RegistrationPCL<PointSource, PointTarget>>;
#else
    using Ptr = boost::shared_ptr<RegistrationPCL<PointSource, PointTarget>>;
    using ConstPtr = boost::shared_ptr<const RegistrationPCL<PointSource, PointTarget>>;
#endif

protected:
    using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::max_iterations_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::final_transformation_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
    using pcl::Registration<PointSource, PointTarget, Scalar>::converged_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RegistrationPCL();

    virtual ~RegistrationPCL();

    inline Eigen::Quaterniond so3_exp(const Eigen::Vector3d &omega);

    void setLMLambda(double eps);

    void setRotationEpsilon(double eps);

    void setInitialLambdaFactor(double init_lambda_factor);

    void setDebugPrint(bool lm_debug_print);

    const Eigen::Matrix<double, 6, 6> &getFinalHessian() const;

    double evaluateCost(const Eigen::Matrix4f &relative_pose, Eigen::Matrix<double, 6, 6> *H = nullptr,
                        Eigen::Matrix<double, 6, 1> *b = nullptr);

    virtual void swapSourceAndTarget()
    {}

    virtual void clearSource()
    {}

    virtual void clearTarget()
    {}

public:
    void computeTransformation(PointCloudSource &output, const Matrix4 &guess);

    bool is_converged(const Eigen::Isometry3d &delta) const;

    virtual double linearize(const Eigen::Isometry3d &trans, Eigen::Matrix<double, 6, 6> *H = nullptr,
                             Eigen::Matrix<double, 6, 1> *b = nullptr) = 0;

    virtual double compute_error(const Eigen::Isometry3d &trans) = 0;

    bool step_optimize(Eigen::Isometry3d &x0, Eigen::Isometry3d &delta);

    bool step_gn(Eigen::Isometry3d &x0, Eigen::Isometry3d &delta);

    bool step_lm(Eigen::Isometry3d &x0, Eigen::Isometry3d &delta);

protected:
    double rotation_epsilon_;

    LSQ_OPTIMIZER_TYPE lsq_optimizer_type_;
    int lm_max_iterations_;
    double lm_init_lambda_factor_;
    double lm_lambda_;
    bool lm_debug_print_;

    Eigen::Matrix<double, 6, 6> final_hessian_;
};


template<typename PointSource, typename PointTarget>
class RegistrationPCL_VGICP : virtual public RegistrationPCL<PointSource, PointTarget>
{
public:
    using Scalar = float;
    using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

    using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
    using PointCloudSourcePtr = typename PointCloudSource::Ptr;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

    using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
    using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;
public:
    void setMatchedList(std::vector<GuassinVoxelMatchInfo<PointWithCov>> &matchList){ _matchedGuassinVoxelList = matchList; }

    void setMatchedList( std::vector<MatchOctoTreeInfo> &matchList){ _matchedOctreeList = matchList; }

    bool reg(typename pcl::PointCloud<PointSource>::Ptr srcCloud,
             typename pcl::PointCloud<PointTarget>::Ptr dstCloud,
             Eigen::Matrix4d& InitTrans,
             Eigen::Matrix4d& trans,
             double voxelSize=0.5);

public:

    virtual double linearize(const Eigen::Isometry3d &trans, Eigen::Matrix<double, 6, 6> *H = nullptr, Eigen::Matrix<double, 6, 1> *b = nullptr) override;

    virtual double compute_error(const Eigen::Isometry3d &trans) override;

private:
    std::vector<GuassinVoxelMatchInfo<PointWithCov>> _matchedGuassinVoxelList;
    std::vector<MatchOctoTreeInfo> _matchedOctreeList;
};

#endif //SRC_LSQREGPCL_H
