//
// Created by w on 2022/11/22.
//

#ifndef FAST_LIO_WIN_CLOUDINFO_H
#define FAST_LIO_WIN_CLOUDINFO_H

#include <string>
#include <vector>
#include <map>

#include "Common.h"


template <class ContainerAllocator>
struct cloud_info
{
    typedef cloud_info<ContainerAllocator> Type;

    cloud_info()
            : header()
            , startRingIndex()
            , endRingIndex()
            , pointColInd()
            , pointRange()
            , imuAvailable(0)
            , odomAvailable(0)
            , imuRollInit(0.0)
            , imuPitchInit(0.0)
            , imuYawInit(0.0)
            , initialGuessX(0.0)
            , initialGuessY(0.0)
            , initialGuessZ(0.0)
            , initialGuessRoll(0.0)
            , initialGuessPitch(0.0)
            , initialGuessYaw(0.0)
            , cloud_deskewed()
            , cloud_corner()
            , cloud_surface()
            , key_frame_cloud()
            , key_frame_color()
            , key_frame_poses()
            , key_frame_map()  {
    }
    cloud_info(const ContainerAllocator& _alloc)
            : header(_alloc)
            , startRingIndex(_alloc)
            , endRingIndex(_alloc)
            , pointColInd(_alloc)
            , pointRange(_alloc)
            , imuAvailable(0)
            , odomAvailable(0)
            , imuRollInit(0.0)
            , imuPitchInit(0.0)
            , imuYawInit(0.0)
            , initialGuessX(0.0)
            , initialGuessY(0.0)
            , initialGuessZ(0.0)
            , initialGuessRoll(0.0)
            , initialGuessPitch(0.0)
            , initialGuessYaw(0.0)
            , cloud_deskewed(_alloc)
            , cloud_corner(_alloc)
            , cloud_surface(_alloc)
            , key_frame_cloud(_alloc)
            , key_frame_color(_alloc)
            , key_frame_poses(_alloc)
            , key_frame_map(_alloc)  {
        (void)_alloc;
    }



    typedef  double _header_type;
    _header_type header;

    typedef std::vector<int32_t, typename ContainerAllocator::template rebind<int32_t>::other >  _startRingIndex_type;
    _startRingIndex_type startRingIndex;

    typedef std::vector<int32_t, typename ContainerAllocator::template rebind<int32_t>::other >  _endRingIndex_type;
    _endRingIndex_type endRingIndex;

    typedef std::vector<int32_t, typename ContainerAllocator::template rebind<int32_t>::other >  _pointColInd_type;
    _pointColInd_type pointColInd;

    typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _pointRange_type;
    _pointRange_type pointRange;

    typedef int64_t _imuAvailable_type;
    _imuAvailable_type imuAvailable;

    typedef int64_t _odomAvailable_type;
    _odomAvailable_type odomAvailable;

    typedef float _imuRollInit_type;
    _imuRollInit_type imuRollInit;

    typedef float _imuPitchInit_type;
    _imuPitchInit_type imuPitchInit;

    typedef float _imuYawInit_type;
    _imuYawInit_type imuYawInit;

    typedef float _initialGuessX_type;
    _initialGuessX_type initialGuessX;

    typedef float _initialGuessY_type;
    _initialGuessY_type initialGuessY;

    typedef float _initialGuessZ_type;
    _initialGuessZ_type initialGuessZ;

    typedef float _initialGuessRoll_type;
    _initialGuessRoll_type initialGuessRoll;

    typedef float _initialGuessPitch_type;
    _initialGuessPitch_type initialGuessPitch;

    typedef float _initialGuessYaw_type;
    _initialGuessYaw_type initialGuessYaw;

    typedef  PointCloudXYZI _cloud_deskewed_type;
    _cloud_deskewed_type cloud_deskewed;

    typedef  PointCloudXYZI  _cloud_corner_type;
    _cloud_corner_type cloud_corner;

    typedef  PointCloudXYZI  _cloud_surface_type;
    _cloud_surface_type cloud_surface;

    typedef  PointCloudXYZI  _key_frame_cloud_type;
    _key_frame_cloud_type key_frame_cloud;

    typedef  PointCloudXYZI  _key_frame_color_type;
    _key_frame_color_type key_frame_color;

    typedef  PointCloudXYZI  _key_frame_poses_type;
    _key_frame_poses_type key_frame_poses;

    typedef  PointCloudXYZI  _key_frame_map_type;
    _key_frame_map_type key_frame_map;





    typedef boost::shared_ptr< cloud_info<ContainerAllocator> > Ptr;
    typedef boost::shared_ptr< cloud_info<ContainerAllocator> const> ConstPtr;

}; // struct cloud_info

typedef cloud_info<std::allocator<void> > CloudInfo;

typedef boost::shared_ptr< CloudInfo > CloudInfoPtr;
typedef boost::shared_ptr< CloudInfo const> CloudInfoConstPtr;





#endif //FAST_LIO_WIN_CLOUDINFO_H
