//
// Created by w on 2022/10/19.
//

#ifndef FAST_LIO_DRAWERPCL_H
#define FAST_LIO_DRAWERPCL_H
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define DRAWER_PCL_API __declspec(dllexport)
#else
#define DRAWER_PCL_API __declspec(dllimport)
#endif
#else
#define DRAWER_PCL_API
#endif

class DRAWER_PCL_API DrawerPCL
{
public:
    DrawerPCL(){}
    ~DrawerPCL(){}

};

#endif //FAST_LIO_DRAWERPCL_H
