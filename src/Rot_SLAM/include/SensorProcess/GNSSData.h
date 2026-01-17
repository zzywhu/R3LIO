//
// Created by w on 2023/11/1.
//

#ifndef RIGELSLAM_ROT_GNSSDATA_H
#define RIGELSLAM_ROT_GNSSDATA_H
#include <boost/make_shared.hpp>

namespace SensorMsgs
{
    struct GNSSData
    {
        typedef GNSSData Type;

        GNSSData(): header(),
                    pos(),
                    covariance()
        {
            pos.setZero();
            covariance.setZero();
        }

        GNSSData(const GNSSData& gnssData): header(gnssData.header),
                                            pos(gnssData.pos),
                                            covariance(gnssData.covariance)
        {}


        double header;

        typedef  Eigen::Matrix<double,3,1> _pose_type;
        _pose_type pos;

        typedef Eigen::Matrix<double,36,1> _covariance_type;
        _covariance_type covariance;


        typedef boost::shared_ptr< ::SensorMsgs::GNSSData> Ptr;
        typedef boost::shared_ptr< ::SensorMsgs::GNSSData> ConstPtr;

    }; // struct Imu_

    typedef ::SensorMsgs::GNSSData GNSS;

    typedef boost::shared_ptr< ::SensorMsgs::GNSSData > GNSSPtr;
    typedef boost::shared_ptr< ::SensorMsgs::GNSSData const> GNSSConstPtr;

// constants requiring out of line definition


} // namespace SensorMsgs
#endif //RIGELSLAM_ROT_GNSSDATA_H
