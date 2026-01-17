#include "AbsoluteOrientation.h"


AbsouteOrientation::AbsouteOrientation() {}

AbsouteOrientation::~AbsouteOrientation() {}

bool AbsouteOrientation::runCalSevenParams(std::vector<double> &targetFramePts, std::vector<double> &sourceFramePts,
                                           const size_t ptsNum, double *R, double *t, double &scale)
{
    if (targetFramePts.size() != 3 * ptsNum
        || (sourceFramePts.size() != 3 * ptsNum))
        return false;

    Point3d *tgtPts = new Point3d[ptsNum];
    Point3d *srcPts = new Point3d[ptsNum];

    for (size_t i = 0; i < ptsNum; i++)
    {
        tgtPts[i] = Point3d(targetFramePts[i * 3 + 0], targetFramePts[i * 3 + 1], targetFramePts[i * 3 + 2]);
        srcPts[i] = Point3d(sourceFramePts[i * 3 + 0], sourceFramePts[i * 3 + 1], sourceFramePts[i * 3 + 2]);
    }

    bool bOk = orientation(tgtPts, srcPts, ptsNum, R, t, scale, OrientationMethod::USE_QUATERNION);

    return bOk;
}

bool AbsouteOrientation::orientation(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num,
                                     double *rot, double *trans, double &scale, const OrientationMethod &omethod)
{
    if (pt_num < 3 || pts_ref == nullptr || pts_flt == nullptr || rot == nullptr || trans == nullptr)
    {
        return false;
    }
    //Point3d ct1, ct2;
    double cx1 = 0.0, cy1 = 0.0, cz1 = 0.0;
    double cx2 = 0.0, cy2 = 0.0, cz2 = 0.0;
    for (int i = 0; i < pt_num; i++)
    {
        cx1 += pts_ref[i].x;
        cy1 += pts_ref[i].y;
        cz1 += pts_ref[i].z;

        cx2 += pts_flt[i].x;
        cy2 += pts_flt[i].y;
        cz2 += pts_flt[i].z;
    }
    // ????
    cx1 /= pt_num;
    cy1 /= pt_num;
    cz1 /= pt_num;

    cx2 /= pt_num;
    cy2 /= pt_num;
    cz2 /= pt_num;
    Point3d *bc_pts1 = new Point3d[pt_num];
    Point3d *bc_pts2 = new Point3d[pt_num];
    for (int i = 0; i < pt_num; i++)
    {
        bc_pts1[i].x = pts_ref[i].x - cx1;
        bc_pts1[i].y = pts_ref[i].y - cy1;
        bc_pts1[i].z = pts_ref[i].z - cz1;

        bc_pts2[i].x = pts_flt[i].x - cx2;
        bc_pts2[i].y = pts_flt[i].y - cy2;
        bc_pts2[i].z = pts_flt[i].z - cz2;
    }
    double sum_sqr1 = 0.0;
    double sum_sqr2 = 0.0;
    for (int i = 0; i < pt_num; i++)
    {
        sum_sqr1 += (bc_pts1[i].x * bc_pts1[i].x + bc_pts1[i].y * bc_pts1[i].y + bc_pts1[i].z * bc_pts1[i].z);
        sum_sqr2 += (bc_pts2[i].x * bc_pts2[i].x + bc_pts2[i].y * bc_pts2[i].y + bc_pts2[i].z * bc_pts2[i].z);
    }
    scale = sqrt(sum_sqr1 / sum_sqr2);

    if (scale < 2.0)
        scale = 1.0;

    bool bret;
    switch (omethod)
    {
        case LOCAL_COOR_P3:
        {
            bret = computeRotP3(bc_pts1, bc_pts2, pt_num, rot);
            break;
        }
        case USE_QUATERNION:
        {
            bret = computeRotQuaternion(bc_pts1, bc_pts2, pt_num, rot);
            break;
        }
        case USE_ORTHO_MAT:
        {
            break;
        }
        default:
            break;
    }
    if (bret)
    {
        trans[0] = cx1 - scale * (rot[0] * cx2 + rot[1] * cy2 + rot[2] * cz2);
        trans[1] = cy1 - scale * (rot[3] * cx2 + rot[4] * cy2 + rot[5] * cz2);
        trans[2] = cz1 - scale * (rot[6] * cx2 + rot[7] * cy2 + rot[8] * cz2);
    }
    return bret;
}


bool AbsouteOrientation::computeRotP3(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num, double *rot)
{
    double max_score = 0.0; // std::numeric_limits<double>::max();
    for (int i = 0; i < pt_num - 2; i++)
    {
        for (int j = i + 1; j < pt_num - 1; j++)
        {
            float len1 = dis(pts_ref[i], pts_ref[j]);
            for (int k = j + 1; k < pt_num; k++)
            {
                float len2 = dis(pts_ref[j], pts_ref[k]);
                float len3 = dis(pts_ref[i], pts_ref[k]);
                int pi1 = i, pi2 = k, pi3 = j;
                if (len1 > len2 && len1 > len3)
                {
                    pi1 = i;
                    pi2 = j;
                    pi3 = k;
                } else if (len2 > len3)
                {
                    pi1 = j;
                    pi2 = k;
                    pi3 = i;
                }

                Point3d xaxis1 = (pts_ref[pi2] - pts_ref[pi1]);
                xaxis1.Normalize();
                Point3d yaxis1 = (pts_ref[pi3] - pts_ref[pi1]);
                float dotxy = xaxis1.Dot(yaxis1);
                yaxis1.x = yaxis1.x - dotxy * xaxis1.x;
                yaxis1.y = yaxis1.y - dotxy * xaxis1.y;
                yaxis1.z = yaxis1.z - dotxy * xaxis1.z;
                if (!yaxis1.Normalize())
                    continue;
                Point3d zaxis1 = xaxis1.Cross(yaxis1);

                Point3d xaxis2 = (pts_flt[pi2] - pts_flt[pi1]);
                xaxis2.Normalize();
                Point3d yaxis2 = (pts_flt[pi3] - pts_flt[pi1]);
                dotxy = xaxis2.Dot(yaxis2);
                yaxis2.x = yaxis2.x - dotxy * xaxis2.x;
                yaxis2.y = yaxis2.y - dotxy * xaxis2.y;
                yaxis2.z = yaxis2.z - dotxy * xaxis2.z;
                if (!yaxis2.Normalize())
                    continue;
                Point3d zaxis2 = xaxis2.Cross(yaxis2);
                double R_t[9];
                R_t[0] = (double) xaxis1.x * xaxis2.x + (double) yaxis1.x * yaxis2.x + (double) zaxis1.x * zaxis2.x;
                R_t[1] = (double) xaxis1.x * xaxis2.y + (double) yaxis1.x * yaxis2.y + (double) zaxis1.x * zaxis2.y;
                R_t[2] = (double) xaxis1.x * xaxis2.z + (double) yaxis1.x * yaxis2.z + (double) zaxis1.x * zaxis2.z;

                R_t[3] = (double) xaxis1.y * xaxis2.x + (double) yaxis1.y * yaxis2.x + (double) zaxis1.y * zaxis2.x;
                R_t[4] = (double) xaxis1.y * xaxis2.y + (double) yaxis1.y * yaxis2.y + (double) zaxis1.y * zaxis2.y;
                R_t[5] = (double) xaxis1.y * xaxis2.z + (double) yaxis1.y * yaxis2.z + (double) zaxis1.y * zaxis2.z;

                R_t[6] = (double) xaxis1.z * xaxis2.x + (double) yaxis1.z * yaxis2.x + (double) zaxis1.z * zaxis2.x;
                R_t[7] = (double) xaxis1.z * xaxis2.y + (double) yaxis1.z * yaxis2.y + (double) zaxis1.z * zaxis2.y;
                R_t[8] = (double) xaxis1.z * xaxis2.z + (double) yaxis1.z * yaxis2.z + (double) zaxis1.z * zaxis2.z;

                double sum_dot = 0.0;
                for (int n = 0; n < pt_num; n++)
                {
                    Point3d pt_trans;
                    pt_trans.x = R_t[0] * pts_flt[n].x + R_t[1] * pts_flt[n].y + R_t[2] * pts_flt[n].z;
                    pt_trans.y = R_t[3] * pts_flt[n].x + R_t[4] * pts_flt[n].y + R_t[5] * pts_flt[n].z;
                    pt_trans.z = R_t[6] * pts_flt[n].x + R_t[7] * pts_flt[n].y + R_t[8] * pts_flt[n].z;

                    sum_dot += (pts_ref[n].Dot(pt_trans));
                }
                if (sum_dot > max_score)
                {
                    max_score = sum_dot;
                    memcpy(rot, R_t, 9 * sizeof(double));
                }
            }
        }
    }
    if (max_score > 0.0)
        return true;
    return false;
}

bool AbsouteOrientation::computeRotQuaternion(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num, double *rot)
{
    if (pts_ref == nullptr || pts_flt == nullptr || pt_num < 3 || rot == nullptr)
        return false;

    double Sxx = 0.0, Sxy = 0.0, Sxz = 0.0, Syx = 0.0, Syy = 0.0, Syz = 0.0, Szx = 0.0, Szy = 0.0, Szz = 0.0;
    for (int i = 0; i < pt_num; i++)
    {
        Sxx += (pts_flt[i].x * pts_ref[i].x) / 10000;
        Sxy += (pts_flt[i].x * pts_ref[i].y) / 10000;
        Sxz += (pts_flt[i].x * pts_ref[i].z) / 10000;

        Syx += (pts_flt[i].y * pts_ref[i].x) / 10000;
        Syy += (pts_flt[i].y * pts_ref[i].y) / 10000;
        Syz += (pts_flt[i].y * pts_ref[i].z) / 10000;

        Szx += (pts_flt[i].z * pts_ref[i].x) / 10000;
        Szy += (pts_flt[i].z * pts_ref[i].y) / 10000;
        Szz += (pts_flt[i].z * pts_ref[i].z) / 10000;
    }
    double N[16];
    N[0] = Sxx + Syy + Szz;
    N[1] = Syz - Szy;
    N[2] = Szx - Sxz;
    N[3] = Sxy - Syx;

    N[4] = Syz - Szy;
    N[5] = Sxx - Syy - Szz;
    N[6] = Sxy + Syx;
    N[7] = Szx + Sxz;

    N[8] = Szx - Sxz;
    N[9] = Sxy + Syx;
    N[10] = -Sxx + Syy - Szz;
    N[11] = Syz + Szy;

    N[12] = Sxy - Syx;
    N[13] = Szx + Sxz;
    N[14] = Syz + Szy;
    N[15] = -Sxx - Syy + Szz;

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> matN(N);
    Eigen::EigenSolver<Eigen::MatrixXd> es(matN);

    int max_eval_id = 0;
    double max_eval = es.eigenvalues()[0].real();
    for (int i = 1; i < 4; i++)
    {
        if (max_eval < es.eigenvalues()[i].real())
        {
            max_eval = es.eigenvalues()[i].real();
            max_eval_id = i;
        }
    }
    //std::cout << "eigen values : \n" << es.eigenvalues() << std::endl;
    //std::cout << "eigen vectors : \n" << es.eigenvectors() << std::endl;
    Eigen::VectorXcd evec = es.eigenvectors().col(max_eval_id);
    //std::cout << "eigen vec with max eval : \n" << evec << std::endl;
    double q[4] = {evec[0].real(), evec[1].real(), evec[2].real(), evec[3].real()};
    //std::cout << std::setprecision(6) << "eigen vec: \n" << q[0] << "\t" << q[1] << "\t" << q[2] << "\t" << q[3] << std::endl;
    quaternion2Rotation(q, rot);
    double q_t[4];
    rotation2Quaternion(rot, q_t);
    //std::cout << std::setprecision(6) << "eigen vec: \n" << q_t[0] << "\t" << q_t[1] << "\t" << q_t[2] << "\t" << q_t[3] << std::endl;
/*
	for (int i = 0; i < pt_num; i++) {
		printf("REF: %.2f\t %.2f\t %.2f\n", pts_ref[i].x, pts_ref[i].y, pts_ref[i].z);
		float x = rot[0] * pts_flt[i].x + rot[1] * pts_flt[i].y + rot[2] * pts_flt[i].z;
		float y = rot[3] * pts_flt[i].x + rot[4] * pts_flt[i].y + rot[5] * pts_flt[i].z;
		float z = rot[6] * pts_flt[i].x + rot[7] * pts_flt[i].y + rot[8] * pts_flt[i].z;
		printf("TRANS: %.2f\t %.2f\t %.2f\n", x, y, z);
	}
*/
    return true;
}

bool AbsouteOrientation::computeRotOrthoMat(const Point3d *pts_ref, const Point3d *pts_flt, const int &pt_num, double *R)
{

    return true;
}