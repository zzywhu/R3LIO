//
// Created by w on 2020/3/7.
//

#ifndef READWRITER_H
#define READWRITER_H

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#include <io.h>
#include <sys/stat.h>
#include <direct.h>
#else

#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#endif

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>
#include <vector>
#include <deque>

#include "Common.h"

#ifdef __SHARED_LIBS__
#ifdef __DLL_EXPORTS__
#define READWRITER_API __declspec(dllexport)
#else
#define READWRITER_API __declspec(dllimport)
#endif
#else
#define READWRITER_API
#endif

// class  ReadWriter
class READWRITER_API ReadWriter
{
public:
    struct Comparator
    {
        bool operator()(std::string str1, std::string str2) const
        { return str1 < str2; }
    };

    typedef std::map<std::string, std::string, Comparator> PathStrMap;
    typedef std::map<double, std::string> PathTimeMap;
    typedef std::map<int, std::string> PathIndexMap;
    typedef std::vector<std::string> StrList;

public:
    static FILE *efopen(const char *file, char *mode);

    static bool getFileListFromDir(const std::string &dir, PathIndexMap &filePathList);

    static bool getFileListFromDir(const std::string &dir, PathStrMap &filePathList);

    static int getFileListFromDir(const std::string &dir, PathStrMap &filePathList, const std::string &extStr);

    static int getFileListFromDir(const std::string &dir, PathTimeMap &filePathList, const std::string &extStr);

    static int getDirListFromDir(std::string &dir, PathIndexMap &dirList);

    static int getDirListFromDir(std::string &dir, PathStrMap &dirList);

    static int getDirWithSubName(const PathStrMap &dirList, PathStrMap &dstDirList, const std::string &subName);

    static bool loadImageList(std::string imgDir, StrList &leftImgPathList, StrList &rightImgPathList);

    static bool parseIMUFile(const std::string& imuFilePath, std::deque<SensorMsgs::ImuData::Ptr>& imuBufferAll);

    static bool parseMotorFile(const std::string& motorFilePath, std::list<std::pair<double, double>>& motorAngleBufAll);

    static bool outputCloudFile(const std::string& filePath, PointCloudXYZI& cloud);

    static int cleanDir(const std::string& dir_full_path);

#ifndef WIN32
    static bool remove(const std::string& file_name);
#endif

private:
    static bool readLine(FILE *infile, char *buffer, const int &len);

private:
    static FILE *_INFILE;
    static FILE *_DFILE;
};

#endif // READWRITER_H
