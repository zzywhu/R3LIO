//
// Created by w on 2020/3/6.
//

#include "ReadWriter.h"

FILE *ReadWriter::_INFILE = nullptr;
FILE *ReadWriter::_DFILE = stderr;

const double ANG_TO_RAD = M_PI / 180.;

#ifndef _WIN32
#define _MAX_PATH 260
#define _MAX_DRIVE 3
#define _MAX_DIR 256
#define _MAX_FNAME 256
#define _MAX_EXT 256

static void _split_whole_name(const char *whole_name, char *fname, char *ext)
{
    char *p_ext;

    p_ext = rindex(const_cast<char *>(whole_name), '.');
    if (NULL != p_ext)
    {
        strcpy(ext, p_ext);
        snprintf(fname, p_ext - whole_name + 1, "%s", whole_name);
    }
    else
    {
        ext[0] = '\0';
        strcpy(fname, whole_name);
    }
}

void _splitpath(const char *path, char *drive, char *dir, char *fname, char *ext)
{
    char *p_whole_name;

    drive[0] = '\0';
    if (NULL == path)
    {
        dir[0] = '\0';
        fname[0] = '\0';
        ext[0] = '\0';
        return;
    }

    if ('/' == path[strlen(path)])
    {
        strcpy(dir, path);
        fname[0] = '\0';
        ext[0] = '\0';
        return;
    }

    p_whole_name = rindex(const_cast<char *>(path), '/');
    if (NULL != p_whole_name)
    {
        p_whole_name++;
        _split_whole_name(p_whole_name, fname, ext);

        snprintf(dir, p_whole_name - path, "%s", path);
    }
    else
    {
        _split_whole_name(path, fname, ext);
        dir[0] = '\0';
    }
}

#endif

FILE *ReadWriter::efopen(const char *file, char *mode)
{
    FILE *fp;
    if ((fp = fopen(file, mode)) != NULL)
        return fp;
    fprintf(_DFILE, "couldn't open file %s mode %s\n", file, mode);
    return NULL;
}

bool ReadWriter::readLine(FILE *infile, char *buffer, const int &len)
{
    fgets(buffer, len, infile);

    while (isspace(*buffer))
    {
        if (*buffer == '#' || *buffer == '\0' || *buffer == '\n')
            return false;
        buffer++;
    }
    return true;
}

#ifdef _WIN32

bool ReadWriter::getFileListFromDir(const std::string &dir, PathStrMap &filePathList) {
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    if (dir.empty())
        return false;
    char buffer[_MAX_PATH + 100] = {0};
    __finddata64_t fileinfo;

    std::string strFullName = dir;
    if (strFullName.back() != '/')
        strFullName += '/';
    strFullName += "*";

    intptr_t findRet = _findfirst64(strFullName.c_str(), &fileinfo);
    if (-1 != findRet) {
        do {

            if (stricmp(fileinfo.name, ".") == 0 || 0 == stricmp(fileinfo.name, "..")) {
                continue;
            }

            if (fileinfo.attrib & _A_SYSTEM || fileinfo.attrib & _A_HIDDEN) {
                continue;
            }

            if (fileinfo.attrib & _A_SUBDIR) {
                continue;
            } else {

                _ctime64_s(buffer, _countof(buffer), &fileinfo.time_write);

                std::string fileName = fileinfo.name;

                _splitpath(fileName.c_str(), drive, secDir, fname, ext);
                // if(strcmp(ext, ".bmp"))
                // continue;

                std::string filePath;
                if (dir.back() != '/')
                    filePath = dir + '/' + fileName;
                else
                    filePath = dir + fileName;

                filePathList.insert(std::make_pair(fname, filePath));
            }
        } while (-1 != _findnext64(findRet, &fileinfo));

        _findclose(findRet);
    }
    return true;
}

bool ReadWriter::getFileListFromDir(const std::string &dir, PathIndexMap &filePathList) {
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    if (dir.empty())
        return false;
    char buffer[_MAX_PATH + 100] = {0};
    __finddata64_t fileinfo;

    std::string strFullName = dir;
    if (strFullName.back() != '/')
        strFullName += '/';
    strFullName += "*";

    intptr_t findRet = _findfirst64(strFullName.c_str(), &fileinfo);
    if (-1 != findRet) {
        do {

            if (stricmp(fileinfo.name, ".") == 0 || 0 == stricmp(fileinfo.name, "..")) {
                continue;
            }

            if (fileinfo.attrib & _A_SYSTEM || fileinfo.attrib & _A_HIDDEN) {
                continue;
            }

            if (fileinfo.attrib & _A_SUBDIR) {
                continue;
            } else {

                _ctime64_s(buffer, _countof(buffer), &fileinfo.time_write);

                std::string fileName = fileinfo.name;

                _splitpath(fileName.c_str(), drive, secDir, fname, ext);
                /*	if(strcmp(ext, ".bmp")&& strcmp(ext, ".jpg")&& strcmp(ext, ".png"))
                                continue;*/

                std::string filePath;
                if (dir.back() != '/')
                    filePath = dir + '/' + fileName;
                else
                    filePath = dir + fileName;

                int number = atoi(fname);
                filePathList.insert(std::make_pair(number, filePath));
            }
        } while (-1 != _findnext64(findRet, &fileinfo));

        _findclose(findRet);
    }
    return true;
}

int ReadWriter::getFileListFromDir(const std::string &dir, PathStrMap &filePathList, const std::string &extStr) {
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    if (dir.empty())
        return 0;
    char buffer[_MAX_PATH + 100] = {0};
    __finddata64_t fileinfo;

    std::string strFullName = dir;
    if (strFullName.back() != '/')
        strFullName += '/';
    strFullName += "*";

    intptr_t findRet = _findfirst64(strFullName.c_str(), &fileinfo);
    if (-1 != findRet) {
        do {
            if (stricmp(fileinfo.name, ".") == 0 || 0 == stricmp(fileinfo.name, "..")) {
                continue;
            }
            if (fileinfo.attrib & _A_SYSTEM || fileinfo.attrib & _A_HIDDEN) {
                continue;
            }

            if (fileinfo.attrib & _A_SUBDIR) {
                continue;
            } else {
                _ctime64_s(buffer, _countof(buffer), &fileinfo.time_write);

                std::string fileName = fileinfo.name;

                _splitpath(fileName.c_str(), drive, secDir, fname, ext);

                if (strcmp(ext, extStr.c_str()))
                    continue;

                std::string filePath;
                if (dir.back() != '/')
                    filePath = dir + '/' + fileName;
                else
                    filePath = dir + fileName;

                filePathList.insert(std::make_pair(fname, filePath));
            }
        } while (-1 != _findnext64(findRet, &fileinfo));

        _findclose(findRet);
    }
    return filePathList.size();
}

int ReadWriter::getFileListFromDir(const std::string &dir, PathTimeMap &filePathList, const std::string &extStr) {
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    if (dir.empty())
        return 0;
    char buffer[_MAX_PATH + 100] = {0};
    __finddata64_t fileinfo;

    std::string strFullName = dir;
    if (strFullName.back() != '/')
        strFullName += '/';
    strFullName += "*";

    intptr_t findRet = _findfirst64(strFullName.c_str(), &fileinfo);
    if (-1 != findRet) {
        do {
            if (stricmp(fileinfo.name, ".") == 0 || 0 == stricmp(fileinfo.name, "..")) {
                continue;
            }
            if (fileinfo.attrib & _A_SYSTEM || fileinfo.attrib & _A_HIDDEN) {
                continue;
            }

            if (fileinfo.attrib & _A_SUBDIR) {
                continue;
            } else {
                _ctime64_s(buffer, _countof(buffer), &fileinfo.time_write);

                std::string fileName = fileinfo.name;

                _splitpath(fileName.c_str(), drive, secDir, fname, ext);

                if (strcmp(ext, extStr.c_str()))
                    continue;

                std::string filePath;
                if (dir.back() != '/')
                    filePath = dir + '/' + fileName;
                else
                    filePath = dir + fileName;

                double time = std::stod(fname);
                filePathList.insert(std::make_pair(time, filePath));
            }
        } while (-1 != _findnext64(findRet, &fileinfo));

        _findclose(findRet);
    }
    return filePathList.size();
}

int ReadWriter::getDirListFromDir(std::string &dir, PathIndexMap &dirList) {
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    if (dir.empty())
        return false;
    char buffer[_MAX_PATH + 100] = {0};
    __finddata64_t fileinfo;

    if (dir.back() != '/')
        dir += '/';
    std::string strFullName = dir;
    strFullName += "*";

    intptr_t findRet = _findfirst64(strFullName.c_str(), &fileinfo);
    if (-1 != findRet) {
        do {

            if (stricmp(fileinfo.name, ".") == 0 || 0 == stricmp(fileinfo.name, "..")) {
                continue;
            }

            if (fileinfo.attrib & _A_SYSTEM || fileinfo.attrib & _A_HIDDEN) {
                continue;
            }

            if (fileinfo.attrib & _A_SUBDIR) {
                std::string fname = fileinfo.name;

                // if(strcmp(ext, ".bmp"))
                // continue;
                int number;
                if (fname == "0")
                    number = 0;
                else {
                    number = atoi(fileinfo.name);
                    if (!number)
                        continue;
                }

                std::string dirPath = dir + fileinfo.name;
                dirList.insert(std::make_pair(number, dirPath));
            } else {
                continue;
            }
        } while (-1 != _findnext64(findRet, &fileinfo));

        _findclose(findRet);
    }
    return true;
}

int ReadWriter::getDirListFromDir(std::string &dir, PathStrMap &dirList) {
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    if (dir.empty())
        return false;
    char buffer[_MAX_PATH + 100] = {0};
    __finddata64_t fileinfo;

    if (dir.back() != '/')
        dir += '/';
    std::string strFullName = dir;
    strFullName += "*";

    intptr_t findRet = _findfirst64(strFullName.c_str(), &fileinfo);
    if (-1 != findRet) {
        do {

            if (stricmp(fileinfo.name, ".") == 0 || 0 == stricmp(fileinfo.name, "..")) {
                continue;
            }

            if (fileinfo.attrib & _A_SYSTEM || fileinfo.attrib & _A_HIDDEN) {
                continue;
            }

            if (fileinfo.attrib & _A_SUBDIR) {
                std::string fname = fileinfo.name;

                // if(strcmp(ext, ".bmp"))
                // continue;

                std::string dirPath = dir + fileinfo.name;
                dirList.insert(std::make_pair(fileinfo.name, dirPath));
            } else {
                continue;
            }
        } while (-1 != _findnext64(findRet, &fileinfo));

        _findclose(findRet);
    }
    return true;
}

int ReadWriter::getDirWithSubName(const PathStrMap &srcDirList, PathStrMap &dstDirList, const std::string &subName) {
    for (auto dirItr = srcDirList.begin(); dirItr != srcDirList.end(); dirItr++) {
        const std::string &dirName = dirItr->first;
        if (dirName.find(subName.c_str()) != std::string::npos) {
            dstDirList.insert(*dirItr);
        }
    }
    return dstDirList.size();
}

int ReadWriter::cleanDir(const std::string& dirPath)
{
    struct _finddata_t fb;  
    std::string path;
    long long  handle;
    int noFile;           

    noFile = 0;
    handle = 0;

    struct stat s;
    if (stat(dirPath.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR) 
            std::cout << "it's a directory" << std::endl;    
        else if (s.st_mode & S_IFREG) 
            std::cout << "it's a file" << std::endl;
        else 
            std::cout << "not file not directory" << std::endl;   
    }
    else
    {
        std::cout << "error, doesn't exist" << std::endl;
        return -1;
    }

    path = dirPath + "/*";

    handle = _findfirst(path.c_str(), &fb);
    if (handle != 0)
    {
        while (0 == _findnext(handle, &fb))
        {
            noFile = strcmp(fb.name, "..");

            if (0 != noFile)
            {
                path = dirPath + "/" + fb.name;

                if (fb.attrib == 16)
                    cleanDir(path);
                else
                    remove(path.c_str());
            }
        }

        //_rmdir(dirPath.c_str());

        _findclose(handle);
    }
    return 0;
}



#else

bool ReadWriter::getFileListFromDir(const std::string &dir, PathStrMap &filePathList)
{
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    DIR *pDir = nullptr;
    struct dirent *entry = nullptr;
    struct stat fileinfo;

    if (dir.empty())
        return false;

    if (!(pDir = opendir(dir.c_str())))
    {
        std::cerr << "opendir error: " << dir << std::endl;
        return false;
    }
    while ((entry = readdir(pDir)) != 0)
    {
        lstat(entry->d_name, &fileinfo);

        if (strcmp(entry->d_name, ".") == 0 || 0 == strcmp(entry->d_name, ".."))
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFDIR)
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFLNK)
            continue;

        else
        {
            const std::string &fileName = entry->d_name;
            _splitpath(fileName.c_str(), drive, secDir, fname, ext);

            std::string filePath;
            if (dir.back() != '/')
                filePath = dir + '/' + fileName;
            else
                filePath = dir + fileName;
            filePathList.insert(std::make_pair(fname, filePath));
        }
    }

    closedir(pDir);
    return true;
}

bool ReadWriter::getFileListFromDir(const std::string &dir, PathIndexMap &filePathList)
{
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    DIR *pDir = nullptr;
    struct dirent *entry = nullptr;
    struct stat fileinfo;

    if (dir.empty())
        return false;

    if (!(pDir = opendir(dir.c_str())))
    {
        std::cerr << "opendir error: " << dir << std::endl;
        return false;
    }
    while ((entry = readdir(pDir)) != 0)
    {
        lstat(entry->d_name, &fileinfo);

        if (strcmp(entry->d_name, ".") == 0 || 0 == strcmp(entry->d_name, ".."))
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFDIR)
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFLNK)
            continue;

        else
        {
            std::string fileName = entry->d_name;

            _splitpath(fileName.c_str(), drive, secDir, fname, ext);
            /*	if(strcmp(ext, ".bmp")&& strcmp(ext, ".jpg")&& strcmp(ext, ".png"))
                    continue;*/

            std::string filePath;
            if (dir.back() != '/')
                filePath = dir + '/' + fileName;
            else
                filePath = dir + fileName;

            int number = atoi(fname);
            filePathList.insert(std::make_pair(number, filePath));
        }
    }

    closedir(pDir);
    return true;
}

int ReadWriter::getFileListFromDir(const std::string &dir, PathStrMap &filePathList, const std::string &extStr)
{
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];
    DIR *pDir = nullptr;
    struct dirent *entry = nullptr;
    struct stat fileinfo;
    if (dir.empty())
        return false;

    if (!(pDir = opendir(dir.c_str())))
    {
        std::cerr << "opendir error: " << dir << std::endl;
        return false;
    }
    while ((entry = readdir(pDir)) != 0)
    {
        lstat(entry->d_name, &fileinfo);
        if (strcmp(entry->d_name, ".") == 0 || 0 == strcmp(entry->d_name, ".."))
            continue;

            // else if(S_ISDIR(fileinfo.st_mode))
            // continue;

        else if (S_ISLNK(fileinfo.st_mode))
            continue;

        else
        {
            const std::string &fileName = entry->d_name;

            _splitpath(fileName.c_str(), drive, secDir, fname, ext);

            if (strcmp(ext, extStr.c_str()))
                continue;

            std::string filePath;
            if (dir.back() != '/')
                filePath = dir + '/' + fileName;
            else
                filePath = dir + fileName;

            filePathList.insert(std::make_pair(fname, filePath));
        }
    }
    closedir(pDir);
    return true;
}

int ReadWriter::getFileListFromDir(const std::string &dir, PathTimeMap &filePathList, const std::string &extStr)
{
    char drive[_MAX_DRIVE];
    char secDir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];
    DIR *pDir = nullptr;
    struct dirent *entry = nullptr;
    struct stat fileinfo;
    if (dir.empty())
        return false;

    if (!(pDir = opendir(dir.c_str())))
    {
        std::cerr << "opendir error: " << dir << std::endl;
        return false;
    }
    while ((entry = readdir(pDir)) != 0)
    {
        lstat(entry->d_name, &fileinfo);
        if (strcmp(entry->d_name, ".") == 0 || 0 == strcmp(entry->d_name, ".."))
            continue;

            // else if(S_ISDIR(fileinfo.st_mode))
            // continue;

        else if (S_ISLNK(fileinfo.st_mode))
            continue;

        else
        {
            const std::string &fileName = entry->d_name;

            _splitpath(fileName.c_str(), drive, secDir, fname, ext);

            if (strcmp(ext, extStr.c_str()))
                continue;

            std::string filePath;
            if (dir.back() != '/')
                filePath = dir + '/' + fileName;
            else
                filePath = dir + fileName;

            double time = std::stod(fname);
            filePathList.insert(std::make_pair(time, filePath));
        }
    }

    closedir(pDir);

    return true;
}

int ReadWriter::getDirListFromDir(std::string &dir, PathIndexMap &dirList)
{
    DIR *pDir = nullptr;
    struct dirent *entry = nullptr;
    struct stat fileinfo;
    if (dir.empty())
        return false;

    if (!(pDir = opendir(dir.c_str())))
    {
        std::cerr << "opendir error: " << dir << std::endl;
        return false;
    }
    while ((entry = readdir(pDir)) != 0)
    {
        lstat(entry->d_name, &fileinfo);
        if (strcmp(entry->d_name, ".") == 0 || 0 == strcmp(entry->d_name, ".."))
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFLNK)
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFDIR)
        {
            const std::string &fname = entry->d_name;
            int number;
            if (fname == "0")
                number = 0;
            else
            {
                number = atoi(entry->d_name);
                if (!number)
                    continue;
            }
            std::string dirPath = dir + entry->d_name;
            dirList.insert(std::make_pair(number, dirPath));
        }

        else
            continue;
    }

    closedir(pDir);

    return true;
}

int ReadWriter::getDirListFromDir(std::string &dir, PathStrMap &dirList)
{
    DIR *pDir = nullptr;
    struct dirent *entry = nullptr;
    struct stat fileinfo;

    if (dir.empty())
        return false;

    if (!(pDir = opendir(dir.c_str())))
    {
        std::cerr << "opendir error: " << dir << std::endl;
        return false;
    }
    while ((entry = readdir(pDir)) != 0)
    {
        lstat(entry->d_name, &fileinfo);
        if (strcmp(entry->d_name, ".") == 0 || 0 == strcmp(entry->d_name, ".."))
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFLNK)
            continue;

        else if ((fileinfo.st_mode & S_IFMT) == S_IFDIR)
        {
            std::string fname = entry->d_name;
            std::string dirPath = dir + entry->d_name;
            dirList.insert(std::make_pair(entry->d_name, dirPath));
        }

        else
            continue;
    }

    closedir(pDir);

    return true;
}

int ReadWriter::getDirWithSubName(const PathStrMap &srcDirList, PathStrMap &dstDirList, const std::string &subName)
{
    for (auto dirItr = srcDirList.begin(); dirItr != srcDirList.end(); dirItr++)
    {
        const std::string &dirName = dirItr->first;
        if (dirName.find(subName.c_str()) != std::string::npos)
        {
            dstDirList.insert(*dirItr);
        }
    }
    return dstDirList.size();
}

int ReadWriter::cleanDir(const std::string& path)
{
    DIR* dirp = opendir(path.c_str());
    if (!dirp)
        return -1;
    struct dirent* dir;
    struct stat st;
    while ((dir = readdir(dirp)) != NULL)
    {
        if (strcmp(dir->d_name, ".") == 0
            || strcmp(dir->d_name, "..") == 0)
            continue;
        std::string sub_path = path + '/' + dir->d_name;
        if (lstat(sub_path.c_str(), &st) == -1)
        {
            //Log("rm_dir:lstat ",sub_path," error");
            continue;
        }
        if (S_ISDIR(st.st_mode))
        {
            if (cleanDir(sub_path) == -1) // 如果是目录文件，递归删除
            {
                closedir(dirp);
                return -1;
            }
            rmdir(sub_path.c_str());
        }
        else if (S_ISREG(st.st_mode))
            unlink(sub_path.c_str());     // 如果是普通文件，则unlink
        else
            //Log("rm_dir:st_mode ",sub_path," error");
            continue;
    }
    //    if(rmdir(path.c_str()) == -1)//delete dir itself.
    //    {
    //        closedir(dirp);
    //        return -1;
    //    }
    closedir(dirp);
    return 0;
}

bool ReadWriter::remove(const std::string& filePath)
{
    struct stat st;
    if (lstat(filePath.c_str(), &st) == -1)
        return EXIT_FAILURE;

    if (S_ISREG(st.st_mode))
        if (unlink(filePath.c_str()) == -1)
            return EXIT_FAILURE;

        else if (S_ISDIR(st.st_mode))
        {
            if (filePath == "." || filePath == "..")
                return EXIT_FAILURE;
            if (cleanDir(filePath) == -1)//delete all the files in dir.
                return EXIT_FAILURE;
        }
    return EXIT_SUCCESS;
}

#endif

bool ReadWriter::loadImageList(std::string imgDir, ReadWriter::StrList &leftImgPathList,
                               ReadWriter::StrList &rightImgPathList)
{
    std::string imgLeftDir, imgRightDir;
    if (imgDir.back() != '/')
        imgDir += '/';
    imgLeftDir = imgDir + "L";
    imgRightDir = imgDir + "R";

    PathStrMap leftImgPathMap;
    PathStrMap rightImgPathMap;

    getFileListFromDir(imgLeftDir, leftImgPathMap);
    getFileListFromDir(imgRightDir, rightImgPathMap);

    for (ReadWriter::PathStrMap::iterator itr = leftImgPathMap.begin(); itr != leftImgPathMap.end();)
    {
        std::string filePath = itr->second;
        int dotIndex = filePath.rfind('.');
        std::string ext = filePath.substr(dotIndex, filePath.length() - dotIndex);
        if (ext != ".bmp" && ext != ".jpg" && ext != ".png")
            itr = leftImgPathMap.erase(itr);
        else
            itr++;
    }

    for (ReadWriter::PathStrMap::iterator itr = rightImgPathMap.begin(); itr != rightImgPathMap.end();)
    {
        std::string filePath = itr->second;
        int dotIndex = filePath.rfind('.');
        std::string ext = filePath.substr(dotIndex, filePath.length() - dotIndex);
        if (ext != ".bmp" && ext != ".jpg" && ext != ".png")
            itr = rightImgPathMap.erase(itr);
        else
            itr++;
    }

    if (leftImgPathMap.size() != leftImgPathMap.size())
    {
        std::cout << "Wrong input image files sequence" << std::endl;
        return false;
    }

    for (PathStrMap::iterator fItr = leftImgPathMap.begin(); fItr != leftImgPathMap.end(); fItr++)
        leftImgPathList.emplace_back(fItr->second);
    for (PathStrMap::iterator fItr = rightImgPathMap.begin(); fItr != rightImgPathMap.end(); fItr++)
        rightImgPathList.emplace_back(fItr->second);

    return true;
}


bool ReadWriter::outputCloudFile(const std::string& filePath, PointCloudXYZI& cloud)
{
    std::ofstream fCloudOfs =std::ofstream(filePath, std::ios::trunc | std::ios::in);
    if(!fCloudOfs.is_open())
    {
        std::cerr<<"Failed create cloud file:"<<filePath<<std::endl;
        return false;
    }
    for(auto& pt: cloud)
        fCloudOfs<<std::fixed<<std::setprecision(8)<<pt.x<<","<<pt.y<<","<<pt.z<<","<<pt.curvature<<","<<(int)pt.intensity<<","<<(int)pt.normal_x<<std::endl;

    fCloudOfs.close();
    return true;
}


bool ReadWriter::parseIMUFile(const std::string& imuFilePath, std::deque<SensorMsgs::Imu::Ptr>& imuBufferAll)
{
    imuBufferAll.clear();
    std::ifstream imuFin(imuFilePath);
    while (!imuFin.eof())
    {
        std::string s;
        getline(imuFin, s);
        if (!s.empty())
        {
            std::string item;
            size_t pos = 0;

            double timestamp;
            std::vector<double> imu_data;

            int count = 0;
            while ((pos = s.find(',')) != std::string::npos)
            {
                item = s.substr(0, pos);
                if (count == 0)
                    timestamp = stod(item) / 1000000.0;
                else
                    imu_data.push_back(stod(item));
                s.erase(0, pos + 1);

                ++count;
                if (count > 6)
                {
                    std::cout << "\033[31m"
                              << "parse imu file error"
                              << "\033[0m" << std::endl;
                    imuFin.close();
                    return false;
                }
            }
            item = s.substr(0, pos);
            imu_data.push_back(stod(item));

            SensorMsgs::Imu::Ptr msg(new SensorMsgs::Imu());

            msg->header = timestamp;
            msg->angular_velocity[0] = imu_data[0];
            msg->angular_velocity[1] = imu_data[1];
            msg->angular_velocity[2] = imu_data[2];
            msg->linear_acceleration[0] = imu_data[3];
            msg->linear_acceleration[1] = imu_data[4];
            msg->linear_acceleration[2] = imu_data[5];

            imuBufferAll.push_back(msg);
        }
    }
    imuFin.close();

    return true;
}

bool ReadWriter::parseMotorFile(const std::string& motorFilePath, std::list<std::pair<double, double>>& motorAngleBufAll)
{
    motorAngleBufAll.clear();
    std::ifstream motorFin(motorFilePath);
    while (!motorFin.eof())
    {
        std::string s;
        getline(motorFin, s);
        if (!s.empty())
        {
            std::string item;
            size_t pos = 0;

            std::string data[2];

            int count = 0;
            while ((pos = s.find(',')) != std::string::npos)
            {
                item = s.substr(0, pos);
                data[count] = item;
                s.erase(0, pos + 1);

                ++count;
                if (count > 1)
                {
                    std::cout << "\033[31m"
                              << "parse motor file error"
                              << "\033[0m" << std::endl;
                    motorFin.close();
                    return false;
                }
            }
            item = s.substr(0, pos);
            data[count] = item;
            double timestamp = stod(data[0]) / 1000000.0;
            motorAngleBufAll.emplace_back(timestamp, stod(data[1]) * ANG_TO_RAD);
        }
    }
    motorFin.close();

    return true;
}




