//#############################################################################
//  File:      SL/WAIFileSystem.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <WAIFileSystem.h>
#include <WAIUtils.h>

#ifdef WAI_OS_WINDOWS
#    include <direct.h> //_getcwd
#elif defined(WAI_OS_MACOS)
#    include <unistd.h>
#    include <sys/stat.h>
#    include <dirent.h>
#elif defined(WAI_OS_MACIOS)
#    include <unistd.h> //getcwd
#elif defined(WAI_OS_ANDROID)
#    include <unistd.h> //getcwd
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <dirent.h>
#elif defined(WAI_OS_LINUX)
#    include <unistd.h> //getcwd
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <dirent.h>
#endif

//-----------------------------------------------------------------------------
std::string WAIFileSystem::_externalDir       = "";
bool        WAIFileSystem::_externalDirExists = false;
//-----------------------------------------------------------------------------
/*! Returns true if the directory exists. Be aware that on some OS file and
paths are treated case sensitive.
*/
bool WAIFileSystem::dirExists(const std::string& path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
        return false;
    else if (info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}
//-----------------------------------------------------------------------------
/*! Make a directory with given path
*/
void WAIFileSystem::makeDir(const std::string& path)
{
#ifdef WAI_OS_WINDOWS
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}
//-----------------------------------------------------------------------------
/*! Remove a directory with given path. DOES ONLY WORK FOR EMPTY DIRECTORIES
*/
void WAIFileSystem::removeDir(const std::string& path)
{
#ifdef WAI_OS_WINDOWS
    int ret = _rmdir(path.c_str());
    if (ret != 0)
    {
        errno_t err;
        _get_errno(&err);
        printf("Could not remove directory: %s\nErrno: %s\n", path.c_str(), strerror(errno));
    }
#else
    int         ret       = rmdir(path.c_str());
#endif
}
//-----------------------------------------------------------------------------
/*! Returns true if the file exists.Be aware that on some OS file and
paths are treated case sensitive.
*/
bool WAIFileSystem::fileExists(const std::string& pathfilename)
{
    struct stat info;
    if (stat(pathfilename.c_str(), &info) == 0)
    {
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
std::string WAIFileSystem::getAppsWritableDir()
{
#ifdef WAI_OS_WINDOWS
    std::string appData   = getenv("APPDATA");
    std::string configDir = appData + "/SLProject";
    WAIUtils::replaceString(configDir, "\\", "/");
    if (!dirExists(configDir))
        _mkdir(configDir.c_str());
    return configDir + "/";
#elif defined(WAI_OS_MACOS)
    std::string home      = getenv("HOME");
    std::string appData   = home + "/Library/Application Support";
    std::string configDir = appData + "/SLProject";
    if (!dirExists(configDir))
        mkdir(configDir.c_str(), S_IRWXU);
    return configDir + "/";
#elif defined(WAI_OS_ANDROID)
    // @todo Where is the app data path on Andoroid?
#elif defined(WAI_OS_LINUX)
    // @todo Where is the app data path on Linux?
    std::string home      = getenv("HOME");
    std::string configDir = home + "/.SLProject";
    if (!dirExists(configDir))
        mkdir(configDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    return configDir + "/";
#else
#    error "SL has not been ported to this OS"
#endif
}
//-----------------------------------------------------------------------------
std::string WAIFileSystem::getCurrentWorkingDir()
{
#ifdef WAI_OS_WINDOWS
    int   size   = 256;
    char* buffer = (char*)malloc(size);
    if (_getcwd(buffer, size) == buffer)
    {
        std::string dir = buffer;
        WAIUtils::replaceString(dir, "\\", "/");
        return dir + "/";
    }

    free(buffer);
    return "";
#else
    size_t size   = 256;
    char*  buffer = (char*)malloc(size);
    if (getcwd(buffer, size) == buffer)
        return std::string(buffer) + "/";

    free(buffer);
    return "";
#endif
}
//-----------------------------------------------------------------------------
bool WAIFileSystem::deleteFile(std::string& pathfilename)
{
    if (WAIFileSystem::fileExists(pathfilename))
        return remove(pathfilename.c_str()) != 0;
    return false;
}
//-----------------------------------------------------------------------------
std::vector<std::string> WAIFileSystem::getFileNamesInDir(const std::string dirName)
{
    std::vector<std::string> fileNames;
    DIR*                     dir;
    dir = opendir(dirName.c_str());

    if (dir)
    {
        struct dirent* dirContent;
        int            i = 0;

        while ((dirContent = readdir(dir)) != nullptr)
        {
            i++;
            std::string name(dirContent->d_name);
            if (name != "." && name != "..")
                fileNames.push_back(dirName + "/" + name);
        }
        closedir(dir);
    }
    return fileNames;
}
//-----------------------------------------------------------------------------
void WAIFileSystem::split(const std::string& s, char delimiter, std::vector<std::string>& splits)
{
    std::string::size_type i = 0;
    std::string::size_type j = s.find(delimiter);

    while (j != std::string::npos)
    {
        splits.push_back(s.substr(i, j - i));
        i = ++j;
        j = s.find(delimiter, j);
        if (j == std::string::npos)
            splits.push_back(s.substr(i, s.length()));
    }
}
//-----------------------------------------------------------------------------
bool WAIFileSystem::contains(const std::string container, const std::string search)
{
    return (container.find(search) != std::string::npos);
}
//-----------------------------------------------------------------------------
std::string WAIFileSystem::getFileName(const std::string& pathFilename)
{
    size_t i = 0, i1, i2;
    i1       = pathFilename.rfind('\\', pathFilename.length());
    i2       = pathFilename.rfind('/', pathFilename.length());

    if (i1 != std::string::npos && i2 != std::string::npos)
        i = std::max(i1, i2);
    else if (i1 != std::string::npos)
        i = i1;
    else if (i2 != std::string::npos)
        i = i2;

    return pathFilename.substr(i + 1, pathFilename.length() - i);
}
//-----------------------------------------------------------------------------
//!setters
void WAIFileSystem::setExternalDir(const std::string& dir)
{
    _externalDir       = unifySlashes(dir);
    _externalDirExists = true;
}
//-----------------------------------------------------------------------------
std::string WAIFileSystem::unifySlashes(const std::string& inputDir)
{
    std::string copy = inputDir;
    std::string curr;
    std::string delimiter = "\\";
    size_t      pos       = 0;
    std::string token;
    while ((pos = copy.find(delimiter)) != std::string::npos)
    {
        token = copy.substr(0, pos);
        copy.erase(0, pos + delimiter.length());
        curr.append(token);
        curr.append("/");
    }

    curr.append(copy);
    if (curr.size() && curr.back() != '/')
        curr.append("/");

    return curr;
}
