//#############################################################################
//  File:      SL/WAIFileSystem.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef WAIFileSystem_H
#define WAIFileSystem_H

#include <string>
#include <vector>
#include <algorithm>

#include <WAIHelper.h>

//-----------------------------------------------------------------------------
//! WAIFileSystem provides basic filesystem functions
class WAIFileSystem
{
    public:
    //! Returns true if a directory exists.
    static bool dirExists(const std::string& path);

    //! Make a directory with given path
    static void makeDir(const std::string& path);

    //! Remove a directory with given path
    static void removeDir(const std::string& path);

    //! Returns true if a file exists.
    static bool fileExists(const std::string& pathfilename);

    //! Returns the writable configuration directory
    static std::string getAppsWritableDir();

    //! Returns the working directory
    static std::string getCurrentWorkingDir();

    //! Deletes a file on the filesystem
    static bool deleteFile(std::string& pathfilename);

    //!setters
    static void setExternalDir(const std::string& dir);
    //!getters
    static std::string getExternalDir() { return _externalDir; }
    static bool        externalDirExists() { return _externalDirExists; }

    static std::vector<std::string> getFileNamesInDir(const std::string dirName);
    static void                     split(const std::string& s, char delimiter, std::vector<std::string>& splits);
    static bool                     contains(const std::string container, const std::string search);
    static std::string              getFileName(const std::string& pathFilename);
    static std::string              unifySlashes(const std::string& inputDir);

    private:
    //! Directory to save app data outside of the app
    static std::string _externalDir;
    //! flags, if _externalDir was tested on existence
    static bool _externalDirExists;
};
//-----------------------------------------------------------------------------
#endif
