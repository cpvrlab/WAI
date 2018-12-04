//#############################################################################
//  File:      WAIMapStorage.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <WAIMapStorage.h>

//-----------------------------------------------------------------------------
unsigned int WAIMapStorage::_nextId      = 0;
unsigned int WAIMapStorage::_currentId   = 0;
std::string  WAIMapStorage::_mapPrefix   = "slam-map-";
std::string  WAIMapStorage::_mapsDirName = "slam-maps";
std::string  WAIMapStorage::_mapsDir     = "";
//values used by imgui
std::vector<std::string> WAIMapStorage::existingMapNames;
const char*              WAIMapStorage::currItem       = nullptr;
int                      WAIMapStorage::currN          = -1;
bool                     WAIMapStorage::_isInitialized = false;
//-----------------------------------------------------------------------------
WAIMapStorage::WAIMapStorage()
{
}
//-----------------------------------------------------------------------------
void WAIMapStorage::init(std::string externalDir)
{
    WAIFileSystem::setExternalDir(externalDir);
    existingMapNames.clear();
    vector<pair<int, string>> existingMapNamesSorted;

    //setup file system and check for existing files
    if (WAIFileSystem::externalDirExists())
    {
        _mapsDir = WAIFileSystem::unifySlashes(externalDir + _mapsDirName);

        //check if visual odometry maps directory exists
        if (!WAIFileSystem::dirExists(_mapsDir))
        {
            printf("Making dir: %s\n", _mapsDir.c_str());
            WAIFileSystem::makeDir(_mapsDir);
        }
        else
        {
            //parse content: we search for directories in mapsDir
            std::vector<std::string> content = WAIFileSystem::getFileNamesInDir(_mapsDir);
            for (auto path : content)
            {
                std::string name = WAIFileSystem::getFileName(path);
                //find json files that contain mapPrefix and estimate highest used id
                if (WAIFileSystem::contains(name, _mapPrefix))
                {
                    printf("VO-Map found: %s\n", name.c_str());
                    //estimate highest used id
                    std::vector<std::string> splitted;
                    WAIFileSystem::split(name, '-', splitted);
                    if (splitted.size())
                    {
                        int id = atoi(splitted.back().c_str());
                        existingMapNamesSorted.push_back(make_pair(id, name));
                        if (id >= _nextId)
                        {
                            _nextId = id + 1;
                            printf("New next id: %i\n", _nextId);
                        }
                    }
                }
            }
        }
        //sort existingMapNames
        std::sort(existingMapNamesSorted.begin(), existingMapNamesSorted.end(), [](const pair<int, string>& left, const pair<int, string>& right) { return left.first < right.first; });
        for (auto it = existingMapNamesSorted.begin(); it != existingMapNamesSorted.end(); ++it)
            existingMapNames.push_back(it->second);

        //mark storage as initialized
        _isInitialized = true;
    }
    else
    {
        printf("Failed to setup external map storage!\n");
        printf("Exit in WAIMapStorage::init()");
        std::exit(0);
    }
}
//-----------------------------------------------------------------------------
void WAIMapStorage::saveMap(int id, WAI::ModeOrbSlam2* orbSlamMode, bool saveImgs, cv::Mat nodeOm, std::string externalDir)
{
    if (!_isInitialized)
    {
        printf("External map storage is not initialized, you have to call init() first!\n");
        return;
    }

    if (!orbSlamMode->isInitialized())
    {
        printf("Map storage: System is not initialized. Map saving is not possible!\n");
        return;
    }

    bool errorOccured = false;
    //check if map exists
    string mapName  = _mapPrefix + to_string(id);
    string path     = WAIFileSystem::unifySlashes(_mapsDir + mapName);
    string pathImgs = path + "imgs/";
    string filename = path + mapName + ".json";

    try
    {
        //if path exists, delete content
        if (WAIFileSystem::fileExists(path))
        {
            //remove json file
            if (WAIFileSystem::fileExists(filename))
            {
                WAIFileSystem::deleteFile(filename);
                //check if imgs dir exists and delete all containing files
                if (WAIFileSystem::fileExists(pathImgs))
                {
                    std::vector<std::string> content = WAIFileSystem::getFileNamesInDir(pathImgs);
                    for (auto path : content)
                    {
                        WAIFileSystem::deleteFile(path);
                    }
                }
            }
        }
        else
        {
            //create map directory and imgs directory
            WAIFileSystem::makeDir(path);
        }

        if (!WAIFileSystem::fileExists(pathImgs))
        {
            WAIFileSystem::makeDir(pathImgs);
        }

        //switch to idle, so that map does not change, while we are accessing keyframes
        orbSlamMode->pause();
#if 0
        mapTracking->sm.requestStateIdle();
        while (!mapTracking->sm.hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
#endif

        //save the map
        WAIMapIO::save(filename, *orbSlamMode->getMap(), saveImgs, pathImgs, nodeOm);

        //update list of existing maps
        WAIMapStorage::init(externalDir);
        //update current combobox item
        auto it = std::find(existingMapNames.begin(), existingMapNames.end(), mapName);
        if (it != existingMapNames.end())
        {
            currN    = it - existingMapNames.begin();
            currItem = existingMapNames[currN].c_str();
        }
    }
    catch (std::exception& e)
    {
        string msg = "Exception during slam map storage: " + filename + "\n" +
                     e.what() + "\n";
        printf("%s\n", msg.c_str());
        errorOccured = true;
    }
    catch (...)
    {
        string msg = "Exception during slam map storage: " + filename + "\n";
        printf("%s\n", msg.c_str());
        errorOccured = true;
    }

    //if an error occured, we delete the whole directory
    if (errorOccured)
    {
        //if path exists, delete content
        if (WAIFileSystem::fileExists(path))
        {
            //remove json file
            if (WAIFileSystem::fileExists(filename))
                WAIFileSystem::deleteFile(filename);
            //check if imgs dir exists and delete all containing files
            if (WAIFileSystem::fileExists(pathImgs))
            {
                std::vector<std::string> content = WAIFileSystem::getFileNamesInDir(pathImgs);
                for (auto path : content)
                {
                    WAIFileSystem::deleteFile(path);
                }
                WAIFileSystem::deleteFile(pathImgs);
            }
            WAIFileSystem::deleteFile(path);
        }
    }

    //switch back to initialized state and resume tracking
    orbSlamMode->resume();
}
//-----------------------------------------------------------------------------
bool WAIMapStorage::loadMap(const string& path, WAI::ModeOrbSlam2* orbSlamMode, ORBVocabulary* orbVoc, bool loadKfImgs, cv::Mat& nodeOm)
{
    bool loadingSuccessful = false;
    if (!_isInitialized)
    {
        printf("External map storage is not initialized, you have to call init() first!\n");
        return loadingSuccessful;
    }
    if (!orbSlamMode)
    {
        printf("Map tracking not initialized!\n");
        return loadingSuccessful;
    }

    //reset tracking (and all dependent threads/objects like Map, KeyFrameDatabase, LocalMapping, loopClosing)
    orbSlamMode->requestStateIdle();
    while (!orbSlamMode->hasStateIdle())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    orbSlamMode->reset();

    //clear map and keyframe database
    WAIMap*        map  = orbSlamMode->getMap();
    WAIKeyFrameDB* kfDB = orbSlamMode->getKfDB();

    //extract id from map name
    size_t prefixIndex = path.find(_mapPrefix);
    if (prefixIndex != string::npos)
    {
        std::string name     = path.substr(prefixIndex);
        std::string idString = name.substr(_mapPrefix.length());
        _currentId           = atoi(idString.c_str());
    }
    else
    {
        printf("Could not load map. Map id not found in name: %s\n", path.c_str());
        return loadingSuccessful;
    }

    //check if map exists
    string mapName      = _mapPrefix + to_string(_currentId);
    string mapPath      = WAIFileSystem::unifySlashes(_mapsDir + mapName);
    string currPathImgs = mapPath + "imgs/";
    string filename     = mapPath + mapName + ".json";

    //check if dir and file exist
    if (!WAIFileSystem::dirExists(mapPath))
    {
        string msg = "Failed to load map. Path does not exist: " + mapPath + "\n";
        printf("%s\n", msg.c_str());
        return loadingSuccessful;
    }
    if (!WAIFileSystem::fileExists(filename))
    {
        string msg = "Failed to load map: " + filename + "\n";
        printf("%s\n", msg.c_str());
        return loadingSuccessful;
    }

    try
    {
        WAIMapIO mapIO(filename, orbVoc, loadKfImgs, currPathImgs);
        mapIO.load(nodeOm, *map, *kfDB);

        //if map loading was successful, switch to initialized
        orbSlamMode->setInitialized(true);
        loadingSuccessful = true;
    }
    catch (std::exception& e)
    {
        string msg = "Exception during slam map loading: " + filename +
                     e.what() + "\n";
        printf("%s\n", msg.c_str());
    }
    catch (...)
    {
        string msg = "Exception during slam map loading: " + filename + "\n";
        printf("%s\n", msg.c_str());
    }

    orbSlamMode->resume();
    return loadingSuccessful;
}
//-----------------------------------------------------------------------------
void WAIMapStorage::newMap()
{
    if (!_isInitialized)
    {
        printf("External map storage is not initialized, you have to call init() first!\n");
        return;
    }

    //assign next id to current id. The nextId will be increased after file save.
    _currentId = _nextId;
}
//-----------------------------------------------------------------------------
string WAIMapStorage::mapsDir()
{
    return _mapsDir;
}
