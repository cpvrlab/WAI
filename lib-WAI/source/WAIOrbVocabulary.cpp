//#############################################################################
//  File:      WAIOrbVocabulary.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <WAIOrbVocabulary.h>

WAIOrbVocabulary::~WAIOrbVocabulary()
{
    if (_vocabulary)
        delete _vocabulary;
}
//-----------------------------------------------------------------------------
void WAIOrbVocabulary::initialize(std::string filename)
{
    instance().doInitialize(filename);
}
//-----------------------------------------------------------------------------
void WAIOrbVocabulary::loadFromFile(std::string strVocFile)
{
    _vocabulary   = new ORB_SLAM2::ORBVocabulary();
    bool bVocLoad = _vocabulary->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
    {
        WAI_LOG("Wrong path to vocabulary. Failed to open at: %s", strVocFile.c_str());
        WAI_LOG("WAIOrbVocabulary::loadFromFile: failed to load vocabulary");
        return;
    }
    WAI_LOG("Vocabulary loaded!\n");
}
//-----------------------------------------------------------------------------
ORB_SLAM2::ORBVocabulary* WAIOrbVocabulary::get()
{
    return instance().doGet();
}
//-----------------------------------------------------------------------------
void WAIOrbVocabulary::free()
{
    instance().doFree();
}
//-----------------------------------------------------------------------------
void WAIOrbVocabulary::doFree()
{
    if (_vocabulary)
        delete _vocabulary;
}
//-----------------------------------------------------------------------------
ORB_SLAM2::ORBVocabulary* WAIOrbVocabulary::doGet()
{
    return _vocabulary;
}
//-----------------------------------------------------------------------------
void WAIOrbVocabulary::doInitialize(std::string filename)
{
    if (!_vocabulary)
    {
        loadFromFile(filename);
    }
}
