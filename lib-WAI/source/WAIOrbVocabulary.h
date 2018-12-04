//#############################################################################
//  File:      WAIOrbVocabulary.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef WAI_ORBVOCABULARY_H
#define WAI_ORBVOCABULARY_H

#include <WAI.h>
#include <OrbSlam/ORBVocabulary.h>

/*!Singleton class used to load, store and delete ORB_SLAM2::ORBVocabulary instance.
*/
class WAIOrbVocabulary
{
    public:
    ~WAIOrbVocabulary();

    static void initialize(std::string filename);

    //!get vocabulary
    static ORB_SLAM2::ORBVocabulary* get();
    //!delete vocabulary (free storage)
    static void free();

    private:
    static WAIOrbVocabulary& instance()
    {
        static WAIOrbVocabulary s_instance;
        return s_instance;
    }

    void                      loadFromFile(std::string strVocFile);
    void                      doFree();
    ORB_SLAM2::ORBVocabulary* doGet();
    void                      doInitialize(std::string filename);

    ORB_SLAM2::ORBVocabulary* _vocabulary = nullptr;
};

#endif // !WAI_ORBVOCABULARY_H
