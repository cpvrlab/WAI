#ifndef WAI_INITIALIZER_H
#define WAI_INITIALIZER_H

#include <WAIFrame.h>

namespace WAI
{

struct InitializationResult
{
    bool         success;
    WAIFrame*    currentFrame; // TODO(dgj1): should this be a pointer?
    WAIKeyFrame* kfIni;
    WAIKeyFrame* kfCur;
};

class WAIInitializer
{
    public:
    virtual InitializationResult initialize() = 0;
    virtual void                 reset()      = 0;
};

};

#endif
