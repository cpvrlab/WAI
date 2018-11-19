//#############################################################################
//  File:      AppDemoMainGLFW.cpp
//  Purpose:   The demo application demonstrates most features of the SLProject
//             framework. Implementation of the GUI with the GLFW3 framework
//             that can create a window and receive system event on desktop OS
//             such as Windows, MacOS and Linux.
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <GLFW/glfw3.h>

#include <AppDemoGui.h>
#include <AppWAISceneView.h>

#include <SLCVCapture.h>
#include <SLCVMapNode.h>
#include <SLCVCalibration.h>

#include <SLBox.h>
#include <SLCoordAxis.h>
#include <SLRectangle.h>
#include <SLEnums.h>
#include <SLInterface.h>
#include <SLSceneView.h>
#include <SLApplication.h>
#include <SLLightSpot.h>

#include <WAI.h>

//-----------------------------------------------------------------------------
// GLobal application variables
GLFWwindow* window;                     //!< The global glfw window handle
SLint       svIndex;                    //!< SceneView index
SLint       scrWidth;                   //!< Window width at start up
SLint       scrHeight;                  //!< Window height at start up
SLbool      fixAspectRatio;             //!< Flag if aspect ratio should be fixed
SLfloat     scrWdivH;                   //!< aspect ratio screen width divided by height
SLfloat     scr2fbX;                    //!< Factor from screen to framebuffer coords
SLfloat     scr2fbY;                    //!< Factor from screen to framebuffer coords
SLint       startX;                     //!< start position x in pixels
SLint       startY;                     //!< start position y in pixels
SLint       mouseX;                     //!< Last mouse position x in pixels
SLint       mouseY;                     //!< Last mouse position y in pixels
SLVec2i     touch2;                     //!< Last finger touch 2 position in pixels
SLVec2i     touchDelta;                 //!< Delta between two fingers in x
SLint       lastWidth;                  //!< Last window width in pixels
SLint       lastHeight;                 //!< Last window height in pixels
SLint       lastMouseWheelPos;          //!< Last mouse wheel position
SLfloat     lastMouseDownTime = 0.0f;   //!< Last mouse press time
SLKey       modifiers         = K_none; //!< last modifier keys
SLbool      fullscreen        = false;  //!< flag if window is in fullscreen mode
SLCamera*   cameraNode        = 0;
WAI::WAI    wai;

//-----------------------------------------------------------------------------
//! appDemoLoadScene builds a scene from source code.
/*! appDemoLoadScene builds a scene from source code. Such a function must be
passed as a void*-pointer to slCreateScene. It will be called from within
slCreateSceneView as soon as the view is initialized. You could separate
different scene by a different sceneID.<br>
The purpose is to assemble a scene by creating scenegraph objects with nodes
(SLNode) and meshes (SLMesh). See the scene with SID_Minimal for a minimal
example of the different steps.
*/
void appDemoLoadScene(SLScene* s, SLSceneView* sv, SceneID sceneID)
{
    //SLApplication::sceneID = sceneID;

    // remove scene specific uis
    AppDemoGui::clearInfoDialogs();
    // Initialize all preloaded stuff from SLScene
    s->init();

    switch (sceneID)
    {
        case Scene_Empty:
        {
            s->name("No Scene loaded.");
            s->info("No Scene loaded.");
            s->root3D(nullptr);
            sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));
            sv->camera(nullptr);
            sv->doWaitOnIdle(true);
        }
        break;

        case Scene_WAI:
        {
            // Set scene name and info string
            s->name("Track Keyframe based Features");
            s->info("Example for loading an existing pose graph with map points.");

            s->videoType(VT_MAIN);

            //make some light
            SLLightSpot* light1 = new SLLightSpot(1, 1, 1, 0.3f);
            light1->ambient(SLCol4f(0.2f, 0.2f, 0.2f));
            light1->diffuse(SLCol4f(0.8f, 0.8f, 0.8f));
            light1->specular(SLCol4f(1, 1, 1));
            light1->attenuation(1, 0, 0);

            //always equal for tracking
            //setup tracking camera
            cameraNode = new SLCamera("Camera 1");
            cameraNode->translation(0, 0, 0.1);
            cameraNode->lookAt(0, 0, 0);
            //for tracking we have to use the field of view from calibration
            cameraNode->fov(SLApplication::activeCalib->cameraFovDeg());
            cameraNode->clipNear(0.001f);
            cameraNode->clipFar(1000000.0f); // Increase to infinity?
            cameraNode->setInitialState();
            cameraNode->background().texture(s->videoTexture());

            //the map node contains the visual representation of the slam map
            //SLCVMapNode* mapNode = new SLCVMapNode("map");

            // Save no energy
            sv->doWaitOnIdle(false); //for constant video feed
            sv->camera(cameraNode);

            //add yellow box and axis for augmentation
            SLMaterial* yellow = new SLMaterial("mY", SLCol4f(1, 1, 0, 0.5f));
            SLfloat     l = 0.593f, b = 0.466f, h = 0.257f;
            SLBox*      box1     = new SLBox(0.0f, 0.0f, 0.0f, l, h, b, "Box 1", yellow);
            SLNode*     boxNode  = new SLNode(box1, "boxNode");
            SLNode*     axisNode = new SLNode(new SLCoordAxis(), "axis node");
            boxNode->addChild(axisNode);

            //setup scene
            SLNode* scene = new SLNode("scene");
            scene->addChild(light1);
            scene->addChild(boxNode);
            //scene->addChild(mapNode);

            s->root3D(scene);
        }
        break;

        case Scene_None:
        case Scene_Minimal:
        default:
        {
            // Set scene name and info string
            s->name("Minimal Scene Test");
            s->info("Minimal texture mapping example with one light source.");

            // Create textures and materials
            SLGLTexture* texC = new SLGLTexture("earth1024_C.jpg");
            SLMaterial*  m1   = new SLMaterial("m1", texC);

            // Create a scene group node
            SLNode* scene = new SLNode("scene node");

            // Create a light source node
            SLLightSpot* light1 = new SLLightSpot(0.3f);
            light1->translation(0, 0, 5);
            light1->lookAt(0, 0, 0);
            light1->name("light node");
            scene->addChild(light1);

            // Create meshes and nodes
            SLMesh* rectMesh = new SLRectangle(SLVec2f(-5, -5), SLVec2f(5, 5), 1, 1, "rectangle mesh", m1);
            SLNode* rectNode = new SLNode(rectMesh, "rectangle node");
            scene->addChild(rectNode);

            SLNode* axisNode = new SLNode(new SLCoordAxis(), "axis node");
            scene->addChild(axisNode);

            // Set background color and the root scene node
            sv->sceneViewCamera()->background().colors(SLCol4f(0.7f, 0.7f, 0.7f),
                                                       SLCol4f(0.2f, 0.2f, 0.2f));

            // pass the scene group as root node
            s->root3D(scene);

            // Save energy
            sv->doWaitOnIdle(true);
        }
        break;
    }

    ////////////////////////////////////////////////////////////////////////////
    // call onInitialize on all scene views to init the scenegraph and stats
    for (auto sv : s->sceneViews())
    {
        if (sv != nullptr)
        {
            sv->onInitialize();
        }
    }

    s->onAfterLoad();
}
//-----------------------------------------------------------------------------
/*! 
onClose event handler for deallocation of the scene & sceneview. onClose is
called glfwPollEvents, glfwWaitEvents or glfwSwapBuffers.
*/
void onClose(GLFWwindow* window)
{
    slShouldClose(true);
}
//-----------------------------------------------------------------------------
/*!
onPaint: Paint event handler that passes the event to the slPaint function. 
*/
SLbool onPaint()
{
    //////////////////////////////////////////////////
    bool viewNeedsRepaint;
    viewNeedsRepaint = slUpdateAndPaint(svIndex);
    //////////////////////////////////////////////////

    // Fast copy the back buffer to the front buffer. This is OS dependent.
    glfwSwapBuffers(window);

    // Show the title generated by the scene library (FPS etc.)
    glfwSetWindowTitle(window, slGetWindowTitle(svIndex).c_str());

    return viewNeedsRepaint;
}
//-----------------------------------------------------------------------------
//! Maps the GLFW key codes to the SLKey codes
SLKey mapKeyToSLKey(SLint key)
{
    switch (key)
    {
        case GLFW_KEY_SPACE: return K_space;
        case GLFW_KEY_ESCAPE: return K_esc;
        case GLFW_KEY_F1: return K_F1;
        case GLFW_KEY_F2: return K_F2;
        case GLFW_KEY_F3: return K_F3;
        case GLFW_KEY_F4: return K_F4;
        case GLFW_KEY_F5: return K_F5;
        case GLFW_KEY_F6: return K_F6;
        case GLFW_KEY_F7: return K_F7;
        case GLFW_KEY_F8: return K_F8;
        case GLFW_KEY_F9: return K_F9;
        case GLFW_KEY_F10: return K_F10;
        case GLFW_KEY_F11: return K_F11;
        case GLFW_KEY_F12: return K_F12;
        case GLFW_KEY_UP: return K_up;
        case GLFW_KEY_DOWN: return K_down;
        case GLFW_KEY_LEFT: return K_left;
        case GLFW_KEY_RIGHT: return K_right;
        case GLFW_KEY_LEFT_SHIFT: return K_shift;
        case GLFW_KEY_RIGHT_SHIFT: return K_shift;
        case GLFW_KEY_LEFT_CONTROL: return K_ctrl;
        case GLFW_KEY_RIGHT_CONTROL: return K_ctrl;
        case GLFW_KEY_LEFT_ALT: return K_alt;
        case GLFW_KEY_RIGHT_ALT: return K_alt;
        case GLFW_KEY_LEFT_SUPER: return K_super;  // Apple command key
        case GLFW_KEY_RIGHT_SUPER: return K_super; // Apple command key
        case GLFW_KEY_TAB: return K_tab;
        case GLFW_KEY_ENTER: return K_enter;
        case GLFW_KEY_BACKSPACE: return K_backspace;
        case GLFW_KEY_INSERT: return K_insert;
        case GLFW_KEY_DELETE: return K_delete;
        case GLFW_KEY_PAGE_UP: return K_pageUp;
        case GLFW_KEY_PAGE_DOWN: return K_pageDown;
        case GLFW_KEY_HOME: return K_home;
        case GLFW_KEY_END: return K_end;
        case GLFW_KEY_KP_0: return K_NP0;
        case GLFW_KEY_KP_1: return K_NP1;
        case GLFW_KEY_KP_2: return K_NP2;
        case GLFW_KEY_KP_3: return K_NP3;
        case GLFW_KEY_KP_4: return K_NP4;
        case GLFW_KEY_KP_5: return K_NP5;
        case GLFW_KEY_KP_6: return K_NP6;
        case GLFW_KEY_KP_7: return K_NP7;
        case GLFW_KEY_KP_8: return K_NP8;
        case GLFW_KEY_KP_9: return K_NP9;
        case GLFW_KEY_KP_DIVIDE: return K_NPDivide;
        case GLFW_KEY_KP_MULTIPLY: return K_NPMultiply;
        case GLFW_KEY_KP_SUBTRACT: return K_NPSubtract;
        case GLFW_KEY_KP_ADD: return K_NPAdd;
        case GLFW_KEY_KP_DECIMAL: return K_NPDecimal;
    }
    return (SLKey)key;
}
//-----------------------------------------------------------------------------
/*!
onResize: Event handler called on the resize event of the window. This event
should called once before the onPaint event.
*/
static void onResize(GLFWwindow* window, int width, int height)
{
    if (fixAspectRatio)
    {
        //correct target width and height
        if (height * scrWdivH <= width)
        {
            width  = height * scrWdivH;
            height = width / scrWdivH;
        }
        else
        {
            height = width / scrWdivH;
            width  = height * scrWdivH;
        }
    }

    lastWidth  = width;
    lastHeight = height;

    // width & height are in screen coords.
    // We need to scale them to framebuffer coords.
    slResize(svIndex, (int)(width * scr2fbX), (int)(height * scr2fbY));

    //update glfw window with new size
    glfwSetWindowSize(window, width, height);
}
//-----------------------------------------------------------------------------
/*!
onLongTouch gets called from a 500ms timer after a mouse down event.
*/
void onLongTouch()
{
    // forward the long touch only if the mouse or touch hasn't moved.
    if (SL_abs(mouseX - startX) < 2 && SL_abs(mouseY - startY) < 2)
        slLongTouch(svIndex, mouseX, mouseY);
}
//-----------------------------------------------------------------------------
/*!
Mouse button event handler forwards the events to the slMouseDown or slMouseUp.
Two finger touches of touch devices are simulated with ALT & CTRL modifiers.
*/
static void onMouseButton(GLFWwindow* window,
                          int         button,
                          int         action,
                          int         mods)
{
    SLint x = mouseX;
    SLint y = mouseY;
    startX  = x;
    startY  = y;

    // Translate modifiers
    modifiers = K_none;
    if (mods & GLFW_MOD_SHIFT) modifiers = (SLKey)(modifiers | K_shift);
    if (mods & GLFW_MOD_CONTROL) modifiers = (SLKey)(modifiers | K_ctrl);
    if (mods & GLFW_MOD_ALT) modifiers = (SLKey)(modifiers | K_alt);

    if (action == GLFW_PRESS)
    {
        // simulate double touch from touch devices
        if (modifiers & K_alt)
        {
            // init for first touch
            if (touch2.x < 0)
            {
                int scrW2 = lastWidth / 2;
                int scrH2 = lastHeight / 2;
                touch2.set(scrW2 - (x - scrW2), scrH2 - (y - scrH2));
                touchDelta.set(x - touch2.x, y - touch2.y);
            }

            // Do parallel double finger move
            if (modifiers & K_shift)
            {
                slTouch2Down(svIndex, x, y, x - touchDelta.x, y - touchDelta.y);
            }
            else // Do concentric double finger pinch
            {
                slTouch2Down(svIndex, x, y, touch2.x, touch2.y);
            }
        }
        else // Do standard mouse down
        {
            SLfloat mouseDeltaTime = (SLfloat)glfwGetTime() - lastMouseDownTime;
            lastMouseDownTime      = (SLfloat)glfwGetTime();

            // handle double click
            if (mouseDeltaTime < 0.3f)
            {
                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        slDoubleClick(svIndex, MB_left, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        slDoubleClick(svIndex, MB_right, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        slDoubleClick(svIndex, MB_middle, x, y, modifiers);
                        break;
                }
            }
            else // normal mouse clicks
            {
                // Start timer for the long touch detection
                SLTimer::callAfterSleep(SLSceneView::LONGTOUCH_MS, onLongTouch);

                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        slMouseDown(svIndex, MB_left, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        slMouseDown(svIndex, MB_right, x, y, modifiers);
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        slMouseDown(svIndex, MB_middle, x, y, modifiers);
                        break;
                }
            }
        }
    }
    else
    { // flag end of mouse click for long touches
        startX = -1;
        startY = -1;

        // simulate double touch from touch devices
        if (modifiers & K_alt)
        {
            // Do parallel double finger move
            if (modifiers & K_shift)
            {
                slTouch2Up(svIndex, x, y, x - (touch2.x - x), y - (touch2.y - y));
            }
            else // Do concentric double finger pinch
            {
                slTouch2Up(svIndex, x, y, touch2.x, touch2.y);
            }
        }
        else // Do standard mouse down
        {
            switch (button)
            {
                case GLFW_MOUSE_BUTTON_LEFT:
                    slMouseUp(svIndex, MB_left, x, y, modifiers);
                    break;
                case GLFW_MOUSE_BUTTON_RIGHT:
                    slMouseUp(svIndex, MB_right, x, y, modifiers);
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    slMouseUp(svIndex, MB_middle, x, y, modifiers);
                    break;
            }
        }
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse move event handler forwards the events to slMouseMove or slTouch2Move.
*/
static void onMouseMove(GLFWwindow* window,
                        double      x,
                        double      y)
{
    // x & y are in screen coords.
    // We need to scale them to framebuffer coords
    x *= scr2fbX;
    y *= scr2fbY;
    mouseX = (int)x;
    mouseY = (int)y;

    // Offset of 2nd. finger for two finger simulation

    // Simulate double finger touches
    if (modifiers & K_alt)
    {
        // Do parallel double finger move
        if (modifiers & K_shift)
        {
            slTouch2Move(svIndex, (int)x, (int)y, (int)x - touchDelta.x, (int)y - touchDelta.y);
        }
        else // Do concentric double finger pinch
        {
            int scrW2    = lastWidth / 2;
            int scrH2    = lastHeight / 2;
            touch2.x     = scrW2 - ((int)x - scrW2);
            touch2.y     = scrH2 - ((int)y - scrH2);
            touchDelta.x = (int)x - touch2.x;
            touchDelta.y = (int)y - touch2.y;
            slTouch2Move(svIndex, (int)x, (int)y, touch2.x, touch2.y);
        }
    }
    else // Do normal mouse move
    {
        slMouseMove(svIndex, (int)x, (int)y);
    }
}
//-----------------------------------------------------------------------------
/*!
Mouse wheel event handler forwards the events to slMouseWheel
*/
static void onMouseWheel(GLFWwindow* window,
                         double      xscroll,
                         double      yscroll)
{
    // make sure the delta is at least one integer
    int dY = (int)yscroll;
    if (dY == 0) dY = (int)(SL_sign(yscroll));

    slMouseWheel(svIndex, dY, modifiers);
}
//-----------------------------------------------------------------------------
/*!
Key event handler sets the modifier key state & forwards the event to
the slKeyPress function.
*/
static void onKeyPress(GLFWwindow* window,
                       int         GLFWKey,
                       int         scancode,
                       int         action,
                       int         mods)
{
    SLKey key = mapKeyToSLKey(GLFWKey);

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case K_ctrl: modifiers = (SLKey)(modifiers | K_ctrl); return;
            case K_alt: modifiers = (SLKey)(modifiers | K_alt); return;
            case K_shift: modifiers = (SLKey)(modifiers | K_shift); return;
            default: break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key)
        {
            case K_ctrl: modifiers = (SLKey)(modifiers ^ K_ctrl); return;
            case K_alt: modifiers = (SLKey)(modifiers ^ K_alt); return;
            case K_shift: modifiers = (SLKey)(modifiers ^ K_shift); return;
            default: break;
        }
    }

    // Special treatment for ESC key
    if (key == K_esc && action == GLFW_RELEASE)
    {
        if (fullscreen)
        {
            fullscreen = !fullscreen;
            glfwSetWindowSize(window, scrWidth, scrHeight);
            glfwSetWindowPos(window, 10, 30);
        }
        else
        {
            slKeyPress(svIndex, key, modifiers);
            onClose(window);
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
    }
    else
      // Toggle fullscreen mode
      if (key == K_F9 && action == GLFW_PRESS)
    {
        fullscreen = !fullscreen;

        if (fullscreen)
        {
            GLFWmonitor*       primary = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode    = glfwGetVideoMode(primary);
            glfwSetWindowSize(window, mode->width, mode->height);
            glfwSetWindowPos(window, 0, 0);
        }
        else
        {
            glfwSetWindowSize(window, scrWidth, scrHeight);
            glfwSetWindowPos(window, 10, 30);
        }
    }
    else
    {
        if (action == GLFW_PRESS)
            slKeyPress(svIndex, key, modifiers);
        else if (action == GLFW_RELEASE)
            slKeyRelease(svIndex, key, modifiers);
    }
}
//-----------------------------------------------------------------------------
//! Event handler for GLFW character input
void onCharInput(GLFWwindow*, SLuint c)
{
    slCharInput(svIndex, c);
}
//-----------------------------------------------------------------------------
/*!
Error callback handler for GLFW.
*/
void onGLFWError(int error, const char* description)
{
    fputs(description, stderr);
}
//-----------------------------------------------------------------------------
/*!
The C main procedure running the GLFW GUI application.
*/
int main(int argc, char* argv[])
{
    // set command line arguments
    SLVstring cmdLineArgs;
    for (int i = 0; i < argc; i++)
        cmdLineArgs.push_back(SLstring(argv[i]));

    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(onGLFWError);

    // Enable fullscreen anti aliasing with 4 samples
    glfwWindowHint(GLFW_SAMPLES, 4);

    //You can enable or restrict newer OpenGL context here (read the GLFW documentation)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    scrWidth  = 640;
    scrHeight = 360;

    //we have to fix aspect ratio, because the video image is initialized with this ratio
    fixAspectRatio = true;
    scrWdivH       = (float)scrWidth / (float)scrHeight;

    touch2.set(-1, -1);
    touchDelta.set(-1, -1);

    window = glfwCreateWindow(scrWidth, scrHeight, "WAI Demo", nullptr, nullptr);

    //get real window size
    glfwGetWindowSize(window, &scrWidth, &scrHeight);

    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Get the current GL context. After this you can call GL
    glfwMakeContextCurrent(window);

    // On some systems screen & framebuffer size are different
    // All commands in GLFW are in screen coords but rendering in GL is
    // in framebuffer coords
    SLint fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    scr2fbX = (float)fbWidth / (float)scrWidth;
    scr2fbY = (float)fbHeight / (float)scrHeight;

    // Include OpenGL via GLEW (init must be after window creation)
    // The goal of the OpenGL Extension Wrangler Library (GLEW) is to assist C/C++
    // OpenGL developers with two tedious tasks: initializing and using extensions
    // and writing portable applications. GLEW provides an efficient run-time
    // mechanism to determine whether a certain extension is supported by the
    // driver or not. OpenGL core and extension functionality is exposed via a
    // single header file. Download GLEW at: http://glew.sourceforge.net/
    glewExperimental = GL_TRUE; // avoids a crash
    GLenum err       = glewInit();
    if (GLEW_OK != err)
    {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    glfwSetWindowTitle(window, "SLProject Test Application");
    glfwSetWindowPos(window, 10, 30);

    // Set number of monitor refreshes between 2 buffer swaps
    glfwSwapInterval(2);

    // Get GL errors that occurred before our framework is involved
    GET_GL_ERROR;

    // Set your own physical screen dpi
    int dpi = (int)(142 * scr2fbX);
    cout << "------------------------------------------------------------------" << endl;
    cout << "GUI             : GLFW (Version: " << GLFW_VERSION_MAJOR << "." << GLFW_VERSION_MINOR << "." << GLFW_VERSION_REVISION << ")" << endl;
    cout << "DPI             : " << dpi << endl;

    // get executable path
    SLstring projectRoot = SLstring(SL_PROJECT_ROOT);
    SLstring configDir   = SLFileSystem::getAppsWritableDir();
    slSetupExternalDirectories("../data");

    /////////////////////////////////////////////////////////
    slCreateAppAndScene(cmdLineArgs,
                        projectRoot + "/data/shaders/",
                        projectRoot + "/data/models/",
                        projectRoot + "/data/images/textures/",
                        projectRoot + "/data/videos/",
                        projectRoot + "/data/images/fonts/",
                        projectRoot + "/data/calibrations/",
                        configDir,
                        "AppDemoGLFW",
                        (void*)appDemoLoadScene);
    /////////////////////////////////////////////////////////

    // This load the GUI configs that are locally stored
    AppDemoGui::loadConfig(dpi);

    /////////////////////////////////////////////////////////
    svIndex = slCreateSceneView((int)(scrWidth * scr2fbX),
                                (int)(scrHeight * scr2fbY),
                                dpi,
                                (SLSceneID)SL_STARTSCENE,
                                (void*)&onPaint,
                                nullptr,
                                nullptr,
                                (void*)AppDemoGui::build);
    /////////////////////////////////////////////////////////

    // Set GLFW callback functions
    glfwSetKeyCallback(window, onKeyPress);
    glfwSetCharCallback(window, onCharInput);
    glfwSetWindowSizeCallback(window, onResize);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetCursorPosCallback(window, onMouseMove);
    glfwSetScrollCallback(window, onMouseWheel);
    glfwSetWindowCloseCallback(window, onClose);

    wai.setMode(WAI::ModeType_ORB_SLAM2);

    // Event loop
    while (!slShouldClose())
    {
        // If live video image is requested grab it and copy it
        if (slGetVideoType() != VT_NONE)
        {
            SLCVCapture::grabAndAdjustForSL();
            wai.updateSensor(WAI::SensorType_Camera, &SLCVCapture::lastFrameGray);
        }

        WAI::M4x4 pose;
        bool      iKnowWhereIAm = wai.whereAmI(&pose);

        if (iKnowWhereIAm)
        {
            SLMat4f slPose = SLMat4f(pose.e[0]);
            cameraNode->om(slPose);
        }

        /////////////////////////////
        SLbool doRepaint = onPaint();
        /////////////////////////////

        // if no updated occurred wait for the next event (power saving)
        if (!doRepaint)
            glfwWaitEvents();
        else
            glfwPollEvents();
    }

    AppDemoGui::saveConfig();

    slTerminate();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
//-----------------------------------------------------------------------------
