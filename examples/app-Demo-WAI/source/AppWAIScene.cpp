#include <AppWAIScene.h>

AppWAIScene::AppWAIScene ()
{
    cameraNode = new SLCamera("Camera 1");
    mapNode = new SLNode("map");
    mapPC = new SLNode("MapPC");
    mapMatchedPC = new SLNode("MapMatchedPC");
    mapLocalPC = new SLNode("MapLocalPC");
    keyFrameNode = new SLNode("KeyFrames");
    covisibilityGraph = new SLNode("CovisibilityGraph");
    spanningTree = new SLNode("SpanningTree");
    loopEdges = new SLNode("LoopEdges");
    redMat = new SLMaterial(SLCol4f::RED, "Red");
    greenMat = new SLMaterial(SLCol4f::GREEN, "Green");
    blueMat = new SLMaterial(SLCol4f::BLUE, "Blue");
    covisibilityGraphMat = new SLMaterial("YellowLines", SLCol4f::YELLOW);
    spanningTreeMat = new SLMaterial("GreenLines", SLCol4f::GREEN);
    loopEdgesMat = new SLMaterial("RedLines", SLCol4f::RED);

    mapNode->addChild(mapPC);
    mapNode->addChild(mapMatchedPC);
    mapNode->addChild(mapLocalPC);
    mapNode->addChild(keyFrameNode);
    mapNode->addChild(covisibilityGraph);
    mapNode->addChild(spanningTree);
    mapNode->addChild(loopEdges);
    mapNode->addChild(cameraNode);
}
