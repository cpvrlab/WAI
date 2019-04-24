#include <SLApplication.h>
#include <SLBox.h>
#include <SLLightSpot.h>
#include <SLCoordAxis.h>
#include <SLPoints.h>

#include <SLCVCamera.h>
#include <SLCVCapture.h>
#include <SLFileSystem.h>

#include <WAIMapStorage.h>

#include <AppDemoGui.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiMapStorage.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppDemoCalibrationSceneView.h>

CalibrationSceneView::CalibrationSceneView(WAI::WAI wai, SLCamera* cameraNode, SLSceneView * sv)
{
    (void)wai;
    (void)cameraNode;
}

//-----------------------------------------------------------------------------
void WAISceneView::update()
{


}

//-----------------------------------------------------------------------------
void WAISceneView::renderGraphs()
{
#ifndef DATA_ORIENTED
    std::vector<WAIKeyFrame*> kfs = _mode->getKeyFrames();

    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;
    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const vector<WAIKeyFrame*> vCovKFs = kf->GetCovisiblesByWeight(_minNumOfCovisibles);
        if (!vCovKFs.empty())
        {
            for (vector<WAIKeyFrame*>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
            {
                if ((*vit)->mnId < kf->mnId)
                    continue;
                cv::Mat Ow2 = (*vit)->GetCameraCenter();

                covisGraphPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
                covisGraphPts.push_back(SLVec3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2)));
            }
        }

        //spanning tree
        WAIKeyFrame* parent = kf->GetParent();
        if (parent)
        {
            cv::Mat Owp = parent->GetCameraCenter();
            spanningTreePts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            spanningTreePts.push_back(SLVec3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2)));
        }

        //loop edges
        std::set<WAIKeyFrame*> loopKFs = kf->GetLoopEdges();
        for (set<WAIKeyFrame*>::iterator sit = loopKFs.begin(), send = loopKFs.end(); sit != send; sit++)
        {
            if ((*sit)->mnId < kf->mnId)
                continue;
            cv::Mat Owl = (*sit)->GetCameraCenter();
            loopEdgesPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            loopEdgesPts.push_back(SLVec3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2)));
        }
    }

    if (_covisibilityGraphMesh)
        _covisibilityGraph->deleteMesh(_covisibilityGraphMesh);

    if (covisGraphPts.size())
    {
        _covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", _covisibilityGraphMat);
        _covisibilityGraph->addMesh(_covisibilityGraphMesh);
        _covisibilityGraph->updateAABBRec();
    }

    if (_spanningTreeMesh)
        _spanningTree->deleteMesh(_spanningTreeMesh);

    if (spanningTreePts.size())
    {
        _spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", _spanningTreeMat);
        _spanningTree->addMesh(_spanningTreeMesh);
        _spanningTree->updateAABBRec();
    }

    if (_loopEdgesMesh)
        _loopEdges->deleteMesh(_loopEdgesMesh);

    if (loopEdgesPts.size())
    {
        _loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", _loopEdgesMat);
        _loopEdges->addMesh(_loopEdgesMesh);
        _loopEdges->updateAABBRec();
    }
#endif
}
