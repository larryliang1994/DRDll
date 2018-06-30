//
//  objecttracker.cpp
//  HelloWorld
//
//  Created by LarryLiang on 15/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "objecttracker.hpp"

void ObjectTracker::initTracking(Mat frame, Rect2d sourceRect, TrackingMethod method)
{
    switch (method) {
        case Tracking_CV_BOOSTING:
            tracker = TrackerBoosting::create();
            break;
            
        case Tracking_CV_MIL:
            tracker = TrackerMIL::create();
            break;
            
        case Tracking_CV_KCF:
            tracker = TrackerKCF::create();
            break;
            
        case Tracking_CV_TLD:
            tracker = TrackerTLD::create();
            break;
            
        case Tracking_CV_MEDIANFLOW:
            tracker = TrackerMedianFlow::create();
            break;
            
        case Tracking_CV_GOTURN:
            tracker = TrackerGOTURN::create();
            break;
            
        default:
            break;
    }
    
    tracker->init(frame, sourceRect);
}

Rect2d ObjectTracker::tracking(Mat frame, Rect2d sourceRect)
{
    Rect2d outputRect = sourceRect;
    
    // Update the tracking result
    bool ok = tracker->update(frame, outputRect);
    
    if (!ok)
    {
        outputRect = Rect(-1, -1, -1, -1);
    }
    
    return outputRect;
}
