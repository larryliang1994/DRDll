//
//  objecttracker.hpp
//  HelloWorld
//
//  Created by LarryLiang on 15/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#ifndef objecttracker_hpp
#define objecttracker_hpp

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace std;
using namespace cv;

enum TrackingMethod
{
    // 70fps, less accurate
    // 35fps, dont know lose track
    Tracking_CV_BOOSTING    = 0,
    
    // 25fps, more accurate
    // 20fps, dont know lose track
    Tracking_CV_MIL         = 1,
    
    // 40fps, more accurate
    // 80fps, report lose track
    // general purpose
    Tracking_CV_KCF         = 2,
    
    // 30fps, easy to fail
    // 25fps, dont know lose track
    // can find object after occlusion, but too many false positive
    // good for occlusion
    Tracking_CV_TLD         = 3,
    
    // 200fps, easy to fail
    // 180fps, report lose track
    // small and predictable motion
    Tracking_CV_MEDIANFLOW  = 4,
    
    // CNN method, pre-trained
    // fast, not handle occlusion
    Tracking_CV_GOTURN      = 5,
};

class ObjectTracker
{
private:
    Ptr<Tracker> tracker;
    
public:
    void initTracking(Mat frame, Rect2d sourceRect, TrackingMethod method);
    Rect2d tracking(Mat frame, Rect2d sourceRect);
};

#endif /* objecttracker_hpp */
