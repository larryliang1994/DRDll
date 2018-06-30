//
//  myinpainting.hpp
//  HelloWorld
//
//  Created by LarryLiang on 01/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#ifndef drutil_hpp
#define drutil_hpp

#include <opencv2/opencv.hpp>
#include <iostream>

#include "grabcut.h"
#include "preproccessor.hpp"
#include "objecttracker.hpp"
#include "inpainter.hpp"

using namespace cv;
using namespace std;

class DRUtil
{
    Mat mask;
    Mat inpainted;
    ObjectTracker objectTracker;
    
public:
    Mat selection(Mat image);
    Rect getRect();
    
    Mat inpaint(Mat image, Mat mask, InpaintingMethod method);
    void initTracking(Mat frame, Rect2d bbox, TrackingMethod method);
    Rect2d tracking(Mat frame, Rect2d bbox);
};

#endif /* drutil_hpp */
