//
//  myinpainting.cpp
//  HelloWorld
//
//  Created by LarryLiang on 01/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "drutil.hpp"

Mat DRUtil::selection(Mat image)
{
    Mat mask;
    
    mask = Preproccessor::selection(image);
    
    return mask;
}

Rect DRUtil::getRect()
{
    return Preproccessor::getRect();
}

Mat DRUtil::inpaint(Mat image, Mat mask, InpaintingMethod method)
{
    Mat output;
    
    output = Inpainter::inpainting(image, mask, method);
    
    return output;
}

void DRUtil::initTracking(Mat frame, Rect2d bbox, TrackingMethod method)
{
    objectTracker.initTracking(frame, bbox, method);
}

Rect2d DRUtil::tracking(Mat frame, Rect2d bbox)
{
    // Update the tracking result
    Rect2d bounding = objectTracker.tracking(frame, bbox);
    return bounding;
}
