//
//  preproccessor.hpp
//  HelloWorld
//
//  Created by LarryLiang on 08/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#ifndef preproccessor_hpp
#define preproccessor_hpp

#include <opencv2/opencv.hpp>
#include "grabcut.h"
#include "illumination.hpp"

using namespace cv;
using namespace std;

namespace Preproccessor
{
    Mat selection(Mat image);
    Rect getRect();
    
    Mat createMask(Mat image);
    void createSourceMask(Mat image, Mat mask, int neighbourSize, Mat &newImage, Mat &newMask, Rect &boundingBox);
}

#endif /* preproccessor_hpp */
