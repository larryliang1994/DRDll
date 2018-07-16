//
//  illumination.hpp
//  HelloWorld
//
//  Created by LarryLiang on 09/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#ifndef illumination_hpp
#define illumination_hpp

#include <opencv2/opencv.hpp>
#include "inpainter.hpp"
#include <sys/time.h>

using namespace cv;
using namespace std;

namespace Illumination
{
    double get_timestamp();
    
    Mat normalisation(Mat image, Mat mask);
    Mat compensation(Mat source, Mat inpainted, Mat mask);
    void initAdaptation();
    Mat adaptation(Mat frame0, Mat current, Mat inpainted, Rect bbox, vector<Point> frame0ControlPoints, vector<Point> currentControlPoints);
};

#endif /* illumination_hpp */

