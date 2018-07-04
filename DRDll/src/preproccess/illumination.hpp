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

using namespace cv;
using namespace std;

namespace Illumination
{
    uchar medianMat(Mat Input);
    uchar averageMat(Mat input);
    Mat getMedian(Mat image, Mat mask, int neighbourSize);
    Mat normalisation(Mat image, Mat mask);
    Mat normalisation2(Mat image);
    Mat compensation(Mat source, Mat inpainted, Mat mask);
}

#endif /* illumination_hpp */
