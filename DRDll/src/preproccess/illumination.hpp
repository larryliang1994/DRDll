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

using namespace cv;
using namespace std;

namespace Illumination
{
    float medianMat(Mat Input);
    Mat getMedian(Mat image, Mat mask, int neighbourSize);
    Mat normalisation(Mat image, Mat mask);
    Mat normalisation2(Mat image);
}

#endif /* illumination_hpp */
