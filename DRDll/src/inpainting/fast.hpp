//
//  fast.hpp
//  HelloWorld
//
//  Created by LarryLiang on 08/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#ifndef fast_hpp
#define fast_hpp

#include <opencv2/opencv.hpp>

using namespace cv;

namespace Fast
{
    Mat inpaint(Mat image, Mat mask);
}

#endif /* fast_hpp */
