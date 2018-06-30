//
//  inpaintingresult.h
//  HelloWorld
//
//  Created by LarryLiang on 06/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#ifndef inpaintingresult_h
#define inpaintingresult_h

#include <opencv2/opencv.hpp>
#include <iostream>
#include "inpainter.hpp"

using namespace std;
using namespace cv;

class InpaintingResult
{
public:
    string name;
    Mat image;
    Mat mask;
    Mat inpainted;
    InpaintingMethod method;
    double time;
    
    InpaintingResult();
    InpaintingResult(string name, Mat image, Mat mask, Mat inpainted, InpaintingMethod method, double time);
};

#endif /* inpaintingresult_h */
