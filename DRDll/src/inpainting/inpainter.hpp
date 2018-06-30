//
//  inpainter.hpp
//  HelloWorld
//
//  Created by LarryLiang on 08/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#ifndef inpainter_hpp
#define inpainter_hpp

#include <opencv2/opencv.hpp>
#include "criminisi_inpainter.h"
#include "PixMix.h"
#include "fast.hpp"

enum InpaintingMethod
{
    INPAINTING_CV_TELEA     = 0,
    INPAINTING_CV_NS        = 1,
    INPAINTING_EXEMPLAR     = 2,
    INPAINTING_PIXMIX       = 3,
    INPAINTING_FAST         = 4
};

using namespace cv;

namespace Inpainter
{
    Mat reverseMask(Mat mask);
    
    Mat inpainting(Mat image, Mat mask, InpaintingMethod method);
    
    Mat cvInpainting(Mat image, Mat mask, int method);
    Mat exemplarInpainting(Mat image, Mat mask);
    Mat pixmixInpainting(Mat image, Mat mask);
    Mat fastInpainting(Mat image, Mat mask);
}

#endif /* inpainter_hpp */
