//
//  inpainter.cpp
//  HelloWorld
//
//  Created by LarryLiang on 08/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "inpainter.hpp"

namespace Inpainter
{
    Mat reverseMask(Mat mask)
    {
        for (int i = 0; i < mask.rows; i++)
        {
            for (int j = 0; j < mask.cols; j++)
            {
                if ((int)mask.at<uchar>(i, j) == 0)
                {
                    mask.at<uchar>(i, j) = 255;
                }
                else
                {
                    mask.at<uchar>(i, j) = 0;
                }
            }
        }
        
        return mask;
    }
    
    Mat inpainting(Mat image, Mat mask, InpaintingMethod method)
    {
        Mat output;
        
        switch(method)
        {
            case INPAINTING_CV_TELEA:
                output = cvInpainting(image, mask, INPAINT_TELEA);
                break;
                
            case INPAINTING_CV_NS:
                output = cvInpainting(image, mask, INPAINT_NS);
                break;
                
            case INPAINTING_EXEMPLAR:
                output = exemplarInpainting(image, mask);
                break;
                
            case INPAINTING_PIXMIX:
                output = pixmixInpainting(image, mask);
                break;
                
            case INPAINTING_FAST:
                output = fastInpainting(image, mask);
                break;
                
            default:
                output = image;
                break;
        }
        
        return output;
    }
    
    Mat cvInpainting(Mat image, Mat mask, int method)
    {
        Mat output;
        
        mask = reverseMask(mask);
        
        inpaint(image, mask, output, 5.0, method);
        
        return output;
    }
    
    Mat exemplarInpainting(Mat image, Mat mask)
    {
        Mat output;
        image.copyTo(output);
        
        mask = reverseMask(mask);
        
        Mat sourceMask = Mat();
        sourceMask.setTo(Scalar(0));
        
        //imwrite("mask.png", mask);
        
        Inpaint::inpaintCriminisi(output, mask, sourceMask, 40);
        
        return output;
    }
    
    Mat pixmixInpainting(Mat image, Mat mask)
    {
        Mat_<Vec3b> output;
        
        //mask = reverseMask(mask);
        
        PixMix pm;
        pm.init(image, mask);
        
        pm.execute(output, 0.05f);
        
        return output;
    }
    
    Mat fastInpainting(Mat image, Mat mask)
    {
        Mat output;
        
        output = Fast::inpaint(image, mask);
        
        return output;
    }
}
