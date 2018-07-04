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
    
    Mat inpainting(Mat image, Mat mask, InpaintingMethod method, int parameter)
    {
        Mat output;
        
        switch(method)
        {
            case INPAINTING_CV_TELEA:
                output = cvInpainting(image, mask, INPAINT_TELEA, parameter);
                break;
                
            case INPAINTING_CV_NS:
                output = cvInpainting(image, mask, INPAINT_NS, parameter);
                break;
                
            case INPAINTING_EXEMPLAR:
                output = exemplarInpainting(image, mask, parameter);
                break;
                
            case INPAINTING_PIXMIX:
                output = pixmixInpainting(image, mask, parameter);
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
    
    Mat cvInpainting(Mat image, Mat mask, int method, int inpaintRadius)
    {
        if (inpaintRadius == -1)
        {
            inpaintRadius = 15;
        }
        
        Mat output;
        
        mask = reverseMask(mask);
        
        inpaint(image, mask, output, inpaintRadius, method);
        
        return output;
    }
    
    Mat exemplarInpainting(Mat image, Mat mask, int patchSize)
    {
        if (patchSize == -1)
        {
            patchSize = 31;
        }
        
        Mat output;
        image.copyTo(output);
        
        Mat newMask;
        mask.copyTo(newMask);
        newMask = reverseMask(newMask);
        
        Mat sourceMask;
        sourceMask.create(image.size(), CV_8UC1);
        sourceMask.setTo(Scalar(0));
        
        //imwrite("mask.png", mask);
        
        Inpaint::inpaintCriminisi(output, newMask, sourceMask, patchSize);
        
        return output;
    }
    
    Mat pixmixInpainting(Mat image, Mat mask, int iteration)
    {
        if (iteration == -1)
        {
            iteration = 2;
        }
        
        Mat_<Vec3b> output;
        
        //mask = reverseMask(mask);
        
        PixMix pm;
        pm.init(image, mask);
        
        pm.execute(output, 0.05f, iteration);
        
        return output;
    }
    
    Mat fastInpainting(Mat image, Mat mask)
    {
        Mat output;
        
        output = Fast::inpaint(image, mask);
        
        return output;
    }
}
