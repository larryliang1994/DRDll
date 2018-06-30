//
//  fast.cpp
//  HelloWorld
//
//  Created by LarryLiang on 08/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "fast.hpp"

namespace Fast
{
    Mat inpaint(Mat image, Mat mask)
    {
        Mat output;
        int maxNumOfIter = 100;
        float a = 0.073235f;
        float b = 0.176765f;
        
        Mat kernel = (Mat_<float>(3, 3) <<
                      a, b, a,
                      b, 0.0f, b,
                      a, b, a);
        
        if (mask.type() != CV_8UC3)
        {
            cvtColor(mask, mask, CV_GRAY2RGB);
            mask.convertTo(mask, CV_8UC3);
        }
        
        assert(image.type() == mask.type() && mask.type() == CV_8UC3);
        assert(image.size() == mask.size());
        assert(kernel.type() == CV_32F);
        
        // fill in the missing region with the input's average color
        auto avgColor = sum(image) / (image.cols * image.rows);
        Mat avgColorMat(1, 1, CV_8UC3);
        avgColorMat.at<Vec3b>(0, 0) = Vec3b(avgColor[0], avgColor[1], avgColor[2]);
        resize(avgColorMat, avgColorMat, image.size(), 0.0, 0.0, INTER_NEAREST);
        Mat result = (mask / 255).mul(image) + (1 - mask / 255).mul(avgColorMat);
        
        // convolution
        int bSize = kernel.cols / 2;
        Mat kernel3ch, inWithBorder;
        result.convertTo(result, CV_32FC3);
        cvtColor(kernel, kernel3ch, COLOR_GRAY2BGR);
        
        copyMakeBorder(result, inWithBorder, bSize, bSize, bSize, bSize, BORDER_REPLICATE);
        Mat resInWithBorder = Mat(inWithBorder, Rect(bSize, bSize, result.cols, result.rows));
        
        const int ch = result.channels();
        for (int itr = 0; itr < maxNumOfIter; ++itr)
        {
            copyMakeBorder(result, inWithBorder, bSize, bSize, bSize, bSize, BORDER_REPLICATE);
            
            for (int r = 0; r < result.rows; ++r)
            {
                const uchar *pMask = mask.ptr(r);
                float *pRes = result.ptr<float>(r);
                for (int c = 0; c < result.cols; ++c)
                {
                    if (pMask[ch * c] == 0)
                    {
                        Rect rectRoi(c, r, kernel.cols, kernel.rows);
                        Mat roi(inWithBorder, rectRoi);
                        
                        auto sum = cv::sum(kernel3ch.mul(roi));
                        pRes[ch * c + 0] = sum[0];
                        pRes[ch * c + 1] = sum[1];
                        pRes[ch * c + 2] = sum[2];
                    }
                }
            }
            
            // for debugging
            //        if (itr%10 == 0)
            //        {
            //            imshow("Inpainting...", result / 255.0f);
            //            waitKey(0);
            //        }
        }
        
        result.convertTo(output, CV_8UC3);
        
        return output;
    }
}
