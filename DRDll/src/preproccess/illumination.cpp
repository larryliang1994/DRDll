//
//  illumination.cpp
//  HelloWorld
//
//  Created by LarryLiang on 09/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "illumination.hpp"

namespace Illumination
{
    uchar medianMat(Mat input)
    {
        // spread Input Mat to single row
        Mat output = Mat(1, input.rows*input.cols, CV_8UC1);
        
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                output.at<uchar>(0, i*input.cols+j) = input.at<uchar>(i, j);
            }
        }
        
        cv::sort(output, output, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
        
        //cout << output << endl;
        
        vector<uchar> vecFromMat;
        output.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
        
        std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
        return (uchar)vecFromMat[vecFromMat.size() / 2];
    }
    
    uchar averageMat(Mat input)
    {
        int sum = 0;
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                sum += (int)input.at<uchar>(i, j);
            }
        }
        
        int average = (int)(1.0 * sum / (input.rows * input.cols));
        
        return (uchar)average;
    }
    
    Mat getMedian(Mat image, Mat mask, int blockSize)
    {
        Mat expandedImage;
        
        copyMakeBorder(image, expandedImage, blockSize/2, blockSize/2, blockSize/2, blockSize/2, BORDER_REPLICATE);
    
        Mat output = Mat(image.rows, image.cols, CV_8UC1);
        
        for (int y = blockSize/2; y < expandedImage.rows - blockSize/2; y++)
        {
            for (int x = blockSize/2; x < expandedImage.cols - blockSize/2; x++)
            {
                // background, in source mask
//                if (mask.at<uchar>(y - blockSize/2, x - blockSize/2) == 255)
                {
                    Mat target = expandedImage(Rect(x - blockSize/2, y - blockSize/2, blockSize, blockSize));

                    vector<Mat> planes(3);
                    split(target, planes);
                    
                    uchar median = medianMat(planes[0]);

                    output.at<uchar>(y - blockSize/2, x - blockSize/2) = median;
                }
            }
        }
        
        return output;
    }
    
    Mat normalisation(Mat image, Mat mask)
    {
        // The illumination normalization is carried out in the YUV color space.
        // The size of the median block should be large enough to represent global illumination.
        Mat output;

        Mat source;
        cvtColor(image, source, CV_BGR2YUV);
        
        Mat median;
//        median = getMedian(source, mask, 21);
        
        medianBlur(source, median, 21);

        vector<Mat> source_planes(3);
        split(source, source_planes);

        vector<Mat> median_planes(1);
        split(median, median_planes);

        vector<Mat> output_planes(3);
        
        source_planes[0].convertTo(source_planes[0], CV_16S);
        median_planes[0].convertTo(median_planes[0], CV_16S);

        output_planes[0] = source_planes[0] - median_planes[0] + mean(median_planes[0]).val[0];
        
//        cout << mean(median_planes[0]).val[0] << endl;
        
        // normalise to 0-255
        double min, max;
        minMaxLoc(output_planes[0], &min, &max);
        
        double min_, max_;
        minMaxLoc(source_planes[0], &min_, &max_);
        
        double a = (max_ - min_) / (max - min);
        double b = min_ - a * min;
        
        output_planes[0].convertTo(output_planes[0], CV_8U, a, b);
        
        output_planes[1] = source_planes[1];// - median_planes[1] + mean(median).val[1];
        output_planes[2] = source_planes[2];// - median_planes[2] + mean(median).val[2];
        
        merge(output_planes, output);
        cvtColor(output, output, CV_YUV2BGR);

//        imshow("output", output);
//
//        imwrite("output1.jpg", output);
        
        return output;
    }
    
    Mat normalisation2(Mat image)
    {
        // READ RGB color image and convert it to Lab
        cv::Mat bgr_image = image;
        cv::Mat lab_image;
        cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);
        
        // Extract the L channel
        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]
        
        // apply the CLAHE algorithm to the L channel
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(3);
        cv::Mat dst;
        clahe->apply(lab_planes[0], dst);
        
        // Merge the the color planes back into an Lab image
        dst.copyTo(lab_planes[0]);
        cv::merge(lab_planes, lab_image);
        
        // convert back to RGB
        cv::Mat image_clahe;
        cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
        
        // display the results  (you might also want to see lab_planes[0] before and after).
//        cv::imshow("image original", bgr_image);
//        cv::imshow("image CLAHE", image_clahe);
//        cv::waitKey();
        
        return image_clahe;
    }
    
    Mat compensation(Mat source, Mat inpainted, Mat mask)
    {
        Mat sourceYUV;
        cvtColor(source, sourceYUV, CV_BGR2YUV);
        
        vector<Mat> source_planes(3);
        split(sourceYUV, source_planes);
        
        source_planes[0].copyTo(source_planes[1]);
        source_planes[0].copyTo(source_planes[2]);
        
        Mat sourceAllY;
        merge(source_planes, sourceAllY);
        
        cvtColor(sourceAllY, sourceAllY, CV_YUV2BGR);
        
//        imshow("AllY", sourceAllY);
        
        sourceAllY = Inpainter::inpainting(sourceAllY, mask, InpaintingMethod::INPAINTING_PIXMIX);
        
//        imshow("Inpainted AllY", sourceAllY);
        
        cvtColor(sourceAllY, sourceAllY, CV_BGR2YUV);
        split(sourceAllY, source_planes);
        
        Mat inpaintedYUV;
        cvtColor(inpainted, inpaintedYUV, CV_BGR2YUV);
        
        vector<Mat> output_planes(3);
        split(inpaintedYUV, output_planes);
        
        output_planes[0] = source_planes[0];
        
        Mat output;
        merge(output_planes, output);
        
        cvtColor(output, output, CV_YUV2BGR);
        
//        imshow("Merged", output);
        
//        imshow("inpainted", inpainted);
        
//        waitKey(0);
        
        return output;
    }
}
