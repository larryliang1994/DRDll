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
    float medianMat(Mat input)
    {
        // spread Input Mat to single row
        Mat output = Mat(1, input.rows*input.cols, CV_32FC1);
        
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                output.at<float>(0, i*input.cols+j) = input.at<float>(i, j);
            }
        }
        
        cv::sort(output, output, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
        
        //cout << output << endl;
        
        vector<float> vecFromMat;
        output.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
        
        std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
        return vecFromMat[vecFromMat.size() / 2];
    }
    
    Mat getMedian(Mat image, Mat mask, int neighbourSize)
    {
        Mat expandedImage;
        
        copyMakeBorder(image, expandedImage,
                       neighbourSize/2, neighbourSize/2, neighbourSize/2, neighbourSize/2, BORDER_REPLICATE);
    
        Mat output = Mat(image.rows, image.cols, CV_32FC3);
        
        for (int y = neighbourSize/2; y < expandedImage.rows - neighbourSize/2; y++)
        {
            for (int x = neighbourSize/2; x < expandedImage.cols - neighbourSize/2; x++)
            {
                // background, in source mask
                if (mask.at<uchar>(y - neighbourSize/2, x - neighbourSize/2) == 0)
                {
                    Mat target = expandedImage(Rect(x - neighbourSize/2, y - neighbourSize/2, neighbourSize, neighbourSize));
                    
//                    cout << expandedImage.depth() << "  " << expandedImage.channels() << endl;
//                    cout << target.depth() << "  " << target.channels() << endl;
                    
                    vector<Mat> planes(3);
                    split(target, planes);
                    
                    float median0 = medianMat(planes[0]);
                    float median1 = medianMat(planes[1]);
                    float median2 = medianMat(planes[2]);
                    
                    output.at<Vec3f>(y - neighbourSize/2, x - neighbourSize/2)[0] = median0;
                    output.at<Vec3f>(y - neighbourSize/2, x - neighbourSize/2)[1] = median1;
                    output.at<Vec3f>(y - neighbourSize/2, x - neighbourSize/2)[2] = median2;
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
        median = getMedian(image, mask, 3);

        vector<Mat> source_planes(3);
        split(source, source_planes);

        vector<Mat> median_planes(3);
        split(median, median_planes);

        vector<Mat> output_planes(3);

        output_planes[0] = source_planes[0] - median_planes[0] + mean(median).val[0];
        output_planes[1] = source_planes[1];// - median_planes[1] + mean(median).val[1];
        output_planes[2] = source_planes[2];// - median_planes[2] + mean(median).val[2];

        merge(output_planes, output);

        //normalize(output, output);
        //output = source;
        cvtColor(output, output, CV_YUV2BGR);
        
        output.convertTo(output, CV_8U);
        imshow("output", output);
        
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
}
