//
//  illumination.cpp
//  HelloWorld
//
//  Created by LarryLiang on 09/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "illumination.hpp"

extern int illuminationBlockSize;

extern Mat frame0;
extern Mat inpainted;
extern Rect rect;
extern Mat rectMask;
extern vector<Point> frame0ControlPoints;
extern int controlPointSize;

vector<uchar> frame0ControlPointsMedian;
Mat frame0Y;
Mat inpaintedY;
vector<Mat> inpainted_planes(3);

vector<vector<int>> eightControlPointsMapping;

double Illumination::get_timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec*1e-6;
}

float distancePoint2Line(int x1, int y1, int x2, int y2, int x0, int y0)
{
    float numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1);
    float denominator = sqrt(pow((y2 - y1), 2) + pow((x2 - x1), 2));
    
    float distance = numerator / denominator;
    
    return distance;
}

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
            // if (mask.at<uchar>(y - blockSize/2, x - blockSize/2) == 255)
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

Mat Illumination::normalisation(Mat image, Mat mask)
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

void Illumination::initAdaptation()
{
    Mat inpaintedYUV;
    cvtColor(inpainted, inpaintedYUV, CV_BGR2YUV);
    split(inpaintedYUV, inpainted_planes);
    inpaintedY = inpainted_planes[0];
    
    Mat frame0YUV;
    cvtColor(frame0, frame0YUV, CV_BGR2YUV);
    extractChannel(frame0YUV, frame0Y, 0);
    
    // calculate Median of each control points block M(ci)
    for (int i = 0; i < frame0ControlPoints.size(); i++)
    {
        int x = frame0ControlPoints[i].x - illuminationBlockSize / 2;
        int y = frame0ControlPoints[i].y - illuminationBlockSize / 2;
        
        if (x < 0) { x = 0; }
        if (y < 0) { y = 0; }
        
        int regionSizeWidth = illuminationBlockSize;
        if (x + illuminationBlockSize >= frame0Y.cols)
        {
            regionSizeWidth = frame0Y.cols - x;
        }
        
        int regionSizeHeight = illuminationBlockSize;
        if (y + illuminationBlockSize >= frame0Y.rows)
        {
            regionSizeHeight = frame0Y.rows - y;
        }
        
        Mat input = frame0Y(Rect(x, y, regionSizeWidth, regionSizeHeight));
        
        frame0ControlPointsMedian.push_back(medianMat(input));
    }

    // calculate the mapping of a point in roi to eight control points index
    int corner = controlPointSize / 4;
    float minDistance = INT_MAX;
    int minDistanceIndex = 0;
    Point frame0ControlPoint;
    float distance;
    
    for (int row = rect.y; row < rect.y + rect.height; row++)
    {
        for (int col = rect.x; col < rect.x + rect.width; col++)
        {
            if (rectMask.at<uchar>(row, col) == 0)
            {
                vector<int> pointIndex;
                
                // top
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider left, top, and right
                    if (!(i > corner * 2 && i < corner * 3))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a vertical line
                        distance = distancePoint2Line(col, row, col, 0, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // right
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider top, right and bottom
                    if (!(i > corner * 3 && i < corner * 4))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a horizonal line
                        distance = distancePoint2Line(col, row, 0, row, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // bottom
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider right, bottom and left
                    if (!(i > corner * 0 && i < corner * 1))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a vertical line
                        distance = distancePoint2Line(col, row, col, 0, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // left
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider bottom, left and top
                    if (!(i > corner * 1 && i < corner * 2))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a horizonal line
                        distance = distancePoint2Line(col, row, 0, row, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // top left
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider top and left
                    if ((i >= corner * 0 && i <= corner * 1) || (i >= corner * 3 && i <= corner * 4))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a \ line
                        distance = distancePoint2Line(col, row, col - 2, row - 2, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // top right
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider top and right
                    if ((i >= corner * 0 && i <= corner * 1) || (i >= corner * 1 && i <= corner * 2))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a / line
                        distance = distancePoint2Line(col, row, col + 2, row - 2, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // bottom right
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider bottom and right
                    if ((i >= corner * 2 && i <= corner * 3) || (i >= corner * 1 && i <= corner * 2))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a \ line
                        distance = distancePoint2Line(col, row, col - 2, row - 2, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // bottom left
                for (int i = 0; i < frame0ControlPoints.size(); i++)
                {
                    // only consider bottom and left
                    if ((i >= corner * 2 && i <= corner * 3) || (i >= corner * 3 && i < corner * 4))
                    {
                        frame0ControlPoint = frame0ControlPoints[i];
                        
                        // a / line
                        distance = distancePoint2Line(col, row, col + 2, row - 2, frame0ControlPoint.x, frame0ControlPoint.y);
                        
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minDistanceIndex = i;
                        }
                    }
                }
                pointIndex.push_back(minDistanceIndex);
                minDistance = INT_MAX;
                minDistanceIndex = 0;
                
                // complete a point
                eightControlPointsMapping.push_back(pointIndex);
            }
        }
    }
}

Mat Illumination::adaptation(Mat frame0, Mat current, Mat inpainted, Rect bbox, vector<Point> frame0ControlPoints, vector<Point> currentControlPoints)
{
    double start, end;
    
    start = get_timestamp();
    // extract Y channel
    Mat currentYUV;
    cvtColor(current, currentYUV, CV_BGR2YUV);
    Mat currentY;
    extractChannel(currentYUV, currentY, 0);
    
    Mat outputY;
    inpaintedY.copyTo(outputY);
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("copy time = %f\n", end-start);
    
    // calculate Median of each control points block M(ci)
    vector<uchar> currentControlPointsMedian;
    
    start = get_timestamp();
    for (int i = 0; i < currentControlPoints.size(); i++)
    {
        if (currentControlPoints[i].x < 0 || currentControlPoints[i].y < 0)
        {
            currentControlPointsMedian.push_back(0);
            continue;
        }
        
        int x = currentControlPoints[i].x - illuminationBlockSize / 2;
        int y = currentControlPoints[i].y - illuminationBlockSize / 2;
        
        if (x < 0) { x = 0; }
        if (y < 0) { y = 0; }
        
        int regionSizeWidth = illuminationBlockSize;
        if (x + illuminationBlockSize >= currentY.cols)
        {
            regionSizeWidth = currentY.cols - x;
        }
        
        int regionSizeHeight = illuminationBlockSize;
        if (y + illuminationBlockSize >= currentY.rows)
        {
            regionSizeHeight = currentY.rows - y;
        }
        
        Mat input = currentY(Rect(x, y, regionSizeWidth, regionSizeHeight));
        
        currentControlPointsMedian.push_back(medianMat(input));
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("current median time = %f\n", end-start);
    
    int currentEightControlPointsMappingCount = 0;
    
    start = get_timestamp();
    // Ix = (sum((x - ci)^-4) * Ici) / (sum((x - ci)^-4))
    // Ici = Mcurrent - Mframe0
    for (int row = bbox.y; row < bbox.y + bbox.height; row++)
    {
        for (int col = bbox.x; col < bbox.x + bbox.width; col++)
        {
            if (rectMask.at<uchar>(row, col) == 0)
            {
                // for a point
                Point currentPoint = Point(col, row);
                
                double Ix = 0;
                double numerator = 0;
                double denominator = 0;
                
                double normResult = 0;
                double normPowResult = 0;
                
                vector<int> currentMapping = eightControlPointsMapping[currentEightControlPointsMappingCount];
                currentEightControlPointsMappingCount++;
                
                for (int i = 0; i < currentMapping.size(); i++)
                {
                    int index = currentMapping[i];
                    
                    // this control point is not visible now
                    if (currentControlPoints[index].x < 0 || currentControlPoints[index].y < 0)
                    {
                        continue;
                    }
                    
                    normResult = norm(currentPoint-currentControlPoints[index]);
                    normPowResult = 1.0 / (normResult * normResult * normResult * normResult);
                    
                    // pow is too slow
                    //denominator += pow(normResult, -4);
                    denominator += normPowResult;
                    
                    int Ici = currentControlPointsMedian[index] - frame0ControlPointsMedian[index];
                    
                    //numerator += (pow(normResult, -4) * Ici);
                    numerator += normPowResult * Ici;
                }
                
                if (numerator == 0 || denominator == 0)
                {
                    Ix = 0;
                }
                else
                {
                    Ix = numerator / denominator;
                }
                
                outputY.at<uchar>(row, col) = inpaintedY.at<uchar>(row, col) + Ix;
            }
        }
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("Ix time = %f\n", end-start);
    
    start = get_timestamp();
    // merge back
    Mat output;
    vector<Mat> output_planes(3);
    output_planes[0] = outputY;
    output_planes[1] = inpainted_planes[1];
    output_planes[2] = inpainted_planes[2];
    
    merge(output_planes, output);
    
    cvtColor(output, output, CV_YUV2BGR);
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("merge time = %f\n", end-start);
    
    return output;
}

Mat Illumination::compensation(Mat source, Mat inpainted, Mat mask)
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
    
    sourceAllY = Inpainter::inpainting(sourceAllY, mask, InpaintingMethod::INPAINTING_PIXMIX);
    
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
    
    return output;
}

