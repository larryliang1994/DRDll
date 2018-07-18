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

vector<Vec3i> frame0ControlPointsMedian;
//Mat frame0Y;
//Mat inpaintedY;
vector<Mat> inpainted_planes(3);

vector<vector<int>> eightControlPointsMapping;

Mat inpaintedYUV;

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
//    int sum = 0;
//    for (int i = 0; i < input.rows; i++)
//    {
//        for (int j = 0; j < input.cols; j++)
//        {
//            sum += (int)input.at<uchar>(i, j);
//        }
//    }
//
//    int average = (int)(1.0 * sum / (input.rows * input.cols));
    
    return mean(input).val[0];
    
    //return (uchar)average;
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
    Mat sourceYUV;
    cvtColor(image, sourceYUV, CV_BGR2YUV);
    
    Mat medianYUV;
    medianBlur(sourceYUV, medianYUV, 31);
    
    Mat inter;
    cvtColor(medianYUV, inter, CV_YUV2BGR);
    
//    imshow("medianBGR", inter);
    
    sourceYUV.convertTo(sourceYUV, CV_16S);
    medianYUV.convertTo(medianYUV, CV_16S);
    
    vector<Mat> output_planes(3);
    
    vector<Mat> source_planes(3);
    split(sourceYUV, source_planes);
    
    vector<Mat> median_planes(3);
    split(medianYUV, median_planes);
    
    output_planes[0] = source_planes[0] - median_planes[0] + mean(median_planes[0]).val[0];
    output_planes[1] = source_planes[1] - median_planes[1] + mean(median_planes[1]).val[0];
    output_planes[2] = source_planes[2] - median_planes[2] + mean(median_planes[2]).val[0];
    
    // normalise to 0-255
    double minY, maxY;
    double minY_, maxY_;
    double aY, bY;
    
    // Y
    minMaxLoc(output_planes[0], &minY, &maxY);
    minMaxLoc(source_planes[0], &minY_, &maxY_);
    
    aY = (maxY_ - minY_) / (maxY - minY);
    bY = minY_ - aY * minY;
    
    output_planes[0].convertTo(output_planes[0], CV_8U, aY, bY);
    
//    freopen("debug.txt", "a", stdout);
//    printf("minY = %f, maxY = %f, minY_ = %f, maxY_ = %f\n", minY, maxY, minY_, maxY_);
    
    // U
    double minU, maxU;
    double minU_, maxU_;
    double aU, bU;
    
    minMaxLoc(output_planes[1], &minU, &maxU);
    minMaxLoc(source_planes[1], &minU_, &maxU_);
    
    aU = (maxU_ - minU_) / (maxU - minU);
    bU = minU_ - aU * minU;
    
    output_planes[1].convertTo(output_planes[1], CV_8U, aU, bU);
    
//    freopen("debug.txt", "a", stdout);
//    printf("minU = %f, maxU = %f, minU_ = %f, maxU_ = %f\n", minU, maxU, minU_, maxU_);
    
    // V
    double minV, maxV;
    double minV_, maxV_;
    double aV, bV;
    
    minMaxLoc(output_planes[2], &minV, &maxV);
    minMaxLoc(source_planes[2], &minV_, &maxV_);
    
    aV = (maxV_ - minV_) / (maxV - minV);
    bV = minV_ - aV * minV;
    
    output_planes[2].convertTo(output_planes[2], CV_8U, aV, bV);
    
//    freopen("debug.txt", "a", stdout);
//    printf("minV = %f, maxV = %f, minV_ = %f, maxV_ = %f\n", minV, maxV, minV_, maxV_);
    
    Mat output;
    merge(output_planes, output);
    
//    output.convertTo(output, CV_8U);
    
//    output_planes[0] = source_planes[0] - median_planes[0] + mean(median_planes[0]).val[0];
    
    //        cout << mean(median_planes[0]).val[0] << endl;
    
    // normalise to 0-255
//    double min, max;
//    minMaxLoc(output_planes[0], &min, &max);
//
//    double min_, max_;
//    minMaxLoc(source_planes[0], &min_, &max_);
//
//    double a = (max_ - min_) / (max - min);
//    double b = min_ - a * min;
//
//    output_planes[0].convertTo(output_planes[0], CV_8U, a, b);
//
//    output_planes[1] = source_planes[1];// - median_planes[1] + mean(median).val[1];
//    output_planes[2] = source_planes[2];// - median_planes[2] + mean(median).val[2];
//
//    merge(output_planes, output);
    cvtColor(output, output, CV_YUV2BGR);
    
    //        imshow("output", output);
    //
    //        imwrite("output1.jpg", output);
    
    return output;
}

void Illumination::initAdaptation()
{
    cvtColor(inpainted, inpaintedYUV, CV_BGR2YUV);
    split(inpaintedYUV, inpainted_planes);
//    inpaintedY = inpainted_planes[0];
    
    Mat frame0YUV;
    cvtColor(frame0, frame0YUV, CV_BGR2YUV);
    vector<Mat> frame0_planes;
    split(frame0YUV, frame0_planes);
//    extractChannel(frame0YUV, frame0Y, 0);
    
    // calculate Median of each control points block M(ci)
    for (int i = 0; i < frame0ControlPoints.size(); i++)
    {
        int x = frame0ControlPoints[i].x - illuminationBlockSize / 2;
        int y = frame0ControlPoints[i].y - illuminationBlockSize / 2;
        
        if (x < 0) { x = 0; }
        if (y < 0) { y = 0; }
        
        int regionSizeWidth = illuminationBlockSize;
        if (x + illuminationBlockSize >= frame0YUV.cols)
        {
            regionSizeWidth = frame0YUV.cols - x;
        }
        
        int regionSizeHeight = illuminationBlockSize;
        if (y + illuminationBlockSize >= frame0YUV.rows)
        {
            regionSizeHeight = frame0YUV.rows - y;
        }
        
        Mat inputY = (frame0_planes[0])(Rect(x, y, regionSizeWidth, regionSizeHeight));
        Mat inputU = (frame0_planes[1])(Rect(x, y, regionSizeWidth, regionSizeHeight));
        Mat inputV = (frame0_planes[2])(Rect(x, y, regionSizeWidth, regionSizeHeight));
        
        frame0ControlPointsMedian.push_back(Vec3i(medianMat(inputY), medianMat(inputU), medianMat(inputV)));
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
//    Mat currentY;
    vector<Mat> current_planes;
    split(currentYUV, current_planes);
//    extractChannel(currentYUV, currentY, 0);
    
    Mat outputYUV;
    inpaintedYUV.copyTo(outputYUV);
    
//    Mat outputY;
//    inpaintedY.copyTo(outputY);
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("copy time = %f\n", end-start);
    
    // calculate Median of each control points block M(ci)
    vector<Vec3i> currentControlPointsMedian;
    
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
        if (x + illuminationBlockSize >= currentYUV.cols)
        {
            regionSizeWidth = currentYUV.cols - x;
        }
        
        int regionSizeHeight = illuminationBlockSize;
        if (y + illuminationBlockSize >= currentYUV.rows)
        {
            regionSizeHeight = currentYUV.rows - y;
        }
        
//        Mat input = currentY(Rect(x, y, regionSizeWidth, regionSizeHeight));
//
//        currentControlPointsMedian.push_back(medianMat(input));
        
        Mat inputY = (current_planes[0])(Rect(x, y, regionSizeWidth, regionSizeHeight));
        Mat inputU = (current_planes[1])(Rect(x, y, regionSizeWidth, regionSizeHeight));
        Mat inputV = (current_planes[2])(Rect(x, y, regionSizeWidth, regionSizeHeight));
        
        currentControlPointsMedian.push_back(Vec3i(medianMat(inputY), medianMat(inputU), medianMat(inputV)));
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("current median time = %f\n", end-start);
    
    int currentEightControlPointsMappingCount = 0;
    
    start = get_timestamp();
    // Ix = (sum((x - ci)^-4) * Ici) / (sum((x - ci)^-4))
    // Ici = Mcurrent - Mframe0
    freopen("debug.txt", "a", stdout);
    for (int row = bbox.y; row < bbox.y + bbox.height; row++)
    {
        for (int col = bbox.x; col < bbox.x + bbox.width; col++)
        {
            if (rectMask.at<uchar>(row, col) == 0)
            {
                // for a point
                Point currentPoint = Point(col, row);
                
                double IxY = 0, IxU = 0, IxV = 0;
                double numeratorY = 0, numeratorU = 0, numeratorV = 0;
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
                    
                    if (frame0ControlPointsMedian[index][0] < 0 || frame0ControlPointsMedian[index][0] > 255
                        || frame0ControlPointsMedian[index][1] < 0 || frame0ControlPointsMedian[index][1] > 255
                        || frame0ControlPointsMedian[index][2] < 0 || frame0ControlPointsMedian[index][2] > 255)
                    {
                        printf("frame0ControlPointsMedian %d = %d %d %d\n", index,
                               frame0ControlPointsMedian[index][0], frame0ControlPointsMedian[index][1], frame0ControlPointsMedian[index][2]);
                        continue;
                    }
                    
                    if (currentControlPointsMedian[index][0] < 0 || currentControlPointsMedian[index][0] > 255
                        || currentControlPointsMedian[index][1] < 0 || currentControlPointsMedian[index][1] > 255
                        || currentControlPointsMedian[index][2] < 0 || currentControlPointsMedian[index][2] > 255)
                    {
                        printf("currentControlPointsMedian %d = %d %d %d\n", index,
                               currentControlPointsMedian[index][0], currentControlPointsMedian[index][1], currentControlPointsMedian[index][2]);
                        continue;
                    }
                    
                    
                    normResult = norm(currentPoint - currentControlPoints[index]);
                    normPowResult = 1.0 / (normResult * normResult * normResult * normResult);
                    
                    // pow is too slow
                    // denominator += pow(normResult, -4);
                    denominator += normPowResult;
                    
                    Vec3i Ici = currentControlPointsMedian[index] - frame0ControlPointsMedian[index];
//                    printf("Ici = %d, %d, %d\n", Ici[0], Ici[1], Ici[2]);
                    
                    // numerator += (pow(normResult, -4) * Ici);
                    numeratorY += normPowResult * Ici[0];
                    numeratorU += normPowResult * Ici[1];
                    numeratorV += normPowResult * Ici[2];
                }
                
                if (denominator == 0)
                {
                    IxY = 0;
                    IxU = 0;
                    IxV = 0;
                }
                else
                {
                    IxY = numeratorY / denominator;
                    IxU = numeratorU / denominator;
                    IxV = numeratorV / denominator;
                }
                
                outputYUV.at<Vec3b>(row, col) = (Vec3b)((Vec3f)inpaintedYUV.at<Vec3b>(row, col) + Vec3f(IxY, IxU, IxV));
            }
        }
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("Ix time = %f\n", end-start);
    
    start = get_timestamp();
    // merge back
    Mat output;
//    vector<Mat> output_planes(3);
//    output_planes[0] = outputY;
//    output_planes[1] = inpainted_planes[1];
//    output_planes[2] = inpainted_planes[2];
    
//    merge(output_planes, output);
    
    cvtColor(outputYUV, output, CV_YUV2BGR);
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

