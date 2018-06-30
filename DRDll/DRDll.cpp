//
//  DRDll.cpp
//  DRDll
//
//  Created by LarryLiang on 12/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//


#include "DRDll.h"

DRUtil dRUtil;

double get_timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec*1e-6;
}

vector<InpaintingResult> inpaintingTest(vector<string> imageNames, vector<Mat> images, vector<Mat> masks)
{
    vector<InpaintingResult> results(images.size());
    
    string name;
    Mat image, mask, inpainted;
    double start, end;
    
    for (int i = 0; i < images.size(); i++)
    {
        name  = imageNames[i];
        
        for (int method = 0; method < 4; method++)
        {
            images[i].copyTo(image);
            masks[i].copyTo(mask);
            
            //image = Illumination::normalisation2(image);
            
            start = get_timestamp();
            dRUtil.inpaint(image, mask, (InpaintingMethod)method).copyTo(inpainted);
            end = get_timestamp();
            
            results.push_back(InpaintingResult(name, image, mask, inpainted, (InpaintingMethod)method, end-start));
            
            cout << "image=" << name << "; method=" << method << "; time=" << end-start << "seconds." << endl;
            imwrite("inpaintedm"+to_string((InpaintingMethod)method)+name, inpainted);
        }
    }
    
    return results;
}

vector<InpaintingResult> runInpaintingTest()
{
    vector<string> imageNames{"object1.jpg", "object2.jpg", "object3.jpg"};
    vector<string> imageNamesBound{"object1_bound.jpg", "object2_bound.jpg", "object3_bound.jpg"};
    vector<string> maskNamesFullRect{"object1mask_full_rect.jpg", "object2mask_full_rect.jpg", "object3mask_full_rect.jpg"};
    vector<string> maskNamesFullContour{"object1mask_full_contour.jpg", "object2mask_full_contour.jpg", "object3mask_full_contour.jpg"};
    vector<string> maskNamesBoundContour{"object1mask_bound_contour.jpg", "object2mask_bound_contour.jpg", "object3mask_bound_contour.jpg"};
    
    vector<string> singleImageName{"planar.jpg"};
    vector<string> singleMaskName{"mask.jpg"};
    
    vector<Mat> images, masks;
    
    vector<InpaintingResult> results;
    
    for (int i = 0; i < singleImageName.size(); i++)
    {
        images.push_back(imread(singleImageName[i], IMREAD_COLOR));
        masks.push_back(imread(singleMaskName[i], IMREAD_GRAYSCALE));
    }
    
    results = inpaintingTest(singleImageName, images, masks);
    
    return results;
}

extern "C" int EXPORT_API add(int a, int b)
{
    return a+b;
}

extern "C" void EXPORT_API run()
{
    runInpaintingTest();
}

Mat imageData2Mat(int height, int width, int channels, unsigned char imageData[])
{
    Mat image;
    
    if (channels == 1)
    {
        image = Mat(height, width, CV_8UC1, imageData);
        image.convertTo(image, CV_8UC3);
    }
    else
    {
        image = Mat(height, width, CV_8UC3, imageData);
    }
    
    return image;
}

Rect2d getRect2d(RECT2D source)
{
    Rect2d rect(source.x, source.y, source.width, source.height);
    return rect;
}

Point getPoint(POINT2D source)
{
    Point point(source.x, source.y);
    return point;
}

Point2f getPoint2f(POINT2D source)
{
    Point2f point(source.x, source.y);
    return point;
}

extern "C" int EXPORT_API readImage(int height, int width, int channels, unsigned char imageData[])
{
    Mat image = imageData2Mat(height, width, channels, imageData);
    
    if (image.empty())
    {
        return 0;
    }
    else
    {
        //imshow("show image", image);
        imwrite("captured.jpg", image);
        
        return 1;
    }
}

extern "C" void EXPORT_API drawRect(int height, int width, int channels, unsigned char imageData[], RECT2D bbox)
{
    Mat image = imageData2Mat(height, width, channels, imageData);
    Rect2d rect = getRect2d(bbox);
    
    rectangle(image, rect, Scalar( 255, 0, 0 ), 2, 1 );
    
    imwrite("rect.jpg", image);
}

extern "C" void EXPORT_API initTracking(int height, int width, int channels, unsigned char imageData[], RECT2D bbox, int method)
{
    Mat image = imageData2Mat(height, width, channels, imageData);
    Rect2d rect = getRect2d(bbox);
    
    dRUtil.initTracking(image, rect, (TrackingMethod)method);
}

extern "C" RECT2D EXPORT_API tracking(int height, int width, int channels, unsigned char imageData[], RECT2D bbox)
{
    Mat image = imageData2Mat(height, width, channels, imageData);
    Rect2d rect = getRect2d(bbox);
    
    Rect2d bounding = dRUtil.tracking(image, rect);
    
    RECT2D newBounding;
    newBounding.x = bounding.x;
    newBounding.y = bounding.y;
    newBounding.width = bounding.width;
    newBounding.height = bounding.height;
    
    return newBounding;
}

extern "C" void EXPORT_API averageInpainting(unsigned char* outputData, int height, int width, int channels, unsigned char imageData[], RECT2D bbox)
{
    Mat image = imageData2Mat(height, width, channels, imageData);
    Rect2d rect = getRect2d(bbox);
    
    //Mat roi = Mat(rect.height, rect.width, image.type());
    
//    imshow("before", image);
    
    int neighbour = 3;
    double sum = 0;
//    for (int r = rect.y; r < rect.height + rect.y; r++)
//    {
//        for (int c = rect.x; c < rect.width + rect.x; c++)
//        {
//            sum += (int) image.at<uchar>(r, c);
//        }
//    }
//
    // top
    for (int r = rect.y - neighbour; r < rect.y; r++) {
        if (r > 0) {
            for (int c = rect.x - neighbour; c < rect.x + neighbour; c++) {
                if (c > 0 && c < width)
                {
                    sum += (int) image.at<uchar>(r, c);
                }
            }
        }
    }

    // bottom
    for (int r = rect.height + rect.y; r < rect.height + rect.y + neighbour; r++) {
        if (r > 0 && r < height) {
            for (int c = rect.x - neighbour; c < rect.x + neighbour; c++) {
                if (c > 0 && c < width)
                {
                    sum += (int) image.at<uchar>(r, c);
                }
            }
        }
    }

    // left
    for (int r = rect.y; r < rect.height + rect.y; r++) {
        for (int c = rect.x - neighbour; c < rect.x; c++) {
            if (c > 0)
            {
                sum += (int) image.at<uchar>(r, c);
            }
        }
    }

    // right
    for (int r = rect.y; r < rect.height + rect.y; r++) {
        for (int c = rect.width + rect.x; c < rect.width + rect.x + neighbour; c++) {
            if (c > 0 && c < width)
            {
                sum += (int) image.at<uchar>(r, c);
            }
        }
    }

//    double average = sum / (rect.width * rect.height);
    double average = sum / ((rect.width+2*neighbour) * (rect.height+2*neighbour) -
                            (rect.width * rect.height));
    for (int r = rect.y; r < rect.height + rect.y; r++) {
        for (int c = rect.x; c < rect.width + rect.x; c++) {
            image.at<uchar>(r, c) = (uchar) average;
        }
    }
    
//    imshow("middle", image);
    
//    rectangle(image, rect, Scalar( 255, 0, 0 ), 2, 1 );
    
//    imshow("after", image);
    
//    waitKey(0);
    
    //Convert from RGB to ARGB
    Mat argb_img;
    if (channels == 1)
    {
        cvtColor(image, argb_img, CV_GRAY2BGRA);
    }
    else
    {
        cvtColor(image, argb_img, CV_RGB2BGRA);
    }
    
    vector<Mat> bgra;
    split(argb_img, bgra);
    swap(bgra[0], bgra[3]);
    swap(bgra[1], bgra[2]);
    
    memcpy(outputData, argb_img.data, argb_img.total() * argb_img.elemSize());
}

extern "C" void EXPORT_API fourPointsInpainting(unsigned char* outputData, int height, int width, int channels, unsigned char imageData[], POINT2D fourPoints[])
{
    Mat image = imageData2Mat(height, width, channels, imageData);
    Point points[] = { getPoint(fourPoints[0]), getPoint(fourPoints[1]), getPoint(fourPoints[2]), getPoint(fourPoints[3]) };
    
    line(image, points[0], points[1], Scalar(255), 5);
    line(image, points[1], points[3], Scalar(255), 5);
    line(image, points[3], points[2], Scalar(255), 5);
    line(image, points[2], points[0], Scalar(255), 5);
    
    //Convert from RGB to ARGB
    Mat argb_img;
    if (channels == 1)
    {
        cvtColor(image, argb_img, CV_GRAY2BGRA);
    }
    else
    {
        cvtColor(image, argb_img, CV_RGB2BGRA);
    }
    
    vector<Mat> bgra;
    split(argb_img, bgra);
    swap(bgra[0], bgra[3]);
    swap(bgra[1], bgra[2]);
    
    memcpy(outputData, argb_img.data, argb_img.total() * argb_img.elemSize());
}

extern "C" void EXPORT_API tempFourPointsInpainting(unsigned char* outputData, int height, int width, int channels, unsigned char inpaintedImageData[], POINT2D frame0FourPoints[], unsigned char currentImageData[], POINT2D currentFourPoints[])
{
    Mat inpainted = imageData2Mat(height, width, channels, inpaintedImageData);
    Mat currentImage = imageData2Mat(height, width, channels, currentImageData);
    
    Point2f frame0PointsArray[] = { getPoint2f(frame0FourPoints[0]), getPoint2f(frame0FourPoints[1]), getPoint2f(frame0FourPoints[2]), getPoint2f(frame0FourPoints[3]) };
    Point2f currentPointsArray[] = { getPoint2f(currentFourPoints[0]), getPoint2f(currentFourPoints[1]), getPoint2f(currentFourPoints[2]), getPoint2f(currentFourPoints[3]) };
    
//    vector<Point> frame0Points;
//    frame0Points.push_back(getPoint(frame0FourPoints[0]));
//    frame0Points.push_back(getPoint(frame0FourPoints[1]));
//    frame0Points.push_back(getPoint(frame0FourPoints[3]));
//    frame0Points.push_back(getPoint(frame0FourPoints[2]));
    
    vector<Point> currentPoints;
    currentPoints.push_back(getPoint(currentFourPoints[0]));
    currentPoints.push_back(getPoint(currentFourPoints[1]));
    currentPoints.push_back(getPoint(currentFourPoints[3]));
    currentPoints.push_back(getPoint(currentFourPoints[2]));
    
    Mat M = getPerspectiveTransform(frame0PointsArray, currentPointsArray);
    Mat transformed;
    warpPerspective(inpainted, transformed, M, Size(width, height));
    
    Mat currentMask;
    currentMask.create(currentImage.size(), CV_8UC1);
    currentMask.setTo(Scalar(0));
    
    // Create Polygon from vertices
    vector<Point> currentROI_Poly;
    approxPolyDP(currentPoints, currentROI_Poly, 1.0, true);
    
    // Fill polygon white
    fillConvexPoly(currentMask, currentROI_Poly, Scalar(255));
    
    // Cut out ROI and store it in imageDest
    Mat imageDest;
    currentImage.copyTo(imageDest);
    transformed.copyTo(imageDest, currentMask);
    
    //Convert from RGB to ARGB
    Mat argb_img;
    if (channels == 1)
    {
        cvtColor(imageDest, argb_img, CV_GRAY2BGRA);
    }
    else
    {
        cvtColor(imageDest, argb_img, CV_RGB2BGRA);
    }
    
    vector<Mat> bgra;
    split(argb_img, bgra);
    swap(bgra[0], bgra[3]);
    swap(bgra[1], bgra[2]);
    
    memcpy(outputData, argb_img.data, argb_img.total() * argb_img.elemSize());
}

extern "C" void EXPORT_API initFourPointsInpainting(unsigned char* outputData, int height, int width, int channels, unsigned char frame0ImageData[], POINT2D frame0FourPoints[], int method)
{
    Mat frame0 = imageData2Mat(height, width, channels, frame0ImageData);
    
    vector<Point> frame0Points;
    frame0Points.push_back(getPoint(frame0FourPoints[0]));
    frame0Points.push_back(getPoint(frame0FourPoints[1]));
    frame0Points.push_back(getPoint(frame0FourPoints[3]));
    frame0Points.push_back(getPoint(frame0FourPoints[2]));
    
    Mat frame0Mask;
    frame0Mask.create(frame0.size(), CV_8UC1);
    frame0Mask.setTo(Scalar(255));
    
    // Create Polygon from vertices
    vector<Point> frame0ROI_Poly;
    approxPolyDP(frame0Points, frame0ROI_Poly, 1.0, true);
    
    // Fill polygon white
    fillConvexPoly(frame0Mask, frame0ROI_Poly, Scalar(0));
    
    Mat inpainted;
    cvtColor(frame0, inpainted, CV_GRAY2BGR);
    
    DRUtil dRUtil;
    inpainted = dRUtil.inpaint(inpainted, frame0Mask, (InpaintingMethod)method);
    cvtColor(inpainted, inpainted, CV_BGR2GRAY);

    memcpy(outputData, inpainted.data, inpainted.total() * inpainted.elemSize());
}
