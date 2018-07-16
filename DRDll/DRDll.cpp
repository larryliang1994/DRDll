//
//  DRDll.cpp
//  DRDll
//
//  Created by LarryLiang on 12/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//


#include "DRDll.h"

extern int desiredWidth;
extern int desiredHeight;

extern int height;
extern int width;
extern int channels;

extern int controlPointSize;

extern int illuminationBlockSize;

extern vector<Point> frame0BoundingPoints;
extern vector<Point> frame0ControlPoints;
extern Point2f frame0BoundingPointsArray[4];

extern Mat frame0;

extern Mat inpainted;
extern Mat dilatedContourMask;
extern Rect rect;
extern Mat rectMask;

DRUtil dRUtil;

double get_timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec*1e-6;
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

Point getPoint(POINT2D source, double widthRatio, double heightRatio)
{
    Point point(source.x * widthRatio, source.y * heightRatio);
    return point;
}

Point2f getPoint2f(POINT2D source)
{
    Point2f point(source.x, source.y);
    return point;
}

Point2f getPoint2f(POINT2D source, double widthRatio, double heightRatio)
{
    Point2f point(source.x * widthRatio, source.y * heightRatio);
    return point;
}

void getRectMask(Mat image, vector<Point> points, Rect &rect, Mat &rectMask)
{
    int xMin = INT_MAX, yMin = INT_MAX, xMax = INT_MIN, yMax = INT_MIN;
    for (int i = 0; i < points.size(); i++)
    {
        Point point = points[i];
        if (point.x < xMin) { xMin = point.x; }
        if (point.y < yMin) { yMin = point.y; }
        if (point.x > xMax) { xMax = point.x; }
        if (point.y > yMax) { yMax = point.y; }
    }
    
    rect = Rect(xMin, yMin, xMax - xMin, yMax - yMin);
    
    rectMask.create(image.size(), CV_8UC1);
    rectMask.setTo(Scalar(255));
    
    // Create Polygon from vertices
    vector<Point> ROI_Poly;
    approxPolyDP(points, ROI_Poly, 1.0, true);
    
    // Fill polygon white
    fillConvexPoly(rectMask, ROI_Poly, Scalar(0));
}

void getContourMask(Mat srcImage, Rect rect, Mat &contourMask)
{
    Mat image;
    float kdata[] = {-1,-1,-1, -1,9,-1, -1,-1,-1};
    Mat kernel(3,3,CV_32F, kdata);
    filter2D(srcImage, image, -1, kernel);
    
    Mat mask, bgdModel, fgdModel;
    grabCut( image, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT );
    
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            if ((int)mask.at<uchar>(i, j) == GC_PR_FGD)
            {
                mask.at<uchar>(i, j) = 255;
            }
            else
            {
                mask.at<uchar>(i, j) = 0;
            }
        }
    }
    
    // find bounding boxes for all contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    double largest_area=0;
    int largest_contour_index=0;
    
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        double a = contourArea( contours[i],false);  //  Find the area of contour
        if(a > largest_area)
        {
            largest_area = a;
            largest_contour_index = i;                //Store the index of largest contour
        }
    }
    
    contourMask = Mat(mask.rows, mask.cols, CV_8UC1, Scalar::all(0));
    
    if (channels == 1)
    {
        for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
        {
            drawContours( contourMask, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy );
        }
    }
    else
    {
        drawContours( contourMask, contours,largest_contour_index, Scalar(255,255,255), CV_FILLED, 8, hierarchy );
    }
}

void dilation(Mat image, Mat &dilatedMask, int dilation_elem = 2, int dilation_size = 10)
{
    Mat element = getStructuringElement( (MorphShapes)dilation_elem,
                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        Point( dilation_size, dilation_size ) );
    /// Apply the dilation operation
    dilate( image, dilatedMask, element );
}

double getProbability(double x, double mean, double stddev)
{
    //    double p = exp(-pow(x - mean, 2) / (2 * pow(stddev, 2))) / (stddev * sqrt(2 * M_PI));
    
    double max = mean;
    double min = stddev;
    double p = (x - min) * 1.0 / (max - min);
    
    return p;
}

void surroundingRandomisation(Mat image, Mat inpainted, Mat &output, Mat dilatedContourMask, Mat rectMask, Rect rect)
{
    double start, end;
    
    start = get_timestamp();
    Mat mask;
    bitwise_not(dilatedContourMask, mask);
    
    // find all contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    if (contours.size() == 0)
    {
        inpainted.copyTo(output);
        return;
    }
    
    double largest_area=0;
    int largest_contour_index=0;
    
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        double a = contourArea(contours[i],false);  //  Find the area of contour
        if(a > largest_area)
        {
            largest_area = a;
            largest_contour_index = i;                //Store the index of largest contour
        }
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("contour time = %f\n", end-start);
    
    image.copyTo(output);
    
    vector<double> distances_raw;
    
    vector<Vec3f> distances;
    
    start = get_timestamp();
    for (int y = rect.y; y < rect.y + rect.height; y++)
    {
        for (int x = rect.x; x < rect.x + rect.width; x++)
        {
            // in surrounding
            if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 0)
            {
                Point2i current(x, y);
                
                double distance_raw = -pointPolygonTest(contours[largest_contour_index], current, true);
                
                distances_raw.push_back(distance_raw);
                
//                distances.push_back(Vec3f(x, y, distance_raw));
            }
        }
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("distance time = %f\n", end-start);
    
    start = get_timestamp();
    // standardise data
    double min = *min_element(distances_raw.begin(), distances_raw.end());
    double max = *max_element(distances_raw.begin(), distances_raw.end());
    double mean = accumulate(distances_raw.begin(), distances_raw.end(), 0.0) / distances_raw.size();
    double stddev = 0;
    for (int i = 0; i < distances.size(); i++)
    {
        stddev += pow(distances_raw[i] - mean, 2);
    }
    stddev = sqrt(stddev / (distances_raw.size() - 1));
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("standardise time = %f\n", end-start);
    
    srand ((unsigned)time(NULL));
    
    int distancesCount = 0;
    start = get_timestamp();
    for (int y = rect.y; y < rect.y + rect.height; y++)
    {
        for (int x = rect.x; x < rect.x + rect.width; x++)
        {
            // in surrounding, do randomisation
            if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 0)
            {
                Point2i current(x, y);
                
                //double distance = -pointPolygonTest(contours[largest_contour_index], current, true);
                
                double distance = distances_raw[distancesCount];
                distancesCount++;

                double p = getProbability(distance, max, min);
                
                output.at<Vec3b>(y, x) = image.at<Vec3b>(y, x) * p + inpainted.at<Vec3b>(y, x) * (1 - p);

//                if ((rand() * 1.0 / RAND_MAX) < p)
//                {
//                    //output.at<Vec3b>(y, x) = image.at<Vec3b>(y, x);
//                }
//                else
//                {
//                    output.at<Vec3b>(y, x) = inpainted.at<Vec3b>(y, x);
//                }
            }
            // in between, use original
            else if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 255)
            {
                //output.at<Vec3b>(y, x) = image.at<Vec3b>(y, x);
            }
            // in roi, just copy inpainted
            else
            {
                output.at<Vec3b>(y, x) = inpainted.at<Vec3b>(y, x);
            }
        }
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("output time = %f\n", end-start);
}

void resizeImage(Mat image, Mat &outputImage, double &widthRatio, double &heightRatio)
{
    widthRatio  = desiredWidth * 1.0  / image.cols;
    heightRatio = desiredHeight * 1.0 / image.rows;
        
    resize(image, outputImage, Size(desiredWidth, desiredHeight));
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
//        imwrite("captured.jpg", image);
        
        return 1;
    }
}

extern "C" void EXPORT_API drawRect(int height, int width, int channels, unsigned char imageData[], RECT2D bbox)
{
    Mat image = imageData2Mat(height, width, channels, imageData);
    Rect2d rect = getRect2d(bbox);
    
    rectangle(image, rect, Scalar( 255, 0, 0 ), 2, 1 );
    
//    imwrite("rect.jpg", image);
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

extern "C" void EXPORT_API initParameters(int h, int w, int c, int dh, int dw, int cps, int ibs)
{
    height = h;
    width = w;
    channels = c;
    desiredHeight = dh;
    desiredWidth = dw;
    controlPointSize = cps;
    illuminationBlockSize = ibs;
}

extern "C" void EXPORT_API tempFourPointsInpainting(unsigned char* outputData, unsigned char currentImageData[], POINT2D currentBoundingPoint2ds[], POINT2D currentControlPoint2ds[], bool useIlluminationAdaptation, bool useSurroundingRandomisation)
{
    Mat image = imageData2Mat(height, width, channels, currentImageData);
    
    if (channels == 1)
    {
        cvtColor(image, image, CV_GRAY2BGR);
    }
    
    double start, end;
    
    double widthRatio = 1, heightRatio = 1;
    Mat currentImage;
    
    start = get_timestamp();
    resizeImage(image, currentImage, widthRatio, heightRatio);
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("resizeImage 1 time = %f\n", end-start);
    
    vector<Point> currentBoundingPoints;
    currentBoundingPoints.push_back(getPoint(currentBoundingPoint2ds[0], widthRatio, heightRatio));
    currentBoundingPoints.push_back(getPoint(currentBoundingPoint2ds[1], widthRatio, heightRatio));
    currentBoundingPoints.push_back(getPoint(currentBoundingPoint2ds[3], widthRatio, heightRatio));
    currentBoundingPoints.push_back(getPoint(currentBoundingPoint2ds[2], widthRatio, heightRatio));
    
    Point2f currentBoundingPointsArray[4];
    currentBoundingPointsArray[0] = getPoint2f(currentBoundingPoint2ds[0], widthRatio, heightRatio);
    currentBoundingPointsArray[1] = getPoint2f(currentBoundingPoint2ds[1], widthRatio, heightRatio);
    currentBoundingPointsArray[2] = getPoint2f(currentBoundingPoint2ds[3], widthRatio, heightRatio);
    currentBoundingPointsArray[3] = getPoint2f(currentBoundingPoint2ds[2], widthRatio, heightRatio);
    
    for (int i = 0; i < 4; i++)
    {
        if (currentBoundingPoints[i].x >= desiredWidth || currentBoundingPoints[i].y >= desiredHeight
            || currentBoundingPoints[i].x < 0 || currentBoundingPoints[i].y < 0)
        {
            Mat resizedImageDest;
            resize(currentImage, resizedImageDest, Size(width, height));
            
            Mat argb_img;
            cvtColor(resizedImageDest, argb_img, CV_BGR2BGRA);
            
            vector<Mat> bgra;
            split(argb_img, bgra);
            swap(bgra[0], bgra[3]);
            swap(bgra[1], bgra[2]);
            
            memcpy(outputData, argb_img.data, argb_img.total() * argb_img.elemSize());
            
            return;
        }
    }
    
    vector<Point> currentControlPoints;
    for (int i = 0; i < controlPointSize; i++)
    {
        currentControlPoints.push_back(getPoint(currentControlPoint2ds[i], widthRatio, heightRatio));
    }
    
    for (int i = 0; i < controlPointSize; i++)
    {
        if (currentControlPoints[i].x >= desiredWidth || currentControlPoints[i].y >= desiredHeight
            || currentControlPoints[i].x < 0 || currentControlPoints[i].y < 0)
        {
            currentControlPoints[i].x = -1;
            currentControlPoints[i].y = -1;
        }
    }
    
    start = get_timestamp();
    Mat adaptedInpainted;
    if (useIlluminationAdaptation)
    {
        adaptedInpainted = Illumination::adaptation(frame0, currentImage, inpainted, rect, frame0ControlPoints, currentControlPoints);
    }
    else
    {
        inpainted.copyTo(adaptedInpainted);
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("adaptation time = %f\n", end-start);
    
    start = get_timestamp();
    Mat M = getPerspectiveTransform(frame0BoundingPointsArray, currentBoundingPointsArray);
    Mat transformedInpainted;
    warpPerspective(adaptedInpainted, transformedInpainted, M, Size(desiredWidth, desiredHeight));
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("warpPerspective time = %f\n", end-start);
    
    start = get_timestamp();
    Mat currentMask;
    currentMask.create(currentImage.size(), CV_8UC1);
    currentMask.setTo(Scalar(0));

    // Create Polygon from vertices
    vector<Point> currentROI_Poly;
    approxPolyDP(currentBoundingPoints, currentROI_Poly, 1.0, true);
    
    // Fill polygon white
    fillConvexPoly(currentMask, currentROI_Poly, Scalar(255));

    // Cut out ROI and store it in imageDest
    Mat imageDest;
    currentImage.copyTo(imageDest);
    transformedInpainted.copyTo(imageDest, currentMask);
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("copy time = %f\n", end-start);
    
    start = get_timestamp();
    if (useSurroundingRandomisation)
    {
        Mat transformedDilatedContourMask;
        warpPerspective(dilatedContourMask, transformedDilatedContourMask, M, Size(desiredWidth, desiredHeight));
        
        Rect transformedRect;
        Mat transformedRectMask;
        getRectMask(currentImage, currentBoundingPoints, transformedRect, transformedRectMask);
        
        Mat result;
        surroundingRandomisation(currentImage, imageDest, result, transformedDilatedContourMask, transformedRectMask, transformedRect);
        result.copyTo(imageDest);
    }
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("surroundingRandomisation time = %f\n", end-start);
    
    Mat resizedImageDest;
    
    start = get_timestamp();
    resize(imageDest, resizedImageDest, Size(width, height));
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("resize 2 time = %f\n", end-start);
    
    start = get_timestamp();
    // Convert from RGB to ARGB
    Mat argb_img;
    cvtColor(resizedImageDest, argb_img, CV_BGR2BGRA);
    
    vector<Mat> bgra;
    split(argb_img, bgra);
    swap(bgra[0], bgra[3]);
    swap(bgra[1], bgra[2]);
    
    memcpy(outputData, argb_img.data, argb_img.total() * argb_img.elemSize());
    end = get_timestamp();
//    freopen("debug.txt", "a", stdout);
//    printf("cvtColor time = %f\n", end-start);
}

extern "C" void EXPORT_API initFourPointsInpainting(unsigned char frame0ImageData[], POINT2D frame0BoundingPoint2ds[], POINT2D frame0ControlPoint2ds[], int method, int parameter, bool useSurroundingRandomisation)
{
    frame0 = imageData2Mat(height, width, channels, frame0ImageData);
    
    if (channels == 1)
    {
        cvtColor(frame0, frame0, CV_GRAY2BGR);
    }
    
    double widthRatio = 1, heightRatio = 1;
    resizeImage(frame0, frame0, widthRatio, heightRatio);
    
    frame0BoundingPoints.push_back(getPoint(frame0BoundingPoint2ds[0], widthRatio, heightRatio));
    frame0BoundingPoints.push_back(getPoint(frame0BoundingPoint2ds[1], widthRatio, heightRatio));
    frame0BoundingPoints.push_back(getPoint(frame0BoundingPoint2ds[3], widthRatio, heightRatio));
    frame0BoundingPoints.push_back(getPoint(frame0BoundingPoint2ds[2], widthRatio, heightRatio));
    
    frame0BoundingPointsArray[0] = getPoint2f(frame0BoundingPoint2ds[0], widthRatio, heightRatio);
    frame0BoundingPointsArray[1] = getPoint2f(frame0BoundingPoint2ds[1], widthRatio, heightRatio);
    frame0BoundingPointsArray[2] = getPoint2f(frame0BoundingPoint2ds[3], widthRatio, heightRatio);
    frame0BoundingPointsArray[3] = getPoint2f(frame0BoundingPoint2ds[2], widthRatio, heightRatio);
    
    for (int i = 0; i < controlPointSize; i++)
    {
        frame0ControlPoints.push_back(getPoint(frame0ControlPoint2ds[i], widthRatio, heightRatio));
    }
    
    getRectMask(frame0, frame0BoundingPoints, rect, rectMask);
    
    Mat contourMask;
    getContourMask(frame0, rect, contourMask);
    
    dilation(contourMask, dilatedContourMask, 2, 20);
    
    bitwise_not(dilatedContourMask, dilatedContourMask);
    
//    imshow("rectMask", rectMask);
//    imshow("dilatedContourMask", dilatedContourMask);
    
    if (((InpaintingMethod)method) == InpaintingMethod::INPAINTING_EXEMPLAR)
    {
        Mat sourceMask;
        sourceMask.create(frame0.size(), CV_8UC1);
        sourceMask.setTo(Scalar(0));
        
        for (int row = rect.y - 50; row < rect.y + rect.height + 50; row++)
        {
            for (int col = rect.x - 50; col < rect.x + rect.width + 50; col++)
            {
                if (row >= 0 && row < sourceMask.rows && col >= 0 && col < sourceMask.cols)
                {
                    if (!(row >= rect.y && row <= rect.y + rect.height && col >= rect.x && col <= rect.x + rect.width))
                    {
                        sourceMask.at<uchar>(row, col) = 255;
                    }
                }
            }
        }
        
        inpainted = Inpainter::exemplarInpainting(frame0, rectMask, sourceMask, parameter);
    }
    else
    {
        DRUtil dRUtil;
        inpainted = dRUtil.inpaint(frame0, rectMask, (InpaintingMethod)method, parameter);
    }
    
    if (useSurroundingRandomisation)
    {
        Mat result;
        surroundingRandomisation(frame0, inpainted, result, dilatedContourMask, rectMask, rect);
        result.copyTo(inpainted);
    }
    
//    imshow("inpainted", inpainted);
    
    Illumination::initAdaptation();
}
