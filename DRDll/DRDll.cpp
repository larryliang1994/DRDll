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

extern Mat surroundingRandomisationDistances;
extern Mat transformedSurroundingRandomisationDistances;

extern double surroundingRandomisationMax;
extern double surroundingRandomisationMin;

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

double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);
    
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    
    return 0.5*(1.0 + sign*y);
}

double normalCFD(double value)
{
    return 0.5 * erfc(-value * M_SQRT1_2);
}

double getProbability(double x, double mean, double stddev)
{
    //    double p = exp(-pow(x - mean, 2) / (2 * pow(stddev, 2))) / (stddev * sqrt(2 * M_PI));
    double p = phi(x);
    
//    double max = mean;
//    double min = stddev;
//    double p = (x - min) * 1.0 / (max - min);
    
    return p;
}

void initSurroundingRandomisation(Mat image, Mat dilatedContourMask, Mat rectMask, Rect rect)
{
    Mat mask;
    bitwise_not(dilatedContourMask, mask);
    
    // find all contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
    
    if (contours.size() == 0)
    {
        return;
    }
    
    double largestArea = 0;
    int largestContourIndex = 0;
    
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        double a = contourArea(contours[i],false);  //  Find the area of contour
        if(a > largestArea)
        {
            largestArea = a;
            largestContourIndex = i;                //Store the index of largest contour
        }
    }
    
    surroundingRandomisationDistances.create(image.size(), CV_32FC1);
    surroundingRandomisationDistances.setTo(Scalar(1024));
    
    vector<float> distances_raw;
    
    for (int y = rect.y; y < rect.y + rect.height; y++)
    {
        for (int x = rect.x; x < rect.x + rect.width; x++)
        {
            // in surrounding
            if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 0)
            {
                Point2f current(x, y);
                
                float distance = -pointPolygonTest( contours[largestContourIndex], current, true );
                
                surroundingRandomisationDistances.at<float>(y, x) = distance;
                
                distances_raw.push_back(distance);
            }
        }
    }
    
    // standardise data
    surroundingRandomisationMin = *min_element(distances_raw.begin(), distances_raw.end());
    surroundingRandomisationMax = *max_element(distances_raw.begin(), distances_raw.end());
    double meanVal = accumulate(distances_raw.begin(), distances_raw.end(), 0.0) / distances_raw.size();
    double stddevVal = 0;
    for (int i = 0; i < distances_raw.size(); i++)
    {
        stddevVal += pow(distances_raw[i] - meanVal, 2);
    }
    stddevVal = sqrt(stddevVal / (distances_raw.size() - 1));
    
    for (int y = rect.y; y < rect.y + rect.height; y++)
    {
        for (int x = rect.x; x < rect.x + rect.width; x++)
        {
            // in surrounding
            if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 0)
            {
                float distance = surroundingRandomisationDistances.at<float>(y, x);
                
                distance = (distance - meanVal) / stddevVal;
                
                surroundingRandomisationDistances.at<float>(y, x) = distance;
            }
        }
    }
}

void surroundingRandomisation(Mat image, Mat inpainted, Mat &output, Mat dilatedContourMask, Mat rectMask, Rect rect)
{
    image.copyTo(output);
    
    for (int y = rect.y; y < rect.y + rect.height; y++)
    {
        for (int x = rect.x; x < rect.x + rect.width; x++)
        {
            // in surrounding, do randomisation
            if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 0)
            {
                float distance = transformedSurroundingRandomisationDistances.at<float>(y, x);

                double p = getProbability(distance, 0, 0);
                
                output.at<Vec3b>(y, x) = image.at<Vec3b>(y, x) * p + inpainted.at<Vec3b>(y, x) * (1 - p);
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
}

void resizeImage(Mat image, Mat &outputImage, double &widthRatio, double &heightRatio)
{
    widthRatio  = desiredWidth * 1.0  / image.cols;
    heightRatio = desiredHeight * 1.0 / image.rows;
        
    resize(image, outputImage, Size(desiredWidth, desiredHeight));
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

extern "C" void EXPORT_API currentFrameInpainting(unsigned char* outputData, unsigned char currentImageData[], POINT2D currentBoundingPoint2ds[], POINT2D currentControlPoint2ds[], bool useIlluminationAdaptation, bool useSurroundingRandomisation)
{
//    freopen("debug.txt", "a", stdout);
    
    double start, end;
    
    start = get_timestamp();
    Mat image = imageData2Mat(height, width, channels, currentImageData);
    
    if (channels == 1)
    {
        cvtColor(image, image, CV_GRAY2BGR);
    }
    end = get_timestamp();
//    printf("imageData2Mat time = %f\n", end-start);
    
    double widthRatio = 1, heightRatio = 1;
    Mat currentImage;
    
    if (width != desiredWidth && height != desiredHeight)
    {
        start = get_timestamp();
        resizeImage(image, currentImage, widthRatio, heightRatio);
        end = get_timestamp();
        //    printf("resizeImage 1 time = %f\n", end-start);
    }
    else
    {
        image.copyTo(currentImage);
    }
    
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
//    printf("adaptation time = %f\n", end-start);
    
    start = get_timestamp();
    Mat M = getPerspectiveTransform(frame0BoundingPointsArray, currentBoundingPointsArray);
    Mat transformedInpainted;
    warpPerspective(adaptedInpainted, transformedInpainted, M, Size(desiredWidth, desiredHeight));
    end = get_timestamp();
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
//    printf("copy time = %f\n", end-start);
    
    start = get_timestamp();
    if (useSurroundingRandomisation)
    {
        Mat transformedDilatedContourMask;
        warpPerspective(dilatedContourMask, transformedDilatedContourMask, M, Size(desiredWidth, desiredHeight));
        
        Rect transformedRect;
        Mat transformedRectMask;
        getRectMask(currentImage, currentBoundingPoints, transformedRect, transformedRectMask);
        
        warpPerspective(surroundingRandomisationDistances, transformedSurroundingRandomisationDistances, M, Size(desiredWidth, desiredHeight));
        
        Mat result;
        surroundingRandomisation(currentImage, imageDest, result, transformedDilatedContourMask, transformedRectMask, transformedRect);
        result.copyTo(imageDest);
    }
    end = get_timestamp();
//    printf("surroundingRandomisation time = %f\n\n", end-start);
    
    Mat resizedImageDest;
    if (width != desiredWidth && height != desiredHeight)
    {
        start = get_timestamp();
        resize(imageDest, resizedImageDest, Size(width, height));
        end = get_timestamp();
        //    printf("resize 2 time = %f\n", end-start);
    }
    else
    {
        imageDest.copyTo(resizedImageDest);
    }

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
//    printf("cvtColor time = %f\n", end-start);
}

extern "C" void EXPORT_API initInpainting(unsigned char frame0ImageData[], POINT2D frame0BoundingPoint2ds[], POINT2D frame0ControlPoint2ds[], int method, int parameter, bool useNormalisation)
{
//    freopen("debug.txt", "a", stdout);
    
    double start, end;
    
    start = get_timestamp();
    frame0 = imageData2Mat(height, width, channels, frame0ImageData);
    
    if (channels == 1)
    {
        cvtColor(frame0, frame0, CV_GRAY2BGR);
    }
    
    double widthRatio = 1, heightRatio = 1;
    
    if (width != desiredWidth && height != desiredHeight)
    {
        resizeImage(frame0, frame0, widthRatio, heightRatio);
    }
    
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
    end = get_timestamp();
//    printf("ready time = %f\n", end-start);
    
    start = get_timestamp();
    getRectMask(frame0, frame0BoundingPoints, rect, rectMask);
    
    Mat contourMask;
    getContourMask(frame0, rect, contourMask);
    
    if (channels == 1)
    {
        dilation(contourMask, dilatedContourMask, 2, 20);
    }
    else
    {
        dilation(contourMask, dilatedContourMask, 2, 10);
    }
    
    bitwise_not(dilatedContourMask, dilatedContourMask);
    end = get_timestamp();
//    printf("mask time = %f\n", end-start);
    
    Mat normalisedFrame0;
    if (channels == 3 && useNormalisation)
    {
        normalisedFrame0 = Illumination::normalisation(frame0, rectMask);
    }
    else
    {
        frame0.copyTo(normalisedFrame0);
    }
    
    start = get_timestamp();
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
        
        inpainted = Inpainter::exemplarInpainting(normalisedFrame0, rectMask, sourceMask, parameter);
    }
    else
    {
        DRUtil dRUtil;
        inpainted = dRUtil.inpaint(normalisedFrame0, rectMask, (InpaintingMethod)method, parameter);
    }
    end = get_timestamp();
//    printf("inpaint time = %f\n", end-start);

    start = get_timestamp();
    initSurroundingRandomisation(frame0, dilatedContourMask, rectMask, rect);
    end = get_timestamp();
//    printf("initSurroundingRandomisation time = %f\n", end-start);
    
    start = get_timestamp();
    Illumination::initAdaptation();
    end = get_timestamp();
//    printf("initAdaptation time = %f\n", end-start);
    
    if (channels != 1 && useNormalisation)
    {
        Mat adapted = Illumination::adaptation(normalisedFrame0, frame0, inpainted, rect, frame0ControlPoints, frame0ControlPoints);

        adapted.copyTo(inpainted);
    }
}
