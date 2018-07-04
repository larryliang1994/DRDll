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

void getContourMask(Mat image, Rect rect, Mat &contourMask)
{
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
    
    int largest_area=0;
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
//    drawContours( contourMask, contours,largest_contour_index, Scalar(255,255,255), CV_FILLED, 8, hierarchy );
    
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        drawContours( contourMask, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy );
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
    image.copyTo(output);
    
    Point2i centre(rect.x + rect.width / 2, rect.y + rect.height / 2);
    
    vector<double> distances;
    
    for (int y = rect.y; y < rect.y + rect.height; y++)
    {
        for (int x = rect.x; x < rect.x + rect.width; x++)
        {
            // in surrounding
            if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 0)
            {
                Point2i current(x, y);
                
                double xDistance = abs(current.x - centre.x) * 1.0 / rect.width;
                double yDistance = abs(current.y - centre.y) * 1.0 / rect.height;
                
                double distance = sqrt(pow(xDistance, 2) + pow(yDistance, 2));
                
                distances.push_back(distance);
            }
        }
    }
    
    double min = *min_element(distances.begin(), distances.end());
    double max = *max_element(distances.begin(), distances.end());
    double mean = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    double stddev = 0;
    for (int i = 0; i < distances.size(); i++)
    {
        stddev += pow(distances[i] - mean, 2);
    }
    stddev = sqrt(stddev / (distances.size() - 1));

    srand ((unsigned)time(NULL));
    
    for (int y = rect.y; y < rect.y + rect.height; y++)
    {
        for (int x = rect.x; x < rect.x + rect.width; x++)
        {
            // in surrounding, do randomisation
            if (dilatedContourMask.at<uchar>(y, x) == 255 && rectMask.at<uchar>(y, x) == 0)
            {
                Point2i current(x, y);
                
                double xDistance = abs(current.x - centre.x) * 1.0 / rect.width;
                double yDistance = abs(current.y - centre.y) * 1.0 / rect.height;
                
                double distance = sqrt(pow(xDistance, 2) + pow(yDistance, 2));
                
                double p = getProbability(distance, max, min);
                
                //                cout << p << endl;
                
                //                int pixel = p * 255;
                //                output.at<Vec3b>(y, x) = Vec3b(pixel, pixel, pixel);
                
                if ((rand() * 1.0 / RAND_MAX) < p)
                {
                    //output.at<Vec3b>(y, x) = image.at<Vec3b>(y, x);
                }
                else
                {
                    output.at<Vec3b>(y, x) = inpainted.at<Vec3b>(y, x);
                }
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

extern "C" void EXPORT_API tempFourPointsInpainting(unsigned char* outputData, int height, int width, int channels, unsigned char inpaintedImageData[], unsigned char maskData[], POINT2D frame0FourPoints[], unsigned char currentImageData[], POINT2D currentFourPoints[])
{
    Mat inpainted = imageData2Mat(height, width, channels, inpaintedImageData);
    Mat currentImage = imageData2Mat(height, width, channels, currentImageData);
    Mat dilatedContourMask = imageData2Mat(height, width, 1, maskData);
    
    if (channels == 1)
    {
        cvtColor(inpainted, inpainted, CV_GRAY2BGR);
        cvtColor(currentImage, currentImage, CV_GRAY2BGR);
    }
    
    Point2f frame0PointsArray[] = { getPoint2f(frame0FourPoints[0]), getPoint2f(frame0FourPoints[1]), getPoint2f(frame0FourPoints[2]), getPoint2f(frame0FourPoints[3]) };
    Point2f currentPointsArray[] = { getPoint2f(currentFourPoints[0]), getPoint2f(currentFourPoints[1]), getPoint2f(currentFourPoints[2]), getPoint2f(currentFourPoints[3]) };

    vector<Point> currentPoints;
    currentPoints.push_back(getPoint(currentFourPoints[0]));
    currentPoints.push_back(getPoint(currentFourPoints[1]));
    currentPoints.push_back(getPoint(currentFourPoints[3]));
    currentPoints.push_back(getPoint(currentFourPoints[2]));
    
    Mat M = getPerspectiveTransform(frame0PointsArray, currentPointsArray);
    Mat transformedInpainted;
    warpPerspective(inpainted, transformedInpainted, M, Size(width, height));

    Mat currentMask;
    currentMask.create(currentImage.size(), CV_8UC1);
    currentMask.setTo(Scalar(0));

    // Create Polygon from vertices
    vector<Point> currentROI_Poly;
    approxPolyDP(currentPoints, currentROI_Poly, 1.0, true);
    
    // Fill polygon white
    fillConvexPoly(currentMask, currentROI_Poly, Scalar(255));

    Rect rect;
    Mat rectMask;
    getRectMask(currentImage, currentPoints, rect, rectMask);

    Mat transformedDilatedContourMask;
    warpPerspective(dilatedContourMask, transformedDilatedContourMask, M, Size(width, height));

    Mat output;
    surroundingRandomisation(currentImage, transformedInpainted, output, transformedDilatedContourMask, rectMask, rect);

    // Cut out ROI and store it in imageDest
    Mat imageDest;
    currentImage.copyTo(imageDest);
    output.copyTo(imageDest, currentMask);

    // Convert from RGB to ARGB
    Mat argb_img;
//    if (channels == 1)
//    {
//        cvtColor(imageDest, argb_img, CV_GRAY2BGRA);
//    }
//    else
//    {
        cvtColor(imageDest, argb_img, CV_RGB2BGRA);
//    }
    
    vector<Mat> bgra;
    split(argb_img, bgra);
    swap(bgra[0], bgra[3]);
    swap(bgra[1], bgra[2]);
    
    //return argb_img.data;
    
    memcpy(outputData, argb_img.data, argb_img.total() * argb_img.elemSize());
}

extern "C" void EXPORT_API initFourPointsInpainting(unsigned char* resultData, unsigned char* inpaintedData, unsigned char* maskData, int height, int width, int channels, unsigned char frame0ImageData[], POINT2D frame0FourPoints[], int method, int parameter)
{
    Mat frame0 = imageData2Mat(height, width, channels, frame0ImageData);
    
    if (channels == 1)
    {
        cvtColor(frame0, frame0, CV_GRAY2BGR);
    }
    
    vector<Point> frame0Points;
    frame0Points.push_back(getPoint(frame0FourPoints[0]));
    frame0Points.push_back(getPoint(frame0FourPoints[1]));
    frame0Points.push_back(getPoint(frame0FourPoints[3]));
    frame0Points.push_back(getPoint(frame0FourPoints[2]));
    
    Rect rect;
    Mat rectMask;
    getRectMask(frame0, frame0Points, rect, rectMask);
    
    Mat contourMask;
    getContourMask(frame0, rect, contourMask);

    Mat dilatedContourMask;
    dilation(contourMask, dilatedContourMask, 2, 20);
    
    bitwise_not(dilatedContourMask, dilatedContourMask);
    
    DRUtil dRUtil;
    Mat inpainted = dRUtil.inpaint(frame0, rectMask, (InpaintingMethod)method, parameter);

//    Mat result;
//    surroundingRandomisation(frame0, inpainted, result, dilatedContourMask, rectMask, rect);
    
    if (channels == 1)
    {
//        cvtColor(result, result, CV_BGR2GRAY);
        cvtColor(inpainted, inpainted, CV_BGR2GRAY);
    }
    
//    memcpy(resultData, result.data, result.total() * result.elemSize());
    memcpy(inpaintedData, inpainted.data, inpainted.total() * inpainted.elemSize());
    memcpy(maskData, dilatedContourMask.data, dilatedContourMask.total() * dilatedContourMask.elemSize());
    
//    return output.data;
    
//    Mat frame0Mask;
//    frame0Mask.create(frame0.size(), CV_8UC1);
//    frame0Mask.setTo(Scalar(255));
//
//    // Create Polygon from vertices
//    vector<Point> frame0ROI_Poly;
//    approxPolyDP(frame0Points, frame0ROI_Poly, 1.0, true);
//
//    // Fill polygon white
//    fillConvexPoly(frame0Mask, frame0ROI_Poly, Scalar(0));
//
//    Mat inpainted;
//
//    if (channels == 1)
//    {
//        cvtColor(frame0, inpainted, CV_GRAY2BGR);
//    }
//    else
//    {
//        frame0.copyTo(inpainted);
//    }
//
//    DRUtil dRUtil;
//    inpainted = dRUtil.inpaint(inpainted, frame0Mask, (InpaintingMethod)method);
//
//    if (channels == 1)
//    {
//        cvtColor(inpainted, inpainted, CV_BGR2GRAY);
//    }
//
//    return inpainted.data;
    //memcpy(outputData, inpainted.data, inpainted.total() * inpainted.elemSize());
}
