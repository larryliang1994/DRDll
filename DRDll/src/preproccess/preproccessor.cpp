//
//  preproccessor.cpp
//  HelloWorld
//
//  Created by LarryLiang on 08/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "preproccessor.hpp"

namespace Preproccessor
{
    GrabCut gcapp;
    static void on_mouse( int event, int x, int y, int flags, void* param )
    {
        gcapp.mouseClick( event, x, y, flags, param );
    }
    
    Mat selection(Mat image)
    {
        int run = 1;
        
        const string winName = "selecting roi";
        namedWindow(winName, WINDOW_AUTOSIZE);
        setMouseCallback(winName, on_mouse, 0);
        
        gcapp.setImageAndWinName( image, winName );
        gcapp.showImage();
        
        while(run)
        {
            char c = (char)waitKey(0);
            switch( c )
            {
                case '\x1b':
                    cout << "Exiting ..." << endl;
                    destroyWindow( winName );
                    break;
                case 'r':
                    cout << endl;
                    gcapp.reset();
                    gcapp.showImage();
                    break;
                case 'n':
                    int iterCount = gcapp.getIterCount();
                    int newIterCount = gcapp.nextIter();
                    if( newIterCount > iterCount )
                    {
                        run = 0;
                    }
                    else
                    {
                        cout << "rect must be determined" << endl;
                    }
                    break;
            }
        }
        
        return gcapp.getMask();
    }
    
    Rect getRect()
    {
        return gcapp.getRect();
    }
    
    Mat createMask(Mat image)
    {
        Mat mask = selection(image);
        
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
        
        return mask;
    }
    
    void createSourceMask(Mat image, Mat mask, int neighbourSize, Mat &newImage, Mat &newMask, Rect &boundingBox)
    {
        // find bounding boxes for all contours
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours( mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        }
        
        // leave the biggest bounding box
        Rect biggestBounding = Rect();
        for (int i = 0; i < boundRect.size(); i++)
        {
            if (boundRect[i].width * boundRect[i].height > biggestBounding.width * biggestBounding.height)
            {
                biggestBounding = boundRect[i];
            }
        }

        // expand the bounding box
        Rect expandedBounding = Rect();
        expandedBounding.x = biggestBounding.x - neighbourSize > 0 ?
                biggestBounding.x - neighbourSize : 0;
        expandedBounding.y = biggestBounding.y - neighbourSize > 0 ?
                biggestBounding.y - neighbourSize : 0;
        expandedBounding.width = biggestBounding.x + biggestBounding.width + neighbourSize  < image.rows ?
                biggestBounding.width  + 2 * neighbourSize : image.rows - expandedBounding.x;
        expandedBounding.height = biggestBounding.y + biggestBounding.height + neighbourSize < image.cols ?
                biggestBounding.height + 2 * neighbourSize : image.cols - expandedBounding.y;
        
        Mat drawing = Mat::zeros( mask.size(), CV_8UC3 );
        Scalar color = Scalar( 0, 0, 255 );
        drawContours( drawing, contours_poly, 0, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( drawing, biggestBounding.tl(), biggestBounding.br(), color, 2, 8, 0 );
        rectangle( drawing, expandedBounding.tl(), expandedBounding.br(), color, 2, 8, 0 );
        imwrite("mask2.jpg", drawing);
        
        newImage = image(expandedBounding);
        newMask = mask(expandedBounding);
        boundingBox = expandedBounding;
    }
}
