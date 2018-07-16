//
//  globalvariables.cpp
//  DRDll
//
//  Created by LarryLiang on 13/07/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "globalvariables.h"

int desiredWidth;
int desiredHeight;

int height;
int width;
int channels;

int controlPointSize;

int illuminationBlockSize;

vector<Point> frame0BoundingPoints;
vector<Point> frame0ControlPoints;
Point2f frame0BoundingPointsArray[4];

Mat frame0;

Mat inpainted;
Mat dilatedContourMask;
Rect rect;
Mat rectMask;
