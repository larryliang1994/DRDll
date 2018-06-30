//
//  DRDll.hpp
//  DRDll
//
//  Created by LarryLiang on 12/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#pragma once
#if UNITY_METRO
#define EXPORT_API __declspec(dllexport) __stdcall
#elif UNITY_WIN
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif


#include <iostream>
#include <vector>
#include <sys/time.h>

#include "drutil.hpp"
#include "preproccessor.hpp"
#include "inpaintingresult.h"
#include <stdlib.h>

using namespace cv;
using namespace std;

struct RECT2D {
    double x;
    double y;
    double width;
    double height;
};

struct POINT2D {
    int x;
    int y;
};
