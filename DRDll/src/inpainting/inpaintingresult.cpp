//
//  inpaintingresult.cpp
//  HelloWorld
//
//  Created by LarryLiang on 06/06/2018.
//  Copyright © 2018 LarryLiang. All rights reserved.
//

#include "inpaintingresult.h"

InpaintingResult::InpaintingResult()
{
    
}

InpaintingResult::InpaintingResult(string name, Mat image, Mat mask, Mat inpainted, InpaintingMethod method, double time)
{
    this->name = name;
    this->image = image;
    this->mask = mask;
    this->inpainted = inpainted;
    this->method = method;
    this->time = time;
}
