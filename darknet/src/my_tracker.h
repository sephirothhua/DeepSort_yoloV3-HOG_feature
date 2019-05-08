#ifndef MY_DEMO_H
#define MY_DEMO_H

#include "image.h"
#include "my_demo.h"
#include <vector>
#include "opencv2/opencv.hpp"
#include "hungarian.h"
#include "opencv2/tracking.hpp"
// #include "onlineMIL.hpp"


class OneTracker
{
    public:
        OneTracker(cv::Mat img,cv::Rect2d det){
            track = cv::TrackerKCF::create();
            track->init(img,det);
            id += 1;
        };
        ~OneTracker(){
            track.release();
        };
    private:
        int max_fps = 30;
        static int id;
        int status = 0;
        bool tracked = true;
        cv::Ptr<cv::TrackerKCF> track;
        cv::Rect2d position;
};
int OneTracker::id = 0;

class Track
{
    public:
    private:
        std::vector<int> track_ids;
        std::vector<OneTracker> Trackers;
};

#endif