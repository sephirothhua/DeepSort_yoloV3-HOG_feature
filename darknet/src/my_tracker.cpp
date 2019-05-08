#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
using namespace cv;
using namespace std;

class OneTracker
{
    public:
        void update(vector<Rect2d> bboxes){
            for(int i=0;i<bboxes.size();i++)
            {
                
            }
        };
    private:
        float cal_iou(Rect2d src,Rect2d dst){
            Rect2d p1 = src&dst;
            int area_all = src.area()+dst.area()-p1.area();
            float iou = float(p1.area())/float(area_all);
            return iou;
        };
};