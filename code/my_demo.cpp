
#include "my_demo.h"
#include <vector>
#include "opencv2/opencv.hpp"
#include "time.h"
using namespace cv;
using namespace std;

image ipl2image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

image mat2image(Mat m)
{
    IplImage ipl = m;
    image im = ipl2image(&ipl);
    rgbgr_image(im);
    return im;
}

void detect_image(network *net,Mat m,float classes,float nms,float thresh,float hier_thresh,detection *&det,int *nboxes){
    image im = mat2image(m);
    image resized_im = letterbox_image(im,416,416);
    // image resized_im = letterbox_image(im,320,320);
    // save_image(resized_im, "./1.jpg");
    float *X = resized_im.data;
    network_predict(net, X);
    // int nboxes = 0;
    det = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, nboxes);
    if (nms) do_nms_sort(det, *nboxes, classes, nms);
    free_image(im);
    free_image(resized_im);
    // free(X);
}

int adjust_dets(detection *&dets,int nboxes,int classes,float thresh){
    int box_count = 0;
    int real_count = 0;
    for(int i=0;i<nboxes;i++){
        // printf("Prob: %f",dets[i].prob[0]);
        if (dets[i].prob[0] > thresh){
            box_count++;
            }
        }
    detection *dst_det = (detection *)calloc(box_count, sizeof(detection));
    for(int i=0;i<nboxes;i++){
        if (dets[i].prob[0] > thresh){
            *(dst_det + real_count) = *(dets + i);
            //printf("det ADDRESS: %p; src ADDRESS: %p",);
            // copy prob & mask
            (dst_det[real_count]).prob = new float(classes);
            memcpy((dst_det[real_count]).prob,dets[i].prob,classes*sizeof(float));
            // (*dst_det[box_count]).mask = new float(classes);//clases->l.coords-4
            // memcpy((*dst_det[box_count]).mask,dets[i].mask,classes*sizeof(float));
            real_count++;
            }
        }
    free_detections(dets,nboxes);
    dets = dst_det;
    return box_count;
}

vector<Rect> dets2bbox(detection *dets,int nboxes,int width,int height){
    int box_count = 0;
    int real_count = 0;
    vector<Rect> bboxes;
    for(int i=0;i<nboxes;i++){
        int x1 = (int)((dets[i].bbox.x - dets[i].bbox.w/2)*width);
        int y1 = (int)((dets[i].bbox.y - dets[i].bbox.h/2)*height);
        int w = (int)(dets[i].bbox.w*width);
        int h = (int)(dets[i].bbox.h*height);
        bboxes.push_back(Rect(x1,y1,w,h));
    }
    return bboxes;
}

DS_DetectObjects det2detobj(detection *dets,int nboxes,int width,int height){
    DS_DetectObjects result;
    for (int i=0;i<nboxes;i++){
        DS_Rect rec;DS_DetectObject obj;
        int x = (int)(dets[i].bbox.x*width);
        int y = (int)(dets[i].bbox.y*height);
        int w = (int)(dets[i].bbox.w*width);
        int h = (int)(dets[i].bbox.h*height);
        // int x1 = (int)((dets[i].bbox.x - dets[i].bbox.w/2)*width);
        // int y1 = (int)((dets[i].bbox.y - dets[i].bbox.h/2)*height);
        // int w = (int)(dets[i].bbox.w*width);
        // int h = (int)(dets[i].bbox.h*height);
        int x1 = (int)(x-w/2.0);
        int y1 = (int)(y-h/2.0);
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if((x1+w)>width) w=width-x1;
        if((y1+h)>height) h=height-y1;
        float prob = dets[i].prob[0];
        rec.x = x1;rec.y = y1;rec.width = w;rec.height = h;
        obj.class_id = 1;obj.rect = rec;obj.confidence = prob;
        result.push_back(obj);
    }
    return result;
}

deque<cv::Point> line_point;

void onmouse(int event, int x, int y, int flag, void *img)
{
    if(event == 1){
        line_point.push_back(cv::Point(x,y));
        }
    if(event == 2){
        line_point.pop_back();
    }
}

void draw_lines(Mat frame){
    if(line_point.size() == 0) return;
    if(line_point.size() == 2 || line_point.size() == 3){
        cv::line(frame, line_point[0], line_point[1], cv::Scalar(0, 255, 0), 5);
    }
    else if(line_point.size() == 4){
        cv::line(frame, line_point[0], line_point[1], cv::Scalar(0, 255, 0), 5);
        cv::line(frame, line_point[0], line_point[2], cv::Scalar(0, 255, 0), 5);
        cv::line(frame, line_point[2], line_point[3], cv::Scalar(0, 255, 0), 5);
        cv::line(frame, line_point[1], line_point[3], cv::Scalar(0, 255, 0), 5);
    }
}

void MyDemo(char *cfgfile, char *weightfile,const char *filename,const char *outfile, int classes,float thresh, float hier_thresh,int fps){
    image **alphabet = load_alphabet();
    VideoCapture *p;
    p = new VideoCapture(filename);
    VideoCapture *cap = (VideoCapture *)p;
    Mat frame;
    int frame_id = 0;
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    // layer l = net->layers[net->n-1];
    float nms=.45;
    char str[50]={0};
    // vector<Rect> bboxes;
    // DS_Tracker h_tracker=DS_Create(0.2,100,0.7,50,3);
    DS_DetectObjects detect_objects;
    Deep_sort Tracker = Deep_sort();
	DS_TrackObjects track_objects;
    char text[30];
    int font_face = cv::FONT_HERSHEY_COMPLEX;
    cv::namedWindow("Detect",CV_WINDOW_AUTOSIZE);
    setMouseCallback("Detect", onmouse,&frame);
    vector<cv::Scalar> color_map = {cv::Scalar(0,0,255),cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 0)
                                   ,cv::Scalar(255,255,0),cv::Scalar(255,0,255),cv::Scalar(0,255,255)
                                   ,cv::Scalar(180,54,0),cv::Scalar(54,200,54),cv::Scalar(255,211,155)};

    clock_t start,finish;

    cv::VideoWriter outputVideo;
    outputVideo.open(outfile, CV_FOURCC('M', 'P', '4', '2'), fps, cv::Size(1024,768));
    double totaltime;
    while(cap->isOpened())
    {
        *cap>>frame;
        // frame = cv::imread("/data/mnt/zj/arm_camera/test_data/test_pics/image-001.jpg");
        cv::resize(frame, frame, cv::Size(1024, 768), (0, 0), (0, 0), cv::INTER_LINEAR);
        detection *dets = NULL;
        int nboxes;
        // detect the image with one picture
        start=clock();
        detect_image(net,frame,classes,nms,thresh,hier_thresh,dets,&nboxes);
        // get the final result, the nboxes is the number of people
        nboxes = adjust_dets(dets,nboxes,classes,thresh);
        // bboxes = dets2bbox(dets,nboxes,frame.size[1],frame.size[0]);
        detect_objects = det2detobj(dets,nboxes,frame.size[1],frame.size[0]);
        Tracker.update(detect_objects,line_point,frame);
        track_objects = Tracker.get_detect_obj();
        // DS_Update(h_tracker, detect_objects, track_objects,line_point);
        for(auto oloop : track_objects) 
		{
                int col = int(oloop.track_id%9);
                // cv::Scalar color = color_map[col];
                cv::Scalar color;
                if(oloop.outside){
                    // sprintf(text,"%d  outside",oloop.track_id);
                    color = cv::Scalar(255, 0, 0);}
                else{
                    // sprintf(text,"%d  inside",oloop.track_id);
                    color = cv::Scalar(0, 0, 255);}
                Rect box = Rect(oloop.rect.x,oloop.rect.y,oloop.rect.width,oloop.rect.height);
                cv::rectangle(frame,box,color,2,1);
                Point origin = Point(oloop.rect.x,oloop.rect.y-2);
                // cv::putText(frame, text, origin, font_face, 0.5, color, 1, 8, 0);
                if(oloop.tracklet.size()>=2){
                for(int i=0;i<(oloop.tracklet.size()-1);i++){
                    cv::line(frame, oloop.tracklet[i],oloop.tracklet[i+1], color, 2);
                    cv::circle(frame, oloop.tracklet[i], 3, color);
                }
                }
		}
        // int area_person = Tracker.get_area_count();
        // int out_person = Tracker.get_out_count();
        finish=clock();
        totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
        sprintf(text,"Area Count :%3d",Tracker.get_area_count());
        cv::putText(frame, text, Point(0,15), font_face, 0.7, cv::Scalar(255, 0, 0), 2, 8, 0);
        sprintf(text,"Out Person Count :%3d",Tracker.get_out_count());
        cv::putText(frame, text, Point(0,40), font_face, 0.7, cv::Scalar(255, 0, 0), 2, 8, 0);

        sprintf(text,"FPS :%.3f",1/totaltime);
        cv::putText(frame, text, Point(frame.size[1]-150,15), font_face, 0.7, cv::Scalar(0, 255, 0), 2, 8, 0);
        // for(int i=0;i<bboxes.size();i++){
        //     sprintf(str,"tracker/%d_%d.jpg",frame_id,i);
        //     try
        //     {
        //         Mat ROI = frame(bboxes[i]);
        //         cv::imwrite(str,ROI);
        //         cv::rectangle(frame,bboxes[i],Scalar(255,0,0),2,1);
        //     }
        //     catch(std::exception &e)
        //     {
        //         printf("bboxes[%d] = %d %d %d %d\n", i, bboxes[i].x, bboxes[i].y, bboxes[i].height, bboxes[i].width);
        //     }
        // }
        draw_lines(frame);
        cv::imshow("Detect", frame);
        outputVideo << frame;
        free_detections(dets, nboxes);
        frame_id += 1;
        if(frame.empty())
            break;
        cv::waitKey(20);
    }
    outputVideo.release();
    cv::destroyAllWindows();
}

int main(int argc, char* argv[])
{
    char *cfg = argv[1];
    char *weights = argv[2];
    char *video = argv[3];
    char *outfile = find_char_arg(argc, argv, "-out","demo.avi");
    int fps = find_int_arg(argc, argv, "-fps",20);
    MyDemo(   cfg
            , weights
            , video
            , outfile
            , 1
            , 0.5
            , 0.5
            , fps);

    printf("输入任意字符串结束：%c", getc(NULL));
    return 0;
}
