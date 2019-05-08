
#include "image.h"
#include "my_demo.h"
#include <vector>
// using namespace cv;

#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

extern "C" {
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



void MyDemo(char *cfgfile, char *weightfile,const char *filename,float thresh, float hier_thresh){
    image **alphabet = load_alphabet();
    VideoCapture *p;
    p = new VideoCapture(filename);
    VideoCapture *cap = (VideoCapture *)p;
    Mat frame;
    int frame_id = 0;
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    layer l = net->layers[net->n-1];
    float nms=.45;
    char str[50]={0};
    vector<Rect> bboxes;
    while(1)
    {
        *cap>>frame;
        cv::resize(frame, frame, cv::Size(1024, 768), (0, 0), (0, 0), cv::INTER_LINEAR);
        // image im = mat2image(frame);
        detection *dets = NULL;
        int nboxes;
        // detect the image with one picture
        detect_image(net,frame,l.classes,nms,thresh,hier_thresh,dets,&nboxes);
        // get the final result, the nboxes is the number of people
        nboxes = adjust_dets(dets,nboxes,l.classes,thresh);
        bboxes = dets2bbox(dets,nboxes,frame.size[1],frame.size[0]);
        for(int i=0;i<bboxes.size();i++){
            sprintf(str,"tracker/%d_%d.jpg",frame_id,i);
            
            Mat ROI = frame(bboxes[i]);
            cv::imwrite(str,ROI);
            cv::rectangle(frame,bboxes[i],Scalar(255,0,0),2,1);
        }
        // draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        // show_image(im,"Demo",1);
        cv::imshow("Detect", frame);
        free_detections(dets, nboxes);
        frame_id += 1;
        // free_image(im);
        // free_image(resized_im);
        if(frame.empty())
            break;
        // imshow("video", frame);
        waitKey(20);
    }
}
}
