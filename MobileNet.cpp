#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <iomanip>
using namespace std;

string CLASSES[] = { "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor" };


int main(int argc, char** argv)
{
    CV_TRACE_FUNCTION();
    if (2 != argc) {
        std::cout << "Usage: MObileNet video_file" << std::endl;
        return 1;
    }

    String modelTxt = "MobileNetSSD_deploy.prototxt.txt";
    String modelBin = "MobileNetSSD_deploy.caffemodel";

    //String imageFile = (argc > 1) ? argv[1] : "space_shuttle.jpg";
    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }

    VideoCapture cap;
    cap.open(argv[1]);
    if (!cap.isOpened()) {
        cout << "Can not open video stream: '" << argv[1] << "'" << endl;
        return 2;
    }

    // Open video writer.
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    //int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');
    double fps = 30.f;
    cv::Size videoSize;
    videoSize.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    videoSize.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter vw;
    if (!vw.open("D:\\video.avi", fourcc, fps, videoSize, true)) {
        std::cerr << "Error: Unsupported video format" << std::endl;
        return 1;
    }

    Mat frame;
    for (;;) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Finished reading: empty frame" << endl;
            break;
        }

        int64 t = getTickCount();
        Mat img2;
        resize(frame, img2, Size(300, 300));
        Mat inputBlob = blobFromImage(img2, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);

        net.setInput(inputBlob, "data");
        Mat detection = net.forward("detection_out");
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        t = getTickCount() - t;

        ostringstream ss;
        float confidenceThreshold = 0.2;
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > confidenceThreshold)
            {
                int idx = static_cast<int>(detectionMat.at<float>(i, 1));
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                    (int)(xRightTop - xLeftBottom),
                    (int)(yRightTop - yLeftBottom));

                rectangle(frame, object, Scalar(0, 255, 0), 2);

                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = CLASSES[idx] + ": " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                putText(frame, label, Point(xLeftBottom, yLeftBottom),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            }
        }

        //{
        //    ostringstream buf;
        //    buf << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
        //    putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        //}

        imshow("detections", frame);
        waitKey(1);

        vw.write(frame);
    }

    return 0;
}