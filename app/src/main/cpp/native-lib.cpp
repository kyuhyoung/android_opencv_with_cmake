#include <jni.h>
#include <opencv2/opencv.hpp>
///////////////////////////////////////////////
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
///////////////////////////////////////////////
using namespace cv;
///////////////////////////////////////////////
using namespace dnn;
using namespace std;

///////////////////////////////////////////////

//#include <string>
//
//extern "C" JNIEXPORT jstring JNICALL
//Java_com_example_use_1opencv_1with_1cmake_MainActivity_stringFromJNI(
//        JNIEnv *env,
//        jobject /* this */) {
//    std::string hello = "Hello from C++";
//    return env->NewStringUTF(hello.c_str());
//}

float resize(Mat img_src, Mat &img_resize, int resize_width){
    float scale = resize_width / (float)img_src.cols ;
    if (img_src.cols > resize_width) {
        int new_height = cvRound(img_src.rows * scale);
        resize(img_src, img_resize, Size(resize_width, new_height));
    }
    else {
        img_resize = img_src;
    }
    return scale;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame,
        std::vector<std::string>& classes)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}


void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net,
        float confThreshold, float nmsThreshold, std::vector<std::string>& mClasses)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float* data = (float*)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width * height <= 1)
                    {
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                    boxes.push_back(Rect(left, top, width, height));
                    confidences.push_back(confidence);
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, mClasses);
    }
}

//extern "C"
//JNIEXPORT
std::string
//JNICALL
jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes, env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte* pBytes = env->GetByteArrayElements(stringJbytes, NULL);

    std::string ret = std::string((char *)pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_use_1opencv_1with_1cmake_MainActivity_loadDarknet(
        JNIEnv *env, jobject instance
        , jstring modelPath, jstring configPath, jstring mStrFramework,
        jint mIntBackend, jint mIntTarget
        )
{
    /*
    int a = 0;
    float b = 1;
    std::string c = "abc";
     */
    //int mIntBackend = 0, mIntTarget = 1;

/*
    std::string mFnModel = jstring2string(env, modelPath);
    std::string fn_config = jstring2string(env, configPath);
    std::string frmwork = jstring2string(env, mStrFramework);
*/
    /*
    const char *nativeFileNameString_model = env->GetStringUTFChars(modelPath, 0),
        *nativeFileNameString_cfg = env->GetStringUTFChars(configPath, 0),
        *nativeFileNameString_frmwork = env->GetStringUTFChars(mStrFramework, 0);
    */
    //string baseDir_model("/storage/emulated/0/"), baseDir_cfg("/storage/emulated/0/"), s_frmwork(nativeFileNameString_frmwork);
    //baseDir_model.append(nativeFileNameString_model);
    //baseDir_cfg.append(nativeFileNameString_cfg);
    //const char *pathDir = baseDir.c_str();

    //readNetFromDarknet(baseDir_cfg, baseDir_model);
    //readNetFromDarknet(jstring2string(env, configPath), jstring2string(env, modelPath));

    jlong ret = 0; ret = (jlong) new Net();
    //*((Net *) ret) = readNet(nativeFileNameString_model, nativeFileNameString_cfg, nativeFileNameString_frmwork);
    *((Net *) ret) = readNet(env->GetStringUTFChars(modelPath, 0), env->GetStringUTFChars(configPath, 0), env->GetStringUTFChars(mStrFramework, 0));
    //*((Net *) ret) = readNet(baseDir_model, baseDir_cfg, s_frmwork);
    //*((Net *) ret) = readNet(jstring2string(env, modelPath), jstring2string(env, configPath), jstring2string(env, mStrFramework));
    //*((Net *) ret) = readNet("C:\\Users\\kevin\\Documents\\android_opencv_with_cmake\\app\\src\\main\\cpp\\yolov3.weights", "C:\\Users\\kevin\\Documents\\android_opencv_with_cmake\\app\\src\\main\\cpp\\yolov3.cfg", "darknet");
    //*((Net *) ret) = readNet("C:/Users/kevin/Documents/android_opencv_with_cmake/app/src/main/cpp/yolov3.weights", "C:/Users/kevin/Documents/android_opencv_with_cmake/app/src/main/cpp/yolov3.cfg", "darknet");
    //ret = (jlong) new Net() cv::dnn::readNet(mFnModel, mFnCfg, mStrFramework);
    ((Net *) ret)->setPreferableBackend(mIntBackend);
    ((Net *) ret)->setPreferableTarget(mIntTarget);
    //return (jlong)(&net);
    return ret;
}

        //jlong addrScale,

extern "C"
JNIEXPORT void JNICALL
Java_com_example_use_1opencv_1with_1cmake_MainActivity_yolo(JNIEnv *env,
        jobject instance, jlong matAddrInput, jlong matAddrResult, jlong addrNet,
        //jlong addrScale,
        double mScale,
        jlong addrInpSize,
        //Size mInpSize,
        jlong addrMean,
        bool mSwapRB,
//        jobject jOutNames,
        float mThConf,
        float mThNms,
        jobject jClasses)
{
    // TODO
    // 입력 RGBA 이미지를 GRAY 이미지로 변환
    jclass alCls = env->FindClass("java/util/ArrayList");
    jclass stCls = env->FindClass("java/lang/String");
    if (alCls == nullptr  || stCls == nullptr) {
        return;
    }

    jmethodID alGetId  = env->GetMethodID(alCls, "get", "(I)Ljava/lang/Object;");
    //jmethodID alGetId  = env->GetMethodID(alCls, "get", "(I)Ljava/lang/String;");
    jmethodID alSizeId = env->GetMethodID(alCls, "size", "()I");

    if (alGetId == nullptr || alSizeId == nullptr) {
        env->DeleteLocalRef(alCls);
        env->DeleteLocalRef(stCls);
        return;
    }

    int //n_str_outNames = static_cast<int>(env->CallIntMethod(jOutNames, alSizeId)),
        n_str_classes = static_cast<int>(env->CallIntMethod(jClasses, alSizeId));

    if (/*n_str_outNames < 1 ||*/ n_str_classes < 1 ) {
        env->DeleteLocalRef(alCls);
        env->DeleteLocalRef(stCls);
        return;
    }
/*
    std::vector<std::string> outNames, mClasses;
    for (int i = 0; i < n_str_outNames; ++i) {
        jobject str = env->CallObjectMethod(jOutNames, alGetId, i);
        //jstring str = env->CallObjectMethod(jOutNames, alGetId, i);
        outNames.push_back(jstring2string(env, (jstring)str));
        env->DeleteLocalRef(str);
    }
*/
    std::vector<std::string> mClasses;
    for (int i = 0; i < n_str_classes; ++i) {
        //jstring str = env->CallObjectMethod(jClasses, alGetId, i);
        jobject str = env->CallObjectMethod(jClasses, alGetId, i);
        mClasses.push_back(jstring2string(env, (jstring)str));
        env->DeleteLocalRef(str);
    }



    Mat blob,
        &mMatInput = *(Mat *)matAddrInput,
        &mMatResult = *(Mat *)matAddrResult;

    mMatInput.copyTo(mMatResult);
    //double &mScale = *(double *)addrScale;
    //Size &mInpSize = *(Size *)addrInpSize;
    Scalar &mMmean = *(Scalar *)addrMean;
    //bool &mSwapRB = *(bool *)addrmSwapRB;
    Net &net = *(Net *)addrNet;
    //vector<String> &outNames = *(vector<String> *)addrOutNames;
    //float &th_conf = *(float *)addrmThConf, &th_nms = *(float *)addrmThNms;
    //vector<std::string> &mClasses = *(vector<std::string> *)addrClasses;
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    blobFromImage(mMatInput, blob, mScale, mInpSize, mMmean, mSwapRB, false);

    // Run a model.
    net.setInput(blob);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        //resize(frame, frame, mInpSize);
        resize(mMatInput, mMatInput, mInpSize);
        Mat imInfo = (Mat_<float>(1, 3) << mInpSize.height, mInpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
    std::vector<Mat> outs;
    net.forward(outs, outNames);


    //postprocess(frame, outs, net);
    postprocess(mMatResult, outs, net, mThConf, mThNms, mClasses);

    // Put efficiency information.
    std::vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = format("Inference time: %.2f ms", t);
    putText(mMatResult, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));



#if 1
/////////////////////////////////////////////////////
    //mMatResult = mMatInput;
    //rectangle(mMatResult, Point2d(100, 150), Point2d(200, 300), CV_RGB(255, 0, 0));
/////////////////////////////////////////////////////
#else
    cvtColor(mMatInput, mMatResult, COLOR_RGBA2GRAY);
#endif
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_use_1opencv_1with_1cmake_MainActivity_ConvertRGBtoGray(JNIEnv *env,
        jobject instance, jlong matAddrInput, jlong matAddrResult)
{
    // TODO
    // 입력 RGBA 이미지를 GRAY 이미지로 변환

    Mat &mMatInput = *(Mat *)matAddrInput;

    Mat &mMatResult = *(Mat *)matAddrResult;

    cvtColor(mMatInput, mMatResult, COLOR_RGBA2GRAY);
}