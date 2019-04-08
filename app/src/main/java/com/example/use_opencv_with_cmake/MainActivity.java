package com.example.use_opencv_with_cmake;

import android.annotation.TargetApi;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
//////////////////////////////////////////////////////////////
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import android.os.Environment;
import android.content.res.AssetManager;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.FileNotFoundException;
import java.lang.Object;
//import java.lang.String;

//////////////////////////////////////////////////////////////

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "opencv";
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mMatInput;
    private Mat mMatResult;
    //private Net net;
    private double mScale;
    private Size mInpSize;
    private Scalar mMmean;
    private boolean mSwapRB;
    private ArrayList<String> mClasses;//outNames,
    private float mThConf, mThNms;
    private String mFnClass, mFnModel, mFnCfg, mStrFramework;
    private int mIntBackend, mIntTarget;



    public native void ConvertRGBtoGray(long matAddrInput, long matAddrResult);
    public native void yolo(long matAddrInput, long matAddrResult,
                            long addrNet, double mScale, long addrInpSize,
                            long addrMmean, boolean mSwapRB,
                            float mThConf, float mThNms, ArrayList<String> mClasses);


    //public native long loadCascade(String cascadeFileName );
    //public native long load_yolo_weight(String fn_yolo_weight);
    public native long loadDarknet(String fn_mdoel, String mFnCfg, String mStrFramework, int mIntBackend, int mIntTarget);

    public native void detect(long mCascadeClassifierFace,
                              long mCascadeClassifierEye, long matAddrInput, long matAddrResult);
    public long mCascadeClassifierFace = 0;
    public long mCascadeClassifierEye = 0;
    public long mPtrNet = 0;




    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");
    }


    //private void copyFile(String filename) {
    private String copyFile(String filename) {
        String baseDir = Environment.getExternalStorageDirectory().getPath();
        String pathDir = baseDir + File.separator + filename;

        AssetManager assetManager = this.getAssets();

        InputStream inputStream = null;
        OutputStream outputStream = null;

        try {
            Log.d( TAG, "copyFile :: 다음 경로로 파일복사 "+ pathDir);
            inputStream = assetManager.open(filename);
            outputStream = new FileOutputStream(pathDir);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            inputStream.close();
            inputStream = null;
            outputStream.flush();
            outputStream.close();
            outputStream = null;
        } catch (Exception e) {
            pathDir = null;
            Log.d(TAG, "copyFile :: 파일 복사 중 예외 발생 "+e.toString() );
        }
        return pathDir;
    }

    private void load_classes()
    {
        try {
            Scanner s = new Scanner(new File(mFnClass));
            mClasses = new ArrayList<String>();
            while (s.hasNext()) {
                mClasses.add(s.next());
            }
            s.close();
        }
        catch (FileNotFoundException ex)
        {
            System.out.println("Can not copy yolo config or weight file");
            System.exit(999);
            // insert code to run when exception occurs
        }
        return;
    }


    //private void read_yolo_file()
    private void init_yolo()
    {
        if(0 != mPtrNet)
        {
            return;
        }

        mFnModel = "yolov3.weights";
        mFnCfg = "yolov3.cfg";
        mFnClass = "object_detection_classes_yolov3.txt";
        mStrFramework = "darknet";
        mIntBackend = 0;
        //mIntBackend = 3;
        mIntTarget = 1;

        mFnModel = copyFile(mFnModel);
        mFnCfg = copyFile(mFnCfg);
        mFnClass = copyFile(mFnClass);
        Log.d(TAG, "init_yolo:");

        if (null == mFnModel || mFnModel.isEmpty() || null == mFnCfg || mFnCfg.isEmpty() )
        {
            System.out.println("Can not copy yolo config or weight file");
            System.exit(999);
        }

        //mCascadeClassifierFace = loadCascade( "haarcascade_frontalface_alt.xml");
        mPtrNet = loadDarknet(mFnModel, mFnCfg, mStrFramework, mIntBackend, mIntTarget);

        //if(mClasses.isEmpty())
        //{
            load_classes();
        //}
        mScale = 0.00392;
        mInpSize = new Size(416, 416);//   mInpSize.height = 416;
        mMmean = new Scalar(0, 0, 0, 0);
        mSwapRB = true;
        mThConf = (float)0.5;  mThNms = (float)0.4;
        //mClasses = FileUtils.readLines(new File(mFnClass), "utf-8");

        //Scanner s = new Scanner(new File("filepath"));

        Log.d(TAG, "init_yolo:");

        //mCascadeClassifierEye = loadCascade( "haarcascade_eye_tree_eyeglasses.xml");
    }




    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            //퍼미션 상태 확인
            if (!hasPermissions(PERMISSIONS)) {

                //퍼미션 허가 안되어있다면 사용자에게 요청
                requestPermissions(PERMISSIONS, PERMISSIONS_REQUEST_CODE);
            }
            else if(0 == mPtrNet)
            {
                init_yolo(); //   추가
            }
        }
        else if(0 == mPtrNet)
        {
            init_yolo();   //  추가
        }

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0); // front-camera(1),  back-camera(0)
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mMatInput = inputFrame.rgba();

        //if ( mMatResult != null ) mMatResult.release(); fix 2018. 8. 18

        if ( mMatResult == null )

            mMatResult = new Mat(mMatInput.rows(), mMatInput.cols(), mMatInput.type());

        init_yolo();
        //ConvertRGBtoGray(mMatInput.getNativeObjAddr(), mMatResult.getNativeObjAddr());
        yolo(mMatInput.getNativeObjAddr(), mMatResult.getNativeObjAddr(), mPtrNet, mScale, mInpSize.getNativeObjAddr(), mMmean.getNativeObjAddr(), mSwapRB, mThConf, mThNms, mClasses);
        return mMatResult;
    }



    //여기서부턴 퍼미션 관련 메소드
    static final int PERMISSIONS_REQUEST_CODE = 1000;
    //String[] PERMISSIONS  = {"android.permission.CAMERA"};
    String[] PERMISSIONS  = {"android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE"};


    private boolean hasPermissions(String[] permissions) {
        int result;

        //스트링 배열에 있는 퍼미션들의 허가 상태 여부 확인
        for (String perms : permissions){

            result = ContextCompat.checkSelfPermission(this, perms);

            if (result == PackageManager.PERMISSION_DENIED){
                //허가 안된 퍼미션 발견
                return false;
            }
        }

        //모든 퍼미션이 허가되었음
        return true;
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch(requestCode){

            case PERMISSIONS_REQUEST_CODE:
                if (grantResults.length > 0) {
                    boolean cameraPermissionAccepted = grantResults[0]
                            == PackageManager.PERMISSION_GRANTED;
/*////////////////////////////////////////////////////////////////////////////////////////////////////////
                    if (!cameraPermissionAccepted)
                        showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
*/////////////////////////////////////////////////////////////////////////////////////////////////////////
                    boolean writePermissionAccepted = grantResults[1]
                            == PackageManager.PERMISSION_GRANTED;

                    if (!cameraPermissionAccepted || !writePermissionAccepted) {
                        showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
                        return;
                    }
                    else if (0 == mPtrNet)
                    {
                        init_yolo();
                        //mPtrNet = load_darknet(fn_mdoel, mFnCfg, mStrFramework, mIntBackend, mIntTarget);
                        //read_yolo_file();
                    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////

                }
                break;
        }
    }


    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {

        AlertDialog.Builder builder = new AlertDialog.Builder( MainActivity.this);
        builder.setTitle("알림");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(PERMISSIONS, PERMISSIONS_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }
}



