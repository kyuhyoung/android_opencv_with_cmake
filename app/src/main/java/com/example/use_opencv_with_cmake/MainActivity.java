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
    private Mat matInput;
    private Mat matResult;
    //private Net net;
    private double skale;
    private Size inpSize;
    private Scalar mean;
    private boolean swapRB;
    private ArrayList<String> klasses;//outNames,
    private float thConf, thNms;
    private String fn_class, fn_model, fn_cfg, str_framework;
    private int int_backend, int_target;



    public native void ConvertRGBtoGray(long matAddrInput, long matAddrResult);
    public native void yolo(long matAddrInput, long matAddrResult,
                            long addrNet, double skale, Size inpSize,
                            Scalar mean, boolean swapRB,
                            //ArrayList<String> jOutNames,
                            float thConf, float thNms, ArrayList<String> klasses);


    //public native long loadCascade(String cascadeFileName );
    //public native long load_yolo_weight(String fn_yolo_weight);
    public native long loadDarknet(String fn_mdoel, String fn_cfg, String str_framework, int int_backend, int int_target);
    //public native long loadDarknet();


    public native void detect(long cascadeClassifier_face,
                              long cascadeClassifier_eye, long matAddrInput, long matAddrResult);
    public long cascadeClassifier_face = 0;
    public long cascadeClassifier_eye = 0;
    public long ptr_net = 0;




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

    //private void read_yolo_file()
    private void init_yolo()
    {
        fn_model = "yolov3.weights";
        fn_cfg = "yolov3.cfg";
        fn_class = "object_detection_classes_yolov3.txt";
        str_framework = "darknet";
        int_backend = 0;
        //int_backend = 3;
        int_target = 1;

        fn_model = copyFile(fn_model);
        fn_cfg = copyFile(fn_cfg);
        fn_class = copyFile(fn_class);
        Log.d(TAG, "init_yolo:");

        if (null == fn_model || fn_model.isEmpty() || null == fn_cfg || fn_cfg.isEmpty() )
        {
            System.out.println("Can not copy yolo config or weight file");
            System.exit(999);
        }

        //cascadeClassifier_face = loadCascade( "haarcascade_frontalface_alt.xml");
        ptr_net = loadDarknet(fn_model, fn_cfg, str_framework, int_backend, int_target);

        //klasses = FileUtils.readLines(new File(fn_class), "utf-8");

        //Scanner s = new Scanner(new File("filepath"));
        try {
            Scanner s = new Scanner(new File(fn_class));
            klasses = new ArrayList<String>();
            while (s.hasNext()) {
                klasses.add(s.next());
            }
            s.close();
        }
        catch (FileNotFoundException ex)
        {
            System.out.println("Can not copy yolo config or weight file");
            System.exit(999);
            // insert code to run when exception occurs
        }

        Log.d(TAG, "init_yolo:");

        //cascadeClassifier_eye = loadCascade( "haarcascade_eye_tree_eyeglasses.xml");

        //ptr_net = loadDarknet();
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
            else if(0 == ptr_net)
            {
                init_yolo(); //   추가
            }
        }
        else if(0 == ptr_net)
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

        matInput = inputFrame.rgba();

        //if ( matResult != null ) matResult.release(); fix 2018. 8. 18

        if ( matResult == null )

            matResult = new Mat(matInput.rows(), matInput.cols(), matInput.type());

        //ConvertRGBtoGray(matInput.getNativeObjAddr(), matResult.getNativeObjAddr());
        yolo(matInput.getNativeObjAddr(),
                matResult.getNativeObjAddr(),
                //net.getNativeObjAddr(),
                ptr_net,
                //scale.getNativeObjAddr(),
                skale,
                //InpSize.getNativeObjAddr(),
                inpSize,
                //mean.getNativeObjAddr(),
                mean,
                //swapRB.getNativeObjAddr(),
                swapRB,
                //outNames,
                thConf,
                thNms,
                klasses);

        return matResult;
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
                    else if (0 == ptr_net)
                    {
                        init_yolo();
                        //ptr_net = load_darknet(fn_mdoel, fn_cfg, str_framework, int_backend, int_target);
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



