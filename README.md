# OpenCV Python based camera calibration

[![Github All Releases](https://img.shields.io/github/downloads/chansoopark98/OpenCV-Camera-Calibration/total.svg)]() 


<p align="center">
 <img src="https://img.shields.io/github/issues/chansoopark98/OpenCV-Camera-Calibration">
 <img src="https://img.shields.io/github/forks/chansoopark98/OpenCV-Camera-Calibration">
 <img src="https://img.shields.io/github/stars/chansoopark98/OpenCV-Camera-Calibration">
 <img src="https://img.shields.io/github/license/chansoopark98/OpenCV-Camera-Calibration">
 </p>

<p align="center">
 <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
 <img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>
 <br>
</p>


# Features

## Corner Detection

<table>
    <tr>
        <td><img src="assets/calibration_results/corners_1.jpg" alt="corner_1" width="300"></td>
        <td><img src="assets/calibration_results/corners_2.jpg" alt="corner_2" width="300"></td>
    </tr>
    <tr>
        <td><img src="assets/calibration_results/corners_3.jpg" alt="corner_3" width="300"></td>
        <td><img src="assets/calibration_results/corners_4.jpg" alt="corner_4" width="300"></td>
    </tr>
</table>

## Undistortion Images

![Undistortion_Result_1](assets/calibration_results/undistortion_comparison_2.jpg)

## Calculate Reprojection Errors

![reprojection_errors](assets/calibration_results/reprojection_errors.png)

## Show histogram Errors

![error_histogram](assets/calibration_results/error_histogram.png)


## Calibration Results

```json
{
    "reprojection_error": 1.2273468645903203,
    "camera_matrix": [
        [
            1422.117605057742,
            0.0,
            924.5633003737133
        ],
        [
            0.0,
            1423.204374674482,
            667.6183041323897
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ],
    "dist_coeffs": [
        4.612616555464993,
        152.56623164609113,
        0.039943271525320366,
        -0.008947777240770066,
        -228.0106478053562,
        4.592880469213394,
        151.6208067211176,
        -221.65838356166944,
        0.0010495594571102234,
        0.011289450598695816,
        -0.038397785485157175,
        -0.03986920764202555
    ]
}
```


# Environment
    conda create -n calib python=3.11

    pip install -r requirements.txt

# How to use

1. **Print** Checkerboard images (assets/checkerboard.png) to A4 
    ![CheckerBoard](assets/checkerboard.png)

2. **Recording** and then move video file to **assets/calibration_video.mp4**

3. **Extract** video frames
    ```bash
    python extract_video.py
    ```


4. **Run** calibration
    ```bash
    python camera_calibration.py # simple run

    python camera_calibration_high.py # Best quality
    ```