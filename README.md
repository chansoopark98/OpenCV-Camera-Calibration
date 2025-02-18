    # OpenCV-Camera-Calibration

### OpenCV Python based camera calibration

[![Github All Releases](https://img.shields.io/github/downloads/chansoopark98/OpenCV-Camera-Calibration/total.svg)]() 

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchansoopark98%2FOpenCV-Camera-Calibration&count_bg=%23C83D3D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

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

# Environment
    conda create -n calib python=3.11

    pip install -r requirements.txt

# How to use

1. assets/checkerboard.png A4 용지 인쇄
    ![CheckerBoard](assets/checkerboard.png)

2. 카메라로 촬영 후 assets/calibration_video.mp4 파일 저장

3. 비디오 프레임 추출
    ```bash
    python extract_video.py
    ```


4. 캘리브레이션 실행
    ```bash
    python camera_calibration.py
    ```