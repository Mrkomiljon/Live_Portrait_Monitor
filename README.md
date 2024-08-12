# Live_Portrait_Monitor
If you find this project useful, please give it a star ❤️❤️
# Monitor [results](https://github.com/Mrkomiljon/Live_Portrait_Monitor)

https://github.com/user-attachments/assets/80020a36-6ec9-4efa-abf7-c1adbbfc6f39

https://github.com/user-attachments/assets/471c65a2-567f-4822-93af-882f0d041f18

https://github.com/user-attachments/assets/3bf96941-3a93-475b-9d47-8c70e0bd1e48

https://github.com/user-attachments/assets/23a83942-48a6-4922-8a50-b8f7ebdaa143

https://github.com/user-attachments/assets/c65006c6-bfc2-4c99-b7d0-4d8853f0c9da

# Webcam [result](https://github.com/Mrkomiljon/Webcam_Live_Portrait)

https://github.com/Mrkomiljon/Webcam_Live_Portrait/assets/92161283/4e16fbc7-8c13-4415-b946-dd731ac00b6e




## 🔥 Updates
- **`2024/07/10`**: 🔥 I released the initial version of the inference code for webcam. 
- **`2024/07/16`**: 🔥 I released the initial version of the inference code for monitor.
- **`2024/07/26`**: 🔥 I released windows package for faster inference speed. Continuous updates, stay tuned!


## Introduction
This repo, named **Webcam Live Portrait**, 
Author paper [LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control](https://arxiv.org/pdf/2407.03168).
I am actively updating and improving this repository for research only. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR) 💖.

## 🔥 Getting Started
### 1. Clone the code and prepare the environment
```bash
git clone https://github.com/Mrkomiljon/Live_Portrait_Monitor.git
cd Live_Portrait_Monitor

# create env using conda
conda create -n LivePortrait python==3.9.18
conda activate LivePortrait
# install dependencies with pip
pip install -r requirements.txt
pip install mss # for using monitor
```

### 2. Download pretrained weights
Download pretrained LivePortrait weights and face detection models of InsightFace from [Google Drive](https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib) or [Baidu Yun](https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn). All weights have packed  in one directory 😊. Unzip and place them in `./pretrained_weights` ensuring the directory structure is as follows:
```text
pretrained_weights
├── insightface
│   └── models
│       └── buffalo_l
│           ├── 2d106det.onnx
│           └── det_10g.onnx
└── liveportrait
    ├── base_models
    │   ├── appearance_feature_extractor.pth
    │   ├── motion_extractor.pth
    │   ├── spade_generator.pth
    │   └── warping_module.pth
    ├── landmark.onnx
    └── retargeting_models
        └── stitching_retargeting_module.pth
```

### 3. Inference 🚀

```bash
python inference_org.py
```

If the script runs successfully, you will get an output mp4 file named `animations/s6--d0_concat.mp4`. This file includes the following results: driving video, input image, and generated result.

<p align="center">
  <img src="https://github.com/Mrkomiljon/Webcam_Live_Portrait/assets/92161283/7c4daf41-838d-4eb8-a762-9188cd337ee6">
</p>

# Unrealtime result

https://github.com/Mrkomiljon/Webcam_Live_Portrait/assets/92161283/7c4daf41-838d-4eb8-a762-9188cd337ee6

Or, you can change the input by specifying the `-s` and `-d` arguments come from webcam:

```bash
# to use monitor
python inference_monitor.py -s assets/examples/source/MY_photo.jpg 

# or disable pasting back
python inference_monitor.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4 --no_flag_pasteback

# to use original code for inference 
python inference_org.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4 --no_flag_pasteback

# more options to see
python inference_org.py -h
```
```bash
# For good result, please play with padding (src/crop.py) for driver head.
    y1 = max(0, y1 - 320)                # Upper part of the head
    y2 = min(frame.shape[0], y2 + 550)   # Lower part of the head
    x1 = max(0, x1 - 360)                # Both sides of the head
    x2 = min(frame.shape[1], x2 + 360)   # Both sides of the head

```

### 5. Improved inference speed 🚀🚀🚀


https://github.com/user-attachments/assets/9b674f43-7c3c-4689-828c-311e56038afd


https://github.com/user-attachments/assets/d541c4b4-dd09-4088-a97b-77d5f29f1695

- 2024/07/26: Windows integrated package, one-click run without installation, supporting TensorRT. Thanks to [author](https://github.com/warmshao/FasterLivePortrait) for supporting TensorRT.
- [Optional] If you have already installed CUDA and cuDNN on your Windows computer, please skip this step. I have only verified this on CUDA 12.4. If you haven't installed CUDA or if you encounter CUDA-related errors, you need to follow - 
  these steps for installation::
- Download cuda12.2, double-click the exe and follow the default settings to install.
- Download the cudnn zip file, extract it and copy the lib, bin, and include folders from the cudnn folder to the CUDA12.4 folder (default is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4)
- [Optional] Install ffmpeg
- Download [Live_Portrait_Monitor](https://drive.google.com/drive/folders/16MF3TijndHVG73U3Ms5WBW7QVPcaKElb?usp=drive_link) zip file and extract it.
- Enter FasterLivePortrait_High_FPS
- For real-time Monitor operation, double-click Monitor.bat. If you want to change the target image, run in command line: camera.bat assets/examples/source/s1.jpg
- Note!!! There should be a driver face video on Monitor[1] or [2].
- I have adapted the code for readability on monitors and introduced some functionalities. You can easily use it.



## Acknowledgements
I would like to thank the contributors of [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [SPADE](https://github.com/NVlabs/SPADE), [InsightFace](https://github.com/deepinsight/insightface) repositories, for their open research and main [authors](https://github.com/KwaiVGI/LivePortrait).

（Some demo images/videos above are sourced from image websites/repos. If there is any infringement, I will immediately remove them and apologize.）




