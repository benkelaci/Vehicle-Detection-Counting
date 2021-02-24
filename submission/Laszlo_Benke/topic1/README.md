Preliminary instructions:
- go to my topic1 solution:
cd <path to the repo>/Vehicle-Detection-Counting/submission/Laszlo_Benke/topic1
- download yolov4.weights file my topic1 solution directory
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

- clone stable (compatible) tensorflow-yolov4 fork to my topic1 solution directory:
git clone https://github.com/SoloSynth1/tensorflow-yolov4
(I tried the more elegant way, using fork that has setup.py (pip install yolov4) with conda (git:git install), this repo:  https://github.com/hhk7734/tensorflow-yolov4 . I was able to install it, but there were some tensorflow and keras version conflicts, I assume with some manual version installation it can work. Rather I suggest using the stable fork of https://github.com/hunglc007/tensorflow-yolov4-tflite)


Usage:
- activate conda environment ...topic1/envs/py-gpu.yml
- python3 main.py --image_dir <image_directory_absolute_path>

Expected output:
- result.csv, that contains the vehicle counts for each image file
- _out.jpg file for each image, it is generated into the image_directory

Some comments:
- There are some more advanced variants of yolov4 (yolov4-csp and yolov4x-mish) that is a little bit better on the evaluation dataset (like COCO), but I recommend using the original one as standard. 
I experienced unknown issues at different yolo variants with Deepstream (as the best streaming platform with AI computer vision). So it is possible we can reach better prediction, but if the framework/platform is not compatible, then it would useless to use the new ones.
- There is a hugh impact on prediciton accuracy if the input size if higher, than the default. I assume in the project there will be similar images as in the examples predicitons directory. In this case (if there is enough GPU performance on the Cloud) splitting the images would be a huge advance.
- Version incompatibility caused a lot of issues in the recent years in this era, so be careful when you are choosing a platform/framework and GPU type also.
- I have not checked the provided NMS algo, but it might need to fine-tune - I have not played with that, for a production-ready system of course it needs to setup carefully
- For the production program we need to implement a tracking algorithm (IoU-tracking), with that the false positive bus detection on this image: RoundAbout_52321.jpg probably will be eliminated
