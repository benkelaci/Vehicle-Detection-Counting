import os
orig_dir = os.getcwd()
os.chdir("./tensorflow-yolov4")

import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from PIL import Image
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf
import glob
import csv

flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', os.path.join(orig_dir, 'yolov4.weights'),
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image_dir', './data/', 'path to input image directory')
flags.DEFINE_string('output', 'result.png', 'path to output image')

def main(_argv):
    #TODO: add valid extensions
    directory = os.path.join(FLAGS.image_dir, "*")
    image_list = glob.glob(directory)

    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    input_size = FLAGS.size

    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        if FLAGS.tiny:
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_tiny(model, FLAGS.weights)
        else:
            if FLAGS.model == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights_v3(model, FLAGS.weights)
            elif FLAGS.model == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights(model, FLAGS.weights)

        model.summary()
    fieldnames = [
        "filename",
        "cars",
        "trucks",
        "buses"
    ]
    with open(os.path.join(orig_dir, "result.csv"), 'w+', newline='') as f:
        # Attach a CSV writer to the file with the desired fieldnames
        writer = csv.DictWriter(f, fieldnames, delimiter=";")
        writer.writeheader()

        for image_path in image_list:
            if "_out" in image_path:
                continue
            d = {}
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]

            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            pred_bbox = model.predict(image_data)

            if FLAGS.model == 'yolov4':
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            else:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
            bboxes = utils.nms(bboxes, 0.213, method='nms')
            bboxes_filtered = bboxes.copy()
            l = len(bboxes)
            for i, bbox in enumerate(bboxes_filtered):
                bboxes_filtered = np.delete(bboxes_filtered, l - 1 - i, 0)
            cars = 0
            trucks = 0
            buses = 0
            for i, bbox in enumerate(bboxes):
                class_ind = int(bbox[5])
                if class_ind == 2 or class_ind == 5 or class_ind == 7:
                    bboxes_filtered = np.insert(bboxes_filtered, 0, bbox, axis=0)
                if class_ind == 2:
                    cars += 1
                if class_ind == 5:
                    buses += 1
                if class_ind == 7:
                    trucks += 1
            d["filename"] = image_path
            d["cars"] = cars
            d["trucks"] = trucks
            d["buses"] = buses
            writer.writerow(d)
            image = utils.draw_bbox(original_image, bboxes_filtered)
            image = Image.fromarray(image)
            # image.show()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path.replace(".jpg", "_out.jpg"), image)

if __name__ == '__main__':
    try:
        app.run(main)
        print("finished")
    except SystemExit:
        pass
