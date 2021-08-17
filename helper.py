import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import time

CUSTOM_MODEL_NAME = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
LABEL_MAP_NAME = 'label_map.pbtxt'

root_dir = "static"
paths = {
    'ANNOTATION_PATH': os.path.join(root_dir, 'annot'),
    'CHECKPOINT_PATH': os.path.join(root_dir, CUSTOM_MODEL_NAME)
}
files = {
    'PIPELINE_CONFIG': os.path.join(root_dir, CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

cap = cv2.VideoCapture(0)

def load_model():
    # -- Now lets load pipeline config and build a model for detection
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # -- Load the checkpoints we have saved
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(
        paths['CHECKPOINT_PATH'], 'ckpt-54')).expect_partial()
    
    return detection_model

def detect(detection_model):
    
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        
        prediction_dict = detection_model.predict(image, shapes)
        
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections


    category_index = label_map_util.create_category_index_from_labelmap(
        files['LABELMAP'])

    # try other channels if "0" did not worked, line "1" or "2" !!!!!!!!!!!!!!!
    # cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_width = 800
    output_height = 600

    while cap.isOpened():
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        else:
            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
            image_np_resized = cv2.resize(
                image_np_with_detections, (output_width, output_height))

            ret, buffer = cv2.imencode('.jpg', image_np_resized)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result