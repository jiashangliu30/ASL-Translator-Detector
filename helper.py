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
from collections import Counter
import wordninja

global TEXT
TEXT = ""

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
    
    counter = 1
    alphabet_list = []
    pred_terminal = None
            
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
            

            frame_elapsed_time = time.time() - frame_start_time
            # Put model name and fps text-------------------------------------------
            font = cv2.FONT_HERSHEY_SIMPLEX
            upperLeftCorner = (30, 35)
            fontScale = 0.85
            fontColor = (0, 0, 255)
            lineType = 2

            cv2.putText(image_np_resized, f'FPS: {np.round(1/frame_elapsed_time, 1)}',
                        (output_width-200, 35),  # upper right corner
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            ret, buffer = cv2.imencode('.jpg', image_np_resized)
            frame = buffer.tobytes()
            
            for i in range(len(detections['detection_boxes'])):
                if (detections['detection_scores'][i]) > 0.75:
                    alphabet = detections['detection_classes'] + label_id_offset
                    class_name = category_index[alphabet[i]]['name']
                    if counter%10 != 0:
                        # add letter to alphabet_list
                        alphabet_list.append(class_name)
                        
                    else:
                        # find mode of letters and clear
                        c = Counter(alphabet_list)
                        most_common = c.most_common()
                        if len(most_common) == 0:
                            pred_terminal = None
                        else:
                            pred_terminal = most_common[0][0]
                            global TEXT
                            if pred_terminal is not None:
                                TEXT += pred_terminal

                        print(f"class: {class_name}")
                        print(f"alph: {alphabet_list}")
                        print(f"Counter: {c}")
                        print(f"most common: {most_common}")
                        print(f"pred_terminal: {pred_terminal}")
                        alphabet_list.clear()
                        
                    
            counter = counter + 1        
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') #, pred_terminal)  # concat frame one by one and show result

            # cv2.imshow('object detection', image_np_resized)  # was (800, 600)

        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         cap.release()
        #         cv2.destroyAllWindows()
        #         break

        # cap.release()
        # cv2.destroyAllWindows()
        
        # ajax asynchronous calls for the text on screen

def get_prediction_text():
    global TEXT
    return TEXT

def reset_prediction_text():
    global TEXT
    TEXT = ''

def text_to_speech():
    sliced_text = ' '.join(wordninja.split(TEXT))
    pass
