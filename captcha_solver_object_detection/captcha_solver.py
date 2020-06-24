from object_detection.utils import label_map_util, visualization_utils as vis_util
import tensorflow                                                      as tf
import pandas                                                          as pd
import numpy                                                           as np
import cv2                                                             as cv
import sys
import os
import collections
from pathlib import Path


BASE_DIR = Path(__file__).parent


class CaptchaSolver(object):
    
    
    def __init__(self):
        self.num_classes = 36
        self.labels_path = str((BASE_DIR / 'model/labelmap.pbtxt').resolve())
        self.modelckpt_path = str((BASE_DIR / 'model/frozen_inference_graph.pb').resolve())
        self.tolerance  = 0.6
        self.model = None
        self.detection_graph = None
        
        
    def __load_label_map(self):      
        label_map      = label_map_util.load_labelmap(self.labels_path)
        categories     = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
                
        return category_index
    
    def __load_tfmodel(self):        
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            
            with tf.gfile.GFile(self.modelckpt_path , 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            self.model = tf.Session(graph=self.detection_graph)
            
    def get_boxes_coordinates(self, image, boxes, classes, scores, category_index, instance_masks=None, instance_boundaries=None, keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=6, min_score_thresh=.5, agnostic_mode=False, line_thickness=4, groundtruth_box_visualization_color='black', skip_scores=False, skip_labels=False):
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_instance_boundaries_map = {}
        box_to_score_map = {}
        box_to_keypoints_map = collections.defaultdict(list)
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if instance_boundaries is not None:
                    box_to_instance_boundaries_map[box] = instance_boundaries[i]
                if keypoints is not None:
                    box_to_keypoints_map[box].extend(keypoints[i])
                if scores is None:
                    box_to_color_map[box] = groundtruth_box_visualization_color
                else:
                    display_str = ''
                    if not skip_labels:
                        if not agnostic_mode:
                            if classes[i] in category_index.keys():
                                class_name = category_index[classes[i]]['name']
                            else:
                                class_name = 'N/A'
                                display_str = str(class_name)
                    if not skip_scores:
                        if not display_str:
                            display_str = '{}%'.format(int(100*scores[i]))
                        else:
                            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                    box_to_display_str_map[box].append(display_str)
                    box_to_score_map[box] = scores[i]
                    if agnostic_mode:
                        box_to_color_map[box] = 'DarkOrange'
                    else:
                        box_to_color_map[box] = vis_util.STANDARD_COLORS[classes[i] % len(vis_util.STANDARD_COLORS)]

        coordinates_list = []
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            height, width, channels = image.shape
            coordinate = dict(xmin=int(),xmax=int(),ymin=int(),ymax=int())
            coordinate['ymin'] = int(ymin*height)
            coordinate['ymax'] = int(ymax*height)
            coordinate['xmin'] = int(xmin*width)
            coordinate['xmax'] = int(xmax*width)
            coordinates_list.append(coordinate)
            
        return coordinates_list
    
    def predict_captcha(self, image_path):
        self.__load_tfmodel()

        image_tensor      = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes   = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores  = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections    = self.detection_graph.get_tensor_by_name('num_detections:0')

        image          = cv.imread(image_path)
        image_rgb      = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        (boxes, scores, classes, num) = self.model.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                        feed_dict={image_tensor: image_expanded})

        category_index = self.__load_label_map()

        coordinates = self.get_boxes_coordinates(image, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, min_score_thresh=self.tolerance)
        digits      = self.__get_digits_prediction(category_index, (boxes, scores, classes, num), coordinates)

        solved_captcha = self.__get_solved_captcha(digits)

        return solved_captcha

    def __get_digits_prediction(self, category_index, model_output, coordinates, threshold=0.6):
        digits = []
        for x in range(len(model_output[1][0])):
            if model_output[1][0][x] > threshold:
                digits.append(dict(label=category_index[model_output[2][0][x]]['name'], score=float(model_output[1][0][x]), 
                                    coordenadas=coordinates[x], xmin=coordinates[x]['xmin']))

        return sorted(digits, key=lambda digit:digit['xmin'])

    def __get_solved_captcha(self, digits):
        solved_captcha = ''
        for digit in digits:
            solved_captcha = solved_captcha + digit['label']
        
        return solved_captcha

