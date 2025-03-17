from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import sys
sys.path.append('../')
from utils import get_bbox_width, get_center_of_boxx

from trackers import Tracker


def get_cut_frame(frame, k_top=0.3, k_bot=0.85, k_left=0.1, k_right=0.9): # в аргументах фрейм и кэфы отступа для сетки
    
    hight_frame = len(frame)
    width_frame = len(frame[0])

    # разрежем изображение на 4 части (перед этим верхнюю часть выкинем вовсе, там нет шайбы)
    Cated_frame = []
    
    # порядок приоритета: (центр, лево, право, низ-центр, низ-право, низ-лево) т.е. где в первую, вторую и .. очередь искать шайбу
    Cated_frame.append(frame[int(hight_frame*k_top):int(hight_frame*k_bot), int(width_frame*k_left):int(width_frame*k_right)])
    Cated_frame.append(frame[int(hight_frame*k_top):int(hight_frame*k_bot), :int(width_frame*k_left)])
    Cated_frame.append(frame[int(hight_frame*k_top):int(hight_frame*k_bot), int(width_frame*k_right):])
    Cated_frame.append(frame[int(hight_frame*k_bot):, int(width_frame*k_left):int(width_frame*k_right)])
    Cated_frame.append(frame[int(hight_frame*k_bot):, :int(width_frame*k_left)])
    Cated_frame.append(frame[int(hight_frame*k_bot):, int(width_frame*k_right):])

    return Cated_frame
         


def detect_frames(self, frames):
    batch_size = 20
    detections = []

    for i in range(0, len(frames), batch_size):
        detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.4)
        detections += detections_batch
    
    return detections


def detect_puck_frames(self, frames):
    detections = []

    for frame in range(len(frames)):
        # режем изображение
        Cated_frames = get_cut_frame(frame, k_top=0.3, k_bot=0.85, k_left=0.1, k_right=0.9)

        # ищем часть изображения с шайбой
        for part_frame in Cated_frames:
            pred = self.model_puck.predict(part_frame, conf = 0.4)[0]
            detection = sv.Detections.from_ultralytics(pred)
            if len(detection):
                break
        
        detections.append(detection)
        
    return detections