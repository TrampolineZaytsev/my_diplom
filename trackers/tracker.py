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


class Tracker:
    def __init__(self, model_path, model_puck_path):
        self.model_puck = YOLO(model_puck_path)
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # детектируем моделью
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        detections_puck = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.4)
            detections += detections_batch
            detections_puck_batch = self.model_puck.predict(frames[i:i+batch_size], conf = 0.4)
            detections_puck += detections_puck_batch

        
        return detections, detections_puck


    def puck_interpolate(self, puck_coords):

        # получаем из словарей двумерный массив
        puck_coords = [i.get(1,{}).get('bbox', []) for i in puck_coords]

        # интерполируем кадры без шайбы
        df_puck_coords = pd.DataFrame(puck_coords, columns=['x1', 'y1', 'x2', 'y2'])
        df_puck_coords.interpolate(inplace=True)
        df_puck_coords.bfill(inplace=True)

        # преобразуем обратно в словари для трекеров 
        puck_coords = [{1:{'bbox':i}} for i in df_puck_coords.to_numpy().tolist()]

        return puck_coords


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections, detections_puck = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "goalies":[],
            "goals":[],
            "pucks":[]
        }
############################################################
        for frame_num in range(len(detections)):

            detection = detections[frame_num]
            detection_puck = detections_puck[frame_num]
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}
            
            # вычленяем из предсказания коробки классов
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_puck_supervision = sv.Detections.from_ultralytics(detection_puck)

            # отслеживаем треки коробок
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            #detection_puck_with_tracks = self.tracker.update_with_detections(detection_puck_supervision)

            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["goalies"].append({})
            tracks["goals"].append({})
            tracks["pucks"].append({})


            for cur_box_obj in detection_with_tracks:
                bbox = cur_box_obj[0].tolist()
                cls_id = cur_box_obj[3]
                track_id = cur_box_obj[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['goalie']:
                    tracks["goalies"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['goal']:
                     tracks["goals"][frame_num][1] = {"bbox":bbox}

            for cur_box_puck in detection_puck_supervision:
                bbox = cur_box_puck[0].tolist()
                tracks["pucks"][frame_num][1] = {"bbox":bbox}



            '''for cur_box_obj in detection_supervision: # как будто в один цикл можно засунуть
                bbox = cur_box_obj[0].tolist()
                cls_id = cur_box_obj[3]

                if cls_id == cls_names_inv['goal']:
                    tracks["goals"][frame_num][1] = {"bbox":bbox}'''


        #это для сохранения трекеров в папку чтобы при отладке не ждать заного
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_boxx(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width/2), int(0.15*width)),
            angle=0.0,
            startAngle= -45,
            endAngle=205,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        

        if track_id is not None:

            #статичный размер закоментирован
            r_width = 30
            r_hight = 23
            #r_width = width*0.2
            #r_hight = width*0.15
            r_x1 = int(x_center - r_width//2)
            r_y1 = int(y2 - r_hight//2 + 0.15*width)
            r_x2 = int(x_center + r_width//2)
            r_y2 = int(y2 + r_hight//2 + 0.15*width)

            if len(str(track_id))>1:
                dif_x = -r_width*0.005
            else:
                dif_x = r_width*0.24

            cv2.rectangle(frame,
                        (int(r_x1), int(r_y1)),
                        (int(r_x2), int(r_y2)),
                        color=color,
                        thickness=-1)
            
            cv2.putText(frame,
                        str(track_id),
                        (int(r_x1 + dif_x), int(r_y2 - r_hight*0.1)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.025*r_width,
                        color=(0, 0, 0))
        return frame 
    

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_boxx(bbox)
        
        size_tr = 10
        tr_points = np.array([[x, y],
                             [x - size_tr, y - size_tr],
                             [x + size_tr, y - size_tr]])
        
        cv2.drawContours(frame, [tr_points], 0, color, thickness=-1)
        cv2.drawContours(frame, [tr_points], 0, (0, 0, 0), 2)

        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalie_dict = tracks["goalies"][frame_num]
            goal_dict = tracks["goals"][frame_num]
            puck_dict = tracks["pucks"][frame_num]

            # отрисовка меток
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], player["team_color"], track_id)

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (150, 150, 150))

            for track_id, goalie in goalie_dict.items():
                frame = self.draw_ellipse(frame, goalie["bbox"], (0, 255, 0), frame_num)

            for track_id, goal in goal_dict.items():
                frame = cv2.rectangle(frame, 
                                      (int(goal["bbox"][0]), int(goal["bbox"][1])),
                                      (int(goal["bbox"][2]), int(goal["bbox"][3])),
                                      color=(255,0,0), thickness=2)
            
            for track_id, puck in puck_dict.items():
                frame = self.draw_traingle(frame, puck["bbox"], (105, 0, 198))

            output_video_frames.append(frame)
            
        

        return output_video_frames