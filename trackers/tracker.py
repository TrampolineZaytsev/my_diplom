from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import sys
sys.path.append('../')
from utils import get_bbox_width, get_bbox_hight, get_center_of_boxx, MySlicer
from team_assign import TeamAssigner

from utils import get_center_of_boxx, bbox_is_square #################### убери



class Tracker:
    def __init__(self, model_path, model_puck_path):
        self.model_puck = YOLO(model_puck_path)
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.my_slicer = MySlicer()

    # детектируем моделью
    '''
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
    '''

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.4)
            detections += detections_batch
        
        return detections


    def detect_puck_frames(self, frames):

        detections = []
        
        for num_frame in range(len(frames)):
            # режем изображение
            Cated_frames = self.my_slicer.get_cut_frame(frames[num_frame])

            # ищем часть изображения с шайбой
            num_part_puck = None
            for num_part, part_frame in enumerate(Cated_frames):
                imgsz = (part_frame.shape[0], part_frame.shape[1])
                if num_part == 0:
                    pred = self.model_puck.predict(part_frame, conf=0.3, imgsz=imgsz)[0]
                else:
                    pred = self.model_puck.predict(part_frame, conf=0.4, imgsz=imgsz)[0]
                detection = sv.Detections.from_ultralytics(pred)
                if len(detection):
                    num_part_puck = num_part
                    break
            if len(detection):
                detection = detection[0].xyxy[0].tolist()
                detection = self.my_slicer.scaling_box(num_part_puck, detection)
            else:
                detection = []

            detections.append(detection)
            
        return detections


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

        detections = self.detect_frames(frames) 
        detections_puck = self.detect_puck_frames(frames)

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
            detection_puck_supervision = detections_puck[frame_num]

            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}
            
            # вычленяем из предсказания коробки классов
            detection_supervision = sv.Detections.from_ultralytics(detection)
            #detection_puck_supervision = sv.Detections.from_ultralytics(detection_puck)

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

            if len(detection_puck_supervision):
                tracks["pucks"][frame_num][1] = {"bbox":detection_puck_supervision}

            '''
            for cur_box_puck in detection_puck_supervision:
                bbox = cur_box_puck[0].tolist()
                tracks["pucks"][frame_num][1] = {"bbox":bbox}
            '''


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


    def draw_ellipse(self, frame, bbox, color, track_id=None, is_nimb=False):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_boxx(bbox)
        width = get_bbox_width(bbox)
        hight = get_bbox_hight(bbox)

        # игрок владеющий шайбой имеет нимб)
        if is_nimb:
            # для прозрачности наложим изображение поверх друг друга
            overlay = frame.copy()

            # рисуем на доп изображениии элипс
            axes = (int(hight/10), int(hight/40))
            #axes = (10, 2)
            cv2.ellipse(
                overlay,
                center = (x_center, int(bbox[1])-5),
                axes = (int(hight/10), int(hight/50)),
                angle = 0.0,
                startAngle = 0,
                endAngle = 360,
                color = color,
                thickness = 2,
                lineType = cv2.LINE_AA)

            # накладываем изобр друг на друга с прозрачностью algha:
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        else:
            cv2.ellipse(
                frame,
                center=(x_center, y2),
                axes=(int(width/2), int(0.15*width)),
                angle=0.0,
                startAngle= -45,
                endAngle=205,
                color=color,
                thickness=2,
                lineType=cv2.LINE_4)
        

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
                
                # отрисовка нимба над игроком с шайбой
                if player.get('has_puck', False):
                    #frame = self.draw_traingle(frame, player["bbox"], (255, 0, 0))
                    frame = self.draw_ellipse(frame, player["bbox"], (0,190,255), is_nimb=True)

                
                '''
                # if track_id == 18: 
                #     colorP, dirt, near = self.t_assign.get_player_color(frame, player["bbox"])
                    
                #     cv2.rectangle(frame,
                #                 (1500, 900),
                #                 (1900, 1070),
                #                 color=colorP,
                #                 thickness=-1)
            
                #     cv2.putText(frame,
                #                 str(dirt, ),
                #                 (1500, 1070),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.5,
                #                 color=(0, 0, 0))'
                '''
                

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
    

    def split_team(self, video_frames, players):

        team_assigner = TeamAssigner()
        

        # берем несколько кадров из видео для определения цветов команд
        len_video = len(video_frames)
        player_tracks = []
        frames_team = []
        for i in range(1, 10):
            num_frame_team = int(0.1*i*len_video)
            frames_team.append(video_frames[num_frame_team])

            cur_player_boxes = list(players[num_frame_team].values())
            player_tracks.append(cur_player_boxes)
                
        # получаем цвета команд
        team_assigner.get_teams(frames_team,  player_tracks) 

        # покадрово распределяем игроков по командам и дописываем инфу в треки.
        for num_frame, frame in enumerate(video_frames):
            for track_id, track in players[num_frame].items():
                team_of_cur_player = team_assigner.get_team_for_player(frame, track_id, track["bbox"])
                track["team"] = team_of_cur_player
                track["team_color"] = team_assigner.team_colors[team_of_cur_player]

        #self.t_assign = team_assigner ################ для отладки. не забудь убрать
        return players