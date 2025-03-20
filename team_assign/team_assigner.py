from sklearn.cluster import KMeans
from utils import get_center_of_boxx, bbox_is_square, distance_color
import numpy as np


class TeamAssigner:
    def __init__(self):
        self.dict_team_of_player = {} # словарь уже определенных в предыдущих кадрах игроков
        self.kmean_team_model = None
        self.team_colors = {}
        
    # функция обучения модели кластеризации
    def get_claster_model(self, points, n_clusters):
        img2d = np.array(points)
        if len(img2d.shape) == 3:
            img2d = img2d.reshape(-1, 3)
        kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1)
        kmeans_model.fit(img2d)
        return kmeans_model

    # функция определения цвета игрока
    def get_player_color(self, frame, player_bbox):
        
        # вычленим коробку игрока
        player_bbox = [int(i) for i in player_bbox]
        frame_team = frame[player_bbox[1]:player_bbox[3], player_bbox[0]:player_bbox[2]]

        # нам нужна только верхняя часть (цвет майки)
        hight_img = len(frame_team)
        top_half_img = frame_team[int(hight_img*0.19):int(hight_img*0.5), :]

        # кластеризуем изображение
        kmeans = self.get_claster_model(top_half_img, 3)
        
        # найдем фон, как самый яркий кластер
        centers = kmeans.cluster_centers_.tolist()
        background_color = max(centers, key = lambda x: sum(x))
        backgraund_label = centers.index(background_color)

        # найдем веса кластеров игрока (нужно для среднего взвешенного) 
        labels = kmeans.labels_
        labels_list = labels.tolist()
        label_pl1, label_pl2 = set(labels_list) - set([backgraund_label])
        w1, w2 = labels_list.count(label_pl1), labels_list.count(label_pl2)

        # смешиваем краски)
        player_color = np.array([(i*w1 + j*w2)/(w1 + w2) for i, j in zip(centers[label_pl1], centers[label_pl2])])


        # определим флаги, уведомляющие о том, что мешает назначить корректный цвет
        dirty_back = False # на фоне не только лед
        near_border = False # игрок скраю изображения


        # проверим скраю ли игрок
        box_x, box_y = get_center_of_boxx(player_bbox)
        hight_frame, width_frame = len(frame), len(frame[0])
        if not((width_frame*0.1 < box_x < width_frame*0.9) and (hight_frame*0.1 < box_y < hight_frame*0.9)):
            near_border = True

        # проверим грязный ли фон:
        labels_2d = labels.reshape(top_half_img.shape[0], -1)
        corner_labels = [labels_2d[0, 0], labels_2d[0, -1], labels_2d[-1, 0], labels_2d[-1, -1]]
        if corner_labels.count(backgraund_label) < 4:
            dirty_back = True
        

        return player_color, dirty_back, near_border



    # функция определения команд и их цветов (выполняется один раз)
    def get_teams(self, frames_team, player_detections):

        # массив с качественно определенными по нескольким кадрам цветами игроков 
        player_colors = []
        for frame_team, cur_player_boxes in zip(frames_team, player_detections):
            
            for player_detect in cur_player_boxes:
                player_bbox = player_detect['bbox']
                player_color, dirty_back, near_border = self.get_player_color(frame_team, player_bbox)

                # если фон не грязный и игрок полностью в кадре, то добавляем в список цветов игроков
                if (bbox_is_square(player_bbox) or not(near_border)) and not(dirty_back):
                    player_colors.append(player_color)

        
        # разобъем всех игроков на 2 кластера по цветам
        self.kmean_team_model = self.get_claster_model(player_colors, 2)
        
        self.team_colors[0] = self.kmean_team_model.cluster_centers_[0]
        self.team_colors[1] = self.kmean_team_model.cluster_centers_[1]


    # функция определения команды игрока
    def get_team_for_player(self, frame_team, player_id, player_bbox):

        
        player_color, dirty_back, near_border = self.get_player_color(frame_team, player_bbox)
        
        


        # смотрим, когда нельзя назначать и лучше вернуть этот трек, если он есть:
        # игрок скраю И не полностью виден (не квадратная коробка)
        # на фоне другие игроки (грязный фон)
        # цвет игрока достаточно близко к одной из команд

        # проверим грязный ли цвет игрока:
        dirty_color = False
        dist_to_team_1 = distance_color(player_color, self.team_colors[0])
        dist_to_team_2 = distance_color(player_color, self.team_colors[1])
        if 0.4 < dist_to_team_1/dist_to_team_2 < 1.6:
            dirty_color = True

        # проверим все условия
        if (player_id in self.dict_team_of_player) \
        and (not(bbox_is_square(player_bbox)) and near_border or dirty_back or dirty_color): 
            return self.dict_team_of_player[player_id]
        

        # предсказываем по цвету команду
        player_team = self.kmean_team_model.predict(player_color.reshape(1, -1))[0]
        self.dict_team_of_player[player_id] = player_team

        return player_team
        


        
        