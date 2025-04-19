import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_approx_polinom
from sklearn.cluster import KMeans

class Mark_detector:

    def __init__(self):
        pass 

    def get_contour_line(self, image, line):
        
        # преобразуем в нужное цвет. простр-во
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hue_min, hue_max = None, None
        # фильтруем по цвету нужные элементы на изображении
        if line == 'yellow':
            hue_min = np.array((10, 50, 160))
            hue_max = np.array((25, 220, 250))
        elif line == 'blue':
            hue_min = np.array((10, 50, 160))
            hue_max = np.array((25, 220, 250))
        elif line == 'red':
            hue_min = np.array((10, 50, 160))
            hue_max = np.array((25, 220, 250))
        else:
            hue_min = np.array((10, 50, 160))
            hue_max = np.array((25, 220, 250))

        # наблюдение для желтой линии
        '''
        # # в ргб
        # hue_min = np.array((200, 160, 30))
        # hue_max = np.array((240, 200, 120))

        # в hsv
        # если желтая линяя не за воротами, то можно:
        # hue_min = np.array((10, 150, 160))
        # hue_max = np.array((25, 220, 250))
        '''

        # фильтруем изображение на нужный цвет
        filtered_start = cv2.inRange(image_hsv, hue_min, hue_max)

        # заполняем светлые пятна
        kernel_b = np.ones((5,5), np.uint8)
        filtered = cv2.morphologyEx(filtered_start, cv2.MORPH_OPEN, kernel_b)

        # заполняем темные пятна
        kernel_w = np.ones((5,30), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_w)

        # находим контуры
        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # находим масив точек всех контуров
        merge_contours = np.array([])
        for contour in contours:
            merge_contours = np.append(merge_contours, contour.squeeze())
        merge_contours = merge_contours.reshape(-1, 2)
        
        return merge_contours


    def get_rink_borders(self, frame):

        # получаем точки для аппроксимации
        points_of_line = self.get_contour_line(frame)

        # подбираем полином для аппроксимации 
        coef, mse = get_approx_polinom(points_of_line, 5)
        coef_1, mse_1 = get_approx_polinom(points_of_line, 1)
        if mse_1 < mse+30: 
            coef, mse = coef_1, mse_1
        poly = np.poly1d(coef)


        # строим касательные по краям
        # находим производную
        coef_dif_poly = np.polyder(coef)
        dif_poly = np.poly1d(coef_dif_poly)
        dif2_poly = np.poly1d(np.polyder(coef_dif_poly))
        
        left_x, right_x = 0, 1900
        step_x = (right_x - left_x) // 30
        list_k = [] # список значений производных, когда полином похож на прямую
        list_b = [] #
        list_L = [] # список прямых в составе полинома
        for x in range(left_x, right_x, step_x):
            dif2_poly_x = dif2_poly(x)

            # проверка на непрямой участок
            if dif2_poly_x > 0.000249:

                # если долго была "прямая", параметры усредняем и добавляем прямую
                if len(list_k) > 1:
                    mean_k, mean_b = sum(list_k)/len(list_k), sum(list_b)/len(list_b)
                    list_L.append([mean_k, mean_b])
                    list_k, list_b = [], []

            else:
                dif_poly_x = dif_poly(x)
                list_k.append(dif_poly_x)
                list_b.append(poly(x)-dif_poly_x*x)
        
        # добавляем вторую прямую, если она есть
        if len(list_k) > 1:
            mean_k, mean_b = sum(list_k)/len(list_k), sum(list_b)/len(list_b)
            list_L.append([mean_k, mean_b])


        return list_L