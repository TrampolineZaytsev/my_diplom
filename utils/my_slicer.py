# функция режущая изображение

class MySlicer:
    def __init__(self, k_top=0.3, k_bot=0.85, k_left=0.1, k_right=0.9):
        self.k_top=k_top
        self.k_bot=k_bot
        self.k_left=k_left
        self.k_right=k_right
        self.hight_frame = None
        self.width_frame = None
    def get_cut_frame(self, frame): # в аргументах фрейм и кэфы отступа для сетки
        
        self.hight_frame = len(frame)
        self.width_frame = len(frame[0])

        # разрежем изображение на 4 части (перед этим верхнюю часть выкинем вовсе, там нет шайбы)
        Cated_frame = []
        
        # порядок приоритета: (центр, лево, право, низ-центр, низ-право, низ-лево) т.е. где в первую, вторую и .. очередь искать шайбу
        Cated_frame.append(frame[int(self.hight_frame*self.k_top):int(self.hight_frame*self.k_bot), int(self.width_frame*self.k_left):int(self.width_frame*self.k_right)])
        Cated_frame.append(frame[int(self.hight_frame*self.k_top):int(self.hight_frame*self.k_bot), :int(self.width_frame*self.k_left)])
        Cated_frame.append(frame[int(self.hight_frame*self.k_top):int(self.hight_frame*self.k_bot), int(self.width_frame*self.k_right):])
        Cated_frame.append(frame[int(self.hight_frame*self.k_bot):, int(self.width_frame*self.k_left):int(self.width_frame*self.k_right)])
        #Cated_frame.append(frame[int(hight_frame*k_bot):, :int(width_frame*k_left)]) # для увеличения скорости
        #Cated_frame.append(frame[int(hight_frame*k_bot):, int(width_frame*k_right):]) # логотип на шайбы похож.

        return Cated_frame
    
    def scaling_box(self, num_part, box):
        x1, y1, x2, y2 = box
        if num_part == 0:
            add_x = self.width_frame*self.k_left
            add_y = self.hight_frame*self.k_top
        elif num_part == 1:
            add_x = 0
            add_y = self.hight_frame*self.k_top
        elif num_part == 2:
            add_x = self.width_frame*self.k_right
            add_y = self.hight_frame*self.k_top
        else:
            add_x = self.width_frame*self.k_left
            add_y = self.hight_frame*self.k_bot
        
        scale_box = [x1+add_x, y1+add_y, x2+add_x, y2+add_y]
        return scale_box