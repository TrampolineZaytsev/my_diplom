from utils import read_video, save_video
from trackers import Tracker

import cv2


def main():
    # read video
    video_frames = read_video('input_video/now_input_video.mp4')

    # init tracker
    tracker = Tracker('models/without_puck.pt', 'models/only_puck_1.pt')

    # получаем треки
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')
    #tracks = tracker.get_object_tracks(video_frames)




    # ДЛЯ ОТЛАДКИ
    # num_fr = [149, 216, 344, 380, 400, 450, 477, 624]
    # num_frames = [216, 344]
    # for num_fr in num_frames:
    #     num_tr = 1
    #     print(tracks['goals'][num_fr])
    #     bbox = [int(i) for i in tracks['goals'][num_fr][num_tr]['bbox']]
    #     img = video_frames[num_fr][bbox[1]:bbox[3],bbox[0]:bbox[2]]
    #     cv2.imwrite(f'develop_image/goal_left_{num_fr}.png', img)
    #     #cv2.imwrite(f'develop_image/{num_fr}.png', video_frames[i])



    
    # разбиваем игроков по командам
    # надо настроить алгоритм на поиск наиболее подходящего кадра для
    tracker.split_team(video_frames, tracks["players"])
    

    # интерполяция шайбы   ############# разрезать изображение получше
    tracks['pucks'] = tracker.track_interpolate(tracks, 'pucks')
    tracks['goals'] = tracker.track_interpolate(tracks, 'goals')

    # поиск игрока владеющего шайбой
    tracker.add_have_puck(video_frames, tracks)
    

    # Отрисовка трэккинга
    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    '''
    
    # # сохр. изобр игроков:
    # num_fr = 105
    # num_tr = 18 #18
    # bbox = [int(i) for i in tracks['players'][num_fr][num_tr]['bbox']]
    # img = video_frames[num_fr][bbox[1]:bbox[3],bbox[0]:bbox[2]]
    # cv2.imwrite('output_videos/crop_image.jpg', img)
    '''
    
    
    # сохраним аннотированное видео
    save_video(output_video_frames, 'output_videos/output_video.mp4')

if __name__ == '__main__':
    main()