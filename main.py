from utils import read_video, save_video
from trackers import Tracker
from team_assign import TeamAssigner
from player_have_puck import PlayerPuckAssigner
import cv2
def main():
    # read video
    video_frames = read_video('input_video/input_video.mp4')

    # init tracker
    tracker = Tracker('models/without_puck.pt', 'models/only_puck_1.pt')

    # получаем треки
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')
    #tracks = tracker.get_object_tracks(video_frames)
    
    # разбиваем игроков по командам
    # надо настроить алгоритм на поиск наиболее подходящего кадра для
    tracks["players"] = tracker.split_team(video_frames, tracks["players"])
    

    # интерполяция шайбы
    # получилось лучше чем думал, но надо дообучить модель детекции шайбы и разрезать изображение получше
    tracks['pucks'] = tracker.puck_interpolate(tracks['pucks'])
    
    # поиск игрока владеющего шайбой
    player_puck_assigner = PlayerPuckAssigner()
    for num_frame in range(len(video_frames)):
        
        if tracks['pucks'][num_frame]:
            puck = tracks['pucks'][num_frame][1]['bbox']
        else:
            continue # затычка на случай, если нет интерполяции шайбы
        player_puck_id = player_puck_assigner.assign_player_puck(tracks["players"][num_frame], puck)

        if player_puck_id is not None:
            tracks["players"][num_frame][player_puck_id]['has_puck'] = True

        
        
        
    

    # Отрисовка трэккинга
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # ДЛЯ ОТЛАДКИ
    # # сохр. изобр игроков:
    # num_fr = 105
    # num_tr = 18 #18
    # bbox = [int(i) for i in tracks['players'][num_fr][num_tr]['bbox']]
    # img = video_frames[num_fr][bbox[1]:bbox[3],bbox[0]:bbox[2]]
    # cv2.imwrite('output_videos/crop_image.jpg', img)
    
    
    # сохраним аннотированное видео
    save_video(output_video_frames, 'output_videos/output_video.mp4')

if __name__ == '__main__':
    main()