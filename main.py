from utils import read_video, save_video
from trackers import Tracker
from team_assign import TeamAssigner
import cv2
def main():
    # read video
    video_frames = read_video('input_video/input_video.mp4')
    len_video = len(video_frames)
    # init tracker
    tracker = Tracker('models/without_puck.pt', 'models/only_puck_1.pt')

    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pk1')
    

    #tracks = tracker.get_object_tracks(video_frames) #######################

    team_assigner = TeamAssigner()

    # здесь доработать (в кадре может быть мало игроков или вообще не быть)
    num_frame_team = int(len_video*0.4)
    frame_team = video_frames[num_frame_team]
    team_assigner.get_teams(frame_team, tracks["players"][num_frame_team])

    print(team_assigner.team_colors)

    for num_frame, frame in enumerate(video_frames):
        for track_id, track in tracks["players"][num_frame].items():
            team_cur_player = team_assigner.get_team_for_player(frame, track_id, track["bbox"])
            track["team"] = team_cur_player
            track["team_color"] = team_assigner.team_colors[team_cur_player]




    # Отрисовка трэккинга
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # ДЛЯ ОТЛАДКИ
    # сохр. изобр игроков:
    num_fr = 116
    num_tr = 6 #18
    bbox = [int(i) for i in tracks['players'][num_fr][num_tr]['bbox']]
    img = video_frames[num_fr][bbox[1]:bbox[3],bbox[0]:bbox[2]]
    cv2.imwrite('output_videos/crop_image.jpg', img)
    print(bbox)
    
    
    # сохраним аннотированное видео
    save_video(output_video_frames, 'output_videos/output_video.mp4')

if __name__ == '__main__':
    main()