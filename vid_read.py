import cv2
import os

def vid_read(vid_path):
    try:
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    except:
        vid_name = 'webcam'
        vid_path = 0
    cap= cv2.VideoCapture(vid_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cap.get(cv2.CAP_PROP_FOURCC)
    print('Video Width: {}, Height: {}, FPS: {}, No. Frames: {}, Codec: {}.'.format(width, height, fps, no_frames, codec))

    return vid_name, cap, width, height, fps