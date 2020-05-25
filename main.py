import argparse, os
import cv2
import vid_read as vid_read
import rppg_classes as rppg_classes

def main():
    # load arguments from CLI
    print('Loading arguments from CLI.')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='vid_path', type=str, help='Path to input video. If not use webcam')
    parser.add_argument('-sw', dest='sw_size', type=int, required=True, help='Sliding window size (in seconds)')
    parser.add_argument('-fd', dest='facedet_alg', type=str, help='Face detection algorithm')
    parser.add_argument('-roi', dest='roi_alg', type=str, help='ROI selection algorithm')
    parser.add_argument('-rppg', dest='rppg_alg', default='g', type=str, help='RPPG processing algorithm')
    parser.add_argument('-gui', dest='gui', type=int, default=False, help='Display results or not and how.')
    parser.add_argument('-log', dest='log', type=bool, default=False, help='Log results.')
    args = parser.parse_args()
    
    # do some trys and exceptions blocks, make algos into ints
    vid_path = args.vid_path
    sw_size = args.sw_size
    facedet_alg = args.facedet_alg
    roi_alg = args.roi_alg
    rppg_alg = args.rppg_alg
    gui = args.gui
    log = args.log

    if facedet_alg == 'haar':
        facedet_alg = 0
        face_class = cv2.CascadeClassifier(os.path.join('opencv', 'haarcascade_frontalface_default.xml'))
    if roi_alg == 'skin':
        roi_alg = 0

    # initialize rppg_trad object
    print('Initializing RPPG object.')
    print('Initialization parameters: ', vid_path, sw_size, facedet_alg, roi_alg, 
    rppg_alg, gui, log)
    vid_name, cap, width, height, fps = vid_read.vid_read(vid_path)
    rppg = rppg_classes.RPPG_trad_HR(width, height, fps, sw_size, 
    facedet_alg, roi_alg, rppg_alg, face_class, gui, log)

    # begin reading frames and passing them to rppg_trad object to process
    print('Begin processing video.')
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        hr_calc = rppg.processFrame(frame)
        if i%100==0:
            print('Frame: {}, Predicted HR: {} BPM'.format(i, hr_calc))
        i += 1
    cv2.destroyAllWindows()
    print('Finished processing video and program.')

main()