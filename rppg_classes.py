import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import vid_read as vid_read
import skin_seg_naive as skin_seg_naive
import filters as filters
SEC_PER_MIN = 60
LOW_FREQ = 0.7 # 42 BPM
HIGH_FREQ = 4.0 # 240 BPM
FILTER_ORDER = 6

FONT = cv2.FONT_HERSHEY_SIMPLEX 
FONTSCALE = 1
THICKNESS = 2
# color in BGR
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


class RPPG_trad_HR:
    def __init__(self, width, height, fps, 
    sw_size, facedet_alg, roi_alg, rppg_alg, face_class, 
    gui=False, log=False, debug=False):
        self.width = width
        self.height = height
        self.fps = fps
        self.sw_size = sw_size
        self.facedet_alg = facedet_alg
        self.roi_alg = roi_alg
        self.rppg_alg = rppg_alg
        self.face_class = face_class
        self.gui = gui
        self.log = log
        self.debug = debug
        
        self.buffer_size = int(fps*sw_size)
        self.init_buffer()
        self.low_freq = LOW_FREQ
        self.high_freq = HIGH_FREQ
        self.filt_order = FILTER_ORDER
    
    def init_buffer(self):
        self.signal_buffer = np.zeros((self.buffer_size, 3))
        self.accum = 0

        if (self.gui==2) or (self.log==True):
            self.signal_original = np.zeros((self.buffer_size, 3))
            self.signal_norm = np.zeros((self.buffer_size, 3))
            self.signal_final = np.zeros((self.buffer_size, 1))

    def processFrame(self, frame):

        # face detection and ROI selection
        seg_img, mask = self.detFace(frame)
         
        # if no mask found (no face det) then make self.accum = 0 and 
        # self.signal_buffer full of zeros, aka, start accumulating signal again
        if mask is None:
            self.init_buffer()
        else:    
            # calculate channel mean and index into numpy array, 
            # from 0 until numpy array is full, then put at end
            imgn = np.where(seg_img>0, seg_img, np.nan)
            chan_mean = np.nanmean(imgn, axis=(0,1))
            
            # if accumulator gets full then do the rppg processing
            # to filter signal then calculate HR by freq domain peaks
            if self.accum == (self.buffer_size):
                self.signal_buffer = np.roll(self.signal_buffer, shift=-1, axis=0)
                self.signal_buffer[self.accum-1, :] = chan_mean
                #print(self.signal_buffer)
            
                signal_final = self.rppgProcessing()
                hr_calc = self.calcHR(signal_final)

                # if gui is turned on then it will show the original frame
                # and the frame after masking
                if self.gui == 1:
                    self.disp_gui(frame, 'Original Frame', seg_img, hr_calc)

                return hr_calc

            if self.accum <= (self.buffer_size):
                self.signal_buffer[self.accum, :] = chan_mean
                self.accum += 1

                if self.gui == 1:
                    self.disp_gui(frame, 'Original Frame', seg_img)

                return None
    
    def detFace(self, frame):
        # detect if there's face
        # if yes, roiSelection(bounding_boxes, frame)
        # if not, return nothing/skip to next frame and also
        if self.facedet_alg == 0: # haar
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bounding_boxes = self.face_class.detectMultiScale(gray, 1.3, 5)
            
        if bounding_boxes is None:
            seg_img = None
            mask = None
            return seg_img, mask

        frame_bb = np.copy(frame)
        for (x,y,w,h) in bounding_boxes:
            frame_bb = cv2.rectangle(frame_bb,(x,y),(x+w,y+h),(0,255,0),2)
        if self.gui == 2:
            self.disp_gui(frame_bb, 'Face Detection Frame')

        if self.roi_alg == 0:
            seg_img, mask = self.roiSelection(frame, bounding_boxes)
        else:
            pass # return forehead ROI

        return seg_img, mask
        
    def roiSelection(self, frame, bounding_boxes):
        # use bounding boxes from detFace to apply skin detection
        # or shrink to forehead or other region
        # returns mask of 0s and 1s
        seg_img, mask = skin_seg_naive.skin_seg(frame, bounding_boxes)
        return seg_img, mask
        
    def rppgProcessing(self):
        # returns filtered 1-d time series
        if self.rppg_alg == 'g':
            signal_final = self.rppg_g()
            return signal_final
        
    def rppg_g(self):
        # g algorithm as taken from https://github.com/prouast/heartbeat:
        signal_original = np.copy(self.signal_buffer[:, 1]).reshape((-1, 1))
        # temp and space normalization
        signal_norm = filters.normalization(signal_original)
        # detrend filter (LPF)
        signal_detrend = filters.detrend(signal_norm, self.fps)
        # moving average (LPF)
        signal_mavg = filters.movingAverage(signal_detrend, np.fmax(math.floor(self.fps/6), 2))
        # butterworth bandpass filter (BPF), 6th order
        signal_bpf = filters.butter_bandpass_filter(signal_mavg, 
        self.low_freq, self.high_freq, self.fps, order=self.filt_order)

        signal_final = signal_mavg

        if (self.gui==2) or (self.log==True):
            self.signal_original = signal_original
            self.norm = signal_norm
            self.signal_final = signal_final

        if self.debug == True:
            signals = [signal_original, signal_norm, signal_detrend, signal_mavg, signal_bpf]
            self.plot_static(signals)

        return signal_final

    def calcHR(self, signal_final):
        # transforms filtered 1-d time series to freq domain
        # normalize and use peak finder function
        # return peak and multiply * 60 to obtain HR
        n_rows = signal_final.size
        freq_ticks = self.fps*np.arange(0, math.floor(n_rows/2)+1, 1)/n_rows
        
        signal_time = np.copy(signal_final)
        signal_freq_twosided = np.abs(np.fft.fft(signal_time))/n_rows
        signal_freq_onesided = signal_freq_twosided[:math.floor(n_rows/2)+1]
        signal_freq_onesided[1:-1] = 2*signal_freq_onesided[1:-1]

        low_freq_ind = np.where(freq_ticks>self.low_freq)[0][0] # first low freq
        high_freq_ind = np.where(freq_ticks<self.high_freq)[0][-1] # last high freq
        peak = low_freq_ind + np.argmax(signal_freq_onesided[low_freq_ind:high_freq_ind])
        hr_calc = freq_ticks[peak]*60
        
        #peak = np.argmax(signal_freq_onesided)
        #plt.plot(freq_ticks, signal_freq_onesided)
        #plt.show()

        return hr_calc
        
    def disp_gui(self, frame, name, seg_img=None, hr_calc=None):
        if seg_img is not None:
            cv2.imshow('mask', seg_img)
            if hr_calc is not None:
                    text = "HR (CALC): %.0f BPM" % (hr_calc)
                    #position of bottom left corner of text in pixels
                    org = (int(0.40*self.width), int(0.80*self.height))
                    
                    seg_img = cv2.putText(seg_img, text, 
                    org, FONT, FONTSCALE, RED, THICKNESS, cv2.LINE_AA)
                    cv2.imshow('mask', seg_img)

        else:
            cv2.imshow('{}'.format(name), frame)
        if self.gui == 2:
            pass
            # plot original, normalized and final time signal for g channel
            # plot current section of the signal only so it looks "animated" vs time/frame

        cv2.waitKey(1)

    def plot_static(self, signals):
        # do a for where loops through signals and just names them according to steps
            fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(18, 12))
            axs[0].plot(signal_original)
            axs[1].plot(signal_norm)
            axs[2].plot(signal_detrend)
            axs[3].plot(signal_mavg)
            axs[4].plot(signal_bpf)

            axs[0].title.set_text('Signal original (pre-filtering')
            axs[1].title.set_text('Signal post space and temporal normalization')
            axs[2].title.set_text('Signal post detrend filter')
            axs[3].title.set_text('Signal post moving average filter')
            axs[4].title.set_text('Signal post bandpass filter (final signal)')
            
            plt.xlabel('Time (s)')
            fig.text(0.0075, 0.5, 'Channel mean intensity', rotation='vertical', horizontalalignment='center')
            plt.subplots_adjust(left=0.5, hspace=0.3)
            fig.tight_layout()
            plt.show()  
