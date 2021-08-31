"""
    Demo of YOLO object detection with a WebCam using PySimpleGUI
    Copyright 2019, 2020 PySimpleGUI
    www.PySimpleGUI.com
    Licensed under LGPL3+
    The YOLO detection code was provided courtsey of Dr. Adrian Rosebrock of the pyimagesearch organization.
    https://www.pyimagesearch.com
    If you use this program, you agree to keep this header in place.
    It's OK to build on other people's works, it's not ok to remove their credits and claim the work as your own.
"""

import numpy as np
import time
import cv2
import os
import PySimpleGUI as sg
import queue
import threading
# import PySimpleGUIQt as sg        # Runs on Qt too... just change the import.

from mylib.cnn import *
from mylib.pre_processing import *
import time
import pytorchocr.predict_rec as predict_rec


text_recognizer = predict_rec.TextRecognizer()


img_path = r'alpr.jpg'

sg.theme('LightGreen')
# rtsppath = r'rtsp://garage:Aa1234567890@94.215.54.32:88/videoMain'
rtsppath = r'F:\AI_FL\chinese_code\code\test_plate1.mp4'
cap = None

frame = cv2.imread(img_path)
imgbytes = cv2.imencode('.png', frame)[1].tobytes()

gui_queue = queue.Queue()  # queue used to communicate between the gui and the threads
t_started = False
gvpath = None
timgbytes = imgbytes
event=''
values=''

cap = None
def aiprocess(gui_queue):
    global event,values,cap,imgbytes,gvpath
    frame_i = 0
    W, H = None, None
    gvpath = values['input']
    oldpath = gvpath
    rtsppath = gvpath
    while True:             # Event Loop
        if event in (None, 'Exit'):
            break
        if oldpath is not gvpath:
            print('tt:',rtsppath)
            W, H = None, None            
            if cap is not None:
                cap.release()
                time.sleep(5)

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        cap = cv2.VideoCapture(rtsppath)  # initialize the capture device
        while True:
            plateRet, plateFrame = cap.read()
            if not plateRet:
                print('none')
                break
            #srcimg = copy.copy(plateFrame)
            plateFrame = cv2.resize(plateFrame, (plateFrame.shape[1] // 2, plateFrame.shape[0] // 2))            
            #plateFrame = cv2.imread('F:\\data\\2.jpg')
            frame_i += 1

            if oldpath is not gvpath:
                rtsppath = gvpath
                W, H = None, None            
                if cap is not None:
                    cap.release()
                    time.sleep(5)
                oldpath = gvpath
                break
            t1 = time.time()
            plate_Original, plate_morphEx, edge = preprocess(plateFrame, (15,3), False)
            #cv2.imshow("original", plate_Original)

            img_crop_list = []
            box_list = []
            # find plate countours
            plate_countours,_ = cv2.findContours(plate_morphEx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for plt_countour in plate_countours:

                # ratio of width to hieght
                aspect_ratio_range, area_range = (2.8, 8), (1000, 18000)
                #aspect_ratio_range, area_range = (2.8, 3), (250, 4500)

                # validate the countours, return boolean (True, False)
                if contour_vladition(plt_countour, plate_morphEx, aspect_ratio_range, area_range):
                    rect = cv2.minAreaRect(plt_countour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    #cv2.drawContours(plate_Original, [box], 0, (0,255,0), 1) #change position after CNN
                    x_s, y_s = [i[0] for i in box], [i[1] for i in box]
                    x1, y1 = min(x_s), min(y_s)
                    x2, y2 = max(x_s), max(y_s)

                    angle = rect[2]
                    if angle < -45: angle += 90

                    W, H = rect[1][0], rect[1][1]
                    aspect_ratio = float(W)/H if W > H else float(H)/W

                    center = ((x1+x2)/2, (y1+y2)/2)
                    size = (x2-x1, y2-y1)
                    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
                    tmp = cv2.getRectSubPix(edge, size, center)
                    Tmp_w = H if H > W else W
                    Tmp_h = H if H < W else W
                    tmp = cv2.getRectSubPix(tmp, (int(Tmp_w),int(Tmp_h)), (size[0]/2, size[1]/2))
                    __,tmp = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    white_pixels = 0
                    for x in range(tmp.shape[0]):
                        for y in range(tmp.shape[1]):
                            if tmp[x][y] == 255:
                                white_pixels += 1

                    edge_density = float(white_pixels)/(tmp.shape[0]*tmp.shape[1])

                    tmp = cv2.getRectSubPix(plateFrame, size, center)
                    tmp = cv2.warpAffine(tmp, M, size)
                    Tmp_w = H if H > W else W
                    Tmp_h = H if H < W else W
                    tmp = cv2.getRectSubPix(tmp, (int(Tmp_w),int(Tmp_h)), (size[0]/2, size[1]/2))


                    # tmp = cv2.getRectSubPix(srcimg, (size[0]*2, size[1]*2), (center[0]*2, center[1]*2))
                    # tmp = cv2.warpAffine(tmp, M, (size[0]*2, size[1]*2))
                    # Tmp_w = H if H > W else W
                    # Tmp_h = H if H < W else W
                    # tmp = cv2.getRectSubPix(tmp, (int(Tmp_w * 2),int(Tmp_h * 2)), (size[0], size[1]))           
                    #cv2.imshow("plate_original", plate_Original)
                    

                    tmpcl = copy.copy(tmp)
                    tmp = im_reshape(tmp, plate_img_size, "plate_buffer.jpg")

                    data = tmp.reshape(plate_img_size, plate_img_size, 1)
                    plate_model_out = plate_model.predict([data])[0]
                    t2 = time.time()
                    if not np.argmax(plate_model_out) == 1:
                        img_crop_list.append(tmpcl)
                        box_list.append([box])
                    else:
                        continue
                        
            rec_res, elapse = text_recognizer(img_crop_list)
            print("rec_res num  : {}, elapse : {}".format(
                len(rec_res), elapse))
            # self.print_draw_crop_rec_res(img_crop_list, rec_res)
            filter_rec_res = [], []
            for rec_reuslt, box in zip(rec_res, box_list):
                #cv2.drawContours(plate_Original, box, 0, (0,255,255), 2) #change position after CNN
                text, score = rec_reuslt
                if score >= 0.5:
                    print("{}, {:.3f}".format(text, score))
                    print(box[0])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    imgzi = cv2.putText(plate_Original, text, (box[0][2][0], box[0][2][1]), font, 0.8, (255, 255, 255), 1)
                    cv2.drawContours(plate_Original, box, 0, (0,0,255), 2) #change position after CNN
            
            print('time:', time.time() - t1)

            imgbytes = cv2.imencode('.png', plate_Original)[1].tobytes()
            gui_queue.put('done')  # put a message into queue for GUI


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
sg.popup_quick_message('Loading weights from disk.... one moment...', background_color='red', text_color='white')

layout = [
            [sg.Text('alpr Window', size=(30, 1))],
            [sg.Image(data=imgbytes, key='_IMAGE_')],            
            [sg.Text('rtsp'), sg.In(rtsppath,size=(15,15), key='input'), sg.Button('ok')],
            [sg.Exit()]
        ]
win = sg.Window('Webcam', layout, default_element_size=(14, 1), text_justification='right', auto_size_text=True, finalize=True)
image_elem = win['_IMAGE_']

while True:
    event, values = win.Read(timeout=0)       # wait for up to 100 ms for a GUI event
    if event is None or event == 'Exit':
        break
    
    elif event is 'ok':
        try:
            if not t_started:
                t_started = True
                print('create thread')
                thldai = threading.Thread(target=aiprocess, args=(gui_queue,), daemon=False)
                print('Starting thread')
                thldai.start()
            gvpath = values['input']
        except Exception as e:
            print('Error starting work thread.')
            
    try:
        message = gui_queue.get_nowait()
    except queue.Empty:             # get_nowait() will get exception when Queue is empty
        message = None              # break from the loop if no more messages are queued up
        
    if message:
        timgbytes = imgbytes
    image_elem.update(data=timgbytes)
        

print("[INFO] cleaning up...")
win.close()
if cap is not None:
    cap.release()
