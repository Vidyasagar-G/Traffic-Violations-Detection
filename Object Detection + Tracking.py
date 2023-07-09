import cv2
import numpy as np
import sys
import glob
import keras_ocr

import time
import torch



class YoloDetector():

    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        self.classes = self.model.names
        print(self.classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 1
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame, height, width, confidence=0.3):

        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            if row[4]>=confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                if self.class_to_label(labels[i]) == 'car':

                    x_center = x1 + (x2-x1)
                    y_center = y1 + ((y2-y1) / 2)

                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype = np.float32)
                    confidence = float(row[4].item())
                    feature = 'car'

                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), 'car'))
        
        return frame, detections
    
    
cap = cv2.VideoCapture("C:\\All data\\IITG\\Projects\\Traffic Light\\Techevince Traffic Light Detection\\Resources\\input\\Set01_video01.h264")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

detector = YoloDetector()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from deep_sort_realtime.deepsort_tracker import DeepSort

object_tracker = DeepSort(
    max_age=5,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None)



vehicles_entering = {}


x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
four_cc = cv2.VideoWriter_fourcc(*"MJPG")
out_file = "C:\\All data\\IITG\\Projects\\Traffic Light\\Techevince Traffic Light Detection\\output.avi"
out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))

while cap.isOpened():
    success, img = cap.read()
    

    start = time.perf_counter()

    results = detector.score_frame(img)

    img,detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)

    tracks = object_tracker.update_tracks(detections, frame=img)

    area_1 = [(0,300), (1920,300), (0,560), (1920,560)]
    line_1 = [(0,300), (1920,300)]
    line_2 = [(0,560), (1920,560)]
    cv2.line(img, line_1[0], line_1[1], (0,255,255),2)
    cv2.line(img, line_2[0], line_2[1], (0,0,255), 3)
    area_2 = [(0,0), (1920,0), (0,323), (1920,300)]

    

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]),int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cx = int((bbox[0]+bbox[2])/2)
        cy = int((bbox[1]+bbox[3])/2)
        

        result_1 = cv2.pointPolygonTest(np.array(area_1,np.int32), (int(cx), int(cy)), False)
        result_2 = cv2.pointPolygonTest(np.array(area_2,np.int32), (int(cx), int(cy)), False)
        if result_1 >=0:
            cv2.circle(img,(cx,cy), 5, (0,255,0), -1)
            if track_id not in vehicles_entering:
                vehicles_entering[track_id] = [time.time(), time.time()]
            else:
                vehicles_entering[track_id][1] = time.time()

        if result_2 >=0:
            cv2.circle(img,(cx,cy), 5, (0,0,255), -1)
            distance = 7
            if track_id in vehicles_entering and (vehicles_entering[track_id][1]!=vehicles_entering[track_id][0]):
                speed_ms = distance/(vehicles_entering[track_id][1]-vehicles_entering[track_id][0])
                speed_kmph = speed_ms * 3.6
                speed_kmph = round(speed_kmph,2)
                cv2.putText(img, "Speed: " + str(speed_kmph) + " kmph", (int(bbox[0]-10),int(bbox[3])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0,255,0), 2)
    print(vehicles_entering)
    
    
        
    

    end = time.perf_counter()
    totalTime = end-start
    fps = 1/totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    cv2.imshow('img', img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()