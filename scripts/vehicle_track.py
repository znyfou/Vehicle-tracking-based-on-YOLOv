import copy
import time
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import cv2

#[cx, cy, h, w, dx, dy, dh, dw] X
#[cx, cy, h, w] Z
class KFtracker:
    kfcount = 0
    def __init__(self, init_infor, dt=0.01, maxage=15, auto_id=-1):
        if auto_id < 0:
            self.id = KFtracker.kfcount
            KFtracker.kfcount += 1
        else:
            self.id = auto_id
        self.conf = init_infor[5]
        self.cls = init_infor[4]

        self.age = 0 #跟踪器存在帧数
        self.hit_streak = 0 #连续匹配帧数
        self.hit = True #是否匹配
        self.time_since_update = 0  #自从上次更新以来的帧数
        self.max_age = maxage   #最大存活帧数

        self.dim_x = 8
        self.dim_z = 4
        self.X = np.zeros((self.dim_x, 1))

        self.kf = KalmanFilter(8, 4)
        self.kf.R[:4, :4] *= 5  #测量噪声协方差矩阵
        self.kf.Q[4:, 4:] *= 0.1    #过程噪声协方差矩阵
        self.kf.P = self.kf.P*1000      #初始不确定性较大
        self.dt = dt
        self.kf.F = np.array([[1, 0, 0, 0, self.dt, 0, 0, 0],
                              [0, 1, 0, 0, 0, self.dt, 0, 0],
                              [0, 0, 1, 0, 0, 0, self.dt, 0],
                              [0, 0, 0, 1, 0, 0, 0, self.dt],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])
        z = self.xyxytoZ(init_infor[:4])
        self.kf.x[:4] = z.reshape(-1, 1)  # 位置初始化
        self.kf.x[4:] = 0.0  # 速度初始化为0
        self.history = []

    def update(self, xyxy, dt=0.01):
        if len(xyxy) > 5:
            self.cls = int(xyxy[4])
            self.conf = float(xyxy[5])
        z = self.xyxytoZ(xyxy)
        self.dt = dt
        self.kf.update(z)

        self.hit = True
        self.time_since_update = 0
        self.hit_streak += 1
        self.history.append(xyxy)

    def predict(self):
        self.kf.predict()

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.hit = False

        return self.Xtoxyxy(self.kf.x)

    @classmethod
    def track(cls, detections, trackers):
        if len(trackers) == 0:
            for d in detections:
                t = KFtracker(d)
                trackers.append(t)

            return trackers

        m, umd, umt = cls.match(detections, trackers)
        
        for id, it, R in m:
            if R == 0:
                trackers[it].update(detections[id])
            else:
                t = KFtracker(detections[id], dt=trackers[it].dt, auto_id=trackers[it].id)
                t.history = copy.deepcopy(trackers[it].history)
                t.history.append(detections[id])
                t.hit_streak = trackers[it].hit_streak
                t.hit = True
                t.age = trackers[it].age
                trackers[it] = t


        for id in umd:
            t = KFtracker(detections[id])
            trackers.append(t)

        uidx = []
        for it in umt:
            t = trackers[it]
            t.time_since_update += 1

            if t.time_since_update > t.max_age:
                uidx.append(it)

        for ui in uidx:
            if len(trackers) > ui:
                trackers.pop(ui)

        return trackers

    @classmethod
    def match(cls, detections, trackers, iou_s=0.5):
        """
        返回已匹配的检测框和跟踪器索引， 未匹配的检测框索引， 未匹配的跟踪器索引
        """
        targets = [t.predict() for t in trackers]
        
        iou_m = cls.IOU_M(detections, targets)

        row_idx, col_idx = linear_sum_assignment(iou_m)
        match_ = []
        
        rc = list(zip(row_idx, col_idx))
        
        unmatch_detections = list(np.setdiff1d(np.arange(len(detections)), row_idx))
        unmatch_trackers = list(np.setdiff1d(np.arange(len(trackers)), col_idx))
        for r, c in rc:
            if -iou_m[r, c]+0.15 >= iou_s:
                if -iou_m[r, c] - iou_s < 0.1:#重新匹配
                    match_.append((r, c, 1))
                else:
                    match_.append((r, c, 0))
            else:
                unmatch_detections.append(r)
                unmatch_trackers.append(c)

        return match_, unmatch_detections, unmatch_trackers

    @classmethod
    def IOU_M(cls, tests, targets):
        """
        tests/targets=[xyxy xyxy xyxy...]
        """
        iou_m = np.zeros((len(tests), len(targets)), dtype=np.float32)
        for j, test in enumerate(tests):
            for i, target in enumerate(targets):
                iou = cls.IOU(test, target)
                iou_m[j, i] = -iou

        return iou_m

    @classmethod
    def IOU(self, test, target):
        lx = np.maximum(test[0], target[0])
        ly = np.maximum(test[1], target[1])
        rx = np.minimum(test[2], target[2])
        ry = np.minimum(test[3], target[3])
        w = np.maximum(0, rx - lx)
        h = np.maximum(0, ry - ly)
        iou = w*h/((test[2]-test[0])*(test[3]-test[1])+
                   (target[2]-target[0])*(target[3]-target[1]) - w*h)
        return iou.item()

    @classmethod
    def xyxytoZ(cls, xyxy):
        cx = (xyxy[0] + xyxy[2])/2
        cy = (xyxy[1] + xyxy[3])/2
        h = (xyxy[2] - xyxy[0])
        w = (xyxy[3] - xyxy[1])
        return np.array([cx, cy, h, w])

    @classmethod
    def Xtoxyxy(cls, X):
        lx = X[0] - X[2]/2
        ly = X[1] - X[3]/2
        rx = X[0] + X[2]/2
        ry = X[1] + X[3]/2
        return np.array([lx, ly, rx, ry])

model = YOLO("yolov5su.pt")
video = cv2.VideoCapture("../data/test_1.mp4")
tracks = []
COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
fps = 1
while True:
    ret, frame = video.read()
    if ret or frame is None==0:
        break

    t1 = time.time()
    results = model.predict(frame)
    xyxys = [res.boxes.xyxy[0] for res in results[0]]
    infors = [[float(xyxys[idx][0]), float(xyxys[idx][1]), float(xyxys[idx][2]), float(xyxys[idx][3]),
               int(results[0][idx].boxes.cls), float(results[0][idx].boxes.conf)] for idx in range(len(results[0]))]
    

    tracks = KFtracker.track(infors, tracks)
    count_car = 0
    for t in tracks:
        if not t.hit:
            continue
        count_car += 1
        pred = t.predict().reshape(4)

        color = [int(c) for c in COLORS[t.id % len(COLORS)]]
        cv2.rectangle(frame, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), color, 2)
        cv2.putText(frame, f"{t.id} {model.names[t.cls]} {t.conf:.2f}", (int(pred[0]), int(pred[1]) - 10), 0, 5e-1, color, 1)
    cv2.putText(frame, f"car: {count_car}", (0, 40), 0, 5e-1, (255, 0, 0), 1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (int(0), int(20)), 0, 5e-1, (0,255,0), 2)

    fps = (fps + (1. / (time.time() - t1))) / 2
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()
