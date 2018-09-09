import cv2 as cv
import numpy as np


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    (fx,fy) = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (255,0, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
window_size = 3


frames= []
flows = []



cap = cv.VideoCapture('project_video.mp4')
#initiate frames and flows
for i in range(window_size):
    _,frame = cap.read()
    frames.append(cv.cvtColor(frame,cv.COLOR_BGR2GRAY))

step = 16
h,w = frames[0].shape[:]
y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)


for i in range(window_size-1):
    flow = cv.calcOpticalFlowFarneback(frames[i],frames[i+1],None,0.5,3,15,3,5,1.2,0)
    # (fx,fy) = flow[y,x].T
    flows.append(flow)


while True:
    ret,frame = cap.read()
    org=frame
    frame = cv.cvtColor(frame.copy() , cv.COLOR_BGR2GRAY)
    for i in range(window_size-1):
        frames[i]=frames[i+1]
    frames[window_size-1] = frame
    print(flow)
    for i in range(window_size-3):
        print(i)
        flows[i]=flows[i+1]

    flow=cv.calcOpticalFlowFarneback(frames[window_size-3],frames[window_size-2],None,0.5,3,15,3,5,1.2,0)
    # (fx,fy) = flow[y,x].T
    flows[window_size-2]=flow
    print(flow.shape)
    

    #
    flow_diff = flows[1]-flows[0]
    vis_org = draw_flow(frame.copy(),flow_diff)
    vis=draw_flow(frame.copy(),flows[0])
    vis_1 = draw_flow(frame.copy(),flows[1])
    # vis=draw_hsv(flows[0])
    # vis_1=draw_hsv(flows[1])
    # vis_org = abs(vis_1 - vis)

    tmp = abs(draw_hsv(flows[1])-draw_hsv(flows[0]))
    vis_org = cv.cvtColor(tmp,cv.COLOR_HSV2BGR)


    # cv.imshow('flow0',vis)
    # cv.imshow('flow1',vis_1)
    cv.imshow('frame',vis_org)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
