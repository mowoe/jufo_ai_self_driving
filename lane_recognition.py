import cv2
import numpy as np
import sys
cap = cv2.VideoCapture(sys.argv[1])
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import time


def without_regression(img, bac):
    lines = cv2.HoughLinesP(img, 1, np.pi/180,100,100,10)
    if type(lines) != type(None):
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(bac, (x1,y1),(x2,y2), (0,255,0),3)    
    return bac

def regression(image, bac):
    rets = []
    xvals = []
    yvals = []
    h1 = image[0:image.shape[0],0:image.shape[1]/2]
    bh = np.zeros_like(h1)
    h2 = image[0:image.shape[0],image.shape[1]/2:image.shape[1]]
    h2 = np.concatenate((bh,h2),axis=1)
    hs = [h1,h2]
    for b in range(2):
        x = np.array([])
        y = np.array([])
        m,c = 0,0
        img = hs[b]
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength,maxLineGap)
        if type(lines) != type(None):
            for x1,y1,x2,y2 in lines[0]:
                x = np.append(x,x1)
                y = np.append(y,y1)
                x = np.append(x,x2)
                y = np.append(y,y2)
        A = np.vstack([x, np.ones(len(x))]).T
        if x.shape[0]  != 0:
            m, c = np.linalg.lstsq(A, y)[0]
        rets.append(m)
        rets.append(c)
        xvals.append(x)
        yvals.append(y)
    resp = rets
    try:
        startp = (int(min(xvals[0])),int(min(xvals[0]) * resp[0]+resp[1]))
        endp = (int(max(xvals[0])),int(max(xvals[0]) * resp[0] + resp[1]))
        startp1 = (int(min(xvals[1])),int(min(xvals[1]) * resp[2]+resp[3]))
        endp1 = (int(max(xvals[1])),int(max(xvals[1]) * resp[2] + resp[3]))
        cv2.line(bac, (startp),(endp),(0,255,0),3)
        cv2.line(bac, (startp1),(endp1),(0,255,0),3)
    except ValueError, e:
        print e, "occurred"
    return bac
    
def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    rows, cols = image.shape[:2]
    vertices = np.array([[100,720],[1200,720],[850,500],[450,500]], dtype=np.int32)
    return filter_region(image, [vertices])

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def apply_smoothing(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=5, high_threshold=30):
    return cv2.Canny(image, low_threshold, high_threshold)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)




while cap.isOpened():
    start = time.time()
    ret, img = cap.read()
    bac = np.copy(img)
    img = convert_gray_scale(img)
    cv2.imshow("hi",img)
    cv2.waitKey(0)
    cop = np.zeros(img.shape)
    img = apply_smoothing(img)
    cv2.imshow("hi",img)
    cv2.waitKey(0)
    img = detect_edges(img)
    cv2.imshow("hi",img)
    cv2.waitKey(0)
    img = select_region(img)
    cv2.imshow("hi",img)
    cv2.waitKey(0)
    view1 = regression(img, bac)  
    for p in [(0,680),(1200,720),(850,500),(450,500)]:
        cv2.circle(view1, p, 10, (255,0,0),3)
    view1 = cv2.resize(view1, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("detected",view1)
    view2 = without_regression(img, bac)
    for p in [(0,680),(1200,720),(850,500),(450,500)]:
        cv2.circle(view2, p, 10, (255,0,0),3)
    view2 = cv2.resize(view2, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("detected without regression",view2)
    if cv2.waitKey(33) == ord('q'):
        print "pressed a"
        break

    print "took:", time.time() -start
    if time.time() -start <  0.03:
        while time.time() -start < 0.03:
            time.sleep(0.0001)
