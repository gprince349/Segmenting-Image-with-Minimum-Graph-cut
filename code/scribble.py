import cv2
import numpy as np

fname = "../data/deer.png"
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
bp = []
rp = []
img = np.zeros((4,4))

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,bp,rp

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.circle(img,(x,y),3,(255,0,0),-1)
                bp.append((y,x))
            else:
                cv2.circle(img,(x,y),3,(0,0,255),-1)
                rp.append((y,x))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(img,(x,y),3,(255,0,0),-1)
            bp.append((y,x))
        else:
            cv2.circle(img,(x,y),3,(0,0,255),-1)
            rp.append((y,x))



def scribe(fn):
    global img,mode,bp,rp
    img = cv2.imread(fn)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == ord('s'):
            break

    img = cv2.imread(fn,0)
    bp = list(set(bp))
    rp = list(set(rp))

    bpixval = [img[x,y] for (x,y) in bp]
    rpixval = [img[x,y] for (x,y) in rp]
    cv2.destroyAllWindows()
    return bp,rp,bpixval,rpixval

if __name__ == "__main__":
    a,b,c,d = scribe(fname)