import cv2
import numpy as np


# mouse callback function

# def scribe(fn):
#     drawing = False # true if mouse is pressed
#     mode = True 
#     ix,iy = -1,-1
#     bp = []
#     rp = []
#     img = np.zeros((4,4))
#     # *******************************
#     def draw_circle(event,x,y,flags,param):

#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing = True
#             ix,iy = x,y

#         elif event == cv2.EVENT_MOUSEMOVE:
#             if drawing == True:
#                 if mode == True:
#                     cv2.circle(img,(x,y),3,(255,0,0),-1)
#                     bp.append((y,x))
#                 else:
#                     cv2.circle(img,(x,y),3,(0,0,255),-1)
#                     rp.append((y,x))

#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing = False
#             if mode == True:
#                 cv2.circle(img,(x,y),3,(255,0,0),-1)
#                 bp.append((y,x))
#             else:
#                 cv2.circle(img,(x,y),3,(0,0,255),-1)
#                 rp.append((y,x))
#     # ****************************************
#     img = cv2.imread(fn)
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image',draw_circle)
    
#     while(1):
#         cv2.imshow('image',img)
#         k = cv2.waitKey(1) & 0xFF
#         if k == ord('m'):
#             mode = not mode
#         elif k == ord('s'):
#             break

#     img = cv2.imread(fn,0)
#     bpos = set(bp)
#     bp = list(bpos)
#     rpos = set(rp)
#     rp = list(rpos)

#     bpixval = [img[x,y] for (x,y) in bp]
#     rpixval = [img[x,y] for (x,y) in rp]
#     cv2.destroyAllWindows()
#     return bp,rp,bpixval,rpixval

class Scribe:
    def __init__(self,fname):
        self.fname = fname
        self.bp = []
        self.rp = []
        self.drawing = False # true if mouse is pressed
        self.mode = True
        self.img = cv2.imread(fname)
        self.ix = 0
        self.iy=0
    
    def draw_circle(self,event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix,self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.mode == True:
                    cv2.circle(self.img,(x,y),3,(255,0,0),-1)
                    self.bp.append((y,x))
                else:
                    cv2.circle(self.img,(x,y),3,(0,0,255),-1)
                    self.rp.append((y,x))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == True:
                cv2.circle(self.img,(x,y),3,(255,0,0),-1)
                self.bp.append((y,x))
            else:
                cv2.circle(self.img,(x,y),3,(0,0,255),-1)
                self.rp.append((y,x))

    def startscribe(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_circle)
        while(1):
            cv2.imshow('image',self.img)
            # print("check1")
            k = cv2.waitKey(1) & 0xFF
            # print("check2")
            if k == ord('m'):
                # print("check3")
                self.mode = not self.mode
                # print("check4")
            elif k == ord('s'):
                # print("checks")
                break

        
        img = cv2.imread(self.fname,0)
        bpos = set(self.bp)
        self.bp = list(bpos)
        rpos = set(self.rp)
        self.rp = list(rpos)

        bpixval = [img[x,y] for (x,y) in self.bp]
        rpixval = [img[x,y] for (x,y) in self.rp]
        cv2.destroyAllWindows()
        return self.bp,self.rp,bpixval,rpixval

if __name__ == "__main__":
    s1 = Scribe("../data/deer.png")
    a,b,c,d=s1.startscribe()
    # fname = "deer.png"
    # a,b,c,d = scribe(fname)
    # print(a)
    # # print(c)
