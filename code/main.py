import argparse
import matplotlib.pyplot as plt 
import cv2
import numpy as np 

# from scribble import *
from compute_weights import *
from graph_cut import *

LAMBDA = 10000
SIGMA = 0.05

'''
weights need to be integers because graph algorithms works on 
integral weights
'''


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
            print("check1")
            k = cv2.waitKey(1) and 0xFF
            print("check2")
            if k == ord('m'):
                print("check3")
                self.mode = not self.mode
                print("check4")
            elif k == ord('s'):
                print("checks")
                break

        cv2.destroyAllWindows()
        
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Image to segment')
    args = parser.parse_args()
    file = args.img
    print(file)

    img = cv2.imread(file, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img/256

    # cv2.imshow("", gray_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(file)
    scribe_obj = Scribe(file)
    F_pos, B_pos, list_B, list_F = scribe_obj.startscribe()

    # graph = get_graph(gray_img, SIGMA, LAMBDA, F_pos, B_pos, list_B, list_F)
    # print(graph[:5], len(graph[-2]), len(graph[-1]))

    # g = Graph(graph)
     
    # print ("The maximum possible flow is %d " % g.FordFulkerson(source, sink))
