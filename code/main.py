import argparse
import cv2
import numpy as np 

from scribble import *
from compute_weights import *
from graph_cut import *


'''
weights need to be integers because graph algorithms works on 
integral weights
'''

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
    s1 = Scribe(file)
    F_pos, B_pos, list_B, list_F = s1.startscribe()

    print(F_pos)
    graph = get_graph(gray_img, Sigma, Lambda, F_pos, B_pos, list_B, list_F)
    print(graph[:5], len(graph[-2]), len(graph[-1]))

    # g = Graph(graph)
     
    # print ("The maximum possible flow is %d " % g.FordFulkerson(source, sink))
