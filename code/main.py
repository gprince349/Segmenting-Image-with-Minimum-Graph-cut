from scribble import *
from compute_weights import *
from graph_cut import *

import argparse
import matplotlib.pyplot as plt 
import cv2

LAMBDA = 10000
SIGMA = 0.3

'''
weights need to be integers because graph algorithms works on 
integral weights
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Image to segment')
    args = parser.parse_args()
    file = args.img

    img = cv2.imread(file, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img/256

    F_pos, B_pos, list_B, list_F = scribe("deer.png")

    graph = get_graph(img, SIGMA, LAMBDA, F_pos, B_pos, list_B, list_F):

    g = Graph(graph)
     
    source = 0; sink = 5
      
    print ("The maximum possible flow is %d " % g.FordFulkerson(source, sink))
