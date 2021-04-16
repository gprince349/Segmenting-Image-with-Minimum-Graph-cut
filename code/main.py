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

    img = cv2.imread(file, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img/256

    s1 = Scribe(file)
    F_pos, B_pos, list_B, list_F = s1.startscribe()

    graph = get_graph(gray_img, Sigma, Lambda, F_pos, B_pos, list_B, list_F)

    # convention of adj_list (Ii, [Iup, Idown, Ileft, Iright]) (WiF) (WiB)
    g = Graph(conv_adj_list(graph))

    src = -2
    sink = -1

    g.minCut(src, sink)
