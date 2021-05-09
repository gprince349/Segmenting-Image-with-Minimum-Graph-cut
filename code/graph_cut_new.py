import maxflow
import argparse
import cv2
import numpy as np 

from scribble import *
from compute_weights import *



if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Image to segment')
    args = parser.parse_args()
    file = args.img


    orig_img = cv2.imread(file)
    print(orig_img)
    orig_img = cv2.blur(orig_img, (3,3))
    orig_img = cv2.resize(orig_img, (600, 600), interpolation=cv2.INTER_CUBIC)
    img = orig_img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_orig = gray_img
    gray_img = gray_img/256
    # print("grayimg======================>",gray_img)

    m,n  = gray_img.shape
    s1 = Scribe(img.copy(), gray_img.copy())
    F_pos, B_pos, list_F, list_B = s1.startscribe()
    print("Blist===================>",list_B)

    test_img = orig_img
    Sigma = 0.1
    Lambda = 0.4
    for i in range(1):
        orig_img = test_img
        # Sigma = Sigma + 0.01

        print("Lambda ==> ", Lambda)
        D,U,L,R,WiF,WiB = get_graph(gray_img, Sigma, Lambda, F_pos, B_pos, list_B, list_F)
        cv2.imshow("fore",WiF)
        cv2.imshow("back", WiB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("WiF========================>",WiF)
        print("WiB========================>",WiB)
        print("D=========================>", D)
        graph = maxflow.Graph[float]()

        nodeid = graph.add_nodes(m*n)

        for i in range(m):
            for j in range(n):
                if(j!= n-1):
                    graph.add_edge(i*m+j,i*m+j+1,0,R[i,j])
                if(i!= m-1):
                    graph.add_edge(i*m+j,(i+1)*m+j,0,D[i,j])

        for i in range(m):
            for j in range(n):
                    graph.add_tedge(i*m+j,WiF[i,j],WiB[i,j])

        mask = np.zeros([m,n],dtype='uint8')

        print(graph.maxflow())

        for i in range(m):
            for j in range(n):
                if(graph.get_segment(i*m+j)):
                    orig_img[i,j,:] = 255

        # print(mask)
        cv2.imshow("image",orig_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        Lambda = Lambda+0.03




