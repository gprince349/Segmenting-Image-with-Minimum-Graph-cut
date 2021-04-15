import numpy as np 
import cv2
import math as m
from scribble import *

#weights need to be integers because graph algorithms works on 
#integral weights

Lambda = 10000
Sigma = 0.3

#between pixels
INT_F = 100
MIN = 0
MAX = 1e9

def W_ij(I1, I2, Sigma):
    f = -1/(2*Sigma*Sigma)
    D = np.roll(I2,-1,axis=0)
    U = np.roll(I2,1, axis=0)
    L = np.roll(I2,1,axis=1)
    R = np.roll(I2,-1,axis=1)

    D = (np.exp(f*np.square(I1-D))*INT_F).astype(int)
    U = (np.exp(f*np.square(I1-U))*INT_F).astype(int)
    L = (np.exp(f*np.square(I1-L))*INT_F).astype(int)
    R = (np.exp(f*np.square(I1-R))*INT_F).astype(int)

    return D,U,L,R


def compute_pdfs(scribble_F, scribble_B):
    mean1 = np.mean(scribble_B)
    mean2 = np.mean(scribble_F)
    std1 = np.std(scribble_B)
    std2 = np.std(scribble_F)
    GB = (mean1,std1)
    GF = (mean2,std2)
    print("GB => ",GB)
    print("GF => ",GF)
    return GB,GF

def Gauss(x,mean,std):
        print(mean,std)
        f = (1/(std*m.sqrt(2*m.pi)))*1000
        print("f =>",f)
        mat = (x-mean)/std
        # print("mat====>",mat)
        res =  f*np.exp(-0.5*np.square(mat))
        # print("res====>", res)
        return res
    

def WiFB(img,Lambda,GB,GF):

    Prob_F = Gauss(img, GF[0],GF[1])
    Prob_B = Gauss(img, GB[0],GB[1])
  
    WiF = -1*Lambda*np.log(np.divide(Prob_B,(Prob_F+Prob_B)))
    WiB = -1*Lambda*np.log(np.divide(Prob_F,(Prob_F+Prob_B)))
    
    return WiF,WiB

def filter_weights(WiF,WiB, position_F, position_B,MIN,MAX):
    for p in position_F:
        WiF[p[0],p[1]] = MAX
        WiB[p[0],p[1]] = MIN

    for p in position_B:
        print(p)
        WiF[p[0],p[1]] = MIN
        WiB[p[0],p[1]] = MAX
        
    return WiF,WiB

def get_graph(img, Sigma, Lambda, F_pos, B_pos, list_B, list_F):

    D,U,L,R = W_ij(img,img,Sigma)

    #inter-pixels (with Background and foreground) weight matrix
    GB,GF = compute_pdfs(list_F, list_B)
    WiF,WiB = WiFB(img,Lambda,GB,GF)

    #for known scrible positions making wights infinite and 0 
    WiF,WiB = filter_weights(WiF,WiB, F_pos, B_pos, MIN,MAX)
    WiF = WiF.astype(int)
    WiB = WiB.astype(int)


    return 


if __name__ == "__main__":

    img = cv2.imread('../data/deer.png',cv2.IMREAD_GRAYSCALE)
    img = img/256
    print(img.shape)

    scribble_F_pos, scribble_B_pos, scribble_B, scribble_F = scribe("deer.png")

    # #intra-pixels weight matrix
    D,U,L,R = W_ij(img,img,Sigma)
    # print(D)

    #inter-pixels (with Background and foreground) weight matrix
    GB,GF = compute_pdfs(scribble_F, scribble_B)
    WiF,WiB = WiFB(img,Lambda,GB,GF)

    #for known scrible positions making wights infinite and 0 
    WiF,WiB = filter_weights(WiF,WiB, scribble_F_pos, scribble_B_pos, MIN,MAX)
    WiF = WiF.astype(int)
    WiB = WiB.astype(int)

    # print(WiB)
    print(type(WiF))

    # img = img/256
    # print(img)
    # print(D)
    # cv2.imshow('deer',D)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()