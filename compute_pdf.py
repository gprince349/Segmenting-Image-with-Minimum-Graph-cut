import numpy as np 
import cv2
import math as m

Lambda = 10
sigma = 0.3

#weights need to be integers because graph algorithms works on 
#integral weights

#between pixels
INT_F = 100

def W_ij(I1, I2, sigma):
    f = -1/(2*sigma*sigma)
    D = np.roll(I2,-1,axis=0)
    U = np.roll(I2,1, axis=0)
    L = np.roll(I2,1,axis=1)
    R = np.roll(I2,-1,axis=1)

    D = (np.exp(f*np.square(I1-D))*INT_F).astype(int)
    U = (np.exp(f*np.square(I1-U))*INT_F).astype(int)
    L = (np.exp(f*np.square(I1-L))*INT_F).astype(int)
    R = (np.exp(f*np.square(I1-R))*INT_F).astype(int)

    return D,U,L,R


def compute_pdfs(scrible_F, scrible_B):
    mean1 = np.mean(scrible_B)
    mean1 = np.mean(scrible_F)
    std1 = np.std(scrible_B)
    std2 = np.std(scrible_F)
    GB = (mean1,std1)
    GF = (mean2,std2)
    return GB,GF

def Gauss(x,mean,sigma):
       return  (1/(sigma*m.sqrt(2*m.pi)))*np.exp(np.square((x-mean)/sigma))


def WiFB(img,Lambda,GB,GF):

    Prob_F = Gauss(img, GF[0],GF[1])
    Prob_G = Gauss(img, GB[0],GB[1])
  
    WiF = np.divide(Prob_B,(Prob_F+Prob_G))
    WiB = np.divide(Prob_F,(Prob_F+Prob_G))
    
    return WiF,WiB


# img = cv2.imread('deer.png',cv2.IMREAD_GRAYSCALE)

# img = img/256
# print(img)

# D,U,L,R = W_ij(img,img,sigma)
# print(D)

# cv2.imshow('deer',D)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()