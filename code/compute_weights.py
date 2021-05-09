import numpy as np 
import cv2
import math as m
# from scribble import *

#weights need to be integers because graph algorithms works on 
#integral weights

# Lambda = 20
# Sigma = 0.01

#between pixels
INT_F = 1
MIN = 0
MAX = 100
eps = 1e-8

def W_ij(I1, I2, Sigma):
    m, n = I1.shape
    idx = np.arange(m*n).reshape(m, n)

    f = -1/(2*Sigma*Sigma)
    D = np.roll(I2,-1,axis=0)
    U = np.roll(I2,1, axis=0)
    L = np.roll(I2,1,axis=1)
    R = np.roll(I2,-1,axis=1)

    D = (np.exp(f*np.square(I1-D))*INT_F)
    cv2.imshow("new",D)
    cv2.waitKey(6000)
    cv2.destroyAllWindows()
    U = (np.exp(f*np.square(I1-U))*INT_F)
    L = (np.exp(f*np.square(I1-L))*INT_F)
    R = (np.exp(f*np.square(I1-R))*INT_F)

    # print("D==========>",D)
    # indices switch up <-> down, left <-> right
    U_idx = np.roll(idx,-1,axis=0)
    D_idx = np.roll(idx,1,axis=0)
    R_idx = np.roll(idx,1,axis=1)
    L_idx = np.roll(idx,-1,axis=1)

    return D,U,L,R, D_idx,U_idx,L_idx,R_idx, idx


def compute_pdfs(list_F, list_B):
    mean1 = np.mean(list_B)
    mean2 = np.mean(list_F)
    std1 = np.std(list_B)
    std2 = np.std(list_F)
    GB = (mean1,std1)
    GF = (mean2,std2)
    print("GB => ",GB)
    print("GF => ",GF)
    return GB,GF

def Gauss(x,mean,std):
    # print(mean,std)
    f = (1/(std*m.sqrt(2*m.pi)))
    print("f =>",f)
    mat = (x-mean)/std
    # print("mat====>",mat)
    res =  f*np.exp(-0.5*np.square(mat))
    # print("res====>", res)
    return res


def WiFB(img,Lambda,GB,GF):

    Prob_F = Gauss(img, GF[0],eps+GF[1])
    Prob_B = Gauss(img, GB[0],eps+GB[1])
  
    WiF = -1*Lambda*np.log( np.divide(Prob_B,(eps+Prob_F+Prob_B)))
    WiB = -1*Lambda*np.log( np.divide(Prob_F,(eps +Prob_F+Prob_B)))
    # print("WiB========>",WiB)
    # print("WiF=============>",WiF)
    return WiF,WiB

def filter_weights(WiF,WiB, position_F, position_B,MIN,MAX):
    for p in position_F:
        WiF[p[0],p[1]] = MAX
        WiB[p[0],p[1]] = MIN

    for p in position_B:
        # print(p)
        WiF[p[0],p[1]] = MIN
        WiB[p[0],p[1]] = MAX
        
    return WiF,WiB

def get_graph(img, Sigma, Lambda, F_pos, B_pos, list_B, list_F):
    D,U,L,R, D_idx,U_idx,L_idx,R_idx, idx = W_ij(img,img,Sigma)

    #inter-pixels (with Background and foreground) weight matrix
    GB,GF = compute_pdfs(list_F, list_B)
    WiF,WiB = WiFB(img,Lambda,GB,GF)

    #for known scrible positions making weights infinite and 0 
    WiF,WiB = filter_weights(WiF,WiB, F_pos, B_pos, MIN,MAX)
    # WiF = WiF.astype(int)
    # WiB = WiB.astype(int)

    # flatten everything to fit into adjacency list
    # m, n = img.shape
    # D, U, L, R = D.flatten(), U.flatten(), L.flatten(), R.flatten()
    # D_idx,U_idx,L_idx,R_idx, idx = D_idx.flatten(),U_idx.flatten(),L_idx.flatten(),R_idx.flatten(), idx.flatten()
    # WiF, WiB = WiF.flatten(), WiB.flatten()

    # convention of adj_list (Ii, [Iup, Idown, Ileft, Iright]) (WiF) (WiB)
    # idx_F, idx_B = m*n, m*n + 1
    # graph = [ [(U_idx[i], U[i]), (D_idx[i], D[i]), (L_idx[i], L[i]), (R_idx[i], R[i]), (idx_F, WiF[i]), (idx_B, WiB[i])] for i in range(m*n)]
    # graph.append( list(zip(idx, WiF)) )
    # graph.append( list(zip(idx, WiB)) )

    return D,U,L,R,WiF,WiB



if __name__ == "__main__":

    from scribble import *
    file = "../data/deer.png"
    scribe = Scribe(file)
    F_pos, B_pos, list_B, list_F = scribe.startscribe()

    img = cv2.imread(file, 0)
    # #intra-pixels weight matrix
    D,U,L,R, D_idx,U_idx,L_idx,R_idx, idx = W_ij(img,img,Sigma)

#     # #intra-pixels weight matrix
#     D,U,L,R, D_idx,U_idx,L_idx,R_idx, idx = W_ij(img,img,Sigma)

#     #inter-pixels (with Background and foreground) weight matrix
#     GB,GF = compute_pdfs(list_F, list_B)
#     WiF,WiB = WiFB(img,Lambda,GB,GF)

#     #for known scrible positions making wights infinite and 0 
#     WiF,WiB = filter_weights(WiF,WiB, F_pos, B_pos, MIN,MAX)
#     WiF = WiF.astype(int)
#     WiB = WiB.astype(int)
