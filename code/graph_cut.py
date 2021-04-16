from collections import defaultdict 
import argparse
import cv2
import numpy as np 
import os

from scribble import *
from compute_weights import *

# graph[parent][s]
def find(graph, parent, s):
    for (i, val) in graph[parent]:
        if i == s:
            return val
    return 0

# graph[parent][s] += v
def add(graph_tmp, parent, s, v):
    adj = graph_tmp[parent][:]
    found = False
    for idx, (i, val) in enumerate(adj):
        if i == s:
            adj[idx][1] += v
            found = True
            break

    if not found:
        adj.append([s, v])

    return adj

def conv_adj_list(adj):
    l = len(adj)
    g = np.zeros((l, l))
    for i in range(l):
        print(i)
        for j in range(len(adj[i])):
            g[i][ adj[i][j][0] ] = adj[i][j][1]

    return g.tolist()


def masked_img(img, visited):
    m, n, _ = img.shape
    mask = np.array(visited[:-2]).reshape(m, n).astype("uint8")
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


class Graph: 
  
    def __init__(self, file): 
        fname = os.path.basename(file)
        self.file = file
        self.fname = fname
        self.orig_img = cv2.imread(file)
        self.orig_img = cv2.resize(self.orig_img, (30, 30))

    def initialise_graph(self, graph):
        self.graph = graph # residual graph 
        self.org_graph = [i[:] for i in graph] 
        self.ROW = len(graph) 
        self.COL = len(graph[-1]) 
  
    '''Returns true if there is a path from 
    source 's' to sink 't' in 
    residual graph. Also fills 
    parent[] to store the path '''
    def BFS(self,s, t, parent): 
  
        visited =[False]*(self.ROW) 
  
        queue=[] 
  
        # Mark the source node as visited and enqueue it 
        queue.append(s) 
        visited[s] = True
  
        # Standard BFS Loop 
        while queue: 
  
            #Dequeue a vertex from queue and print it 
            u = queue.pop(0) 
  
            # Get all adjacent vertices of 
            # the dequeued vertex u 
            # If a adjacent has not been
            # visited, then mark it 
            # visited and enqueue it 

# ======================================================================
            for ind, val in enumerate(self.graph[u]): 
            # for (ind, val) in self.graph[u]: 
                if visited[ind] == False and val > 0 : 
                    queue.append(ind) 
                    visited[ind] = True
                    parent[ind] = u 
  
        self.display_image(visited)
        # If we reached sink in BFS starting
        # from source, then return 
        # true, else false 
        return True if visited[t] else False
          
    # Function for Depth first search 
    # Traversal of the graph
    def dfs(self, graph,s,visited):
        visited[s]=True
        for i in range(len(graph)):
# ======================================================================
            if graph[s][i]>0 and not visited[i]:
            # if find(graph, s, i)>0 and not visited[i]:
                self.dfs(graph,i,visited)
  
    # Returns the min-cut of the given graph 
    def minCut(self, source, sink): 
  
        # This array is filled by BFS and to store path 
        parent = [-1]*(self.ROW) 
  
        max_flow = 0 # There is no flow initially 
        s = sink
        # Augment the flow while there is path from source to sink 
        while self.BFS(source, sink, parent) : 
            print(max_flow)
            # Find minimum residual capacity of the edges along the 
            # path filled by BFS. Or we can say find the maximum flow 
            # through the path found. 
            path_flow = float("Inf") 
            s = sink 
            while(s != source): 
# ======================================================================
                path_flow = min (path_flow, self.graph[parent[s]][s])
                # path_flow = min (path_flow, find(self.graph, parent[s], s) ) 
                s = parent[s] 
  
            max_flow += path_flow 
  
            # update residual capacities of the edges and reverse edges 
            # along the path 
            v = sink 
            while(v != source): 
                u = parent[v] 
# ======================================================================
                self.graph[u][v] -= path_flow 
                self.graph[v][u] += path_flow 
                # self.graph[u] = add(self.graph, u, v, -path_flow)
                # self.graph[v] = add(self.graph, v, u, path_flow)
                v = parent[v] 
  
        visited=len(self.graph)*[False]
        self.dfs(self.graph,s,visited)
  
        # print the edges which initially had weights 
        # but now have 0 weight 
        for i in range(self.ROW): 
            for j in range(self.COL): 
# ======================================================================
                if self.graph[i][j] == 0 and self.org_graph[i][j] > 0 and visited[i]: 
                # if find(self.graph, i, j) == 0 and find(self.org_graph, i, j) > 0 and visited[i]: 
                    print(str(i) + " - " + str(j) )
        return visited

    def display_image(self, visited, save=False):
        out_img = masked_img(self.orig_img, visited)
        cv2.imshow("Segmented Image", out_img)
        cv2.waitKey(1)
        if save:
            cv2.imwrite(out_img, "../results/" + self.fname)

    def segment_image(self):
        img = self.orig_img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img/256

        s1 = Scribe(img.copy())
        F_pos, B_pos, list_B, list_F = s1.startscribe()

        graph = get_graph(gray_img, Sigma, Lambda, F_pos, B_pos, list_B, list_F)

        print(len(graph), len(graph[-1]))
        graph = conv_adj_list(graph)

        # convention of adj_list (Ii, [Iup, Idown, Ileft, Iright]) (WiF) (WiB)
        self.initialise_graph(graph)
        ROW = self.ROW
        src = ROW - 2
        sink = ROW - 1

        visited = self.minCut(src, sink)
        
        self.display_image(visited)
        # cv2.waitKey(-1)
        k = cv2.waitKey(-1) & 0xFF
        if k == ord('s'):
            self.display_image(visited, save=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Image to segment')
    args = parser.parse_args()
    file = args.img

    # Create a graph given in the above diagram 
    # graph = [[0, 16, 13, 0, 0, 0], 
    #         [0, 0, 10, 12, 0, 0], 
    #         [0, 4, 0, 0, 14, 0], 
    #         [0, 0, 9, 0, 0, 20], 
    #         [0, 0, 0, 7, 0, 4], 
    #         [0, 0, 0, 0, 0, 0]] 

    # adj_list = [[[1, 16], [2, 13]],
    #             [[2, 10], [3, 12]],
    #             [[1, 4], [4, 14]],
    #             [[2, 9], [5, 20]],
    #             [[3, 7], [5, 4]],
    #             []]

    # c = conv_adj_list(adj_list)
    # source = 0; sink = 5

    g = Graph(file) 
    g.segment_image()