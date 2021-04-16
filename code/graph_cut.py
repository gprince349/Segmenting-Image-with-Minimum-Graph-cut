from collections import defaultdict 
import numpy as np

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

class Graph: 
  
    def __init__(self,graph): 
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
  
        # Augment the flow while there is path from source to sink 
        while self.BFS(source, sink, parent) : 
  
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
  

def conv_adj_list(adj):
    l = len(adj)
    g = [[0 for _ in range(l)] for _ in range(l)]
    for i in range(l):
        for j in range(len(adj[i])):
            g[i][ adj[i][j][0] ] = adj[i][j][1]
    return g


if __name__ == "__main__":
    # Create a graph given in the above diagram 
    graph = [[0, 16, 13, 0, 0, 0], 
            [0, 0, 10, 12, 0, 0], 
            [0, 4, 0, 0, 14, 0], 
            [0, 0, 9, 0, 0, 20], 
            [0, 0, 0, 7, 0, 4], 
            [0, 0, 0, 0, 0, 0]] 

    adj_list = [[[1, 16], [2, 13]],
                [[2, 10], [3, 12]],
                [[1, 4], [4, 14]],
                [[2, 9], [5, 20]],
                [[3, 7], [5, 4]],
                []]

    c = conv_adj_list(adj_list)
    print(c)

    g = Graph(c) 
      
    source = 0; sink = 5
      
    g.minCut(source, sink) 