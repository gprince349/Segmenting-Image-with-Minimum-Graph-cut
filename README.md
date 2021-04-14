# Segmenting-Image-with-Minimum-Graph-cut
Implementing a segmentation algorithm for separating object and  background in a image
we are using minimum graph cut algorithm (Max-flow algorithm) for this task
converting Images into undirected weighted graph where weights are decided based on similarity 
between neighbourhood pixels and then applying the min-cut algorithm to find two partitions
of graph.

