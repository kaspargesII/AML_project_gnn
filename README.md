#### Resources 

Paper about GNNs for EEG data: https://arxiv.org/abs/2310.02152
syllabus: [M-PML], chapter 23.-23.2, 23.4
Blog posts:
https://distill.pub/2021/gnn-intro/
https://distill.pub/2021/understanding-gnns/

SOTA challenges:
https://ieeexplore.ieee.org/document/9339909
General GNN paper:
https://arxiv.org/abs/2104.13478



Notes for blog posts:
https://distill.pub/2021/gnn-intro/

A neural network designed to leverage the structure of graphs.

Vertex nodes
Edges
Master node (global information about graph, number of node, longest path)

Graphs are useful for representing structures that might have variable sizes, where things are not fully connected or where connectivity might vary for each node. 

Graph level task: Predict something about the graph
Node level task: Predict an attribute about a node, it's identity or role in the graph. Analagous to a segmentation task. 
Edge level task: Predict the existence of an edge or property of an edge. What is the role of the edge in a graph or should edge be there? Input fully connected graph and output relevant edges and/or their roles

How do we represent graphs for DL tasks:
We can choose to include four types of information: Vertex information, edge information, global information and connectivity. 
It is not easy to encode connectivity because several different permutations of adjacency matrices can encode for the same connectivity. Also adjacency matrices will often be sparse.

A better way is to use adjacency list. 
Whilst attributes of nodes/edges/global can be represented as scalars, they are usually encoded as vectors. They are usually represented for the whole graph: [𝑛𝑛𝑜𝑑𝑒𝑠,𝑛𝑜𝑑𝑒𝑑𝑖𝑚]

GNN defintion:
A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances)
GNNs take a graph as input and output a graph as well. They transform the embeddings on the node, edge and global level without altering the connectivity of the graph. 

# Simple GNN
Simple GNN: Has an MLP or another model on each component of the graph; a GNN Layer. For each node vector we apply the model and get back a learned node vector. We do the same for each edge and also for the global context vector. 
The subsequent graph is now a new layer of the model. We can stack layers on top of each other. 
Since the connectivity is unchanged the dimension of input and output matches. 

GNN Pooling:
If we want to predict something about the nodes, but we do not have node information we can use pooling. If we have node information but we do not have edge information we can use pooling. 
procedure:
- For each item we want to pool (what is relevant for the node or edge) gather each of their embeddings and concat into a matrix
- Aggregate embeddings, usually via sum operation
Pooling for global prediction: Gather all avaialable node or edge information together and aggregate.

We want to produce a graph and then pass aggregated graph information into a classification layer that can output a prediction. 

# Mode advanced GNNs










