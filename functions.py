# from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from skimage.future.graph import RAG
import json
from skimage.measure import regionprops_table
import pandas as pd
import networkx
from sklearn.metrics import confusion_matrix

nx=networkx

class InvalidKColoringError(Exception):
    """Raised when the k-coloring does not exist"""
    pass

class NoSolutionException(Exception):
    """Raised when no solution is found"""
    pass

def greedy_k_color(graph: nx.Graph, k: int, fair: bool = False) -> dict:
    """
    Greedy algorithm to find a k-coloring for a given graph.
    If fair, Chooses available colors by least frequency of occurrence.
    If not fair, the NetworkX greedy color with the strategy largest_first is used as a basis. This is extended \
        by selecting the most used color and divide it 50:50 with a new color. \
        That is repeated until all colors are used \
    Raises NoSolutionException if the algorithm can not find coloring for the given k.
    :param graph: NetworkX graph
    :param k: Number of colors
    :param fair: Tries to assign colors equitably. Caution: There might be solutions with fair = False but no solution \
        with fair = True
    :return: Color assignment
    """
    if k > graph.number_of_nodes():
        raise InvalidKColoringError(f"Graph has no {k}-coloring as it only has {graph.number_of_nodes()} vertices")
    coloring = {}
    available_colors = {c: 0 for c in range(k)}
    nodes = sorted(graph, key=graph.degree) # , reverse=True)
    if fair:
        for u in nodes:
            # Set to keep track of colors of neighbours
            neighbour_colors = {coloring[v] for v in graph[u] if v in coloring}
            for color in dict(sorted(available_colors.items(), key=lambda item: item[1])):
                if color not in neighbour_colors:
                    available_colors[color] = available_colors[color] + 1
                    break
            else:
                raise NoSolutionException("No more colors")
            # Assign the new color to the current node.
            coloring[u] = color
    else:
        coloring = nx.greedy_color(graph)
        if max(coloring.values()) + 1 > k:
            raise NoSolutionException(f"Minimal solution needs more colors than k={k} < {max(coloring.values()) + 1}")

        for c in range(max(coloring.values()) + 1, k):
            color_dist = Counter(coloring.values())
            most_used_color = max(color_dist, key=color_dist.get)
            if color_dist[most_used_color] <= 1:
                raise NoSolutionException("Not possible to replace colors")
            nodes_with_most_used_color = [k for k, v in coloring.items() if v == most_used_color]
            replace_color_nodes = default_rng().choice(nodes_with_most_used_color,
                                                       size=math.floor(color_dist[most_used_color] / 2),
                                                       replace=False)
            for node in replace_color_nodes:
                coloring[node] = c

    return coloring

#%%

from skimage.segmentation import find_boundaries

def get_coloring(lbl, per_subgraph=True):
    # FIXME: per_subgraph will need to be ignored (true always)
    
    gr = RAG(lbl,connectivity=2)
    gr.remove_node(0)
    
    A = gr.adj
    islands = []
    for nd in gr.nodes:
        if gr.degree(nd)==0:
            islands.append(nd)
            
    [gr.remove_node(nd) for nd in islands]

    lbl_islands = np.where(np.logical_or.reduce([lbl==x for x in islands]), lbl, 0) 

    lbl_cl = np.zeros(lbl.shape,np.uint8) # 8 channel 
    
    shp = lbl.shape
    msk_2c = np.zeros((shp[0],shp[1],2),'bool')
    msk_2c[...,0] = lbl_islands>0
    
    
    if not per_subgraph:
        coloring = networkx.coloring.equitable_color(gr,min(8,max(dict(gr.degree).values())+1))
        
        for nd,cl in coloring.items():
            lbl_cl = np.where(lbl==nd,cl+2,lbl_cl)
    
        
    else:
        coloring = {}
        subgraphs=[gr.subgraph(comp).copy() for comp in networkx.connected_components(gr)]
        
        for subgr in subgraphs:
    
            sel = list(subgr)
            for x in sel:
                msk_2c[...,1] = msk_2c[...,1] | find_boundaries(lbl==x,mode='inner',connectivity=2)
                
            # grp = np.where(np.logical_or.reduce([lbl==x for x in sel]), lbl, 0) 
            if max(dict(subgr.degree).values())<3:
                coloring_sub = networkx.coloring.greedy_color(subgr,strategy='largest_first')
            else:
                nc = min(8,max(dict(subgr.degree).values())+1)
                # coloring_sub = networkx.coloring.equitable_color(subgr,nc)
                coloring_sub = greedy_k_color(subgr,nc,fair=True)
            
            for nd,cl in coloring_sub.items():
                lbl_cl = np.where(lbl==nd,cl+2,lbl_cl)
                coloring[nd]=cl+2
                
            
            
    lbl_cl[lbl_islands>0]=1
    for nd in islands:
        coloring[nd]=1
        
    return lbl_cl, msk_2c, islands, gr, coloring




