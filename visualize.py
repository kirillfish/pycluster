# visualize.py
# -*- coding: utf-8 -*-

import sys
import time
import json, pickle
import networkx as nx
import argparse
from math import log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import sign

VersionTime = str(int(time.time())/10)[3:] + "_one"

def loadGraph(graphPath):
    print "\nloading pickled graph..."
    Graph = pickle.load(open(graphPath, "r"))
    #print type(Graph)
    #print Graph.nodes()
    print "done"
    print "nodes: ", Graph.order()
    print "edges: ", Graph.size()
    return Graph


def loadDomains(filepath):
    print "\nloading domains: ", filepath
    domains_total = {}
    with open(filepath, "r") as D:
        for line in D:
            dom = line.strip().split()
            try:
                domains_total[dom[0]] = int(dom[2])
            except ValueError:
                domains_total[dom[0]] = 1
    print "\nnumber of domains extracted: ", len(domains_total)

    return domains_total


def loadClusters(filepath, alg='clara'):
    # getting data of clustering algorithm
    print "getting clustering results..."
    if alg == 'clara':
        claraOut = json.load(open(filepath, "r"))
        best_cost, best_choice, best_medoids, mod, harm_centrality, isolates = claraOut[0:6]
        print "clara results loaded"
        print "medoids: ", best_choice
        return best_cost, best_choice, best_medoids, mod, harm_centrality, isolates
    elif alg == 'cores':
        # FIX IT !!!
        return 1
    else:
        print "Error: unknown algorithm results: ", alg
        return 1


def nodeColor(best_choice, best_clust, centralityIntensity=True,
              harm_centrality=None, G=None, labelIntensity=False):
    # compute the ordered list of resulting node colors
    # with or without varying intensities
    print "\ncomputing node colors..."
    if centralityIntensity:
        if harm_centrality == None:
            print "Error: no intra-cluster centrality specified"
            return 1

        # construct a dict of cluster colormaps
        colList = ['binary', 'Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'BuPu', 'PuBuGn', 'pink_r', 'bone_r', 'afmhot_r', 'YlOrBr']
        k = len(best_choice)
        fittedColList = (int(float(k)/len(colList)) + 1 ) * colList
        fittedColList = fittedColList[0:k]
        colDict = dict(zip(best_choice, fittedColList))

        # compute color intensity
        maxCentG = {}
        for j in best_choice:
            values = dict(harm_centrality[j]).values()
            try: maxCentG[j] = (max(values))
            except: maxCentG[j] = 0.0

        node_intensity_dict={}
        for j in best_choice:
            node_intensity_dict[j]=[]
            for i in best_clust[j]:
                try:
                    node_intensity_dict[j].append(float(dict(harm_centrality[j])[i])/float(maxCentG[j]))
                except:
                    node_intensity_dict[j].append(0.1)

        print "done"

        if labelIntensity:  # unused!!!!
            # label intensity is currently unused due to unability of networx
            # to take a list for label color
            label_intensity = {}
            for i in G.nodes():
                for j in best_choice:
                    if i in best_clust[j]:
                        try: node_int = float(dict(harm_centrality[j])[i])/float(maxCentG[j])
                        except: node_int = 0
                        exec 'opposite=list(plt.cm.'+colDict[j]+'('+str(node_int)+'))'
                        label_intensity[i] = (1.0-opposite[0], 1.0-opposite[1], 1.0-opposite[2], 1.0)
            return node_intensity_dict, label_intensity

        return node_intensity_dict, colDict

    else:
        if G == None:
            print "Error: graph G is not specified"
            return 1
        colRange = range(len(best_choice))
        node_color = []
        for i in G.nodes():
            matched = 0
            for j in best_choice:
                if i in best_clust[j]:     # O(n**2) search ...
                    node_color.append( float( colRange[best_choice.index(j)] ) )
                    matched = 1
            if matched == 0:
                node_color.append(float(len(best_choice)+1))
        print "done"
        return node_color

def edgeProc(G, best_clust, colDict=None, colList=None, reweight=False,
             centralityIntensity=True, power=0.8):
    print "\nprocess edges (colors, other attributes)..."
    #if reweight:
    edgeweightsG = [i[2]['weight'] for i in G.edges(data=True)]
    avG = sum(edgeweightsG)*1./len(edgeweightsG)
    #print [G.edges(data=True)[i][2]['weight'] for i in [0,1,39,487]]
    print "average edge weight in the graph: ", avG

    # determine edge colors
    counter=0
    edgeColor = []
    notPassed = []

    if centralityIntensity:
        for i in G.edges(data=True):
            # minimum threshold for edge weight for clustering (additional edge filtering)
            #if reweight: thr = avG * thr
            #else: thr = 0
            if i[2]['weight'] >= 0:
                found = 0
                for j in best_clust:
                    if i[0] in best_clust[j]:
                        if i[1] in best_clust[j]:
                            #if reweight: i[2]['pos_weight']= (i[2]['weight']**(0.7))
                            # the color of an intra-cluster edge
                            exec 'edgeColor.append(plt.cm.'+colDict[j]+'(1.0))'
                        else:
                            #if reweight: i[2]['pos_weight']= i[2]['weight']**(0.7)
                            edgeColor.append((0.0,0.0,0.0,1.))
                        found = 1
                        break
                if found == 0:
                    print "node wasn't found in clusters: ", i[0]
                    edgeColor.append((0.0,0.0,0.0,1.))
            else:
                #if reweight: i[2]['pos_weight'] = 0.
                edgeColor.append((0.0,0.0,0.0,1.))
            counter += 1
            if counter%1000 == 0: print "edges processed: ", counter
        print "\nedges that didn't pass the threshold: ", len(notPassed)

    else:
        for i in G.edges(data=True):
            counter += 1
            if counter % 5000 == 0: print "edges processed: ", counter
            found = 0
            for j in best_clust:
                if i[0] in best_clust[j]:
                    if i[1] in best_clust[j]:
                        exec 'edgeColor.append(plt.cm.Accent(' + str(colList[G.nodes().index(i[0])] / float(len(best_clust)+1) ) + '))'
                    else:
                        edgeColor.append((0.,0.,0.,1.))
                    found = 1
                    break
            if found == 0:
                print "node isn't in clusters: ", i[0], "edge: ", i
                edgeColor.append((0.,0.,0.,1.))

    print len(edgeColor), len(G.edges())

    # computing edge widths as weights
    diffsign = sign(i[2]['weight']-avG)
    edgeWidth = [max(0.3, 1 + 10*diffsign*abs( (i[2]['weight'] - avG)/avG )**power)
                 for i in G.edges(data=True)]
    averageEdgeWidth = sum(edgeWidth)*1./len(edgeWidth)
    print "average edgewidth: " , averageEdgeWidth
    edgeWidth = [i*1./averageEdgeWidth/1.5 for i in edgeWidth]
    return edgeColor, edgeWidth

def nodeSize(G, best_choice, best_clust, isolates,
             domains_total, centralityIntensity=True,
             multiplier=80, power=0.7):
    # construct a list (or dict of lists) of node sizes
    print "\ncompute node sizes..."
    print domains_total.items()[0]
    avVisits = sum(domains_total.values())/len(domains_total)
    minval = min(domains_total.values())

    if centralityIntensity:
        node_size_dict={}
        for j in best_clust:
            node_size_dict[j] = []
            for i in best_clust[j]:
                try: node_size_dict[j].append(domains_total[i])
                except:
                    print "the node in the graph is not presented in domains_total: ", i
                    node_size_dict[j].append(minval)
        node_size_dict_iso = []
        for i in isolates:
            try: node_size_dict_iso.append(domains_total[i])
            except:
                print "the isolated node in the graph is not presented in domains_total: ", i
                node_size_dict_iso.append(minval)

        # transforming domain size into the node size itself
        log_node_size_dict = {}
        ordered_clustered_nodes = {}
        for j in best_clust:
            log_node_size_dict[j] = []
            ordered_clustered_nodes[j] = []
            for i in xrange(len(node_size_dict[j])):
                try:
                    node_size_dict[j][i] = (float(node_size_dict[j][i]))
                except:
                    node_size_dict[j][i] = 1.
                a = node_size_dict[j][i] * 20. / avVisits
                log_node_size_dict[j].append(log(float(a)+2.0)*(a**power)*multiplier)
                ordered_clustered_nodes[j].append(i)

        log_node_size_dict_iso = []
        ordered_clustered_nodes_iso = []
        for i in xrange(len(node_size_dict_iso)):
            try:
                node_size_dict_iso[i] = (float(node_size_dict_iso[i]))
            except:
                node_size_dict_iso[i] = 1.
            a = node_size_dict_iso[i]*20. / avVisits
            log_node_size_dict_iso.append(log(float(a)+2.0)*(a**power)*multiplier)
            ordered_clustered_nodes_iso.append(i)

        forreturn = [log_node_size_dict, log_node_size_dict_iso]

    else:
        if domains_total == None:
            print "Error: domain visits weren't specified"
            return 1

        print "len(domains_total): ", len(domains_total)
        print "G.order(): ", G.order()

        node_size = []
        for i in G.nodes():
            try: node_size.append(domains_total[i])
            except:
                print "the node in the graph is not present in domains_total: ", i
                node_size.append(minval)

        log_node_size = []
        for i in xrange(len(node_size)):
            try:
                node_size[i] = float(node_size[i])
            except:
                node_size[i] = 1.0
            a = node_size[i]*20. / avVisits
            log_node_size.append(log(float(a)+2.0)*(a**power)*multiplier)
            #print i, a, log(float(a)+2.0)**(a**power)*multiplier
        avSize = sum(log_node_size)/len(log_node_size)
        print "average node size: ", avSize
        # normalize node sizes!
        log_node_size = [nodeSize*1800./avSize for nodeSize in log_node_size]

        forreturn = log_node_size

    print "done"
    return forreturn


def drawLabels(G, best_clust):
    node_labels = {}
    for i in G.nodes():
        lab = "iso"
        for j in best_clust:
            if i in best_clust[j]:
                lab = str(best_clust.keys().index(j))
        node_labels[i] = lab + "\n" + i

    return node_labels


def drawGraph(G, best_choice, best_clust, isolates, node_labels, edgeColor, colDict,
         edgeWidth, node_intensity_dict=None, node_color=None,
         log_node_size_dict=None, log_node_size_dict_iso=None,
         log_node_size=None, centralityIntensity=True, scale=None,
         figureFormat='png'):
    # graph drawing
    print "\nDrawing initialization..."
    #edgesum = sum([G.edges(data=True)[i][2]['weight'] for i in range(G.size())])
    if scale==None:
        #scale = 1. / (G.size())**0.5
        scale=0.03 * nx.density(G) / (0.0246 / 2)        # 0.0246 for non-sparsified 1285-node graph
        # 4 for 10121 node-graph, power=0.5, figsize=(150,150)
        # 8 for 10121 node-graph, power=0.25, figsize=(170,170)
    print scale, nx.density(G)
    pos=nx.spring_layout(G, k=scale, weight='pos_weight')
    plt.figure(figsize=(120,120)) #(120,120)

    print "\ndrawing edges..."
    nx.draw_networkx_edges(G, pos,
                        edge_color=edgeColor,
                        alpha=0.4,
                        width=edgeWidth)

    print "\ndrawing nodes..."
    if centralityIntensity:
        for med in best_choice:
            nx.draw_networkx_nodes(G, pos,
                                nodelist=best_clust[med],
                                node_size=log_node_size_dict[med], #[G.degree(i)*4+50 for i in G.nodes()],
                                alpha=1.0,
                                node_color=node_intensity_dict[med],
                                cmap=colDict[med])

        nx.draw_networkx_nodes(G,pos,
                            nodelist=isolates,
                            node_size=log_node_size_dict_iso,
                            alpha=1.0,
                            node_color='w')

    else:
        nx.draw_networkx_nodes(G, pos,
                               node_size=log_node_size,
                               alpha=1.0,
                               node_color=node_color,
                               cmap=plt.cm.Accent)

    print "\ndrawing labels..."
    nx.draw_networkx_labels(G, pos,
                            labels=node_labels,
                            #font_color=label_intensity,
                            font_size=8) # 10 for 1285, 8 -- for 10000

    print "\nsaving the graph..."
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.axis('off')
    plotname = 'graph_' + str(len(G.nodes())) + '_' + VersionTime + '.' + figureFormat
    plt.savefig(plotname)
    print "\nthe graph saved: ", plotname
    #plt.show()

    return 0

'''
def loadGraph(graphPath):
def loadDomains(filepath):
def loadClusters(filepath, alg='clara'):
def nodeColor(k, best_choice, best_clust, centralityIntensity=True,
              harm_centrality=None, G=None, labelIntensity=False):
def edgeproc(G, reweight=False, multiplier=1.0):
def nodesize(G, best_choice, best_clust, isolates,
             centralityIntensity=True, domains_total=None,
             multiplier=80, power=0.7):
def drawgraph(G, best_choice, best_clust, node_labels, edgeColor, colListDict,
         node_intensity_dict=None, node_color=None,
         log_node_size_dict=None, log_node_size_dict_iso=None,
         log_node_size=None, centralityIntensity=True):
'''

def read_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-FG', '--graphPath', type=str, required=True)
    parser.add_argument('-FD', '--domPath', type=str, required=True)
    parser.add_argument('-FC', '--clustPath', type=str, required=True)

    #parser.add_argument('-med', '--best_choice', type=list)
    #parser.add_argument('-clust', '--best_clust', type=dict)
    #parser.add_argument('-iso', '--isolates', type=dict)
    #parser.add_argument('-G', '--G', type=nx.Graph())
    parser.add_argument('-centr', '--harm_centrality', type=dict)

    parser.add_argument('-mul', '--multiplier', default=150)
    parser.add_argument('-pow', '--power', default=0.5)

    parser.add_argument('-li', '--labelIntensity', action='store_true', default=False) # remove
    parser.add_argument('-rw', '--reweight', action='store_true', default=False) # remove
    parser.add_argument('-ci', '--centrIntensity', action='store_true', default=False)

    parser.add_argument('-alg', '--alg', choices=['clara','cores'], default='clara')

    parser.add_argument('-ff', '--figureFormat', choices=['png','svg','pdf'], default='png')
    args = parser.parse_args()
    return args


def main():
    args = read_arguments()
    G = loadGraph(args.graphPath)
    #domains_total = None
    #if not args.centrIntensity:
    domains_total = loadDomains(args.domPath)

    best_cost, best_choice, best_clust, med, harm_centr, isolates = loadClusters(args.clustPath, alg=args.alg)

    nodecol = nodeColor(best_choice, best_clust,
                        centralityIntensity=args.centrIntensity,
                        harm_centrality=harm_centr,
                        G=G)

    if args.centrIntensity:
        nodecol, colDict = nodecol[:2]
    else:
        colDict = None

    #print type(nodecol)
    #print nodecol

    edgecol, edgewid = edgeProc(G=G, colDict=colDict, best_clust=best_clust,
                                centralityIntensity=args.centrIntensity,
                                colList=nodecol)

    nodesize = nodeSize(G=G, best_choice=best_choice,
                        best_clust=best_clust, isolates=isolates,
                        centralityIntensity=args.centrIntensity,
                        domains_total=domains_total,
                        multiplier=args.multiplier,
                        power=args.power)

    nodelab = drawLabels(G=G, best_clust=best_clust)

    # some arguments are mutually exclusive
    drawGraph(G=G, best_choice=best_choice, best_clust=best_clust, isolates=isolates,
              node_labels=nodelab, edgeColor=edgecol, colDict=colDict,
              edgeWidth=edgewid, node_intensity_dict=nodecol, node_color=nodecol,
              log_node_size_dict=nodesize[0], log_node_size_dict_iso=nodesize[1],
              log_node_size=nodesize,
              centralityIntensity=args.centrIntensity,
              figureFormat=args.figureFormat)

if __name__ == '__main__':
    main()
