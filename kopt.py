# kopt.py
# Clara improvements

from util import importData, direct_distance, similarity_distance, AutoVivification
from pam import kmedoids
from clara import targetFunction, modularity, intra_cluster_centrality, clara
#from collections import Counter
import sys
import time
from itertools import chain
from datetime import datetime
import json, pickle
from math import log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
#import scipy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree # what is hcluster?
from scipy.spatial.distance import is_valid_dm, is_valid_y, pdist
import random

# Global variables
bagSize = 460
costType = "modularity"
degreeSelection = True
Kmax = 24
trials = 20
Version = str(int(time.time())/10)[3:] + "_one"
trials_decay = False

def ModularityProfile(data, COST=4, Kmin=1, Kmax=Kmax, Klist=None, edgeDict={},
                      simDict={}, bagSize=bagSize, trials=trials,
                      trials_decay=trials_decay):
    '''
    Function for determining the optimal number of k in clara algorithm.
    Currently by means of modularity maximization over various numbers of k.
    Input: all Clara.py input + bag size per run, list of k's to go over (not
    obligatory range(1:Kmax)), number of trials per k instance.
    trials_decay indicates whether trials number subside with k. If true, it
    decreases evenly up to 3 for k==Kmax.
    Output: modularity distributions + errorplot saved on disk.
    '''
    # runs clara prespecified number of times on a range of k's
    if Klist == None:
        Klist = range(Kmin,Kmax+1)
    if costType != "modularity":
        print "cost function is not modularity - cannot start"
        return(1)
    modMax = []
    mod_lists = []
    modStdev = []
    modMean = []
    for k in Klist:
        # setting up the number of attempts to infer about modularity distribution
        # given the number of clusters k
        if trials_decay: ktrials = int(trials*(1-k / float(Kmax))) + 3
        else: ktrials = trials
        min_cost, best_choice, best_res, cost_list, isolates = clara(data, k, COST,
                                                           simDictClara=simDict,
                                                           affinities=edgeDict,
                                                           bagSize=bagSize,
                                                           claraLoopNum=ktrials,
                                                           noIsolates=True,
                                                           saveAllResults=False,
                                                           acceleration=2)

        modMax.append(-min_cost)  # negative sign due to minimization in clara
        mod_lists.append(cost_list)
        num_cost_list = np.array(cost_list)
        modStdev.append( (1.+1./ktrials)**0.5 * np.std(num_cost_list) )
        print "multiple: ", modStdev[-1]
        print "np.std(num_cost_list): ", np.std(num_cost_list)
        print "isolates: ", isolates
        modMean.append(-np.mean(num_cost_list)) # again negative

    # picking the number of clusters maximizing modularity
    Kopt = 0
    for k in Klist[:-1]:
        if modMean[Klist.index(k)] >= modMean[Klist.index(k)+1] - modStdev[Klist.index(k)+1]:
            Kopt = k
            break
    if not Kopt:
        Kopt = Klist[len(Klist)-1]

    # plotting and saving
    plt.errorbar(x=Klist, y=modMean, yerr=modStdev, fmt = '-o')
    plt.plot(Klist, modMax, '-o')
    plt.plot(Klist.index(Kopt)+1, modMean[Klist.index(Kopt)], 'ro')
    plt.savefig("kink.png")

    return Kopt, mod_lists, modStdev, modMean, modMax


def SGJRIcores(data, COST=4, K=12, edgeDict={}, simDict={}, bagSize=bagSize,
               distDict={}, trials=100, threshold=None, segments=None,
               loadedCores=False, filepathCores=None, dendroFormat='png',
               acceleration=0):
    '''
    Average Clara results from lots of trials, and for each pair
    of nodes compute proportion of times they appear in the same cluster.
    Then discard the edges for which the proportion is lower than prespecified
    threshold alpha. After all, hierarchical structure of connected
    components is obtained, along with stable cores, whose contents
    remained the same whatever the partition is.
    Input
    '''
    print "SGJRI COST: ", COST

    # if the number of resulting segments is not specified, it
    # will be equal to K in clara, which provides most sensible results:
    if segments == None:
        segments = K
        print "as you didn't specify the number of segments, it will equal K: ", segments

    if loadedCores:
        if filepathCores == None:
            print "Error: no filepath for loading cores specified"
            return 1
        commonCluster = pickle.load(open(filepathCores, 'r'))[1]
        clustered = commonCluster.shape[0]
        clustData = []
        for i in xrange(clustered):
            clustData.append(commonCluster[i][0]['A'])
    else:
        isolates, allResults = clara(
                                    data, COST=COST, k=K, simDictClara=simDict,
                                    affinities=edgeDict,
                                    bagSize=bagSize,
                                    claraLoopNum=trials,
                                    noIsolates=True,
                                    saveAllResults=True,
                                    acceleration=acceleration)[4:6]

        # now the data list is already pruned, so there are no need to prune it
        # again. the code is excessive
        clustered = len(data) # - len(isolates)
        clustData = [i for i in data if i not in isolates]
        commonCluster = np.zeros(shape=(clustered,clustered),
                                dtype=[('count','f8'), ('A','a100'), ('B','a100')])

        print "isolates: ", len(isolates)
        #print "data: ", len(data)
        print "clustered: ", clustered

        # determining node names for dendrogram labels
        for i in xrange(clustered):
            for j in xrange(clustered): #xrange(i+1, clustered):
                commonCluster[i][j]['A'] = clustData[i]
                commonCluster[i][j]['B'] = clustData[j]

        print "co-appearance counting and writing it into numpy array..."
        # co-appearance counting and writing it into numpy array
        count = 0
        for res in xrange(len(allResults)):
            for clus in allResults[res]:
                clusContents = list(allResults[res][clus])
                for nodeA in allResults[res][clus]:
                    # reduce the loop more than twice:
                    clusContents.remove(nodeA)
                    Aid = clustData.index(nodeA)
                    for nodeB in clusContents:
                        Bid = clustData.index(nodeB)
                        # making matrix symmetric with zero diagonal
                        commonCluster[Aid][Bid]['count'] += 1. / trials
                        commonCluster[Bid][Aid]['count'] += 1. / trials
            count += 1
            if count % 10 == 0: print "clusterings processed: %d out of %d" % (count, trials)
    """
    # transforming the matrix of co-occurences into matrix of distances
    for i in xrange(clustered):
        for j in xrange(clustered): #xrange(i+1, clustered):
            if commonCluster[i][j]['count'] == 0.: commonCluster[i][j]['count'] = 20.
            else: commonCluster[i][j]['count'] = 1. / commonCluster[i][j]['count']
    """

    # some debug output
    print "shape: ", commonCluster.shape
    #print is_valid_dm(commonCluster)   # -- invalid
    #print is_valid_y(commonCluster)

    # now obtaining the tree structure
    t = time.time()
    print "deriving three structure..."
    treeStruct = linkage(commonCluster, method='ward')
    print "tree structure obtained in %f seconds" % (time.time() - t)

    # then visualise the tree (prune according to common sense)
    t = time.time()
    print "constructing a dendrogram..."
    if threshold == None:
        threshold = treeStruct[len(treeStruct)-segments, 2] + 0.01
    #print len(treeStruct[len(treeStruct)-segments:])

    magnify = (float(clustered)/400.)**0.5
    if dendroFormat == 'png': leaf_font_size = 13
    if (dendroFormat == 'svg') or (dendroFormat == 'pdf'): leaf_font_size = 2 * (3**0.5) / magnify

    dendro = dendrogram(treeStruct, labels=clustData, leaf_font_size=leaf_font_size,
                            color_threshold=threshold)
    print "dendrogram constructed in %f seconds" % (time.time() - t)

    t=time.time()
    print "saving the dendrogram..."
    figure = plt.gcf()
    if dendroFormat == 'png':
        figure.set_size_inches(120*magnify,60*magnify)
    elif dendroFormat == 'svg' or dendroFormat == 'pdf':
        figure.set_size_inches(20*magnify,10*magnify)

    plt.savefig("dendrogram_" + Version + "." + dendroFormat)

    print "dendrogram saved in %f seconds" % (time.time() - t)

    # then construct an array with clustered domains
    threshold = treeStruct[len(treeStruct)-segments, 2] - 0.000001
    clustIDs = fcluster(treeStruct, threshold, 'distance')

    segmentDict = {}
    for i in xrange(segments):
        segmentDict[i] = [clustData[j] for j in range(len(clustData)) if clustIDs[j] == i+1]

    # trying to get by without distDict
    if distDict != {}:
        # searching the most central nodes within the clusters
        harmCentr = intra_cluster_centrality(clustData, distDict=distDict,
                                            medoids=segmentDict)

        # combine results batch into mongo-acceptable format
        date = datetime.now().strftime("%Y-%m-%d")
        clusters = []
        for i in harmCentr:
            clDict = {'number' : i}
            clDict['domains'] = harmCentr[i]
            clusters.append(clDict)

        mongo = { 'date' : date,
                'clusters' : clusters }

    else:
        print "Warning: no harmCentr and mongo!"
        harmCentr = {}
        mongo = {}

    return clustData, commonCluster, treeStruct, segmentDict, harmCentr, mongo


def map_external_to_clusters(segmentDict=None, segmentPath=None, importancePath=None, importance_idx=0,
                             source = 'NOT openstat'):
    if segmentDict == None:
        segmentDict = pickle.load(open(segmentPath, 'r'))[3]   # beware of index changing !!!!

    loaded = pickle.load(open(importancePath, 'r'))
    importances = (loaded['importances'][importance_idx])
    names = (loaded['names'][importance_idx])
    clusteredFeatures = {}

    for seg in segmentDict:
        clusteredFeatures[seg] = ([], [])
    lastSeg = len(clusteredFeatures)
    clusteredFeatures[lastSeg] = ([], [])
    notFound = {}

    for name in names:
        found = 0
        for seg in segmentDict:
            if name in segmentDict[seg]:
                clusteredFeatures[seg][0].append(name)
                clusteredFeatures[seg][1].append(importances[names.index(name)])
                found = 1
        if found == 0:
            notFound[name] = importances[names.index(name)]
            clusteredFeatures[lastSeg][0].append(name)
            clusteredFeatures[lastSeg][1].append(importances[names.index(name)])

    print "domains not found: ", len(notFound)
    maxValue = np.array(importances).max()

    fig, ax = plt.subplots(len(clusteredFeatures)/5+1, 5, figsize = (50,50))
    #fig = plt.figure(figsize=(50,50))
    fig.subplots_adjust(left=0.2, hspace=0.3, wspace=0.5)


    for seg in clusteredFeatures:
        if len(clusteredFeatures[seg][0]) == 0:
            print "no domains for cluster ", seg
            continue

        if seg == lastSeg:
            color = 'g'
        else:
            color = 'b'

        feature_names = np.array(clusteredFeatures[seg][0])
        feature_importance = np.array(clusteredFeatures[seg][1])
        feature_importance = 100.0 * (feature_importance / maxValue)

        if len(feature_names) < 16:
            repl = 16 - len(feature_names)        # kostyl !!!!
            feature_names = np.concatenate((feature_names, np.repeat([""], repl)))
            feature_importance = np.concatenate((feature_importance, np.repeat([-0.00001], repl)))

            sorted_idx = np.argsort(feature_importance)  # [-50:]
            pos = np.arange(sorted_idx.shape[0]) + .5
        else:
            sorted_idx = np.argsort(feature_importance)[-16:]
            pos = np.arange(sorted_idx.shape[0]) + .5

        print "plotting..."
        #fig = plt.figure(figsize=(10,16))
        #plt.subplots_adjust(left=0.2)
        #ax122 = plt.subplot(111)
        #ax = plt.subplot2grid((len(clusteredFeatures)/5, len(clusteredFeatures)%5), (seg/5, seg%5))
        print seg, seg/5, seg%5
        plt.sca(ax[ seg/5, seg%5 ])

        for tick in ax[seg/5, seg%5].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        plt.barh(pos, feature_importance[sorted_idx], align='center', color=color)
        plt.xlim(0,100)
        plt.yticks(pos, feature_names[sorted_idx])    # add variable names!!!
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
    for seg in xrange(len(clusteredFeatures), (len(clusteredFeatures)/5+1)*5):
        print seg, seg/5, seg%5
        fig.delaxes(ax[seg/5, seg%5])

    print "saving..."
    #plotname = str(seg) + "_" + source + "_" + "feature_importance_" + Version
    plotname = "segmentWise_" + source + "_" + "feature_importance_" + Version
    plt.savefig(plotname)
    print "\n plots saved: ", plotname

    print notFound

    return notFound


def treetod3(clustData, treeStruct, hierList=[3,8,20,30],
             domPath=None, savePath='zoomable_circles_'+Version+'.json',
             rename=False, renameLevel=3, mapPath=None, clickSizes=False, clickPath=None):
    # method for transforming hierarchical clustering into several-level packing,
    # to feed it to zoomable circle packing algorithm in d3.js

    thresholds = [treeStruct[len(treeStruct)-i, 2] - 0.000000 for i in hierList]
    clustIDs = [fcluster(treeStruct, i, 'distance') for i in thresholds]
    segmentDicts = []

    for level in xrange(len(hierList)):
        segmentDicts.append({})
        for i in xrange(hierList[level]):
            segmentDicts[level][i] = [clustData[k] for k in xrange(len(clustData)) if clustIDs[level][k] == i+1]

    for level in xrange(len(segmentDicts)):
        for cln in segmentDicts[level].keys():
            segmentDicts[level][str(hierList[level]) + "_" + str(cln)] = segmentDicts[level].pop(cln)

    def add_node(node, parent=None, root=False):
        if not root and parent == None:
            print "Error: non-root node without a parent"
            return 1
        if root:
            rootNode = dict(name=node, children=[])
            return rootNode
        else:
            newNode = dict(name=node, children=[])
            parent['children'].append(newNode)
            return parent, newNode

    # create root level with single node
    rootNode = add_node('domain_levels', root=True)

    # create the first level referring to the root
    levelNodes = {}
    levelNodes[0] = []
    for cln in segmentDicts[0]:
        #print cln
        rootNode, newNode = add_node(cln, rootNode)
        levelNodes[0].append(newNode)

    # create all the rest. [0] element in each childen serves as a key to it's
    # parent
    for level in xrange(1,len(segmentDicts)):
        levelNodes[level] = []
        print "\n\n", segmentDicts[level].keys()
        for clnChi in segmentDicts[level]:
            found = 0
            print list(segmentDicts[level][clnChi])[0]
            for clnPar in segmentDicts[level-1]:
                if list(segmentDicts[level][clnChi])[0] in segmentDicts[level-1][clnPar]:
                    parent, newNode = add_node(clnChi, [i for i in levelNodes[level-1] if i['name']==clnPar][0])
                    levelNodes[level].append(newNode)
                    found = 1
            if found == 0: print "Not found!!!", clnChi
        print "levelNodes[level]: ", level, levelNodes[level]

    # loading domains_total from DOMAINS and determining node sizes, similar to visualize.py
    domains_total = {}
    with open(domPath, 'r') as D:
        for line in D:
            dictline = line.strip().split()
            domains_total[dictline[0]] = float(dictline[2])   # beware of index changes!!!

    # calculate the circle size
    avVisits = sum(domains_total.values()) / len(domains_total)
    for dom in domains_total:
        a = domains_total[dom] * 20. / avVisits
        domains_total[dom] = log(a + 2.) * (a ** 0.7) * 80

    # fill the last level with leaf nodes (domains)
    if clickSizes == False:
        level = len(segmentDicts)-1
        for i in levelNodes[level]:
            i['children'] = segmentDicts[level][i['name']]
            for leafnum in xrange(len(i['children'])):
                i['children'][leafnum] = { "name": i['children'][leafnum],
                                        "size": domains_total[i['children'][leafnum]]}
                print "!!!!", i['children'][leafnum]
    else:
        domainSizes = {}
        notFound = {}
        found = 0
        totalClicks = {}
        foundClicks = 0
        with open(clickPath, 'r') as M:
            M.readline()
            M.readline()
            for line in M:
                dictline = line.strip().split()
                a = float(dictline[1]) * 20 / avVisits
                domainSizes[dictline[0]] = log(a+2.) * (a ** 0.3) * 100
                totalClicks[dictline[0]] = float(dictline[1])

        level = len(segmentDicts)-1
        for i in levelNodes[level]:
            i['children'] = segmentDicts[level][i['name']]
            for leafnum in xrange(len(i['children'])):
                try:
                    i['children'][leafnum] = { "name": i['children'][leafnum],
                                            "size": domainSizes[i['children'][leafnum]]}
                    #print "i: ", i['children'][leafnum]
                    #print "????", i['children'][leafnum]  --- OK
                    found += 1
                    #print "domain sizes:   ", domainSizes[i['children'][leafnum]['name']]
                    foundClicks += totalClicks[i['children'][leafnum]['name']]
                except:
                    #print i['children'][leafnum]  # --- OK
                    i['children'][leafnum] = { "name": i['children'][leafnum],
                                            "size": (random.random()+1)/8.}
                    #print "!!!!", i['children'][leafnum]
                    #print "nonhashable? ", i['children'][leafnum]['name']
                    notFound[i['children'][leafnum]['name']] = domains_total[i['children'][leafnum]['name']]

        print "\nnot found: ", len(notFound)
        print "found: ", found

        print "\ntotalClicks: ", sum(totalClicks.values())
        print "\nfoundClicks: ", foundClicks

    if rename:
        rootNode = rename_clusters(rootNode, levelNodes, renameLevel, mapPath)

    json.dump(rootNode, open(savePath, 'w'))
    return levelNodes[0], rootNode


def rename_clusters(rootNode, levelNodes, levelNum, mapPath):
    # load a manually filled file and rename (low-level) clusters
    mapping = {}
    with open(mapPath, 'r') as M:
        for line in M:
            mapline = line.strip().split()
            mapping[mapline[0]] = mapline[1]
    for i in levelNodes[levelNum]:
        i['name'] = mapping[i['name']]

    # rename high-level clusters
    for level in xrange(levelNum-1, -1, -1):
        for i in levelNodes[level]:
            newname = ''
            for child in i['children']:
                newname = newname + child['name'] + "\n"
            i['name'] = newname

    return rootNode


def juxtapose(clustData, segmentsA, segmentsB, harmCentrA, harmCentrB,
              topmin=8):
    '''
    compare 2 different clusterings and try to match each other's
    cluster indices (to ensure that all clusters evolve smoothly)
    '''
    coincidence = {}
    matchingA = {}
    matchingB = {}
    mostCentralA = {}
    mostCentralB = {}
    topA = {}
    topB = {}

    # finding most central nodes for clustering B
    for numB in segmentsB:
        lengthB = len(segmentsB[numB])
        if lengthB < topmin:
            print "Error: too small cluster detected in the second partition"
            print segmentsB[numB]
            return 1
        print "B: clust %s length is %d" % (str(numB), lengthB)

        topB[numB] = max(int(lengthB * 0.4), int(topmin))
        mostCentralB[numB] = [i[0] for i in harmCentrB[numB][0:topB[numB]]]

        print "\ntop: ", numB, topB[numB]
        print "B: most central: ", mostCentralB[numB]

        coincidence[numB] = {}
        matchingB[numB] = -1

    # finding most central nodes for clustering A
    for numA in segmentsA:
        lengthA = len(segmentsA[numA])
        if lengthA < topmin:
            print "Error: too small cluster detected in the first partition"
            print segmentsA[numA]
            return 1
        print "\n\nA: clust %s length is %d" % (str(numA), lengthA)

        topA[numA] = max(int(lengthA * 0.4), int(topmin))
        mostCentralA[numA] = [i[0] for i in harmCentrA[numA][0:topA[numA]]]

        print "\ntop: ", numA, topA[numA]
        print "A: most central: ", mostCentralA[numA]

        coincidence[numA] = {}
        matchingA[numA] = -1

    # for each cluster in B find the most close in A (many-to-one implies
    # clusters that have been merged)
    for numB in segmentsB:
        for numA in segmentsA:
            coincidence[numB][numA] = 0
            for i in mostCentralB[numB]:
                if i in mostCentralA[numA]:
                    coincidence[numB][numA] += 1
            if coincidence[numB][numA] > topB[numB] / 2.:
                matchingB[numB] = numA
                print "\nmore than half matched! B: %s, A: %s" % (numB, numA)
                break

        print "coincidence calculated\n"

        if matchingB[numB] == -1:
            nonzero = dict([coincidence[numB].items()[i] for i in xrange(len(coincidence[numB])) if coincidence[numB].values()[i]>0])
            moreThanOne = dict([coincidence[numB].items()[i] for i in xrange(len(coincidence[numB])) if coincidence[numB].values()[i]>1])
            if len(nonzero) == 0:
                print "\nB: cluster %s cannot be matched with A" % numB
                continue
            if len(moreThanOne) == 0:
                print "\nB: no more than 1 central node in cluster %s can be matched with central nodes in A: BAD MATCHING" % numB
                continue

            #matchingB[numB] = sorted(nonzero,
            #                         key=nonzero.get,
            #                         reverse=True)[:min(3,len(nonzero))]
            matchingB[numB] = sorted(moreThanOne,
                                     key=moreThanOne.get,
                                     reverse=True)[:min(3,len(moreThanOne))]

    # for each cluster in A find the most close in B (many-to-one
    # relationship implies that the clusters have been merged)
    for numA in segmentsA:
        for numB in segmentsB:
            coincidence[numA][numB] = 0
            for i in mostCentralA[numA]:
                if i in mostCentralB[numB]:
                    coincidence[numA][numB] += 1
            if coincidence[numA][numB] > topA[numA] / 2.:
                matchingA[numA] = numB
                print "\nmore than half matched! A: %s, B: %s" % (numA, numB)
                break

        print "coincidence calculated\n"

        if matchingA[numA] == -1:
            #matching[numA] = dict(sorted(coincidence.iteritems(),
            #       key=operator.itemgetter(1), reverse=True)[:3]).keys()
            nonzero = dict([coincidence[numA].items()[i] for i in xrange(len(coincidence[numA])) if coincidence[numA].values()[i]>0])
            if len(nonzero) == 0:
                print "\nA: cluster %s cannot be matched with B" % numA
                continue
            if len(moreThanOne) == 0:
                print "\nA: no more than 1 central node in cluster %s can be matched with central nodes in B: BAD MATCHING" % numA
                continue

            #matchingA[numA] = sorted(nonzero,
            #                        key=nonzero.get,
            #                        reverse=True)[:min(3,len(nonzero))]
            matchingA[numA] = sorted(moreThanOne,
                                     key=moreThanOne.get,
                                     reverse=True)[:min(3,len(moreThanOne))]

    return matchingA, matchingB


def read_arguments():
    '''
    multiple subparsers in order to permit execution of a certain part
    of the script rather than the entire script
    '''
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    sublist = []

    # create a subparser for each function
    kopt_parser = subparsers.add_parser('kopt')
    kopt_parser.set_defaults(method='kopt')
    sublist.append(kopt_parser)

    cores_parser = subparsers.add_parser('cores')
    cores_parser.set_defaults(method='cores')
    sublist.append(cores_parser)

    #print args.method
    for p in sublist:
        p.add_argument('-d', '--data', required=False)
        p.add_argument('-a', '--edgeDict', required=False)
        p.add_argument('-s', '--simDict', required=False)
        p.add_argument('-dd', '--distDict', required=False, default=None) # required=True)
        p.add_argument('-t', '--trials', default=20, type=int)
        p.add_argument('-b', '--bagSize', default=460, type=int)
        p.add_argument('-acc', '--acceleration', default=0, choices=['0','2'])
        p.add_argument('-load', '--loadedCores', action='store_true', default=False)

    #kopt_parser.add_argument('-dd', '--distDict', type=str) # not used as yet
    kopt_parser.add_argument('-td', '--trials_decay', action='store_true', default=False)
    kopt_parser.add_argument('-Kmax', '--Kmax', default=24, type=int)
    kopt_parser.add_argument('-Kmin', '--Kmin', default=1, type=int)
    kopt_parser.add_argument('-Kl', '--KlastSeq', type=int)
    kopt_parser.add_argument('-L', '--Kupper', type=list)

    cores_parser.add_argument('-K', '--K', default=24, type=int)
    cores_parser.add_argument('-alpha', '--threshold', default=None, type=int)
    cores_parser.add_argument('-fc', '--filepathCores', default=None, type=str)
    cores_parser.add_argument('-form', '--dendroFormat', type=str, choices=['png', 'svg', 'pdf'], default='png')
    cores_parser.add_argument('-sCC', '--saveCommonClusters', action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = read_arguments()

    data = []
    affinities = {}
    simDict = {}

    if args.distDict != None:
        distDict = importData(args.distDict, ifjson=1)
        print "\n pairwise distances imported"
    else:
        distDict = {}

    if not args.loadedCores:
        data = importData(args.data, ifjson=1)
        affinities = importData(args.edgeDict, ifjson=1)
        print "\n affinities imported"
        simDict = importData(args.simDict, ifjson=1)
        print "\n pairwise similarities imported\n"

    #~~~~~~~~~~~~ method for finding the optimal K ~~~~~~~~~~~#
    if args.method == 'kopt':
        if args.trials_decay: trials_decay = True
        else: trials_decay = False

        if args.KlastSeq:
            if args.Kupper:
                Klist = range(1,args.KlastSeq+1) + args.Kupper
            else:
                Klist = range(1,args.KlastSeq+1)
                Klist.append(args.Kmax)
        else:
            Klist = range(1,args.Kmax+1)

        startTime = time.time()
        Kopt, mod_lists, modStdev, modMean, modMax = ModularityProfile(data,
                                                                    Kmin=args.Kmin,
                                                                    Kmax=args.Kmax,
                                                                    Klist=Klist,
                                                                    edgeDict=affinities,
                                                                    simDict=simDict,
                                                                    bagSize=args.bagSize,
                                                                    trials=args.trials,
                                                                    trials_decay=trials_decay)
        endTime = time.time()

        fordump = [Kopt, mod_lists, modStdev, modMean, modMax]
        json_filename = "kopt_json_version_" + Version
        json.dump(fordump, open(json_filename, "w"))

        print '\n\n'
        print 'saved as: ', json_filename
        print 'time: ', endTime - startTime
        print 'optimal K: ', Kopt
        print 'modMax: ', modMax
        for i in xrange(len(Klist)):
            print "mod_list for k=%d is:  %s"% (Klist[i], mod_lists[i])
        print '\n\n'

    #~~~~~~~~~~~ method for averaging clustering results ~~~~~~~~~~~#
    if args.method == "cores":
        print "gotcha"
        result = SGJRIcores(data=data,
                       K=args.K, edgeDict=affinities,
                       simDict=simDict, distDict=distDict,
                       bagSize=args.bagSize,
                       trials=args.trials, threshold=args.threshold,
                       loadedCores=args.loadedCores, filepathCores=args.filepathCores,
                       dendroFormat=args.dendroFormat,
                       acceleration=int(args.acceleration))

        if 0==0: #args.distDict != None:
            clustData, commonCluster, treeStruct, segmentDict, harmCentr = result[:-1]
            mongo = result[-1]
            if args.saveCommonClusters:
                fordump = [clustData, commonCluster, treeStruct, segmentDict, harmCentr, mongo]
            else:
                fordump = [clustData, ['commonCluster_placeholder'],
                           treeStruct, segmentDict, harmCentr, mongo]
        else:
            clustData, commonCluster, treeStruct, segmentDict = result
            fordump = [clustData, commonCluster, treeStruct, segmentDict]

        pkl_filename = "cores_pickled_version_" + Version
        with open(pkl_filename, 'w') as f:
            pickle.dump(fordump, f, pickle.HIGHEST_PROTOCOL)

        print "Saved as: ", pkl_filename
        #json.dump(fordump, open(json_filename, "w"))

if __name__ == '__main__':
    main()
