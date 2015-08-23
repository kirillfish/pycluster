# getdata.py
# -*- coding: utf-8 -*-
'''
Getting and cleaning data for domain clustering
'''

import argparse
import time
from datetime import datetime as dt
import json, pickle
from collections import defaultdict, Counter
import networkx as nx

try:
    import accessor
except:
    print "Warning: accessor module not imported."
import numpy as np
import ast

try:
    import pathos.multiprocessing as mp
except:
    print "Warning: pathos.multiprocessing not imported. Parallel algorithms will be unavailable."

# DEPRECATED
def networkX_to_Giraph(graph=None, filepathGraph=None, outputFormat='VertexInputFormat',
                       filepathOutput=None, extension='txt'):
    '''
    Function that converts networkX graph into the format apt for Apache Giraph
    '''

    if graph == None:
        print "loading the graph..."
        graph = pickle.load(open(filepathGraph, 'r'))

    if outputFormat == 'VertexInputFormat':
        output = {}
        nodeIntMap = {}

        print "creating additional data structures..."
        edgeData = ( edge[2]['weight'] for edge in graph.edges(data=True) )
        edgeDict = dict(zip(graph.edges(), edgeData))

        print "initialising output..."
        for ID, node in enumerate(graph.nodes()):
            output[ID] = [ID, 1., []]
            nodeIntMap[node] = ID

        # only one edge from (node, nei) and (nei, node) is found, but loop
        # gets each of them once -- that's OK
        print "filling output with edges..."
        for ID, node in enumerate(graph.nodes()):
            for nei in graph.neighbors(node):
                if (node, nei) in edgeDict:
                    output[ID][2].append([nodeIntMap[nei], edgeDict[(node, nei)]])

        if filepathOutput == None:
            filepathOutput = "GIRAPH_" + VersionTime

        filepathNodeInt = "NODEINTMAP_" + VersionTime

        if extension == 'txt':
            f = open(filepathOutput, 'w')
            for ID in output:
                f.write("%s\n" % str(output[ID]).encode('utf-8'))
            f.close()
            print "giraph saved: ", filepathOutput
            print "format: ", outputFormat
            json.dump(nodeIntMap, open(filepathNodeInt, 'w'))
            print "nodeIntMap saved: ", filepathNodeInt

        elif extension == 'json':
            json.dump(output.values(), open(filepathOutput, 'w'))
            print "giraph saved: ", filepathOutput
            print "format: ", outputFormat
            json.dump(nodeIntMap, open(filepathNodeInt, 'w'))
            print "nodeIntMap saved: ", filepathNodeInt

        else:
            print "Warning: output not written in file"

        forreturn = {'nodeIntMap': nodeIntMap, 'output': output}
        return forreturn

    elif outputFormat == 'SpinnerEdgeInputFormat':
        nodeIntMap = {}

        print "initialising output..."
        for ID, node in enumerate(graph.nodes()):
            nodeIntMap[node] = ID

        if filepathOutput == None:
            filepathOutput = "GIRAPH_" + extension + outputFormat + VersionTime

        if extension == 'txt':
            f = open(filepathOutput, 'w')
            for edge in graph.edges():
                f.write("%s\t%s\n" % ( str(nodeIntMap[edge[0]]),
                                       str(nodeIntMap[edge[1]]) ))
            f.close()
            print "giraph saved: ", filepathOutput
            print "format: ", outputFormat

            '''
            elif extension == 'json':
                json.dump(graph.edges(), open(filepathOutput, 'w'))
                print "giraph saved: ", filepathOutput
                print "format: ", outputFormat
            '''

        else:
            print "Warning: output not written in file"

        forreturn = {'nodeIntMap': nodeIntMap}
        return forreturn

    else:
        print "unknown output format"
        return 1


# DEPRECATED
def get_paths(data, edgeDict=None, graph=None, algorithm='Floyd', weights='aff',
              saveResults=False, filepathDist=None):
    '''
    a function that computes all pairwise distances (shortest paths)
    and puts it into a dictionary.
    Input: graph with edges as distances OR
    (default) dict with pairwise affinities (not distances)
    Output: shortest paths in a dict of dicts.
    '''
    unmatched = []
    distGraph = nx.Graph()
    if weights == 'aff':
        print "computing 1/affinities..."
        for i in data:
            matched = 0
            if i in edgeDict:
                distGraph.add_node(i)
                matched = 1
                for j in edgeDict[i]:
                    if edgeDict[i][j] == 0:
                        distGraph.add_edge(i, j, weight='inf')
                    else:
                        distGraph.add_edge(i, j, weight=1.0 / edgeDict[i][j])
            if matched == 0: unmatched.append(i)
    elif weights == 'dist':
        distGraph = graph
    else:
        print "Error: weight should be either 'aff' or 'dist'"
        return 1

    print "number of unmatched nodes: ", len(unmatched)
    print unmatched

    if algorithm == 'Floyd':
        print "\ncomputing all shortest paths with Floyd-Warshall algorithm..."
        dist = nx.floyd_warshall(distGraph)
        print "pairwise distances obtained"
    elif algorithm == "Dijkstra":
        print "\ncomputing all shortest paths with Dijkstra algorithm..."
        dist = nx.all_pairs_dijkstra_path_length(distGraph)
        print "pairwise distances obtained"
    else:
        print "\nError: cannot recognize what algorithm to use"
        return 1

    print "num of rows in dist matrix: ", len(dist.keys())
    # print dist.keys()

    if saveResults:
        if filepathDist == None:
            #filepathDist = "/home/krybachuk/SHORTESTPATH_" + VersionTime
            filepathDist = "SHORTESTPATH_" + "latest"
        json.dump(dist, open(filepathDist, 'w'))
        print "shortest path dict saved\n\n"

    return dist


class PyclusterGraphConstructor():
    def __init__(self):
        self.users_count = 10000
        self.min_users = 15
        self.min_visits = 15
        self.min_aff = 20
        self.specified_size = False
        self.node_num = 1200
        self.edge_num = None
        self.Version = dt.fromtimestamp(time.time()).strftime('%Y-%m-%d__%H_%M_%S')

        self.users = []
        self.users_file_path = None
        self.sessions_file_path=None
        self.dom_file_path = None
        self.aff_file_path = None
        self.graph_file_path = None
        self.graph_file_path_nodes = None
        self.file_path_sim = None

        self.domains_common = defaultdict(lambda: Counter())
        self.domains_total_raw = Counter()
        self.domains_total = Counter()
        self.session_aware = True

        self.G = nx.Graph()
        self.SPG = nx.Graph()
        self.affinities = None
        self.similarities = None
        self.split_files = False  # if similarities computed in parallel are stored piecewise or combined into one file

    def get_users(self, file_path=None):
        '''
        Function that loads and saves user data. Can be omitted if the script is
        called from bash (if you tell it to load a previously saved file).
        '''
        print "\ngetting all data about users\n"
        users = []
        count = 0
        t = time.time()

        if file_path == None:
            file_path = "USERS_" + self.Version
        file_path_sessions = "USERS_SESSIONS_" + self.Version

        U = open(file_path, 'w')
        S = open(file_path_sessions, 'w')
        while count < self.users_count:
            try:
                for user in accessor.get_sample_users(5000):
                    count += 1
                    if count % 1000 == 0: print "users sampled: ", count, "   total time: ", time.time() - t
                    attrs = vars(user)['data']['domains']
                    if attrs != []:
                        print >> U, json.dumps(attrs)
                    attrs_sessions = vars(user)['data']['domains_list']
                    if attrs_sessions != []:
                        attrs_sessions = [(v['domain'], v['timestamp']) for v in attrs_sessions]
                        print >> S, json.dumps(attrs_sessions)
            except UnicodeDecodeError:
                print 'UnicodeDecodeError catched'
                continue

        print "USERS saved: ", file_path, "\n\n"
        print "users sampled in total: ", count

        self.users_file_path = file_path
        self.sessions_file_path = file_path_sessions

    def get_domains(self, dom_file_path=None,
                    saveResults=True, give_common=True):
        '''
        Compute total attendance for each domain
        and common attendance for each pair.
        Input: users (result of get_users())
        Output: a counter and defaultdict with unfiltered data.
        '''

        toExclude = {'ams1.ib.adnxs.com', 'fra1.ib.adnxs.com', 'ib.adnxs.com', 'cache.betweendigital.com',
                     '&referrer=http:', "&referrer=${referer_url}", 'https:', 'http:', 'masterh1.adriver.ru',
                     'masterh2.adriver.ru', 'masterh4.adriver.ru', 'masterh5.adriver.ru', 'masterh7.adriver.ru',
                     'masterh6.adriver.ru', 'mh6.adriver.ru', 'mh8.adriver.ru', 'bel1.adriver.ru', 'un1.adriver.ru',
                     'cbn2.tbn.ru', 'cdn.etgdta.com', 'delivery.a.switchadhub.com'}

        print "\nextracting domain data from users\n"
        count = 0
        with open(self.users_file_path, 'r') as U:
            for line in U:
                dict_line = ast.literal_eval(line)
                for domainA in dict_line:  # ['data']['domains']:
                    self.domains_total_raw[domainA] += 1
                count += 1
                if count % 1000 == 0:
                    print "users processed (domain filtering): ", count

        # determine threshold for constructing filtered set domains_total
        if self.specified_size:
            if self.node_num != None:
                node_hist = self.domains_total_raw.values()
                self.min_users = self.graphScaleOutOfThreshold(node_num=self.node_num,
                                                               node_hist=node_hist)[0]
                print "min_users corresponding to your node number: ", self.min_users, "\n"
                if self.min_users < 15: print "ACHTUNG: min_users < 15 !!!"
        else:
            print "min_users: ", self.min_users

        # construct domains_total
        # now domains(nodes) are filtered at the getdomains() stage,
        # not on getaff() !!! But edges are still filtered at getaff()
        self.domains_total = filter(
            lambda domain: (self.domains_total_raw[domain] >= self.min_users and domain not in toExclude
                            and 'adriver' not in domain and 'am15.net' not in domain), self.domains_total_raw)
        self.domains_total = {k: self.domains_total_raw[k] for k in self.domains_total}
        print "total number of domains: ", len(self.domains_total_raw)
        print "number of domains with >=%d visits: %d" % (self.min_users, len(self.domains_total))

        domains_common = defaultdict(lambda: Counter())
        if self.session_aware:
            print "[DEBUG] session_file_path: ", self.sessions_file_path
            with open(self.sessions_file_path, 'r') as U:
                count = 0
                for line in U:
                    dict_line = ast.literal_eval(line)  # json.loads(line)
                    """
                    if len(dict_line) >= 2:
                        for i, visit in enumerate(dict_line[1:]):
                            # 0 -- domain, 1-- timestamp
                            if visit[0] in self.domains_total \
                                and dict_line[i-1][0] in self.domains_total \
                                and visit[0] != dict_line[i-1][0] \
                                and visit[1] >= dict_line[i-1][1] - 120 * 60 * 1000 \
                                and visit[1] <= dict_line[i-1][1] - 2 * 1000:
                                domainA = visit[0]
                                domainB = dict_line[i-1][0]
                                domains_common[domainA][domainB] += 1
                    if len(dict_line) >= 3:
                        for i, visit in enumerate(dict_line[2:]):
                            # 0 -- domain, 1-- timestamp
                            if visit[0] in self.domains_total \
                                and dict_line[i-1][0] in self.domains_total \
                                and visit[0] != dict_line[i-1][0] \
                                and dict_line[i-1][1] - visit[1] < min((dict_line[i-2][1] - dict_line[i-1][1])*100, 5*60*60*1000) \
                                and visit[1] <= dict_line[i-1][1] - 2 * 1000:
                                domainA = visit[0]
                                domainB = dict_line[i-1][0]
                                domains_common[domainA][domainB] += 1
                    """
                    for visitA in dict_line:
                        for visitB in dict_line:
                            # we are not interested to the self-edges, even if they are different visits to one domain!
                            if visitA[0] != visitB[0] \
                                and visitA[0] in self.domains_total \
                                and visitB[0] in self.domains_total \
                                and abs(visitA[1] - visitB[1]) < 20 * 60 * 1000 \
                                and abs(visitA[1] - visitB[1]) > 2 * 1000:
                                domains_common[visitA[0]][visitB[0]] += 1
                    count += 1
                    if count % 1000 == 0: print "users processed (common audience): ", count
        else:
            with open(self.users_file_path, 'r') as U:
                count = 0
                for line in U:
                    dict_line = ast.literal_eval(line)
                    for domainA in dict_line:  # ["data"]["domains"]:
                        if domainA in self.domains_total:
                            for domainB in dict_line:  # ["data"]["domains"]:
                                if domainB in self.domains_total and domainA != domainB:
                                    domains_common[domainA][domainB] += 1
                    count += 1
                    if count % 1000 == 0: print "users processed (common audience): ", count

        if give_common:
            self.domains_common = domains_common
        # when you want to save the results without calling the bash script
        if saveResults:
            if dom_file_path == None:
                dom_file_path = "DOMAINS_" + self.Version
            out = open(dom_file_path, 'w')
            for domainA in self.domains_total:
                for domainB in domains_common[domainA]:
                    try:
                        out.write("%s\t%s\t%s\t%s\t%s\n" % (domainA.encode('utf-8'),
                                                            domainB.encode('utf-8'),
                                                            self.domains_total[domainA.encode('utf-8')],
                                                            self.domains_total[domainB.encode('utf-8')],
                                                            domains_common[domainA.encode('utf-8')][
                                                                domainB.encode('utf-8')]))
                    except KeyError:
                        print "KeyError occured: ", domainA, domainB
            out.close()
            print "domain data saved: ", dom_file_path, "\n\n"
            self.dom_file_path = dom_file_path

    @staticmethod
    def graphScaleOutOfThreshold(edge_hist=None, node_hist=None,
                                 edge_num=None, node_num=None):
        '''
        Funtion that allows to directly specify the number of nodes/edges one
        wants the graph to contain.
        '''
        result = []
        if edge_hist != None and edge_num != None:
            print "\ncomputing aff_border for the graph to contain exact number of edges you told me"
            edgeA = float(edge_num) / len(edge_hist)
            np_edge = np.array(edge_hist)
            edgeQ = np.percentile(np_edge,
                                  100. * (1. - 2 * edgeA))  # cuz the same edge enters twice in getaff() to edge_hist
            result.append(edgeQ)

        if node_hist != None and node_num != None:
            print "\ncomputing min_users for the graph to contain exact number of nodes you told me"
            nodeA = float(node_num) / len(node_hist)
            np_node = np.array(node_hist)
            nodeQ = np.percentile(np_node, 100. * (1. - nodeA))
            result.append(nodeQ)

        if result == []:
            print "\nError: invalid parameters combination\n"
            return 1
        else:
            return result

    def getaff(self, aff_file_path=None,
               graph_file_path=None, saveResults=False, return_aff=False):
        '''
        Compute affinities, prune out weak ones
        and construct a graph.
        Input: results of getdomains(), thresholds on domain visits (nodes) and
        on affinities (edges), or directly stated number of nodes/edges.
        Output: a graph for clustering, affinity dict of dicts
        (the reshaped graph data).
        Nodes and edges are added separately
        (hence, high thresholds on affinity means higher number of isolates)
        '''
        print "computing affinities, pruning out weak ones and cunstructing a graph"

        graph = nx.Graph()
        # affinity data for histograms is not truncated, in order to understand
        # what threshold for affinity to set

        # the block for the case you directly specify number of nodes/edges rather
        # than calculate thresholds

        if self.specified_size:
            aff_hist = []
            if self.edge_num != None:
                # selected = [i for i in domains_total if domains_total[i]>=min_users]
                for domainA in self.domains_total:
                    for domainB in self.domains_total:  #[i for i in domains_common[domainA] if i in selected]:
                        aff = (1.0 * self.domains_common[domainA][domainB] * self.users_count) / \
                              (self.domains_total[domainA] * self.domains_total[domainB])
                        aff_hist.append(aff)  #the same item enters twice

                self.min_aff = self.graphScaleOutOfThreshold(edge_num=self.edge_num,
                                                             edge_hist=aff_hist)[0]
                print "aff_border corresponding to your number of edges: ", self.min_aff, "\n"
            print "minimum affinity: ", self.min_aff

        else:
            print "minimum affinity: ", self.min_aff

        print "selected domains: ", len(self.domains_total)
        print "creating a graph..."
        for domainA in self.domains_common:
            graph.add_node(domainA)
            for domainB in self.domains_common[domainA]:
                # if domains_total[domainB] < min_users:
                #    continue
                #if domainB != domainA:
                aff = (1.0 * self.domains_common[domainA][domainB] * self.users_count) / \
                    (self.domains_total[domainA] * self.domains_total[domainB])
                if (aff > self.min_aff):
                    graph.add_edge(domainA, domainB, weight=aff)
        print "the graph is created\n\n"
        print "[DEBUG] pairs in domains_common: ", sum([len(self.domains_common[d])for d in self.domains_common])
        print "[DEBUG] min affinity: ", self.min_aff
        print "graph order: ", graph.order()
        print "graph size: ", graph.size()
        print "\n"

        # reshape affinity data from list of lists into dict
        print "reshaping affinity into dict of dicts..."
        affinities = defaultdict(lambda: defaultdict(lambda: 0))
        counter = 0
        exceptions = 0
        print "processing pairs..."
        for node_pair in graph.edges(data=True):
            counter += 1
            if counter <= 3: print "\ndata example: ", node_pair, type(float(node_pair[2]['weight'])), repr(
                float(node_pair[2]['weight'])), 1.0 / float(node_pair[2]['weight'])
            try:
                affinities[node_pair[0]][node_pair[1]] = float(node_pair[2]['weight'])
                affinities[node_pair[1]][node_pair[0]] = float(node_pair[2]['weight'])
            except ValueError:
                exceptions += 1
                continue
        print "\npairs in total:", counter
        print "pairs omitted due to exceptions: ", exceptions

        if saveResults:
            if aff_file_path == None:
                aff_file_path = "AFFINITIES_" + self.Version

            if graph_file_path == None:
                graph_file_path = "GRAPH_" + self.Version

            graph_file_path_nodes = graph_file_path + "_NODES"

            with open(graph_file_path, 'w') as f:
                pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

            json.dump(graph.nodes(), open(graph_file_path_nodes, "w"))
            json.dump(affinities, open(aff_file_path, "w"))

            print "graph and affinity dict are saved: ", aff_file_path, graph_file_path, graph_file_path_nodes, "\n\n"
            self.aff_file_path = aff_file_path
            self.graph_file_path = graph_file_path
            self.graph_file_path_nodes = graph_file_path_nodes

        self.G = graph
        if return_aff:
            self.affinities = affinities

    def pairwise_similarities(self, sim_type='weighted_jaccard', file_path_sim=None, attr='weight',
                              save_results=False, skip=None, limit=None):

        similarities = defaultdict(lambda: defaultdict(lambda: 0))
        counter = 0
        print "computing pairwise similarities..."
        print "similarity function chosen: ", sim_type

        if limit == None or skip == None:
            parallel = False
            limit = self.G.number_of_nodes()
            skip = 0
        else:
            parallel = True

        node_subset = self.G.nodes()[skip:skip + limit]
        if sim_type == 'weighted_jaccard':
            for nodeA in node_subset:  # self.G.nodes_iter():
                neighbA = set(self.G.neighbors(nodeA))
                for nodeB in self.G.nodes_iter():
                    neighbB = set(self.G.neighbors(nodeB))
                    #union = neighbB | neighbA
                    xor = neighbB ^ neighbA
                    inters = neighbB & neighbA
                    xor.discard(nodeA)
                    xor.discard(nodeB)
                    intersum = 0.0
                    xorsum = 0.0

                    for node in inters:  # может, не среднее, а минимум
                        #intersum += graph[nodeA][node][attr]
                        #intersum += graph[nodeB][node][attr]
                        intersum += min(self.G[nodeA][node][attr], self.G[nodeB][node][attr])

                    for node in xor:
                        if node in self.G[nodeA]:
                            xorsum += self.G[nodeA][node][attr]
                        if node in self.G[nodeB]:
                            xorsum += self.G[nodeB][node][attr]

                    #intersum = 0.5*intersum
                    unionsum = intersum + xorsum

                    if nodeA == nodeB:
                        similarities[nodeA][nodeB] = 1.0
                    elif unionsum == 0 or intersum == 0:
                        continue
                        #similarities[nodeA][nodeB] = 0.0
                    else:
                        similarities[nodeA][nodeB] = intersum / unionsum
                counter += 1
                if counter % 100 == 0:
                    print "nodes processed: ", counter

        elif sim_type == 'unweighted_jaccard':
            for nodeA in node_subset:  # self.G.nodes_iter():
                neighbA = set(self.G.neighbors(nodeA))
                for nodeB in self.G.nodes_iter():
                    neighbB = set(self.G.neighbors(nodeB))
                    inters = neighbB & neighbA
                    union = neighbB | neighbA
                    union.discard(
                        nodeA)  # remove A and B from the union, to make JS honest (otherwise the absence of an edge would improve JS -- nonsense)
                    union.discard(nodeB)

                    if nodeA == nodeB:
                        similarities[nodeA][nodeB] = 1.0
                    elif len(union) == 0 or len(inters) == 0:  # making matrix sparse, but everything else is 0
                        continue
                        #similarities[nodeA][nodeB] = 0.0
                    else:
                        similarities[nodeA][nodeB] = float(len(inters)) / len(union)
                counter += 1
                if counter % 100 == 0:
                    print "nodes processed: ", counter
        else:
            print "Error: unknown pairwise similarity measure: ", sim_type
            return (1)

        print "number of rows in sim matrix: ", len(similarities.keys())

        if save_results:
            if file_path_sim == None:
                file_path_sim = "SIMILARITIES_" + self.Version
            json.dump(similarities, open(file_path_sim, 'w'))
            print "similarities saved: ", file_path_sim
            self.file_path_sim = file_path_sim

        if not parallel:
            self.similarities = similarities
        else:
            return similarities

    def parallel_sim(self, cores, save_results=False, sim_type='weighted_jaccard'):  # , split_files=False):
        if not self.split_files:
            localVars = {'sim_type': sim_type,
                         'save_results': False}
            localVars = [localVars] * cores

            skips = []
            limits = []
            l = 0
            for coreNum in xrange(cores - 1):
                skips.append(l)
                l += int(self.G.number_of_nodes() / cores)
                limits.append(int(self.G.number_of_nodes() / cores))

            skips.append(l)
            limits.append(self.G.number_of_nodes() - l)
            rangeList = map(None, skips, limits, localVars)

            def pairwise_similarities_wrapper(bigTuple):
                sim = self.pairwise_similarities(skip=bigTuple[0], limit=bigTuple[1], **bigTuple[2])
                return sim

            # pool = multiprocessing.Pool(cores)
            pool = mp.Pool(cores)
            similarities = pool.map(pairwise_similarities_wrapper, rangeList)

            similarities = {k: v for d in similarities for k, v in d.items()}
            print "please check the number of nodes in the graph: ", len(similarities)

            if save_results:
                if self.file_path_sim == None:
                    self.file_path_sim = "SIMILARITIES_" + self.Version
                json.dump(similarities, open(self.file_path_sim, 'w'))
                print "similarity dict saved: ", self.file_path_sim, "\n\n"
        else:
            localVars = {'sim_type': sim_type,
                         'save_results': save_results}
            localVars = [localVars] * cores

            skips = []
            limits = []
            filePathes = []
            l = 0
            if self.file_path_sim == None:
                self.file_path_sim = "SIMILARITIES_" + self.Version
            for coreNum in xrange(cores - 1):
                skips.append(l)
                l += int(self.G.number_of_nodes() / cores)
                limits.append(int(self.G.number_of_nodes() / cores))
                filePathes.append(self.file_path_sim + '_' + str(coreNum))

            skips.append(l)
            limits.append(self.G.number_of_nodes() - l)
            filePathes.append(self.file_path_sim + '_' + str(cores - 1))

            self.sim_file_path = filePathes
            rangeList = map(None, skips, limits, filePathes, localVars)

            def pairwise_similarities_wrapper(bigTuple):
                sim = self.pairwise_similarities(skip=bigTuple[0], limit=bigTuple[1], file_path_sim=bigTuple[2],
                                                 **bigTuple[3])
                return sim

            pool = mp.Pool(cores)
            similarities = pool.map(pairwise_similarities_wrapper, rangeList)

            similarities = {k: v for d in similarities for k, v in d.items()}
            print "please check the number of nodes in the graph: ", len(similarities)
        # not saving united similarities into a single file, because they were
        # already saved piecewise
        self.similarities = similarities
        # return similarities

    def load_splitted_sim(self, file_path_sim, cores):
        import os
        import os.path

        similarities = []
        if os.path.exists(file_path_sim):
            similarities = {}
            similarities.update(json.load(open(file_path_sim, 'r')))
            print "\nnon splitted similarity file found: ", file_path_sim
            print "the search will stop here"
            return similarities
        for i in xrange(cores):
            if os.path.exists(file_path_sim + '_' + str(i)):
                similarities.append(json.load(open(file_path_sim + '_' + str(i), 'r')))
                print "Loaded: ", file_path_sim + '_' + str(i)
            else:
                print "Warning: no such file: ", file_path_sim + '_' + str(i)
        print "the number of loaded files: ", len(similarities)
        print [type(simPiece) for simPiece in similarities]
        similarities = {k: v for d in similarities for k, v in d.items()}
        self.similarities = similarities
        return similarities

    def sparsification(self, local=True, power=0.5, saveResults=True,
                       filepathGraph=None, multiplyByWeight=False):
        '''
        a function that smartassly removes some edges remaining all vertices in place
        needed to emphasize cluster structure (reduce number of iterations in clustering algorithms,
        and directly improve running time of edge-based algorithms
        '''
        if local:

            print "\nsparsification..."
            # add attribute to edges indicating whether the edge is to be retained
            attrDict = {}
            for edge in self.G.edges():
                attrDict[edge] = 0
            nx.set_edge_attributes(self.G, 'retain', attrDict)

            print "finding edges to retain..."
            # find edges to retain
            count = 0
            exceptions = 0
            for nodeA in self.similarities:
                count += 1
                if count % 500 == 0: print "sparsification: nodes processed: ", count

                keys = set(self.G.neighbors(nodeA))
                if multiplyByWeight:
                    neighbWeight = defaultdict(lambda: defaultdict(lambda: {}))
                    for e in self.G.edges_iter(nodeA, data=True):
                        neighbWeight[e[0]][e[1]] = e[2]['weight']
                        neighbWeight[e[1]][e[0]] = e[2]['weight']
                    neighbSim = {node: self.similarities[nodeA][node] * neighbWeight[nodeA][node]
                                 for node in keys}
                else:
                    neighbSim = {node: self.similarities[nodeA][node] for node in keys}
                numRetained = max(1, int(np.ceil(self.G.degree(nodeA) ** power)))
                keysRet = dict(sorted(neighbSim.items(), key=lambda x: x[1],
                                      reverse=True)[:numRetained]).keys()

                # each edge is checking twice
                # if for at least 1 node this edge appears to be at the top d_i^e edges
                # by edge similarity, this edge is marked as to be retained

                if nodeA in self.G.nodes():
                    # for neighb in self.G.neighbors(nodeA):
                    #    if neighb in keysRet:
                    #        self.G[nodeA][neighb]['retain'] = 1
                    for neighb in keysRet:
                        self.G[nodeA][neighb]['retain'] = 1
                else:
                    exceptions += 1

            print "nodes in similarities that are not in the graph: ", exceptions

            print "creating sparsified graph..."
            newEdges = [edge for edge in self.G.edges(data=True) if edge[2]['retain'] == 1]
            for edge in newEdges:
                del edge[2]['retain']
            spg = nx.Graph()
            spg.add_nodes_from(self.G.nodes())
            spg.add_edges_from(newEdges)

        else:
            print "only local sparsification currently available"
            return 1

        print "saving sparsified graph..."

        if saveResults:
            if filepathGraph == None:
                filepathGraph = "SPARSIFIED_GRAPH_" + self.Version
            fordump = spg
            with open(filepathGraph, 'w') as f:
                pickle.dump(fordump, f, pickle.HIGHEST_PROTOCOL)

        print "graph saved: ", filepathGraph

        self.G = spg


def read_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--users', type=int, default=15000)
    parser.add_argument('-m', '--min_users', type=int, default=15)
    parser.add_argument('-mv', '--min_visits', type=int, default=15)
    parser.add_argument('-a', '--min_aff', type=int, default=20)
    parser.add_argument('-ses', '--session_aware', action='store_true', default=False)

    # parser.add_argument('-gd', '--give_common', action='store_true', default=False)
    parser.add_argument('-save', '--saveResults', action='store_true', default=False)
    parser.add_argument('-nousers', '--omit_users', action='store_true', default=False)

    parser.add_argument('-FU', '--filepathUs', type=str, default=None)
    parser.add_argument('-FS', '--filepathSes', type=str, default=None)

    parser.add_argument('-sim', '--simType', choices=['weighted_jaccard', 'unweighted_jaccard'],
                        default='weighted_jaccard')

    parser.add_argument('-scale', '--scale', action='store_true', default=False)
    parser.add_argument('-n', '--node_num', type=int)
    parser.add_argument('-e', '--edge_num', type=int)

    parser.add_argument('-sparsify', '--sparsify', action='store_true', default=False)
    parser.add_argument('-sp', '--sparsifyPower', type=float, default=0.5)
    # parser.add_argument('-minhash', '--minhash', action='store_true', default=False)

    parser.add_argument('-parsim', '--parallel_similarity_computation', action='store_true', default=False)
    parser.add_argument('-cores', '--cores', type=int, default=1)
    parser.add_argument('-split', '--splitSimilarity', action='store_true', default=False)

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = read_arguments()
    # execute all functions successively
    start = time.time()

    pgc = PyclusterGraphConstructor()
    pgc.users_count = args.users
    pgc.min_aff = args.min_aff
    pgc.min_visits = args.min_visits
    pgc.min_users = args.min_users
    pgc.specified_size = args.scale
    pgc.node_num = args.node_num
    pgc.edge_num = args.edge_num
    pgc.session_aware = args.session_aware
    pgc.split_files = args.splitSimilarity

    if not args.omit_users:
        print 'get_users() function with no RAM footprint...'
        pgc.get_users()
    else:
        pgc.users_file_path = args.filepathUs
        if args.session_aware:
            pgc.sessions_file_path = args.filepathSes

    t1 = time.time()

    pgc.get_domains(saveResults=args.saveResults)

    t2 = time.time()
    pgc.getaff(saveResults=args.saveResults)

    t3 = time.time()
    if not args.parallel_similarity_computation:
        pgc.pairwise_similarities(sim_type=args.simType, save_results=args.saveResults)
    else:
        pgc.parallel_sim(cores=args.cores, sim_type=args.simType, save_results=args.saveResults)

    if args.sparsify:
        pgc.sparsification(saveResults=args.saveResults, power=args.sparsifyPower)

        pgc.file_path_sim = 'SPARSIFIED_SIMILARITIES_' + pgc.Version
        if not args.parallel_similarity_computation:
            pgc.pairwise_similarities(sim_type=args.simType, save_results=args.saveResults)
        else:
            pgc.parallel_sim(cores=args.cores, sim_type=args.simType,
                             save_results=args.saveResults)

    finish = time.time()

    print "get_users time: ", t1 - start
    print "getdomains time: ", t2 - t1
    print "getaff time: ", t3 - t2
    print "similarities time: ", finish - t3
    print "\n\n"


if __name__ == '__main__':
    main()
