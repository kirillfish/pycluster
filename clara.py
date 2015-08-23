
from util import importData, euclidean_distance, manhattan_distance, pearson_distance, direct_distance, similarity_distance
from pam import kmedoids
import random
import sys
import time
import json
from scipy.stats import bernoulli
try:
    from getdata import load_splitted_sim
except:
    print "Warning: load_splitted_sim not imported"

# Global variables
debugEnabled = True    # Debug
claraLoopNum = 2
distances_cache = {}    # Save the tmp distance for acceleration (idxMedoid, idxData) -> distance
CostType = "modularity"        # variants: total, average, modularity
BagSize = 10098           # speed-quality tradeoff
degreeSelection = True
coresLoaded = 5
takeAllNodes = True  # renders BagSize irrelevant

def targetFunction(data, costF_idx, medoids_idx, cacheOn=False, distDict={},
                   simDict={}, affinities={}, costType=CostType,
                   namedPoints=True):
    '''
    Compute the average cost of medoids based on certain cost function
    and do the clustering given the medoids
    '''
    if costType not in ["total", "average", "modularity"]:
        print "unknown target function - check the global variables in the code"
        return(1)

    # Init the cluster
    size = len(data)
    total_cost = {}
    medoids = {}
    for idx in medoids_idx:
        medoids[idx] = []
        total_cost[idx] = 0.0
    assignErrors = []

    # Compute the distance and do the clustering
    for i in range(size):
        choice = -1
        # Make a big number
        min_cost = float('inf')
        # medoids themselves are also included into resulting cluster lists
        for m in medoids:
            if cacheOn == True:
                # Check for cache
                tmp = distances_cache.get((m,i), None)
            if cacheOn == False or tmp == None:
                if costF_idx == 0:
                    # euclidean_distance
                    tmp = euclidean_distance(data[m], data[i])
                elif costF_idx == 1:
                    # manhattan_distance
                    tmp = manhattan_distance(data[m], data[i])
                elif costF_idx == 2:
                    # pearson_distance
                    tmp = pearson_distance(data[m], data[i])
                elif costF_idx == 3:
                    # direct_distance
                    tmp = direct_distance(data[m], data[i], distDict)
                elif costF_idx == 4:
                    # similarity_distance
                    tmp = similarity_distance(data[m], data[i], simDict)
                else:
                    print('Error: unknown cost function idx: ' % (costF_idx))
            if cacheOn == True:
                # Save the distance for acceleration
                distances_cache[(m,i)] = tmp
            # Clustering

            # Randomization for nodes/points isolated from all the medoids
            # in order to assign them to random clusters. Hope averaging will
            # be able to glean cases for which some medoids did appear in the
            # same connected component, and group those nodes together.
            if tmp==0.0 and min_cost==0.0: # no connection to either medoid
                rv = bernoulli.rvs(1./len(medoids_idx), size=1)
                if rv[0]==1.: choice = m
            elif tmp < min_cost:
                #if tmp < min_cost:
                choice = m
                min_cost = tmp
        # Done the clustering
        if choice == -1:
            print "ERROR: the node cannot be assigned"
            assignErrors.append(i)
        else:
            medoids[choice].append(i)
            total_cost[choice] += min_cost

    # Compute the target function
    if costType == "total":
        #print total_cost
        return(sum(total_cost.values()), medoids)

    elif costType == "average":
    # Compute the average cost
        avg_cost = 0.0
        for idx in medoids_idx:
            avg_cost += total_cost[idx] / len(medoids[idx])
        # Return the average cost and clustering
        return(avg_cost, medoids)

    elif costType == "modularity":
        # If the points are named, display the names
        if namedPoints == True:
            named_medoids = {}
            for medID in medoids_idx:
                named_medoids[data[medID]] = []
                for pointID in medoids[medID]:
                    named_medoids[data[medID]].append(data[pointID])
            # "-" because we maximize modularity
            mod = -modularity(data, COST=costF_idx, distDict=distDict, edgeDict=affinities, medoids=named_medoids)
        else:
            mod = -modularity(data, COST=costF_idx, distDict=distDict, edgeDict=affinities, medoids=medoids)
        print "modularity computed"

    else:
        print "unknown target function"
        return(1)

    if len(assignErrors) > 0:
        print "unassigned nodes: ", assignErrors
    else:
        print "no unassigned nodes, all right"

    return(mod, medoids)

def modularity(data, COST=4, distDict={}, edgeDict={}, medoids={},
               edges_are_shortest_paths = False):  # as yet, only for named points
    '''
    Algorithm for modularity computation in order
    to assess clustering performance
    '''
    # modularity itself
    Q = 0.0
    # total number of edges (times 2, since edges are counted twice)
    M = 0.0
    degree = {}

    # if edges of the graph in hand are constructed from shortest paths in the
    # original graph ...   (rarely used)
    if edges_are_shortest_paths == True:
        iDist = 0.0
        iFuzzyAff = {}   # kinda affinity (not 1/(1/aff),  but reciprocal to average shortest paths on edges 1/aff)
        for i in data:
            iDist = distDict[i].values()
            iFuzzyAff[i] = [1.0/j for j in iDist if j!=0]
            # weighted degree
            degree[i] = sum(iFuzzyAff[i])
            # since edges are shortest paths on affinities:
            M += degree[i]

        for med in medoids:
            for pointA in medoids[med]:
                for pointB in medoids[med]:    # Should one include pointA == pointB terms ??? FIX IT
                    if pointA != pointB:
                        # 0.5 due to double computation (modularity of node pairs
                        # ab and ba enter the sum separately)
                        Q += 0.5*( 1.0 / distDict[pointA][pointB] - degree[pointA]*degree[pointB] / (2*0.5*M) )
        Q = Q / (2*0.5*M)

    # if edges of the graph in hand are edges from original graph
    else:
        for i in data:
            # weighted degree
            degree[i] = sum(edgeDict[i].values())
            M += degree[i]
        print "M: ", M

        for med in medoids:
            #print "current Q: ", Q
            for pointA in medoids[med]:
                #print "current Q: ", Q
                for pointB in medoids[med]:    # Should one include pointA == pointB terms ??? FIX IT ...FIXED! yes, it should!
                    #if pointA != pointB:
                        try:
                            Aab = edgeDict[pointA][pointB]
                        except:
                            Aab = 0
                        # 0.5 due to double computation (modularity of node pairs
                        # ab and ba enter the sum separately)
                        Q += 0.5*( Aab - degree[pointA]*degree[pointB] / (2*0.5*M) )
                        #print "current delta Q: ", 0.5*( Aab - degree[pointA]*degree[pointB] / (2*0.5*M) )

        Q = Q / (2*0.5*M)
        print "final modularity Q: ", Q
    return(Q)

def intra_cluster_centrality(data, COST=4, distDict={}, medoids={},
                             namedPoints = True):  # as yet, only for direct distance and for named points
    '''
    searching the most important points/nodes in the clusters based
    on certain centrality measures
    '''
    # Harmonic centrality  --  add more measures in future!
    harmCentr = {}
    harmCentrRaw = {}
    def getCentr(item):
        return item[1]
    #print "MEDOIDS: ", medoids
    for med in medoids:
        harmCentrRaw[med] = {}
        #print "MED: ", med, "\n"
        for point in medoids[med]:
            #print "point: ", point
            harmCentrRaw[med][point] = 0    # integer: less memory at the cost of precision loss
            for aim in medoids[med]:
                if point != aim:
                    harmCentrRaw[med][point] += float(1.0/distDict[point][aim])
        harmCentr[med] = sorted(harmCentrRaw[med].items(), key=getCentr, reverse=True)

    print "Harmonic centralities obtained"
    return(harmCentr)


def clara(data, k, COST=0, distDictClara={}, simDictClara={},
          affinities={}, bagSize=BagSize, namedPoints=True,
          degreeSelection=degreeSelection, claraLoopNum=claraLoopNum,
          noIsolates=True, saveAllResults=False, acceleration=0, take_all_nodes=False):
    '''
    CLARA implemenation
    1. For i = 1 to 5, repeat the following steps:
    2. Draw a sample of 40 + 2k objects randomly from the
        entire data set,2 and call Algorithm PAM to find
        k medoids of the sample.
    3. For each object Oj in the entire data set, determine
        which of the k medoids is the most similar to Oj.
    4. Calculate the average dissimilarity of the clustering
        obtained in the previous step. If this value is less
        than the current minimum, use this value as the
        current minimum, and retain the k medoids found in
        Step 2 as the best set of medoids obtained so far.
    5. Return to Step 1 to start the next iteration.
    '''
    size = len(data)
    min_cost = float('inf')
    best_choice = []
    best_res = {}
    sampling_idx = []
    cost_list = []
    isolates = []

    print "clara COST: ", COST

    print "take all nodes: ", take_all_nodes
    if take_all_nodes:
        bagSize = len(affinities)

    if saveAllResults:
        allResults=[]

    def IDtoname(data, best_med):
        best_med_names = {}
        best_choice_names = []
        for medID in best_med.keys():
            #best_choice_names.append(data[medID])
            best_med_names[data[medID]] = []
            for pointID in best_med[medID]:
                best_med_names[data[medID]].append(data[pointID])
        #best_choice = best_choice_names
        best_med = best_med_names
        return best_med

    # if degreeSelection == True, then 4*k nodes with highest degree will be
    # sampled anyway (the rest of nodes for the subsample (the number is Bag_Size)
    # are sampled as usual)
    if degreeSelection:
        degree = {}
        sampling_data_permanent = []
        sampling_data = []
        def getDegree(item):
            return item[1]

        # Compute sorted list of node degrees
        for i in list(data): # in order to not run out of range
            # remove singletons (treat them as isolates)
            # first compute weighted degree
            # TODO: find out the reasons for it
            try:
                degree[i] = sum(affinities[i].values())
            except KeyError:
                print "not in affinities: ", i
                degree[i] = 0
            #print "degree of %s is %f" % (i, degree[i])
            if degree[i]==0.0 and noIsolates == True:
                #print "~~~~~~~~~~EUREKA!!!!~~~~~~~~~~"
                isolates.append(i)
                data.remove(i)

            # Then remove paired nodes the same way as singletons.
            # As algorithm doesn't require absolute connectivity,
            # we deliberately leave components with 3+ nodes, hoping
            # that they will appear to be separate clusters after
            # averaging multiple randomized clara's results with kopt.py.
        for i in list(data):
            try:
                if len(affinities[i])==1 and len(affinities[affinities[i].keys()[0]])==1:
                    isolates.append(i)
                    data.remove(i)
            except: print "LOOK OUT: ", len(affinities[i]), affinities[i]
        # list rather than dict because dict cannot be ordered
        degree = sorted(degree.items(), key=getDegree, reverse=True)
        print "degrees obtained"

        # Obtain the bag of most prominent nodes for Clara clustering
        limit = k*4
        if k*4 > len(data):
            print "used up all data points for degree selection: %d points instead of 4k = %d" %(len(data), 4*k)
            limit = len(data)
        for point in degree:
            if len(sampling_data_permanent) >= limit:
                break
            sampling_data_permanent.append(point[0])
        therest = [point for point in data if point not in sampling_data]
        print "len(therest): ", len(therest)
        print "len(data): ", len(data)
        print "bagSize , bagSize - k*4: ", bagSize, bagSize-k*4

    iterspot = 0
    for i in range(claraLoopNum):
        iterspot += 1
        print "\n\nRUN No.", iterspot
        # Construct the sampling subset
        if degreeSelection == False:
            sampling_data = []
            sampling_idx = random.sample([i for i in range(size)], bagSize)
            for idx in sampling_idx:
                sampling_data.append(data[idx])
        else:
            sampling_data = list(sampling_data_permanent)
            sampling_idx = random.sample([i for i in range(len(therest))], bagSize-k*4)
            for idx in sampling_idx:
                sampling_data.append(therest[idx])
        print "all nodes/points:                  ", len(sampling_data)
        print "permanently selected nodes/points: ", len(sampling_data_permanent)
        # Run kmedoids for the sampling
        pre_cost, pre_choice, pre_medoids = kmedoids(sampling_data, k, COST, distDictKM=distDictClara,
                                                     simDictKM=simDictClara, namedPoints=False,
                                                     acceleration=acceleration)
        if debugEnabled == True:
            print('pre_cost: ', pre_cost)
            print('pre_choice: ', pre_choice)
            print('pre_medoids: ', pre_medoids) # pre_medoids are not too long to display

        # Convert the pre_choice from sampling_data to the whole data
        pre_choice2 = []
        for idx in pre_choice:
            #print sampling_data[idx]
            idx2 = data.index(sampling_data[idx])
            pre_choice2.append(idx2)
        if debugEnabled == True:
            print('pre_choice2: ', pre_choice2)

        # Clustering for all data set
        tmp_cost, tmp_medoids = targetFunction(data, COST, pre_choice2, distDict=distDictClara,
                                               simDict=simDictClara, affinities=affinities)
        cost_list.append(tmp_cost)
        if debugEnabled == True:
            print 'tmp_cost: ', tmp_cost
            print 'tmp_medoids: ', 'OK' #tmp_medoids)

        # If the points are named, display the names (refactor it)
        if namedPoints:
            tmp_medoids = IDtoname(data, tmp_medoids)
            pre_choice2 = tmp_medoids.keys()

        # Update the best
        if tmp_cost <= min_cost:
            min_cost = tmp_cost
            best_choice = list(pre_choice2)
            best_res = dict(tmp_medoids)

        if saveAllResults:
            allResults.append(tmp_medoids)

    if saveAllResults:
        return(min_cost, best_choice, best_res, cost_list, isolates, allResults)
    else:
        return(min_cost, best_choice, best_res, cost_list, isolates)


def main():
    '''
    Main function for Clara
    '''
    print "sys.argv: ", sys.argv
    print "len(sys.argv): ", len(sys.argv)
    #if len(sys.argv) not in [4,6,7,8]:
    #    print('Error: invalid number of parameters. Your parameters should be: \n path_to_node_names  k  cost_type [path_to_distance_matrix [path_to_similarity_matrix]  path_to_edge_matrix_(affinity)]')
    #    return(1)

    if len(sys.argv) == 4 and sys.argv[3] == 3:
        print('Error: cost based on distance/similarity matrix without the matrix specified')
        return(1)

    # Get the parameters
    try:
        splittedSim = sys.argv[8]
    except:
        splittedSim = False
        print "no value for splittedSim parameter specified, so it is set to False"
    filePath = sys.argv[1]
    k = int(sys.argv[2])
    COST = int(sys.argv[3])
    if len(sys.argv) >= 6:
        if COST == 3:
            distDictPath = sys.argv[4]
            affinityPath = sys.argv[5]
            print "distDictPath: ", distDictPath
            print "affinityPath: ", affinityPath
        elif COST == 4:
            distDictPath = sys.argv[4]
            simDictPath = sys.argv[5]
            affinityPath = sys.argv[6]
            try:
                acceleration = int(sys.argv[7])
                print "acceleration degree: ", acceleration
            except: acceleration = 0
            print "distDictPath: ", distDictPath
            print "simDictPath: ", simDictPath
            print "affinityPath: ", affinityPath
        else:
            print "Error: I dunno if you pass 'similarity' (COST=4) or you pass 'distance matrix' (COST=3)"
            return(1)
        #affinityPath = sys.argv[5]

    if debugEnabled == True:
        print 'filePath: ', filePath
        print 'k: ', k
        print "Cost Function: ", COST

    # Run Clara
    if COST in [3,4]:
        affinitiesOur = importData(affinityPath, ifjson=1)
        print "\n affinities imported"
        data = importData(filePath, ifjson=1)
        distDictOur = importData(distDictPath, ifjson=1)
        #distDictOur = None
        print "\n pairwise distances imported"
        if COST == 4:
            #simDictOur = importData(simDictPath, ifjson=1)
            if not splittedSim:
                simDictOur = {}
                with open(simDictPath, 'r') as S:
                    for line in S:
                        simDictOur.update(json.loads(line))
            else:
                simDictOur = load_splitted_sim(simDictPath, coresLoaded)
            print "\n pairwise similarities imported\n"
    else:
        data = importData(filePath)
    if debugEnabled == True:
        for i in range(10):
            print('example_data=', data[i])

    # Add timing
    startTime = time.time()
    if COST == 3:
        best_cost, best_choice, best_medoids, cost_list, isolates = clara(
            data, k, COST, distDictClara=distDictOur, simDictClara={},
            affinities=affinitiesOur, saveAllResults=False, acceleration=acceleration,
            take_all_nodes=takeAllNodes)
    elif COST == 4:
        best_cost, best_choice, best_medoids, cost_list, isolates = clara(
            data, k, COST, distDictClara=distDictOur, simDictClara=simDictOur,
            affinities=affinitiesOur, saveAllResults=False, acceleration=acceleration,
            take_all_nodes=takeAllNodes)
    else:
        best_cost, best_choice, best_medoids, cost_list, isolates = clara(
            data, k, COST, saveAllResults=False,
            take_all_nodes=takeAllNodes)
    endTime = time.time()

    mod = None
    harmonic_centrality = None
    '''
    #don't delete!!!!!!!!!!!!~~~~~
    # Compute modularity and display it
    startMod = time.time()
    if COST in [3,4]:
        mod = modularity(data, COST=COST, distDict=distDictOur,
                         edgeDict=affinitiesOur, medoids=best_medoids)
    else:
        print "no modularity for this regime, sorry"
        pass
    endMod = time.time()
    # comment up to this ~~~~~~~~~~

    startCentr = time.time()
    harmonic_centrality =  intra_cluster_centrality(data, COST=COST,
                            distDict=distDictOur, medoids=best_medoids)
    endCentr = time.time()
    '''
    # Save the result
    fordump = [best_cost, best_choice, best_medoids, mod, harmonic_centrality, isolates]
    json_filename = "clara_json_version_" + str(int(time.time())/10)[3:] + "_one" #time.strftime("%d %b %Y %H:%M:%S", time.gmtime())
    json.dump(fordump, open(json_filename, "w"))

    # Print the result
    diff = (endTime - startTime)
    best_cluster_length = []
    for i in best_choice:
        best_cluster_length.append(len(best_medoids[i]))

    print '\n\n'
    print 'best_time: ', diff
    #print 'best_modularity_time: ', endMod - startMod
    #print 'best_centrality_time: ', endCentr - startCentr
    #print 'best_modularity: ', mod
    print 'best_cost: ', best_cost
    print 'best_choice: ', best_choice
    print 'best_cluster_lengths: ', best_cluster_length
    print 'clustered_nodes: ', sum(best_cluster_length)
    #print 'all_nodes: ', len(data)
    print 'isolates: ', len(isolates), isolates
    print '\n\n'
    print 'best_medoids: ', best_medoids
    print '\n\n'
    #print 'sorted_harmonic_centrality: ', harmonic_centrality[best_choice[0]]

if __name__ == '__main__':
    main()
