
from util import AutoVivification, importData, pearson_distance, euclidean_distance, manhattan_distance, direct_distance, similarity_distance
import random
import sys
import time
import json

# Global variables
initMedoidsFixed = False  # Fix the init value of medoids for performance comparison
debugEnabled = True  # Debug
#namedPoints = True
CacheOn=False
distances_cache = {}  # Save the tmp distance for acceleration (idxMedoid, idxData) -> distance


def totalCost(data, costF_idx, medoids_idx, cacheOn=CacheOn, distDict={}, simDict={}, acceleration=0):
    '''
    Compute the total cost and do the clustering based on certain cost function
    (that is, assign each data point to certain cluster given the medoids)
    '''
    # Init the cluster
    size = len(data)
    total_cost = 0.0
    medoids = {}
    for idx in medoids_idx:
        medoids[idx] = []
    # medoids['unassigned'] = []
    unassigned = []
    tmp = None

    # Compute the distance and do the clustering
    for i in xrange(size):
        choice = -1
        # Make a big number
        min_cost = float('inf')
        for m in medoids:
            if cacheOn == True:
                # Check for cache
                tmp = distances_cache.get((m, i), None)
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
                    try:
                        tmp = similarity_distance(data[m], data[i], simDict)
                    except:
                        print m, i
                        print data[m]
                        print data[i]
                else:
                    print('Error: unknown cost function idx: %d' % (costF_idx))
            if cacheOn == True:
                # Save the distance for acceleration
                distances_cache[(m, i)] = tmp
            # Clustering
            if tmp < min_cost:
                choice = m
                min_cost = tmp
        # Done the clustering
        if min_cost == 0:  # 0 similarity to all the medoids
            unassigned.append(i)  # medoids['unassigned'].append(i)
        else:
            medoids[choice].append(i)
        total_cost += min_cost

    if acceleration == 2:
        transformed_medoids = {} #dict(medoids)
        for i, m in enumerate(medoids.keys()):
            #print i, m
            transformed_medoids[str(i)] = {'med': m, 'nodes': medoids[m]}
            #transformed_medoids[i] = transformed_medoids.pop(m)
        return (total_cost, transformed_medoids)

    # Return the total cost and clustering
    return (total_cost, medoids )


def kmedoids(data, k, COST=0, distDictKM={}, simDictKM={}, namedPoints=True,
             acceleration=0, sampleSize=2, waitingTerm=2.):
    '''
    kMedoids - PAM implemenation
    See more : http://en.wikipedia.org/wiki/K-medoids
    The most common realisation of k-medoid clustering is the Partitioning Around Medoids (PAM) algorithm and is as follows:[2]
    1. Initialize: randomly select k of the n data points as the medoids
    2. Associate each data point to the closest medoid. ("closest" here is defined using any valid distance metric, most commonly Euclidean distance, Manhattan distance or Minkowski distance)
    3. For each medoid m
        For each non-medoid data point o
            Swap m and o and compute the total cost of the configuration
    4. Select the configuration with the lowest cost.
    5. repeat steps 2 to 4 until there is no change in the medoid.
    '''
    print "\n\nPAM acceleration: ", acceleration, type(acceleration)
    size = len(data)
    #medoids_idx = []
    if initMedoidsFixed == False:
        medoids_idx = random.sample([i for i in xrange(size)], k)
    else:
        medoids_idx = list(range(k))
    #print json.dumps(distDictKM)
    pre_cost, medoids = totalCost(data, COST, medoids_idx, cacheOn=CacheOn, distDict=distDictKM, simDict=simDictKM)
    if debugEnabled:
        print('pre_cost: ', pre_cost)
    #print('medoids: ', medoids)
    current_cost = pre_cost
    best_choice = []
    best_res = {}
    iter_count = 0

    maxStable = min( k * waitingTerm, 25)
    if acceleration == 0:

        while True:
            for m in medoids:
                for item in medoids[m]:
                    # NOTE: both m and item are idx!  (you meant from the same cluster?)
                    if item != m:
                        # Swap m and o - save the idx
                        idx = medoids_idx.index(m)
                        # This is m actually...
                        swap_temp = medoids_idx[idx]
                        medoids_idx[idx] = item
                        tmp_cost, tmp_medoids = totalCost(data, COST, medoids_idx, cacheOn=CacheOn,
                                                          distDict=distDictKM, simDict=simDictKM)
                        # Find the lowest cost
                        if tmp_cost < current_cost:
                            best_choice = list(medoids_idx)  # Make a copy
                            best_res = dict(tmp_medoids)  # Make a copy
                            current_cost = tmp_cost
                        # Re-swap the m and o
                        medoids_idx[idx] = swap_temp
            # Increment the counter
            iter_count += 1
            if debugEnabled == True:
                print('current_cost: ', current_cost)
                print('iter_count: ', iter_count)

            if best_choice == medoids_idx:
                # Done the clustering
                break

            # Update the cost and medoids
            if current_cost <= pre_cost:
                pre_cost = current_cost
                medoids = best_res
                medoids_idx = best_choice

    elif acceleration == 2:
        print medoids.keys()
        firstmed = dict(medoids)
        # firstkeys = list(medoids.keys())
        # transforming medoids
        for i, m in enumerate(medoids.keys()):
            medoids[m] = {'med': m, 'nodes': medoids[m]}
            medoids[str(i)] = medoids.pop(m)

        print medoids.keys()
        stableSequence = 0
        cycle_count = 0

        while True:
            maxStablePerCycle = 0
            for i in medoids:
                if len(medoids[i]['nodes']) <= sampleSize + 1:   # medoids[m]
                    sample_idx = range(len(medoids[i]['nodes']))
                else:
                    sample_idx = random.sample(xrange(len(medoids[i]['nodes'])), sampleSize)
                    #print "\nmedoids[m]: ", medoids[m]
                    #print "in sample_idx: ", sample_idx, [medoids[i]['nodes'][k] for k in sample_idx]

                try:
                    for item in (medoids[i]['nodes'][k] for k in sample_idx):
                        # NOTE: both m and item are idx!
                        if item != medoids[i]['med']:
                            # Swap m and o - save the idx
                            idx = medoids_idx.index(medoids[i]['med'])
                            # This is m actually...
                            swap_temp = medoids_idx[idx]
                            medoids_idx[idx] = item
                            tmp_cost, tmp_medoids = totalCost(data, COST, medoids_idx, cacheOn=CacheOn,
                                                            distDict=distDictKM, simDict=simDictKM,
                                                            acceleration=2)
                            # Find the lowest cost
                            if tmp_cost < current_cost:
                                best_choice = list(medoids_idx)  # Make a copy
                                best_res = dict(tmp_medoids)  # Make a copy
                                current_cost = tmp_cost
                            # Re-swap the m and o
                            medoids_idx[idx] = swap_temp
                except:    # delete this gotcha after debug
                    print "\n"
                    print "k, i: ", k, i
                    print "medoids[i]['nodes']: ", medoids[i]['nodes']
                    print medoids #",".join(medoids)
                    print "firstmed: ", firstmed
                    print "\n\n\n"
                    for i, m in enumerate(firstmed.keys()):
                        firstmed[m] = {'med': m, 'nodes': firstmed[m]}
                        firstmed[str(m)] = firstmed.pop(m)
                        firstmed[i] = firstmed.pop(str(m))
                        print "i,m: ", i, m
                        print firstmed
                        print "\n\n\n"
                    print "firstmed: ", firstmed
                    return 0

                iter_count += 1
                if debugEnabled == True:
                    #print 'current_cost: ', current_cost
                    #print 'iter_count: ', iter_count
                    pass

                if best_choice == medoids_idx:
                    stableSequence += 1
                    maxStablePerCycle = max(stableSequence, maxStablePerCycle)
                else:
                    stableSequence = 0
                print "stableSequence: ", stableSequence, maxStablePerCycle
                # if K clusters in a row cannot change partition for better,
                # algorithm stops

                # Update the cost and medoids
                if current_cost < pre_cost:
                    pre_cost = current_cost
                    #print "BEST RES: ", best_res.keys(), best_res[0]
                    medoids = best_res
                    medoids_idx = best_choice

            if maxStablePerCycle >= maxStable: # len(medoids) * waitingTerm:
                break

            cycle_count += 1
            if debugEnabled == True:
                print 'cycle_count: ', cycle_count
                print 'current_cost: ', current_cost
                print '\n'

    else:
        print "Error: unknown acceleration parameter value"
        return 1

    # If points are named, you'd rather see the names
    if namedPoints == True:
        best_res_names = {}
        best_choice_names = []
        for medID in best_choice:
            best_choice_names.append(data[medID])
            best_res_names[data[medID]] = []
            for pointID in best_res[medID]:
                best_res_names[data[medID]].append(data[pointID])
        best_choice = best_choice_names
        best_res = best_res_names

    return(current_cost, best_choice, best_res)


def main():
    '''
    Main function for PAM
    '''
    print sys.argv
    print len(sys.argv)
    if len(sys.argv) == 4 and sys.argv[3] == 3:
        print 'Error: cost based on distance matrix without distance matrix specified'
        return (1)

    if len(sys.argv) not in [4,5]:
        print 'Error: invalid number of parameters. Your parameters should be: \n path_to_node_names  k  cost_type  [pairwise_matrix_(distance_or_similarity)]'
        return (1)

    # Get the parameters
    filePath = sys.argv[1]
    k = int(sys.argv[2])
    COST = int(sys.argv[3])  # here it's obligatory, but in kmedoids(), optional. FIX IT
    if len(sys.argv) == 5:
        if COST == 4:   # k-medoids based on similarity matrix (e.g Jaccard score)
            simDictPath = sys.argv[4]
            print simDictPath
        elif COST == 3:   # k-medoids based on distance matrix (e.g. Average shortest path)
            distDictPath = sys.argv[4]
            print distDictPath # distDictPath is not a file - it's a path to file (string)
        else:
            print "Error: I dunno whether you pass affinities to compute 'similarity' (COST=4) or you pass 'distance matrix' (COST=3)"
            return(1)

    if debugEnabled == True:
        print('filePath: ', filePath)
        print('k: ', k)
        print('cost function number: ', COST)  # better yet, display the name, not number FIX IT

    # Run PAM for europe.txt
    distDictOur = {}
    simDictOur = {}
    if COST == 3:
        data = importData(filePath, ifjson=1)  # actually, ifjson=1 is not tantamount to  direct distance method FIX IT
        distDictOur = importData(distDictPath, ifjson=1)
        # print "distance: ", distDict.items()[0]
    elif COST == 4:
        data = importData(filePath, ifjson=1)
        simDictOur = importData(simDictPath, ifjson=1)
        print "pairwise similarities imported"
    else:
        data = importData(filePath)
    if debugEnabled:
        for i in range(10):
            print('data=', data[i])

    # Add timing here
    startTime = time.time()
    if COST not in [3,4]:
        best_cost, best_choice, best_medoids = kmedoids(data, k, COST)
    elif COST == 3:
        best_cost, best_choice, best_medoids = kmedoids(data, k, COST, distDictOur, simDictKM={})
    elif COST == 4:
        best_cost, best_choice, best_medoids = kmedoids(data, k, COST, distDictKM={}, simDictKM=simDictOur)
    endTime = time.time()

    # Saving the result into new file
    fordump = [best_cost, best_choice, best_medoids]
    json_filename = "pam_json_version " + str(int(time.time())/10)[3:]    # find normal time format FIX IT
    json.dump(fordump, open(json_filename, "w"))

    best_cluster_length = []
    for i in best_choice:
        try:
            best_cluster_length.append(len(best_medoids[i]))
        except KeyError:
            best_cluster_length.append("KeyError ;)")

    print 'best_time: ', (endTime - startTime)
    print 'best_cost: ', best_cost
    print 'best_choice: \n', best_choice
    print 'best_cluster_lengths: \n', best_cluster_length
    print 'best_cluster_contents: \n', best_medoids


if __name__ == '__main__':
    main()
