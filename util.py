
from math import sqrt
import json


class AutoVivification(dict):
    # "dynamic" dictionary
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def importData(filePath='europe.txt', ifjson=0):
    if ifjson==0:
        data = []
        try:
            fnObj = open(filePath, 'r')
            for line in fnObj:
                line = line.strip().split()
                point = []
                for c in line:
                    point.append(float(c))
                data.append(tuple(point))
        finally:
            fnObj.close()
        return(data)
    elif ifjson==1:
        data = json.load(open(filePath))
        return(data)
    else:
        print "ifjson parameter can be either 0 or 1"
        return(1)


def euclidean_distance(vector1, vector2):
    dist = 0
    for i in range(len(vector1)):
        dist += (vector1[i] - vector2[i])**2
    return(dist)


def manhattan_distance(vector1, vector2):
    dist = 0
    for i in range(len(vector1)):
        dist += abs(vector1[i] - vector2[i])
    return(dist)


def pearson_distance(vector1, vector2):
    """
    Calculate distance between two vectors using pearson method
    See more : http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
    """
    sum1 = sum(vector1)
    sum2 = sum(vector2)

    sum1Sq = sum([pow(v,2) for v in vector1])
    sum2Sq = sum([pow(v,2) for v in vector2])

    pSum = sum([vector1[i] * vector2[i] for i in range(len(vector1))])

    num = pSum - (sum1*sum2/len(vector1))
    den = sqrt((sum1Sq - pow(sum1,2)/len(vector1)) * (sum2Sq - pow(sum2,2)/len(vector1)))

    if den == 0 : return 0.0
    return(1.0 - num/den)


def direct_distance(name1, name2, distDict):
    """
    Return the distance directly from distance matrix (e.g. in graphs).
    This version is suitable primarily for AVERAGE SHORTEST PATH-based distances.
    Distance should be dict of dicts, like the output of Floyd-Warshall algorithm in networkX
    """
    # negative sign for distance minimization
    dist = distDict[name1][name2]
    return(dist)


def similarity_distance(name1, name2, simDict):
    """
    Return the distance directly from distance matrix (e.g. in graphs).
    This version is suitable for SIMILARITY-based distances, like Jaccard score
    Distance should be dict of dicts, like the output of pairwise_similarities function.
    """
    # negative sign for distance minimization
    try:
        dist = -simDict[name1][name2]
    except KeyError:
        dist = 0
    return(dist)

