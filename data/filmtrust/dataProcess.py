from recMath.RatingMatrix import RatingMatrix
import time

userId_userIdx = {}
itemId_itemIdx = {}
ratingMat = RatingMatrix()

truster_trustee_pairs = []

def readRatings():
    """
    :return: rating matrix
    """
    baseUserIdx, baseItemIdx = 0, 0
    with open('ratings.txt') as file:
        lines = file.readlines()
    for line in lines:
        userId, itemId, rating = line.strip().split()
        rating = float(rating)
        if userId not in userId_userIdx.keys():
            userId_userIdx[userId]=baseUserIdx
            baseUserIdx += 1
        if itemId not in itemId_itemIdx.keys():
            itemId_itemIdx[itemId]=baseItemIdx
            baseItemIdx += 1
        ratingMat.put(userId_userIdx[userId], itemId_itemIdx[itemId], rating)


def readTrusts():
    """
    :return: truster-trustee pairs
    """
    with open('trusts.txt') as file:
        lines = file.readlines()
    for line in lines:
        trusterId, trusteeId, value = line.strip().split()
        if trusterId in userId_userIdx and trusteeId in userId_userIdx:
            truster_trustee_pairs.append([userId_userIdx[trusterId], userId_userIdx[trusteeId]])




def constructEdgeList():
    """
    :return: node-node pair, node can be user or item
    """
    edgeList = []
    edgeListToWrite = []

    readRatings()
    readTrusts()

    for pair in truster_trustee_pairs:
        edgeList.append(pair)

    offSet = ratingMat.numUser
    for userIdx, itemIdx in ratingMat:
        edgeList.append([userIdx, itemIdx+offSet])
    with open('edgeList.txt', 'w') as file:
        file.writelines('{} {}\n'.format(node1, node2) for node1, node2 in edgeList)

# constructEdgeList()
readRatings()
# userIdxs, itemIdxs, ratings = [], [], []
# for userIdx, itemIdx in ratingMat:
#     userIdxs.append(userIdx)
#     itemIdxs.append(itemIdx)
#     ratings.append(ratingMat.getRating(userIdx, itemIdx))
# from scipy.sparse.csc import csc_matrix
# csc_mat = csc_matrix((ratings, (userIdxs, itemIdxs)))
# coo_mat = csc_mat.tocoo()
from sklearn.utils import shuffle as skshuffle
begin = time.time()
skshuffle(ratingMat)
end = time.time()
print('{} done'.format(end-begin))




