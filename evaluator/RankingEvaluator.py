from multiprocessing.pool import ThreadPool
import time
import numpy as np

class RankingEvaluator:
    def __init__(self, groundTruthLists, topK):
        """
        :param groundTruthLists: {userId:[itemId1, itemId2,...], ...}
        :param user_items_train: {userId: [itemId1, itemId2, ...], ...}
        :param itemInTestSet: set() (itemIdx1, itemIdx2, ...)
        :param topK: Integer
        :param testMatrix: {(userId, itemId):rating ...}
        """
        self.groundTruthLists = groundTruthLists
        # self.user_items_train = user_items_train
        self.predLists = None
        self.indexRange = len(self.groundTruthLists)
        # self.itemInTestSet = itemInTestSet
        self.topK = topK
        # self.testMatrix = testMatrix
        self.pool = ThreadPool()

    def setPredLists(self, predLists):
        self.predLists = predLists
        # assert len(self.predLists) == len(self.groundTruthLists)

    def calPrecision(self):
        results = []

        for userIdx in self.groundTruthLists:
            results.append(self.pool.apply_async(self.calculate_a_precision, (userIdx,)).get())

        precisionSum = 0
        for res in results:
            # print('a precision', res)
            precisionSum += res
        end = time.time()
        # print(end-start)
        return precisionSum / len(results)

    def calculate_a_precision(self, userIdx):
        hitNum = 0
        realList = self.groundTruthLists[userIdx]
        predList = self.predLists[userIdx]

        for itemId in predList:
            if itemId in realList:
                hitNum += 1
        return hitNum / self.topK

    '''Recall'''

    def calRecall(self):
        results = []
        for userIdx in self.groundTruthLists:
            results.append(self.calculate_a_Recall(userIdx))

        recallSum = 0
        for result in results:
            recallSum += result

        return recallSum / len(results)

    def calculate_a_Recall(self, userIdx):
        hitNum = 0
        userTrueList = self.groundTruthLists[userIdx]
        userPredList = self.predLists[userIdx]

        for itemId in userPredList:
            if itemId in userTrueList:
                hitNum += 1
        return hitNum / len(userTrueList)

    def calMAP(self):
        """
        :return: Mean average precision
                    precision@k * rel(k) / min(topk, Iu)
        """
        results = []

        for userIdx in self.groundTruthLists:
            results.append(self.pool.apply_async(self.calculate_average_precision, (userIdx,)).get())

        precisionSum = 0
        for res in results:
            precisionSum += res
        return precisionSum / len(results)

    def calculate_average_precision(self, userIdx):
        user_realList = self.groundTruthLists[userIdx]
        user_predList = self.predLists[userIdx]
        sum_precisionAtK = 0

        hitNum = 0

        for indexOfItem in range(self.topK):
            if user_predList[indexOfItem] in user_realList:
                hitNum += 1
                sum_precisionAtK += 1.0 * hitNum / (indexOfItem + 1)
        average_precision = sum_precisionAtK / min(self.topK, len(user_realList))
        return average_precision


    # def calculate_precisionAtK(self, userIdx, k):
    #     topK_predList = self.predLists[userIdx][0:k]
    #     realList = self.groundTruthLists[userIdx]
    #
    #     hitNum = 0
    #     for itemId in topK_predList:
    #         if itemId in realList:
    #             hitNum += 1
    #     return hitNum / len(topK_predList)

    '''NDCG'''

    def calNDCG(self):
        # start = time.time()
        results = []

        for userIdx in self.groundTruthLists:
            results.append(self.pool.apply_async(self.calculate_a_NDCG, (userIdx,)))
        ndcgSum = 0
        for result in results:
            ndcgSum += result.get()

        if len(results) > 0:
            return ndcgSum / len(results)
        else:
            return 0


    def calculate_a_NDCG(self, userIdx):
        userRealList = self.groundTruthLists[userIdx]
        numUserRatedItems = len(userRealList)
        userTopK_PredList = self.predLists[userIdx]

        # calculate Ideal DCG
        IDCG = 0.0
        if numUserRatedItems >= self.topK:
            for idx in range(self.topK):
                IDCG += 1.0 / np.log2(idx + 2)
        else:
            for idx in range(numUserRatedItems):
                IDCG += 1.0 / np.log2(idx + 2)

        # calculate DCG
        DCG = 0.0
        for index in range(self.topK):
            itemIdx = userTopK_PredList[index]
            if itemIdx in userRealList:
                DCG += 1.0 / np.log2(index + 2)
        NDCG = DCG / IDCG
        # print('user {} topK={}: realList={} topK_predList={} IDCG={} DCG={} NDCG={}'.
        #       format(userIdx, topK, userRealList, userTopK_PredList, IDCG, DCG, NDCG))
        return NDCG




if __name__ == '__main__':
    user_items_train = {
        0: [0, 1, 2, 3, 4, 5],
        1: [0, 1, 2],
        2: [0, 1, 2]
    }
    groundTruthList = {
        0: [7, 8, 9],
        1: [7, 8, 9],
        2: [7, 8, 9]
    }
    topK = 3
    itemInTestSet = set()
    itemInTestSet.add(4)
    itemInTestSet.add(5)
    itemInTestSet.add(7)
    itemInTestSet.add(8)
    itemInTestSet.add(9)

    testMatrix = {
        # (0, 7): 5,
        # (0, 8): 4,
        # (0, 9): 3,
        (1, 7): 5,
        (1, 8): 4,
        (1, 9): 3,
        (2, 7): 5,
        (2, 8): 4,
        (2, 9): 3,
    }
    # predLists = {
    #     0: [7, 8],
    #     1: [7, 4],
    #     2: [4, 7]
    # }
    predLists = {
        0: [1, 2, 3],
        1: [7, 8, 9],
        2: [4, 7, 8]
    }
    evaluator = RankingEvaluator(groundTruthList, topK, )
    evaluator.setPredLists(predLists)
    print(evaluator.calNDCG())