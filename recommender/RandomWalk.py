from .AbstractRecommender import AbstractRecommender
import recMath.graph as graph
from collections import defaultdict
import random
import time
from datamodel.SocialDataModel import SocialDataModel


class RandomWalk(AbstractRecommender):
    def __init__(self, config, dataModel):
        super(RandomWalk, self).__init__(config, dataModel)
        self.G = graph.Graph()
        self.dataModel = dataModel
        self.logger = dataModel.logger
        self.walks = defaultdict(list)
        self.predLists = {}
        self.numUser = self.preferenceMatrix.numUser

    def __constructEdgeList(self):
        """
        :return: node-node pair, node can be user or item
        """
        edgeList = []

        if type(self.dataModel) == SocialDataModel :
            for pair in self.dataModel.wholeTrustDataSet:
                edgeList.append(pair)


        for userIdx, itemIdx in self.trainMatrix:
            edgeList.append([userIdx, itemIdx + self.numUser])
        return edgeList

    def buildGraph(self):
        edges = self.__constructEdgeList()
        for edge in edges:
            self.G[edge[0]].append(edge[1])
            self.G[edge[1]].append(edge[0])

        self.logger.info('nodes:{} edges:{}'.format(len(self.G.nodes()), len(edges)))

    def build_random_walks(self, numWalks, walkLength):
        start = time.time()
        walks = defaultdict(list)
        nodes = list(self.G.nodes())
        for cnt in range(numWalks):
            rand = random.Random(time.time())
            rand.shuffle(nodes)
            for node in nodes:
                walks[node].extend(self.G.random_walk(walkLength, rand=rand, start=node))
        end = time.time()
        self.logger.info('numWalks:{}, walkLength:{}, time:{}'.format(numWalks, walkLength, end-start))
        return walks

    def predict(self, walks):
        for userIdx in self.testMatrix.userIdxs:
            if userIdx in self.trainMatrix.userIdxs:
                walk = walks[userIdx]
                itemIdx_freq = defaultdict(int)
                for node in walk:
                    if node >= self.numUser:
                        itemIdx_freq[node] += 1

                sorted_res = sorted(itemIdx_freq.items(), key=lambda item: item[1], reverse=True)
                topN = [item_freq[0]-self.numUser for item_freq in sorted_res[0:self.topK]]
                self.predLists[userIdx] = topN

    def calculate_a_precision(self, userIdx):
        hitNum = 0
        realList = self.testMatrix.getItemsOfUser(userIdx)
        predList = self.predLists[userIdx]

        for itemId in predList:
            if itemId in realList:
                hitNum += 1
        return hitNum / self.topK

    def evaluate(self):
        precisionSum = 0
        counter = 0

        for userIdx in self.testMatrix.userIdxs:
            if userIdx in self.trainMatrix.userIdxs:
                precisionSum += self.calculate_a_precision(userIdx)
                counter += 1
        avgPrecision = precisionSum / counter
        self.logger.info('precision:{}'.format(avgPrecision))
        return avgPrecision


