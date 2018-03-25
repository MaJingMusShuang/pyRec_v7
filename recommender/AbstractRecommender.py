import numpy as np
import tensorflow as tf
import time
from recMath.RatingMatrix import RatingMatrix
import random


class AbstractRecommender:
    def __init__(self, config, dataModel, evaluator=None):
        self.isRanking = False
        self.numUser = dataModel.numUser
        self.numItem = dataModel.numItem

        self.preferenceMatrix = dataModel.preferenceMatrix
        self.trainMatrix = dataModel.trainMatrix
        self.minRating = dataModel.minRating
        self.maxRating = dataModel.maxRating
        self.globalMean = dataModel.globalMean
        self.testMatrix = dataModel.testMatrix

        self.testUserIdxs_warm_start = []
        self.testUserIdxs_cold_start = []
        for userIdx in self.testMatrix.userIdxs:
            if userIdx not in self.trainMatrix.userIdxs:
                self.testUserIdxs_cold_start.append(userIdx)
            else:
                self.testUserIdxs_warm_start.append(userIdx)

        self.batch_size = dataModel.batch_size
        # if self.batch_size:
        #     self.batch_num = int(self.trainMatrix.size // self.batch_size) + 1


        self.optimizer = None
        self.evaluator = evaluator
        self.config = config

        self.learnRate = config.get('learnRate')
        self.isBasedGlobalMean = config.get('isBasedGlobalMean')

        # train model config
        self.optiType = config.get('optiType')
        self.lossType = config.get('lossType')
        self.outputType = config.get('outputType')
        self.maxIter = config['maxIter'] if config.get('maxIter') else 100
        self.corruption_rate = config.get('corruption_rate')

        if self.corruption_rate:
            self.corruptedTrainMatrix = self.getCorruptedTrainMatrix()


        #pred
        self.predUserItemRating = {} #for users in test set
        for (userIdx, itemIdx) in self.preferenceMatrix:
            self.predUserItemRating[userIdx, itemIdx]=self.globalMean

        # evaluation
        self.evaluator = evaluator
        self.rmse = None
        self.mae = None
        self.rmse_cold_start = None
        self.mae_cold_start = None
        self.map = None
        self.ndcg = None

        self.min_epoch_rmse = 0
        self.min_batchId_rmse = 0
        self.min_rmse = 100

        self.min_epoch_mae = 0
        self.min_batchId_mae = 0
        self.min_mae = 100

        self.min_epoch_rmse_cold_start = 0
        self.min_batchId_rmse_cold_start = 0
        self.min_rmse_cold_start = 100

        self.min_epoch_mae_cold_start = 0
        self.min_batchId_mae_cold_start = 0
        self.min_mae_cold_start = 100


        self.max_epoch_map = 0
        self.max_batchId_map = 0
        self.max_map = -1

        self.max_epoch_ndcg = 0
        self.max_batchId_ndcg = 0
        self.max_ndcg = -1

        self.max_epoch_recall = 0
        self.max_batchId_recall= 0
        self.max_recall = -1

        self.topK = config.get('topK')


    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        sum_rmse = 0
        sum_mae = 0
        testSize = self.testMatrix.size
        for userIdx in self.testMatrix.userIdxs:
            itemIdxs = self.testMatrix.getItemsOfUser(userIdx)
            for itemIdx in itemIdxs:
                realRating = self.testMatrix.userItem_rating[userIdx, itemIdx]
                predictRating = self.predUserItemRating[userIdx, itemIdx]
                predictRating = self.minRating if predictRating < self.minRating else predictRating
                predictRating = self.maxRating if predictRating > self.maxRating else predictRating
                sum_rmse += (realRating - predictRating)**2
                sum_mae += np.abs(realRating - predictRating)
                # print('(user{}, item{}, label:{}):{}'.format(userIdx, itemIdx, realRating, predictRating))

        self.rmse = (sum_rmse / testSize) ** 0.5
        self.mae = sum_mae / testSize




    def getCorruptedTrainMatrix(self):
        corruptedTrainMatrix = RatingMatrix()
        for (userIdx, itemIdx) in self.preferenceMatrix.userItem_rating:
            rdm = random.uniform(0, 1)
            rating = self.preferenceMatrix.userItem_rating[userIdx, itemIdx]
            if rdm <= self.corruption_rate:
                if self.outputType=='rating':
                    corruptedTrainMatrix.put(userIdx, itemIdx, rating+random.uniform(-1, 1))
                else:
                    corruptedTrainMatrix.put(userIdx, itemIdx, 0)
            else:
                corruptedTrainMatrix.put(userIdx, itemIdx, rating)
        return corruptedTrainMatrix

    def getUserRatesVector_Corrupted(self, userIdx):
        implicit_vec = np.zeros(shape=self.numItem, dtype=np.short)
        if userIdx in self.corruptedTrainMatrix.user_items:
            items = self.corruptedTrainMatrix.user_items[userIdx]
            for itemIdx in items:
                implicit_vec[itemIdx] = self.corruptedTrainMatrix.getRating(userIdx, itemIdx)
        return implicit_vec

    def getUserImplicitVector(self, userIdx):
        implicit_vec = np.zeros(shape=self.numItem, dtype=np.short)
        if userIdx in self.trainMatrix.user_items:
            items = self.trainMatrix.user_items[userIdx]
            for itemIdx in items:
                implicit_vec[itemIdx] = 1
        return implicit_vec


    def getUserImplicitVectors(self, batch_u):
        """
        :param batch_u: [[0],[1],...] user idxs in this batch
        :return:
        """
        implicitVecs = []
        for i in range(len(batch_u)):
            thisUserIdx = batch_u[i][0]
            implicitVec = self.getUserImplicitVector(thisUserIdx)
            implicitVecs.append(implicitVec)
        return implicitVecs

    def getUserRatesVector(self, userIdx):
        rates_vec = np.zeros(shape=self.numItem, dtype=np.float32)
        items = self.trainMatrix.getItemsOfUser(userIdx)
        if self.isBasedGlobalMean:
            for itemIdx in items:
                rates_vec[itemIdx] = self.trainMatrix.getRating(userIdx, itemIdx) - self.globalMean
        else:
            for itemIdx in items:
                rates_vec[itemIdx] = self.trainMatrix.getRating(userIdx, itemIdx)
        return rates_vec


    def getUserRatesVectorTest(self, userIdx):
        rates_vec = np.zeros(shape=self.numItem, dtype=np.short)
        if userIdx in self.trainMatrix.user_items:
            items = self.trainMatrix.user_items[userIdx]
            for itemIdx in items:
                rates_vec[itemIdx] = self.trainMatrix.getRating(userIdx, itemIdx)
        return rates_vec

    def getUserImplicitVectorTest(self, userIdx):
        implicit_vec = np.zeros(shape=self.numItem, dtype=np.short)
        if userIdx in self.testMatrix.user_items:
            items = self.testMatrix.user_items[userIdx]
            for itemIdx in items:
                implicit_vec[itemIdx] = 1
        return implicit_vec


    def getItemImplicitVector(self, itemIdx):
        implicit_vec = np.zeros(shape=self.numUser, dtype=np.short)
        if itemIdx in self.trainMatrix.item_users:
            users = self.trainMatrix.item_users[itemIdx]
            for userIdx in users:
                implicit_vec[userIdx] = 1
        return implicit_vec


    def getItemImplicitVectors(self, batch_i):
        implicitVecs = []
        for i in range(len(batch_i)):
            thisItemIdx = batch_i[i][0]
            implicitVec = self.getItemImplicitVector(thisItemIdx)
            implicitVecs.append(implicitVec)
        return implicitVecs

    def getPredLists(self, test_u, topK_predLists):
        """
        :param test_u: np.array[[1], [2], [3],...] user idxs
        :param topK_predLists:
        :return: dict {userIdx: [topK itemIdxs]}
        """
        predLists = {}
        for i in range(len(test_u)):
            userIdx = test_u[i][0]
            if userIdx not in predLists:
                predLists[userIdx] = topK_predLists[i]
            predLists[userIdx] = topK_predLists[i]
        return predLists


    def getTrainUIR(self, batchId):
        """
        :param batchId:
        :return: type np.array (userIdxs, itemIdxs, ratings)
        """
        start = batchId * self.batch_size
        end = start + self.batch_size
        if end > self.trainSize:
            end = self.trainSize

        batch_u = self.trainSet[start:end, 0:1].astype(np.int32)
        batch_i = self.trainSet[start:end, 1:2].astype(np.int32)
        batch_r = self.trainSet[start:end, 2:3]
        return batch_u, batch_i, batch_r


    def getTestUIR(self):
        test_u = self.testSet[:, 0:1].astype(np.int32)
        test_i = self.testSet[:, 1:2].astype(np.int32)
        test_r = self.testSet[:, 2:3]
        return test_u, test_i, test_r


    def saveMinRmse(self, epoch, batchId, rmse):
        if rmse < self.min_rmse:
            self.min_epoch_rmse = epoch
            self.min_batchId_rmse = batchId
            self.min_rmse = rmse

    def saveMinMae(self, epoch, batchId, mae):
        if mae < self.min_mae:
            self.min_epoch_mae = epoch
            self.min_batchId_mae = batchId
            self.min_mae = mae

    def saveMinRmse_coldStart(self, epoch, batchId, rmse):
        if rmse < self.min_rmse_cold_start:
            self.min_epoch_rmse_cold_start = epoch
            self.min_batchId_rmse_cold_start = batchId
            self.min_rmse_cold_start = rmse

    def saveMinMae_coldStart(self, epoch, batchId, mae):
        if mae < self.min_mae_cold_start:
            self.min_epoch_mae_cold_start = epoch
            self.min_batchId_mae_cold_start = batchId
            self.min_mae_cold_start = mae

    def saveMaxMap(self, epoch, batchId, map):
        if map > self.max_map:
            self.max_epoch_map = epoch
            self.max_batchId_map = batchId
            self.max_map = map

    def saveMaxNdcg(self, epoch, batchId, ndcg):
        if ndcg > self.max_ndcg:
            self.max_epoch_ndcg = epoch
            self.max_batchId_ndcg = batchId
            self.max_ndcg = map

    def saveMaxRecall(self, epoch, batchId, recall):
        if recall > self.max_recall:
            self.max_epoch_recall = epoch
            self.max_batchId_recall = batchId
            self.max_recall = recall

    def getActiv(self, input, typeString):
        if typeString == 'tanh':
            return tf.nn.tanh(input)
        elif typeString == 'relu':
            return tf.nn.relu(input)
        elif typeString == 'sigmoid':
            return tf.nn.sigmoid(input)
        elif typeString == 'identical':
            return input
        else:
            return None


    def getOptimizer(self):
        if self.optiType == 'gd':
            return tf.train.GradientDescentOptimizer(self.learnRate, name='GD_optimizer').minimize(self.cost)
        elif self.optiType == 'adam':
            return tf.train.AdamOptimizer(self.learnRate, name='Adam-optimizer' + str(time.time())).minimize(self.cost)
        elif self.optiType == 'adadelta':
            return tf.train.AdadeltaOptimizer(self.learnRate, name='Adadelta_optimizer').minimize(self.cost)
        elif self.optiType == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.learnRate, name='RMSProp_optimizer').minimize(self.cost)
        else:
            return None