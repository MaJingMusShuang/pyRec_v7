import tensorflow as  tf
import numpy as np
import time
import math
import random

class BiasedMF:
    def __init__(self, config, dataModel):
        self.name = 'BiasedMF'
        # evaluation
        self.min_epoch_rmse = 0
        self.min_batchId_rmse = 0
        self.min_epoch_mae = 0
        self.min_batchId_mae = 0
        self.min_epoch_rmseBounded = 0
        self.min_batchId_rmseBounded = 0
        self.min_epoch_maeBounded = 0
        self.min_batchId_maeBounded = 0
        self.min_rmse = 100
        self.min_mae = 100
        self.min_bounded_rmse = 100
        self.min_bounded_mae = 100
        self.resultList = None

        # model hyper parameter
        self.numFactor = config['numFactor']
        self.alpha = config['alpha'] # ???
        self.lam1 = config['lam1']
        self.lam2 = config['lam2']
        self.learnRate = config['learnRate']
        self.maxIter = config['maxIter']
        self.batchSize = config['batchSize']
        self.trainType = config['trainType']
        self.optiType = config['optiType']
        self.outputType = config['outputType']

        # data model config
        self.logger = dataModel.logger
        self.numUser = dataModel.numUserBeforeSplit
        self.numItem = dataModel.numItemBeforeSplit
        self.minRating = dataModel.minRating
        self.maxRating = dataModel.maxRating
        self.trainSet = np.array(dataModel.trainSet)
        self.validSet = np.array(dataModel.validSet)
        self.testSet = np.array(dataModel.testSet)
        self.trainValidSet = np.array(dataModel.trainValidSet)
        self.trainSize = len(self.trainSet)
        self.trainValidSize = len(self.trainValidSet)
        self.user_items_Train = dataModel.user_items_Train
        self.user_items_TrainValid = dataModel.user_items_TrainValid

        self.globalMean = 0
        if self.trainType == 'test':
            self.globalMean = dataModel.trainValidGlobalMean
        else:
            self.globalMean = dataModel.trainGlobalMean

        self.batch_num = 0
        if self.trainType == 'test':
            self.batch_num = int(self.trainValidSize // self.batchSize) + 1
        else:
            self.batch_num = int(self.trainSize // self.batchSize) + 1

        # placeholders
        self.u_input = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.i_input = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.r_label = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # weights
        scale = 1 / math.sqrt(self.numUser)
        tf.set_random_seed(config['seed'])
        random.seed(config['seed'])
        self.weights ={
            'u_factor': tf.Variable(tf.random_uniform([self.numUser, self.numFactor], -scale, scale),
                                    name='user_factor', dtype=tf.float32),
            'i_factor': tf.Variable(tf.random_uniform([self.numItem, self.numFactor], -scale, scale),
                                    name='item_factor', dtype=tf.float32),
            'u_bias': tf.Variable(tf.random_uniform([self.numUser], -scale, scale),
                                    name='user_bias', dtype=tf.float32),
            'i_bias': tf.Variable(tf.random_uniform([self.numItem], -scale, scale),
                                  name='item_bias', dtype=tf.float32),
            'implicit_vectors': tf.Variable(tf.random_uniform([self.numItem, self.numFactor], -scale, scale),
                                  name='implicit_vectors', dtype=tf.float32),
        }

        self.r_pred = None
        self.cost = None
        self.rmse = None
        self.mae = None
        self.optimizer = None

    def buildModel(self):

        u_factor = tf.reshape(tf.nn.embedding_lookup(self.weights['u_factor'], self.u_input), [-1, self.numFactor])
        i_factor = tf.reshape(tf.nn.embedding_lookup(self.weights['i_factor'], self.i_input), [-1, self.numFactor])

        u_bias = tf.reshape(tf.nn.embedding_lookup(self.weights['u_bias'], self.u_input), [-1, 1])
        i_bias = tf.reshape(tf.nn.embedding_lookup(self.weights['i_bias'], self.i_input), [-1, 1])

        self.r_pred = tf.reduce_sum(tf.multiply(u_factor, i_factor), 1, keep_dims=True) + u_bias + i_bias + self.globalMean

        self.cost = tf.reduce_sum(tf.square(self.r_label - self.r_pred)) \
                    + self.lam1 * tf.nn.l2_loss(self.weights['u_factor']) \
                    + self.lam1 * tf.nn.l2_loss(self.weights['i_factor']) \
                    + self.lam2 * tf.nn.l2_loss(self.weights['u_bias']) \
                    + self.lam2 * tf.nn.l2_loss(self.weights['i_bias'])
        self.cost = self.cost * 0.5

        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.r_label - self.r_pred)))
        self.mae = tf.reduce_mean(tf.abs(self.r_label - self.r_pred))

    def trainModel(self):
        self.logger.info("--------------------Begin Training--------------------")
        self.printConfig()

        sess = tf.InteractiveSession()
        self.optimizer = self.getOptimizer()
        sess.run(tf.global_variables_initializer())

        if self.trainType == 'test':
            test_u, test_i, test_r = self.getTestData()
        else:
            test_u, test_i, test_r = self.getValidData()

        start_time = time.time()
        isNaN = False
        overFitCount = 0
        isOverFitting = False
        for epoch in range(self.maxIter):
            for batchId in range(self.batch_num):
                if self.trainType == 'test':
                    batch_u, batch_i, batch_r = self.getTrainValidData(batchId)
                else:
                    batch_u, batch_i, batch_r = self.getTrainData(batchId)

                self.optimizer.run(feed_dict={
                    self.u_input: batch_u,
                    self.i_input: batch_i,
                    self.r_label: batch_r
                })
                loss = self.cost.eval(feed_dict={
                    self.u_input: batch_u,
                    self.i_input: batch_i,
                    self.r_label: batch_r
                })
                rmse = self.rmse.eval(feed_dict={
                    self.u_input: test_u,
                    self.i_input: test_i,
                    self.r_label: test_r
                })
                mae = self.mae.eval(feed_dict={
                    self.u_input: test_u,
                    self.i_input: test_i,
                    self.r_label: test_r
                })
                predList = sess.run(self.r_pred, feed_dict={
                    self.u_input: test_u,
                    self.i_input: test_i
                })
                # print('BeforeBound', '\n', predList)
                self.boundRating(predList)
                # print('AfterBound:', '\n', predList)

                bounded_rmse = self.rmse.eval(feed_dict={
                    self.r_pred: predList,
                    self.r_label: test_r
                })

                bounded_mae = self.mae.eval(feed_dict={
                    self.r_pred: predList,
                    self.r_label: test_r
                })
                self.saveMinRmse(epoch, batchId, rmse)
                self.saveMinMae(epoch, batchId, mae)
                self.saveMinRmseBounded(epoch, batchId, bounded_rmse, predList)
                self.saveMinMaeBounded(epoch, batchId, bounded_mae)
                self.logger.info(
                    "[epoch:{} batchId:{}] loss:{:.4f}, rmse:{:.4f},mae:{:.4f},rmseBounded:{:.4f},maeBounded:{:.4f}"
                        .format(epoch, batchId, loss, rmse, mae, bounded_rmse, bounded_mae)
                )
                if math.isnan(bounded_rmse) or  (bounded_rmse > 2000 and epoch > 50):
                    isNaN = True
                    break
                if bounded_rmse > self.min_rmse:
                    overFitCount += 1
                    self.logger.info('overFitCount: '+ str(overFitCount))
                else:
                    overFitCount = 0

                if overFitCount > 300:
                    isOverFitting = True
                    break
            if isNaN or isOverFitting:
                break
        end_time = time.time()
        trainTime = end_time - start_time
        self.printMinRmse(isNaN, trainTime)
        self.printConfig()
        self.logger.info('done!!!!')

    def printMinRmse(self, isNaN, trainTime):
        if isNaN:
            self.logger.info('NaN error!!')
        self.logger.info('trainTime: '+str(trainTime)+' seconds')

        self.logger.info(
            "converge at epoch {} batchId {}--> min_rmse:{:.4f}, min_bounded_rmse:{:.4f},"
            " min_mae:{:.4f}, min_bounded_mae:{:.4f}".format(self.min_epoch_rmseBounded, self.min_batchId_rmseBounded,
            self.min_rmse, self.min_bounded_rmse, self.min_mae, self.min_bounded_mae)
        )


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

    def saveMinRmseBounded(self, epoch, batchId, rmseBounded, predList):
        if rmseBounded < self.min_bounded_rmse:
            self.min_epoch_rmseBounded = epoch
            self.min_batchId_rmseBounded = batchId
            self.min_bounded_rmse = rmseBounded
            self.resultList = predList

    def saveMinMaeBounded(self, epoch, batchId, maeBounded):
        if maeBounded < self.min_bounded_mae:
            self.min_epoch_maeBounded = epoch
            self.min_batchId_maeBounded = batchId
            self.min_bounded_mae = maeBounded

    def boundRating(self, ratingList):
        for i in range(len(ratingList)):
            if ratingList[i][0] > self.maxRating:
                ratingList[i][0] = self.maxRating
            if ratingList[i][0] < self.minRating:
                ratingList[i][0] = self.minRating

    def getTrainData(self, batchId):
        start = batchId * self.batchSize
        end = start + self.batchSize
        if end > self.trainSize:
            end = self.trainSize

        batch_u = self.trainSet[start:end, 0:1].astype(np.int32)
        batch_i = self.trainSet[start:end, 1:2].astype(np.int32)
        batch_r = self.trainSet[start:end, 2:3]
        return batch_u, batch_i, batch_r

    def getTrainValidData(self, batchId):
        start = batchId * self.batchSize
        end = start + self.batchSize
        if end > self.trainValidSize:
            end = self.trainValidSize

        batch_u = self.trainValidSet[start:end, 0:1].astype(np.int32)
        batch_i = self.trainValidSet[start:end, 1:2].astype(np.int32)
        batch_r = self.trainValidSet[start:end, 2:3]
        return batch_u, batch_i, batch_r

    def getValidData(self):
        valid_u = self.validSet[:, 0:1].astype(np.int32)
        valid_i = self.validSet[:, 1:2].astype(np.int32)
        valid_r = self.validSet[:, 2:3]
        return valid_u, valid_i, valid_r

    def getTestData(self):
        test_u = self.testSet[:, 0:1].astype(np.int32)
        test_i = self.testSet[:, 1:2].astype(np.int32)
        test_r = self.testSet[:, 2:3]
        return test_u, test_i, test_r

    def getOptimizer(self):
        if self.optiType == 'gd':
            return tf.train.GradientDescentOptimizer(self.learnRate, name='GD_optimizer').minimize(self.cost)
        elif self.optiType == 'adam':
            return tf.train.AdamOptimizer(self.learnRate, name='Adam_optimizer').minimize(self.cost)
        elif self.optiType == 'adadelta':
            return tf.train.AdadeltaOptimizer(self.learnRate, name='Adadelta_optimizer').minimize(self.cost)
        elif self.optiType == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.learnRate, name='GD_optimizer').minimize(self.cost)
        else:
            return None

    def printConfig(self):
        self.logger.info("Recommender: "+str(self.name))
        self.logger.info("learnRate: "+str(self.learnRate))
        self.logger.info("globalMean: "+str(self.globalMean))
        self.logger.info("numFactor: "+str(self.numFactor))
        self.logger.info("alpha: "+str(self.alpha))
        self.logger.info("lam1: "+str(self.lam1))
        self.logger.info("lam2: "+str(self.lam2))
        self.logger.info("optType: "+str(self.optiType))
        self.logger.info("outputType: "+str(self.outputType))
        self.logger.info("trainType: "+str(self.trainType))
        if self.trainType == 'test':
            self.logger.info('trainValidSize: '+str(self.trainValidSize))
        else:
            self.logger.info('trainSize: '+str(self.trainSize))

