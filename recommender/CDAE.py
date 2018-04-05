from .AbstractRecommender import AbstractRecommender
import tensorflow as tf
import random
import math
import numpy as np
import time

class CDAE(AbstractRecommender):
    def __init__(self, config, dataModel, evaluator=None):
        super().__init__(config, dataModel, evaluator)
        self.name = 'CDAE'

        self.logger = dataModel.logger
        self.sess = tf.InteractiveSession()

        # model hyper parameter
        self.numFactor = config['numFactor']  # latent factors number of layer 1
        self.lam = config['lam']
        # placeholders
        self.u_input = tf.placeholder(dtype=tf.int32, shape=[None])
        self.u_rates_input = tf.placeholder(dtype=tf.float32, shape=[None, self.numItem])
        # weights
        scale = 1 / math.sqrt(self.numItem * self.numUser)
        tf.set_random_seed(config['seed'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])

        self.weights = {
            'encoder_L1_R': tf.Variable(tf.random_uniform([self.numItem, self.numFactor], -scale, scale),
                                        name='u_rates_encoder', dtype=tf.float32),
            'decoder_L2_R': tf.Variable(tf.random_uniform([self.numFactor, self.numItem], -scale, scale),
                                        name='item_factor', dtype=tf.float32),
            'u_factor': tf.Variable(tf.random_uniform([self.numUser, self.numFactor], -scale, scale),
                                    name='user_factor', dtype=tf.float32),
        }

        self.bias = {
            'encoder_b_R': tf.Variable(tf.random_uniform([self.numFactor], -scale, scale),
                                        name='encoder_u_rates_bias', dtype=tf.float32),
            'decoder_b_R': tf.Variable(tf.random_uniform([self.numItem], -scale, scale),
                                        name='decoder_u_rates_bias', dtype=tf.float32),
        }

        self.cost = None
        self.decoder_R = None
        self.u_factor = None

    def buildModel(self):
        u_factor = tf.reshape(tf.nn.embedding_lookup(self.weights['u_factor'], self.u_input),[-1, self.numFactor])
        layer1 = tf.add(tf.add(tf.matmul(self.u_rates_input, self.weights['encoder_L1_R']), self.bias['encoder_b_R']), u_factor)
        encoder_R_L1 = self.getActiv(layer1, 'sigmoid')
        layer2 = tf.add(tf.matmul(encoder_R_L1, self.weights['decoder_L2_R']), self.bias['decoder_b_R'])
        decoder_R = self.getActiv(layer2, 'identical')
        self.decoder_R = decoder_R
        if self.lossType=='mse':
            loss =  tf.reduce_sum(tf.square((self.u_rates_input - decoder_R)* self.u_rates_input ))
        elif self.lossType=='cross_entropy':
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.u_rates_input, logits=decoder_R) * self.u_rates_input)

        self.cost = loss + self.lam * self.layerReg(self.weights) + self.lam * self.layerReg(self.bias)


    def trainModel(self):
        self.logger.info("--------------------Begin Training--------------------")
        self.logger.info("topK:{}".format(self.topK))
        self.printConfig()
        self.optimizer = self.getOptimizer()
        self.sess.run(tf.global_variables_initializer())

        trainUserIdxs, trainUserRatesInput = self.getTrainData()
        iterNum = 1
        while iterNum <= self.maxIter:
            #train model param
            self.optimizer.run(
                feed_dict={
                    self.u_input: trainUserIdxs,
                    self.u_rates_input: trainUserRatesInput
                }
            )
            loss = self.sess.run(self.cost, feed_dict={
                self.u_input: trainUserIdxs,
                self.u_rates_input: trainUserRatesInput
            })

            recommendLists = self.recommendTopkForUsers()
            self.evaluator.setPredLists(recommendLists)
            recall = self.evaluator.calRecall()
            MAP = self.evaluator.calMAP()

            self.saveMaxRecall(iterNum, 0, recall)
            self.saveMaxMap(iterNum, 0, MAP)

            self.logger.info(
                "[iter:{}] loss:{:.4f}, recall:{:.4f}, MAP:{:.4f}"
                    .format(iterNum, loss, recall, MAP)
            )
            iterNum += 1

        self.printConfig()
        self.logger.info("maxRecall:{:.4f} at iter {},maxMap:{:.4f}".format(self.max_recall, self.max_epoch_recall, self.max_map))
        self.logger.info("topK:{}".format(self.topK))
        self.logger.info('done!!!!')

    def recommendTopkForUsers(self):
        recommendLists = {}

        u_rates_batch = []

        testUserIdxs = self.testMatrix.userIdxs
        testUserNum = self.testMatrix.numUser
        for userIdx in testUserIdxs:
            u_rates_batch.append(self.getUserRatesVector(userIdx).tolist())

        pred = self.sess.run(self.decoder_R, feed_dict={
            self.u_input: list(testUserIdxs),
            self.u_rates_input: u_rates_batch,
        })

        for i, userIdx in zip(range(testUserNum), testUserIdxs):
            trainItems = self.trainMatrix.getItemsOfUser(userIdx)
            sortedItemIdxs = np.argsort(pred[i])
            recommendItems = []
            counter = self.numItem - 1
            numRecItems = 0
            while counter >= 0:
                if sortedItemIdxs[counter] not in trainItems:
                    recommendItems.append(sortedItemIdxs[counter])
                    numRecItems += 1
                if numRecItems >= self.topK:
                    break
                counter -= 1
            if counter == 0:
                print("warning for user{}->recommendItemsLen{}".format(userIdx, numRecItems))
            recommendLists[userIdx] = recommendItems
            # print("recList->user{}:{}".format(userIdx, recommendItems))
        return recommendLists

    def layerReg(self, layer_params):
        regs = []
        for x in layer_params.values():
            regs.append(tf.nn.l2_loss(x))
        sum = tf.add_n(regs)
        return sum

    def getTrainData(self):
        # compute start and end
        u_batch = []
        u_rates_batch =[]

        userIdxs = self.trainMatrix.userIdxs
        for userIdx in userIdxs:
            u_batch.append(userIdx)
            u_rates_batch.append(self.getUserImplicitVector(userIdx).tolist())
        return u_batch, u_rates_batch

    def printConfig(self):
        self.logger.info('Recommender: ' + str(self.name))
        self.logger.info('learnRate: ' + str(self.learnRate))
        self.logger.info('numFactor: ' + str(self.numFactor))
        self.logger.info('lam: ' + str(self.lam))
        self.logger.info('optiType: ' + str(self.optiType))
        self.logger.info('lossType: ' + str(self.lossType))




