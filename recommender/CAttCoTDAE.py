import tensorflow as tf
import numpy as np
import math
import random
import time
import os
from .AbstractRecommender import AbstractRecommender

class CAttCoTDAE(AbstractRecommender):
    def __init__(self, config, dataModel, evaluator=None):
        super().__init__(config, dataModel, evaluator)
        self.name = 'CollaborativeAttCoTDAE'

        self.logger = dataModel.logger
        self.sess = tf.InteractiveSession()
        self.interval = config.get('interval')
        # self.logger.info('cold start users:{}'.format(len(self.coldStartUserIdxs())))

        # data model config
        self.userIdx_FriendIndicies = dataModel.userIdx_FriendIndicies

        self.max_num_items = self.compute_max_num_items() #self.compute_max_num_items()
        self.max_num_items_trust = config['max_num_items_trust']

      # model hyper parameter
        self.numFactor_L1 = config['numFactor_L1']  # latent factors number of layer 1
        self.numFactor_L2 = config['numFactor_L2']
        self.alpha = config['alpha']  # the proportion of trust information
        self.lam = config['lam']
        self.beta = config['beta']



        # placeholders
        self.u_input = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id_input')
        self.u_rates_input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_num_items], name='u_rates_input')
        self.u_item_ids_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_num_items], name='u_item_ids')
        self.u_rates_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.max_num_items], name='u_rates_mask')
        self.u_item_ids_input_pred = tf.placeholder(dtype=tf.int32, shape=[None, self.max_num_items], name='u_item_ids_pred')

        self.trustUser_rates_input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_num_items_trust], name='trustUser_rates_input')
        self.trust_item_ids_input = tf.placeholder(dtype=tf.int32, shape=[None, self.max_num_items_trust], name='trust_item_ids')
        self.trust_rates_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.max_num_items_trust], name='trust_rates_mask')



        self.seed = dataModel.seed
        # weights
        scale = 1 / math.sqrt(self.numItem * self.numUser)
        if self.seed:
            tf.set_random_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.u_factor =  (2 * scale)* np.random.random_sample((self.numUser, self.numFactor_L2)) - scale

        self.weights = {
            #   R means rates, T means trusts
            'encoder_L1_R': tf.Variable(tf.random_uniform([self.numItem, self.numFactor_L1], -scale, scale),
                                            name='u_rates_encoder', dtype=tf.float32),
            'encoder_L1_T': tf.Variable(tf.random_uniform([self.numItem, self.numFactor_L1], -scale, scale),
                                             name='item_factor', dtype=tf.float32),
            'encoder_L2_R': tf.Variable(tf.random_uniform([self.numFactor_L1, self.numFactor_L2], -scale, scale),
                                            name='item_factor', dtype=tf.float32),
            'encoder_L2_T': tf.Variable(tf.random_uniform([self.numFactor_L1, self.numFactor_L2], -scale, scale),
                                             name='item_factor', dtype=tf.float32),
            'decoder_L1_R': tf.Variable(tf.random_uniform([self.numFactor_L2, self.numFactor_L1], -scale, scale),
                                            name='item_factor', dtype=tf.float32),
            'decoder_L1_T': tf.Variable(tf.random_uniform([self.numFactor_L2, self.numFactor_L1], -scale, scale),
                                             name='item_factor', dtype=tf.float32),
            'decoder_L2_R': tf.Variable(tf.random_uniform([self.numFactor_L1, self.numItem], -scale, scale),
                                            name='item_factor', dtype=tf.float32),
            'decoder_L2_T': tf.Variable(tf.random_uniform([self.numFactor_L1, self.numItem], -scale, scale),
                                             name='item_factor', dtype=tf.float32),
            'u_weights':tf.Variable(tf.random_uniform([self.numUser, self.numFactor_L1], -scale, scale),
                                    name='user_factor', dtype=tf.float32)
        }

        self.bias = {
            'encoder_b1_R': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale),
                                        name='u_rates_bias', dtype=tf.float32),
            'encoder_b1_T': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale),
                                        name='u_trusts_bias', dtype=tf.float32),
            'encoder_b2': tf.Variable(tf.random_uniform([self.numFactor_L2], -scale, scale),
                                      name='u_trusts_bias', dtype=tf.float32),
            'decoder_b1_R': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale),
                                        name='u_trusts_bias', dtype=tf.float32),
            'decoder_b1_T': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale),
                                        name='u_trusts_bias', dtype=tf.float32),
            'decoder_b2_R': tf.Variable(tf.random_uniform([self.numItem], -scale, scale),
                                        name='u_trusts_bias', dtype=tf.float32),
            'decoder_b2_T': tf.Variable(tf.random_uniform([self.numItem], -scale, scale),
                                        name='u_trusts_bias', dtype=tf.float32),
        }
        self.theta = {
            'theta_0': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale), dtype=tf.float32),
            'theta_1': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale), dtype=tf.float32),
            'theta_2': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale), dtype=tf.float32),
            'theta_3': tf.Variable(tf.random_uniform([self.numFactor_L1], -scale, scale), dtype=tf.float32),
        }
        self.trainType = config['trainType']
        if self.trainType=='train':
            temp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            self.save_path = './saver' + '/' + dataModel.dataDirectory + '/' + self.name + '_' + temp_time
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.save_name = temp_time + '.ckpt'
            self.global_step = tf.Variable(0, trainable=False)
        else:
            self.save_path = config['save_path']
        self.sess.run(tf.global_variables_initializer())



        self.r_pred = None
        self.u_feature_pred = None
        self.cost = None
        self.rmse = None
        self.mae = None
        self.optimizer = None


    def buildModel(self):

        u_factor = tf.reshape(tf.nn.embedding_lookup(self.weights['u_weights'], self.u_input), [-1, self.numFactor_L1])

        # encode user rating layer 1
        u_selected_weights_first = tf.nn.embedding_lookup(self.weights['encoder_L1_R'], self.u_item_ids_input)
        reshape = tf.reshape(self.u_rates_input, shape=[-1, 1, self.max_num_items])
        multiply = tf.matmul(reshape, u_selected_weights_first)
        mat = tf.reshape(multiply, shape=[-1, self.numFactor_L1])
        u_layer1 = tf.add(tf.add(mat, self.bias['encoder_b1_R']), u_factor)
        encoder_R_L1 = self.getActiv(u_layer1, 'sigmoid')


        # encode trust user rating layer 1
        trust_selected_weights_first = tf.nn.embedding_lookup(self.weights['encoder_L1_T'], self.trust_item_ids_input)
        mat_trust = \
            tf.reshape(
                tf.matmul(
                    tf.reshape(self.trustUser_rates_input, shape=[-1, 1, self.max_num_items_trust]),
                    trust_selected_weights_first
                ),
                shape=[-1, self.numFactor_L1]
            )
        trust_layer1 = tf.add(mat_trust, self.bias['encoder_b1_T'])
        encoder_T_L1 = self.getActiv(trust_layer1, 'sigmoid')

        # encoder user and trust into middle layer
        encoder_User = self.getActiv(self.multiple_input_encoder([[encoder_R_L1, self.weights['encoder_L2_R']],
                                                      [encoder_T_L1, self.weights['encoder_L2_T']]],
                                                     biases=self.bias['encoder_b2']), 'sigmoid')

        # decode user rating layer 1
        decoder_R_L1 = self.getActiv(self.single_input_encoder(encoder_User, self.weights['decoder_L1_R'],
                                                   self.bias['decoder_b1_R']), 'sigmoid')
        decoder_R_L1 = tf.reshape(decoder_R_L1, shape=[-1, 1, self.numFactor_L1])
        # decode trust rating layer 1
        decoder_T_L1 = self.getActiv(self.single_input_encoder(encoder_User, self.weights['decoder_L1_T'],
                                                   self.bias['decoder_b1_T']), 'sigmoid')
        decoder_T_L1 = tf.reshape(decoder_T_L1, shape=[-1, 1, self.numFactor_L1])



        # decode user rating
        u_selected_weights_last = tf.nn.embedding_lookup(tf.transpose(self.weights['decoder_L2_R']), self.u_item_ids_input)
        mat_decoder = tf.reshape(tf.matmul(decoder_R_L1, u_selected_weights_last, transpose_b=True),shape=[-1, self.max_num_items])
        u_selected_bias = tf.nn.embedding_lookup(self.bias['decoder_b2_R'], ids=self.u_item_ids_input)
        decoder_R = tf.add(mat_decoder, u_selected_bias)

        # decode user rating for predict
        u_selected_weights_last = tf.nn.embedding_lookup(tf.transpose(self.weights['decoder_L2_R']), self.u_item_ids_input_pred)
        mat_decoder = tf.reshape(tf.matmul(decoder_R_L1, u_selected_weights_last, transpose_b=True), shape=[-1, self.max_num_items])
        u_selected_bias = tf.nn.embedding_lookup(self.bias['decoder_b2_R'], ids=self.u_item_ids_input_pred)
        self.r_pred = tf.add(mat_decoder, u_selected_bias)



        # decode trust rating
        trust_selected_weights_last = tf.nn.embedding_lookup(tf.transpose(self.weights['decoder_L2_T']), self.trust_item_ids_input)
        trust_mat_decoder = tf.reshape(tf.matmul(decoder_T_L1, trust_selected_weights_last, transpose_b=True), shape=[-1, self.max_num_items_trust])
        trust_selected_bias = tf.nn.embedding_lookup(self.bias['decoder_b2_T'], ids=self.trust_item_ids_input)
        decoder_T = tf.add(trust_mat_decoder, trust_selected_bias)



        regCo = tf.add_n([tf.nn.l2_loss(tf.subtract(encoder_R_L1, encoder_T_L1 * self.theta['theta_0'])),
                          tf.nn.l2_loss(tf.subtract(encoder_T_L1, encoder_R_L1 * self.theta['theta_1'])),
                          tf.nn.l2_loss(tf.subtract(decoder_R_L1, decoder_T_L1 * self.theta['theta_2'])),
                          tf.nn.l2_loss(tf.subtract(decoder_T_L1, decoder_R_L1 * self.theta['theta_3']))])

        loss = (1 - self.alpha) * tf.reduce_sum(
                                                tf.square((tf.reshape(self.u_rates_input, shape=[-1, self.max_num_items]) - decoder_R) * self.u_rates_mask)
                                                )\
                + self.alpha * tf.reduce_sum(
                                              tf.square((tf.reshape(self.trustUser_rates_input, shape=[-1, self.max_num_items_trust])- decoder_T) * self.trust_rates_mask)
                                            )

        self.cost = loss + self.lam * self.layerReg(self.weights) + self.lam * self.layerReg(self.bias) \
                    + self.lam * self.layerReg(self.theta) + self.beta * regCo
        self.cost = self.cost * 0.5


        # self.r_pred = decoder_R * self.u_rates_mask
        self.u_feature_pred = encoder_User
        # self.rmse = tf.sqrt(tf.reduce_sum(tf.square(self.u_rates_input - self.r_pred))  / tf.reduce_sum(self.u_rates_mask))
        # self.mae = tf.reduce_sum(tf.abs(self.u_rates_input - self.r_pred)) /tf.reduce_sum(self.u_rates_mask)

    def feed_dict(self):
        batch_u, batch_uRates, batch_u_item_ids, batch_uRates_mask, \
        batch_trustUsersRates, batch_trust_item_ids, batch_trust_rates_mask = self.getTrainData(0)
        feed_dict = {
            self.u_input: batch_u,
            self.u_rates_input: batch_uRates,
            self.u_item_ids_input: batch_u_item_ids,
            self.u_rates_mask: batch_uRates_mask,
            self.trustUser_rates_input: batch_trustUsersRates,
            self.trust_item_ids_input: batch_trust_item_ids,
            self.trust_rates_mask: batch_trust_rates_mask
        }
        return feed_dict


    def predict(self):
        """
        predict rating in test matrix
        :return:
        """
        batch_num = self.testMatrix.numUser // self.batch_size + 1


        for batch_id in range(batch_num):
            batch_u, batch_uRates, batch_u_item_ids, batch_uRates_mask, \
            batch_trustUsersRates, batch_trust_item_ids, batch_trust_rates_mask, \
            pred_u_item_ids = self.getTestData(batch_id)

            feed_dict = {
                self.u_input: batch_u,
                self.u_rates_input: batch_uRates,
                self.u_item_ids_input: batch_u_item_ids,
                self.u_rates_mask: batch_uRates_mask,
                self.trustUser_rates_input: batch_trustUsersRates,
                self.trust_item_ids_input: batch_trust_item_ids,
                self.trust_rates_mask: batch_trust_rates_mask,
                self.u_item_ids_input_pred: pred_u_item_ids
            }

            pred = self.sess.run(self.r_pred, feed_dict=feed_dict)

            batch_size = len(pred)
            for userIdx, batch_idx in zip(batch_u, range(batch_size)):
                pred_item_idxs = self.testMatrix.getItemsOfUser(userIdx)
                for itemIdx, i in zip(pred_item_idxs, range(len(pred_item_idxs))):
                    self.predUserItemRating[userIdx, itemIdx] = pred[batch_idx, i]

    def recommendTopkForUsers(self):
        recommendLists = {}

        u_rates_batch = []
        trustUser_rates_batch = []
        testUserIdxs = self.testMatrix.userIdxs
        testUserNum = self.testMatrix.numUser
        for userIdx in testUserIdxs:
            u_rates_batch.append(self.getUserRatesVector(userIdx).tolist())
            trustUser_rates_batch.append(self.getTrustUsersAverageRates(userIdx).tolist())

        pred = self.sess.run(self.decoder_R, feed_dict={
            self.u_input:list(testUserIdxs),
            self.u_rates_input: u_rates_batch,
            self.trustUser_rates_input: trustUser_rates_batch
        })

        for i, userIdx in zip(range(testUserNum), testUserIdxs):
            trainItems = self.trainMatrix.getItemsOfUser(userIdx)
            sortedItemIdxs = np.argsort(pred[i])
            recommendItems = []
            counter = self.numItem-1
            numRecItems = 0
            while counter>=0:
                if sortedItemIdxs[counter] not in trainItems:
                    recommendItems.append(sortedItemIdxs[counter])
                    numRecItems += 1
                if numRecItems >= self.topK:
                    break
                counter -= 1
            if counter==0:
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

    def evaluateModel(self):
        saver = tf.train.Saver()

        isNaN = False
        overFitCount = 0
        step = 0

        start_time = time.time()
        while True:
            ckpt = tf.train.get_checkpoint_state(self.save_path)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                self.predict()
            else:
                print('no check point')
                return
            self.evaluate()
            self.saveMinRmse(step, 0, self.rmse)
            self.saveMinMae(step, 0, self.mae)

            if math.isnan(self.rmse) or (self.rmse > 2000 and global_step > 50):
                break

            if self.rmse > self.min_rmse:
                overFitCount += 1
                self.logger.info('overFitCount: ' + str(overFitCount))
            else:
                overFitCount = 0

            if overFitCount > 300:
                break
            step += 1

            self.logger.info('[global_step:{}, step{}]: rmse:{:.4f}, mae:{:.4f}'.format(global_step, step, self.rmse, self.mae))
            time.sleep(self.interval)

        end_time = time.time()
        trainTime = end_time - start_time
        self.printMinRmse(isNaN, trainTime)
        self.printConfig()



    def trainModel(self):
        self.logger.info("--------------------Begin Training--------------------")
        self.printConfig()

        batch_num = self.trainMatrix.numUser // self.batch_size + 1

        saver = tf.train.Saver()
        self.optimizer = self.getOptimizer()
        self.sess.run(tf.global_variables_initializer())

        overFitCount = 0
        isNaN = False
        start = time.time()

        for epoch in range(self.maxIter):
            for batch_id in range(batch_num):
                batch_u, batch_uRates, batch_u_item_ids, batch_uRates_mask,\
                batch_trustUsersRates, batch_trust_item_ids, batch_trust_rates_mask = self.getTrainData(
                    batch_id)
                feed_dict = {
                    self.u_input: batch_u,
                    self.u_rates_input: batch_uRates,
                    self.u_item_ids_input: batch_u_item_ids,
                    self.u_rates_mask: batch_uRates_mask,
                    self.trustUser_rates_input: batch_trustUsersRates,
                    self.trust_item_ids_input: batch_trust_item_ids,
                    self.trust_rates_mask: batch_trust_rates_mask
                }
                self.sess.run(self.optimizer, feed_dict=feed_dict)
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                self.logger.info('[epoch{},batch{} ]loss:{:.4f}'.format(epoch, batch_id, loss))
                saver.save(self.sess, self.save_path + '/' + self.save_name, global_step=self.global_step)
            self.predict()
            self.evaluate()
            self.saveMinRmse(epoch, 0, self.rmse)
            self.saveMinMae(epoch, 0, self.mae)

            if math.isnan(self.rmse) or (self.rmse > 2000 and epoch > 50):
                isNaN = True
                break

            if self.rmse > self.min_rmse:
                overFitCount += 1
                self.logger.info('overFitCount: ' + str(overFitCount))
            else:
                overFitCount = 0

            if overFitCount > 300:
                break
            self.logger.info('[epoch{}]                         rmse:{:.4f} mae:{:.4f}'.format(epoch, self.rmse, self.mae))
        end = time.time()
        self.printMinRmse(isNaN, end-start)
        self.printConfig()


    def trainModelRankingBased(self):
        self.logger.info("--------------------Begin Training--------------------")
        self.logger.info("topK:{}".format(self.topK))
        self.logger.info("beta:{}".format(self.beta))
        self.printConfig()


        self.optimizer = self.getOptimizer()
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.maxIter):
            for batch_id in range(self.batch_num):
                batch_u, batch_uRates,batch_trustUsersRates,batch_uImplicitVecs,batch_socialImplicitVecs = self.getTrainData(batch_id)

                self.optimizer.run(feed_dict={
                    self.u_input: batch_u,
                    self.u_rates_input: batch_uRates,
                    self.trustUser_rates_input: batch_trustUsersRates,
                    self.u_rates_mask: batch_uImplicitVecs,
                    self.trustUser_rates_mask: batch_socialImplicitVecs
                })

                pred_u_feature = self.u_feature_pred.eval(feed_dict={
                    self.u_input: batch_u,
                    self.u_rates_input: batch_uRates,
                    self.trustUser_rates_input: batch_trustUsersRates,
                    self.u_rates_mask: batch_uImplicitVecs,
                    self.trustUser_rates_mask: batch_socialImplicitVecs
                })

                flatten_batch_u = batch_u

                self.u_factor[flatten_batch_u] = pred_u_feature  # update user feature mat

                loss = self.cost.eval(feed_dict={
                    self.u_input: batch_u,
                    self.u_rates_input: batch_uRates,
                    self.trustUser_rates_input: batch_trustUsersRates,
                    self.u_rates_mask: batch_uImplicitVecs,
                    self.trustUser_rates_mask: batch_socialImplicitVecs
                })
                recommendLists = self.recommendTopkForUsers()
                self.evaluator.setPredLists(recommendLists)
                recall = self.evaluator.calRecall()
                map = self.evaluator.calMAP()

                self.saveMaxRecall(epoch, batch_id, recall)
                self.saveMaxMap(epoch, batch_id, map)

                self.logger.info("[epoch:{}, batch:{}] loss:{:.4f}".format(epoch, batch_id, loss))

        self.printConfig()
        self.logger.info("beta:{}".format(self.beta))
        self.logger.info("maxRecall:{:.4f}".format(self.max_recall))
        self.logger.info("maxMAP:{:.4f}".format(self.max_map))
        self.logger.info("topK:{}".format(self.topK))
        self.logger.info('done!!!!')


    def printMinRmse(self, isNaN, trainTime):
        if isNaN:
            self.logger.info('NaN error!!')
        self.logger.info('trainTime: '+str(trainTime)+' seconds')

        self.logger.info(
            "converge at epoch {} batchId {}--> min_rmse:{:.4f}"
                .format(self.min_epoch_rmse, self.min_batchId_rmse, self.min_rmse)
        )

        self.logger.info(
            "Min mae at epoch {} batchId {}--> min_mae:{:.4f}"
                .format(self.min_epoch_mae, self.min_batchId_mae, self.min_mae)
        )


    def getTrainData(self, batch_id):
        # compute start and end
        u_idxs_batch = self.trainMatrix.next_batch(batch_id)
        u_rates_batch =[]
        u_items_batch = []
        u_item_mask_batch = []
        trust_rates_batch = []
        trust_item_batch = []
        trust_item_mask_batch = []

        for userIdx in u_idxs_batch:
            u_rates, u_item_ids, u_item_mask = self.getRates_Id_Mask_Vector(userIdx)
            trust_rates, trust_item_ids, trust_item_mask = self.getTrustUsersAverageRates(userIdx)
            u_rates_batch.append(u_rates)
            u_items_batch.append(u_item_ids)
            u_item_mask_batch.append(u_item_mask)
            trust_rates_batch.append(trust_rates)
            trust_item_batch.append(trust_item_ids)
            trust_item_mask_batch.append(trust_item_mask)

        return u_idxs_batch, u_rates_batch, u_items_batch, u_item_mask_batch,\
               trust_rates_batch, trust_item_batch, trust_item_mask_batch


    def getTestData(self, batch_id):
        # compute start and end
        u_idxs_batch = self.testMatrix.next_batch(batch_id)
        u_rates_batch =[]
        u_items_batch = []
        u_item_mask_batch = []
        trust_rates_batch = []
        trust_item_batch = []
        trust_item_mask_batch = []
        pred_u_item_ids_batch = []

        for userIdx in u_idxs_batch:
            u_rates, u_item_ids, u_item_mask = self.getRates_Id_Mask_Vector(userIdx)
            trust_rates, trust_item_ids, trust_item_mask = self.getTrustUsersAverageRates(userIdx)
            u_rates_batch.append(u_rates)
            u_items_batch.append(u_item_ids)
            u_item_mask_batch.append(u_item_mask)
            trust_rates_batch.append(trust_rates)
            trust_item_batch.append(trust_item_ids)
            trust_item_mask_batch.append(trust_item_mask)
            pred_u_item_ids = []
            pred_items = self.testMatrix.getItemsOfUser(userIdx)
            for itemIdx in pred_items:
                pred_u_item_ids.append(itemIdx)
            if len(pred_items)<self.max_num_items:
                for i in range(self.max_num_items - len(pred_items)):
                    pred_u_item_ids.append(0)
            pred_u_item_ids_batch.append(pred_u_item_ids)

        return u_idxs_batch, u_rates_batch, u_items_batch, u_item_mask_batch,\
               trust_rates_batch, trust_item_batch, trust_item_mask_batch, pred_u_item_ids_batch


    def getRates_Id_Mask_Vector(self, userIdx):
        rates_vec = []
        item_ids = []
        item_mask = []
        items = self.trainMatrix.getItemsOfUser(userIdx)

        for itemIdx in items:
            item_ids.append(itemIdx)
            rates_vec.append(self.trainMatrix.getRating(userIdx, itemIdx))
            item_mask.append(1)
        for i in range(self.max_num_items - len(items)):
            item_ids.append(0)
            rates_vec.append(0)
            item_mask.append(0)
        return rates_vec, item_ids, item_mask


    def getSocialImplicitVector(self, userIdx):
        socialImplicit = np.zeros(self.numItem)
        if userIdx in self.userIdx_FriendIndicies:
            for friendIdx in self.userIdx_FriendIndicies[userIdx]:
                for friendItemIdx in self.trainMatrix.getItemsOfUser(friendIdx):
                    socialImplicit[friendItemIdx] = 1
        return socialImplicit

    def getTrustUsersAverageRates(self, thisUserIdx):
        thisUserfeature = self.u_factor[thisUserIdx]
        trustUsersAttention = []
        trustUsersRate = np.zeros(self.numItem)


        # get this user's friends
        if thisUserIdx in self.userIdx_FriendIndicies:
            for trustUserIdx in self.userIdx_FriendIndicies[thisUserIdx]:
                 # get trust users' feature
                trustUserFeature = self.u_factor[trustUserIdx]
                trustUserAttention = thisUserfeature.dot(trustUserFeature)
                trustUsersAttention.append(trustUserAttention)
        # compute attention
        import recMath.vector
        trustUsersAttention = recMath.vector.softmax(trustUsersAttention)

        # get weighted_mean average rating
        if thisUserIdx in self.userIdx_FriendIndicies:
            for trustUserIdx, attention in zip(self.userIdx_FriendIndicies[thisUserIdx], trustUsersAttention):
                trustUsersRate += self.getUserRatesVector(trustUserIdx) * attention

        nonZeroIdxs = np.nonzero(trustUsersRate)
        nonZeroIdxs = nonZeroIdxs[0].tolist()
        if len(nonZeroIdxs) > self.max_num_items_trust:
            nonZeroIdxs = random.sample(nonZeroIdxs, self.max_num_items_trust)
        rates_vec = trustUsersRate[nonZeroIdxs].tolist()
        items_vec = nonZeroIdxs
        items_mask = [1 for j in range(len(rates_vec))]
        if len(rates_vec) < self.max_num_items_trust:
            for i in range(self.max_num_items_trust - len(rates_vec)):
                rates_vec.append(0)
                items_vec.append(0)
                items_mask.append(0)
        return rates_vec, items_vec, items_mask


    def multiple_input_encoder(self, input_weight_piars, biases, activ='sigmoid'):
        """
        :param input_weight_piars: [[input1, weight1], [input2, weight2],...]
        :param biases: w1*x1+w2*x2+bias
        :return: w1*x1+w2*x2+bias
        """
        matmul_list = []
        for input_weight in input_weight_piars:
            matmul = tf.matmul(input_weight[0], input_weight[1])  # 0:input 1:weight
            matmul_list.append(matmul)
        res = tf.add(tf.add_n(matmul_list), biases)
        return res

    def single_input_encoder(self, input, weights, bias):
        output = tf.add(tf.matmul(input, weights), bias)
        return output

    def compute_max_num_items(self):
        max_num_items = 0

        for userIdx in self.preferenceMatrix.userIdxs:
            userItems = self.preferenceMatrix.getItemsOfUser(userIdx)
            numItems = len(userItems)
            if numItems>max_num_items:
                max_num_items =numItems

        self.logger.info("max_num_items: {}".format(max_num_items))
        return max_num_items

    def compute_max_num_items_trust(self):
        max_num_items = 0

        for userIdx in self.preferenceMatrix.userIdxs:
            userItems = self.preferenceMatrix.getItemsOfUser(userIdx)
            numItems = len(userItems)
            if numItems>max_num_items:
                max_num_items =numItems

            if userIdx in self.userIdx_FriendIndicies.keys():
                itemSet_trustUsers = set()
                trustUserIdxs = self.userIdx_FriendIndicies[userIdx]
                for trustUserIdx in trustUserIdxs:
                    trustUserItems = self.preferenceMatrix.getItemsOfUser(trustUserIdx)
                    itemSet_trustUsers.update(trustUserItems)
                if len(itemSet_trustUsers)>max_num_items:
                    max_num_items = len(itemSet_trustUsers)
        self.logger.info("max_num_items_trust: {}".format(max_num_items))
        return max_num_items

    def getTrustUserItems(self, userIdx):
        itemSet_trustUsers = set(self.trainMatrix.getItemsOfUser(userIdx))
        if userIdx in self.userIdx_FriendIndicies.keys() and userIdx in self.trainMatrix.userIdxs:
            trustUserIdxs = self.userIdx_FriendIndicies[userIdx]
            for trustUserIdx in trustUserIdxs:
                if trustUserIdx in self.trainMatrix.userIdxs:
                    trustUserItems = self.trainMatrix.getItemsOfUser(trustUserIdx)
                    itemSet_trustUsers.update(trustUserItems)

        return list(itemSet_trustUsers)

    def getTrustUserRates(self, userIdx, trustItemIds):
        rates = []
        for itemIdx in trustItemIds:
            rates.append(self.trainMatrix.getRating(userIdx, itemIdx))
        return rates

    def printConfig(self):
        self.logger.info('Recommender: ' + str(self.name))
        self.logger.info('num cold start users: ' + str(len(self.testUserIdxs_cold_start)))
        self.logger.info('num warm start users: ' + str(len(self.testUserIdxs_warm_start)))
        self.logger.info('learnRate: ' + str(self.learnRate))
        self.logger.info('numFactor_L1: ' + str(self.numFactor_L1))
        self.logger.info('numFactor_L2: ' + str(self.numFactor_L2))
        self.logger.info('alpha: ' + str(self.alpha))
        self.logger.info("beta: {}".format(self.beta))
        self.logger.info('lam: ' + str(self.lam))
        self.logger.info('optiType: ' + str(self.optiType))
        self.logger.info('lossType: ' + str(self.lossType))
        self.logger.info('outputType: ' + str(self.outputType))
        self.logger.info('isBasedGlobalMean: '+str(self.isBasedGlobalMean))
        self.logger.info('save_path: {}'.format(self.save_path))

    def printTensor(self, name, tensor, feed_dict):
        tensorVal = self.sess.run(tensor, feed_dict=feed_dict)
        print(name, '\n', tensorVal)
        return tensorVal