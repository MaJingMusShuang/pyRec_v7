import logging
import time
import os.path
import random
from recMath.RatingMatrix import RatingMatrix

class DataModel:
    def __init__(self, config):
        self.config = config
        self.dataDirectory = config['dataDirectory']
        self.logger = self.__initLogger()
        self.ratingInputPath = './data' + '/' +self.dataDirectory + '/ratings.txt'
        self.seed = config.get('seed')

        self.userId_userIdx = {}
        self.itemId_itemIdx = {}
        self.numUser = 0
        self.numItem = 0
        self.trainRatio = config['trainRatio']
        self.batch_size = config.get('batch_size')
        self.preferenceMatrix = RatingMatrix(batch_size=self.batch_size)
        self.trainMatrix = RatingMatrix(batch_size=self.batch_size)
        self.testMatrix = RatingMatrix(batch_size=self.batch_size)
        self.trainSet = []
        self.testSet = []
        self.ratingScaleSet = set()
        self.minRating = 0
        self.maxRating = 0
        self.globalMean = 0
        self.binThold_drop = config.get('binThold_drop') #drop ratings less than 4 star
        self.cold_drop = config.get('cold_drop') #drop users or items with less than 5 ratings
        self.binThold = config.get('binThold') #binary threshold


    def buildDataModel(self, **crossValidation):
        if self.binThold_drop:
            self.readRatingsGtThold()
        else:
            self.readRatingData()

        if self.cold_drop:
            self.dropColdUsersItems(self.cold_drop)

        if crossValidation:
            self.splitDataCross(self.config['crossValidation'])
        else:
            if self.trainRatio>0 and self.trainRatio<1:
                self.splitData()
            else:
                self.logger.info('!!!!!!!splitting data fails, split ratio: {}'.format(self.trainRatio))
        self.logger.info("===========================Data model built successfully!!!============================")
        self.logger.info(str(self))
        self.logger.info("========================================================================================")

    def readRatingData(self):
        self.logger.info('Reading rating dataset:{}'.format(self.dataDirectory))

        baseUserIdx = 0
        baseItemIdx = 0
        with open(self.ratingInputPath) as inputFile:
            for line in inputFile:
                userId, itemId, rating = line.strip().split(' ')
                rating = float(rating)

                if userId not in self.userId_userIdx:
                    self.userId_userIdx[userId] = baseUserIdx
                    baseUserIdx += 1
                if itemId not in self.itemId_itemIdx:
                    self.itemId_itemIdx[itemId] = baseItemIdx
                    baseItemIdx += 1
                userIdx = self.userId_userIdx[userId]
                itemIdx = self.itemId_itemIdx[itemId]
                self.preferenceMatrix.put(userIdx, itemIdx, rating)
                self.ratingScaleSet.add(rating)

        self.numUser = self.preferenceMatrix.numUser
        self.numItem = self.preferenceMatrix.numItem
        self.minRating = min(self.ratingScaleSet)
        self.maxRating = max(self.ratingScaleSet)
        self.globalMean = self.preferenceMatrix.globalMean

        if self.binThold:
                self.preferenceMatrix.binarize(self.binThold)
                self.minRating = 0
                self.maxRating = 1
                self.logger.info("binThold:{}".format(self.binThold))

        self.logger.info('loading dataset successfully!!! dataset info')
        self.logger.info('Rating Matrix\n'+ str(self.preferenceMatrix))
        self.logger.info('Rating Matrix density: ' + str(self.preferenceMatrix.getDensity()))
        self.logger.info('MaxRating: '+ str(self.maxRating))
        self.logger.info('MinRating: ' + str(self.minRating))
        self.logger.info('Global Mean: ' + str(self.globalMean))
        
    def readRatingsGtThold(self):
        self.logger.info('Reading rating dataset:{}'.format(self.dataDirectory))
        self.logger.info('and drop ratings less than:{}'.format(self.binThold_drop))
        baseUserIdx = 0
        baseItemIdx = 0
        with open(self.ratingInputPath) as inputFile:
            for line in inputFile:
                userId, itemId, rating = line.strip().split(' ')
                rating = float(rating)
                if rating >= self.binThold_drop:
                    if userId not in self.userId_userIdx:
                        self.userId_userIdx[userId] = baseUserIdx
                        baseUserIdx += 1
                    if itemId not in self.itemId_itemIdx:
                        self.itemId_itemIdx[itemId] = baseItemIdx
                        baseItemIdx += 1
                    userIdx = self.userId_userIdx[userId]
                    itemIdx = self.itemId_itemIdx[itemId]
                    rating = 1
                    self.preferenceMatrix.put(userIdx, itemIdx, rating)
                    self.ratingScaleSet.add(rating)

        self.numUser = self.preferenceMatrix.numUser
        self.numItem = self.preferenceMatrix.numItem
        self.minRating = min(self.ratingScaleSet)
        self.maxRating = max(self.ratingScaleSet)
        self.trainGlobalMean = self.preferenceMatrix.globalMean

        self.logger.info('loading dataset successfully!!! dataset info')
        self.logger.info('Rating Matrix\n' + str(self.preferenceMatrix))
        self.logger.info('Rating Matrix density: ' + str(self.preferenceMatrix.getDensity()))

    def dropColdUsersItems(self, threshold=5):
        self.logger.info("======================Drop cold users and items with ratings less than:{}".format(threshold))

        preferenceMatrix = RatingMatrix()
        userId_userIdx = {}
        itemId_itemIdx = {}

        baseUserIdx = 0
        baseItemIdx = 0
        with open(self.ratingInputPath) as inputFile:
            for line in inputFile:
                userId, itemId, rating = line.strip().split(' ')
                rating = float(rating)

                if rating >= self.binThold_drop:
                    userIndex = self.userId_userIdx[userId]
                    itemIndex = self.itemId_itemIdx[itemId]

                    if len(self.preferenceMatrix.user_items[userIndex]) < self.cold_drop \
                            or len(self.preferenceMatrix.item_users[itemIndex]) < self.cold_drop:
                        continue

                    if userId not in userId_userIdx:
                        userId_userIdx[userId] = baseUserIdx
                        baseUserIdx += 1
                    if itemId not in itemId_itemIdx:
                        itemId_itemIdx[itemId] = baseItemIdx
                        baseItemIdx += 1
                    userIdx = userId_userIdx[userId]
                    itemIdx = itemId_itemIdx[itemId]
                    rating = 1
                    preferenceMatrix.put(userIdx, itemIdx, rating)


        self.preferenceMatrix = preferenceMatrix
        self.userId_userIdx = userId_userIdx
        self.itemId_itemIdx = itemId_itemIdx

        self.numUser = self.preferenceMatrix.numUser
        self.numItem = self.preferenceMatrix.numItem

        self.logger.info('Drop cold users and items successfully')
        self.logger.info('Rating Matrix\n' + str(self.preferenceMatrix))
        self.logger.info('Rating Matrix density: ' + str(self.preferenceMatrix.getDensity()))
        self.logger.info('-------------------------------------------------------------------')

    def dropUserOrItemWithLessThan5Ratings(self):
        userItem_rating = self.preferenceMatrix.userItem_rating
        lines = []
        for (userIdx, itemIdx) in userItem_rating.keys():
            if len(self.preferenceMatrix.getItemsOfUser(userIdx)) < 5 or len(self.preferenceMatrix.getUsersOfItem(itemIdx)) < 5:
                continue
            lines.append('{} {} {}\n'.format(userIdx, itemIdx, userItem_rating[userIdx, itemIdx]))
        with open('ratings.txt', 'w') as outputFile:
            outputFile.writelines(lines)




    def __dropColdUsersItems(self, threshold=5):
        self.logger.info("======================Drop cold users and items with ratings less than:{}".format(threshold))
        wholeDataSet = []
        baseUserIdx = 0
        baseItemIdx = 0
        temp_userId_userIdx = {}
        temp_itemId_itemIdx = {}
        with open(self.ratingInputPath) as inputFile:
            for line in inputFile:
                userId, itemId, rating = line.strip().split(' ')
                rating = float(rating)
                if self.binThold_drop:
                    if rating >= self.binThold_drop:
                        rating = 1
                    else:
                        continue
                userIdx = self.userId_userIdx[userId]
                itemIdx = self.itemId_itemIdx[itemId]
                if len(self.user_items[userIdx]) < threshold or len(self.item_users[itemIdx]) < threshold:
                    continue

                if userId not in temp_userId_userIdx:
                    temp_userId_userIdx[userId] = baseUserIdx
                    baseUserIdx += 1

                if itemId not in temp_itemId_itemIdx:
                    temp_itemId_itemIdx[itemId] = baseItemIdx
                    baseItemIdx += 1

                temp_userIdx = temp_userId_userIdx[userId]
                temp_itemIdx = temp_itemId_itemIdx[itemId]
                wholeDataSet.append([temp_userIdx, temp_itemIdx, rating])
        self.userId_userIdx = temp_userId_userIdx
        self.itemId_itemIdx = temp_itemId_itemIdx
        self.numRatingBeforeSplit = len(wholeDataSet)
        self.numUserBeforeSplit = baseUserIdx
        self.numItemBeforeSplit = baseItemIdx
        self.minRating = min(self.ratingScaleSet)
        self.maxRating = max(self.ratingScaleSet)

        # collect user_items mappings
        user_items = {}
        # collect item_users mappings
        item_users = {}
        # collect userItem_rating mappings
        userItem_rating = {}
        for uir in wholeDataSet:
            userIdx, itemIdx, rating = uir
            if userIdx not in user_items.keys():
                user_items[userIdx] = []
            user_items[userIdx].append(itemIdx)

            if itemIdx not in item_users.keys():
                item_users[itemIdx] = []
            item_users[itemIdx].append(userIdx)
            userItem_rating[userIdx, itemIdx] = rating

        self.user_items = user_items
        self.item_users = item_users
        self.userItem_rating = userItem_rating
        self.logger.info("======================Drop cold users and items done!!!")


    def splitData(self):
        self.logger.info('Splitting data...TrainRatio: '+str(self.trainRatio))
        if self.seed:
            random.seed(self.seed)
        # split data
        # raw data
        for (userIdx, itemIdx) in self.preferenceMatrix.userItem_rating:
            rdm = random.uniform(0, 1)
            rating = self.preferenceMatrix.userItem_rating[userIdx, itemIdx]
            if rdm <= self.trainRatio:
                self.trainMatrix.put(userIdx, itemIdx, rating)
                self.trainSet.append([userIdx, itemIdx, rating])
            else:
                if userIdx not in self.trainMatrix.userIdxs or itemIdx not in self.trainMatrix.itemIdxs:
                    continue                  #filter out cold start users and items
                self.testMatrix.put(userIdx, itemIdx, rating)
                self.testSet.append([userIdx, itemIdx, rating])

        self.logger.info('Splitting data done!!!')
        self.logger.info('trainMatrix:\n' + str(self.trainMatrix))
        self.logger.info('testMatrix:\n'+ str(self.testMatrix))

    def __splitDataCross(self):
        random.seed(123)

    def __initLogger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(levelname)s] %(message)s')

        # create console handler and set level to info
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # create debug file handler and set level to debug

        fh = logging.FileHandler(
            os.path.join('./log',
                         time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        return logger

    def __str__(self):
        if self.seed:
            self.logger.info('************************************************* seed:{}'.format(self.seed))
        return 'dataset: ' + str(self.dataDirectory) + '\n' \
                + 'ratingScale: ' + str(self.ratingScaleSet) + '\n'




