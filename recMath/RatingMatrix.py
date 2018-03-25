import numpy as np
from scipy.sparse.coo import coo_matrix
userIdxs, itemIdxs, ratings = [], [], []

class RatingMatrix(coo_matrix):
    def __init__(self, batch_size=None):
        self.userItem_rating = {}
        self.user_items = {}
        self.item_users = {}
        self.numUser = 0
        self.numItem = 0
        self.size = 0 #number of none-zero entries
        self.batch_size = batch_size


    @property
    def userIdxs(self):
        return list(self.user_items.keys())

    @property
    def itemIdxs(self):
        return list(self.item_users.keys())

    def put(self, userIdx, itemIdx, rating):
        self.userItem_rating[userIdx, itemIdx] = rating

        if userIdx not in self.user_items:
            self.user_items[userIdx] = []
            self.numUser += 1
        self.user_items[userIdx].append(itemIdx)

        if itemIdx not in self.item_users:
            self.item_users[itemIdx] = []
            self.numItem += 1
        self.item_users[itemIdx].append(userIdx)
        self.size = len(self.userItem_rating)

        # userIdxs, itemIdxs, ratings =[], [], []
        # for userIdx, itemIdx in self:
        #     userIdxs.append(userIdx)
        #     itemIdxs.append(itemIdx)
        #     ratings.append(rating)
        # super(RatingMatrix, self).__init__((ratings, (userIdxs, itemIdxs)))


    def getRating(self, userIdx, itemIdx):
        return self.userItem_rating[userIdx, itemIdx]

    def getDensity(self):
        if self.numUser == 0:
            return 0
        else:
            return len(self.userItem_rating) / (self.numUser * self.numItem)

    @property
    def globalMean(self):
        return sum(self.userItem_rating.values()) / len(self.userItem_rating)

    def size(self):
        return len(self.userItem_rating)

    def getItemsOfUser(self, userIdx):
        if userIdx in self.user_items:
            return self.user_items[userIdx]
        else:
            return []

    def getNumItems(self, userIdx):
        return len(self.user_items[userIdx])

    def getUserRates(self, userIdx):
        rates = []
        items = self.user_items[userIdx]
        for itemIdx in items:
            rates.append(self.getRating(userIdx, itemIdx))
        return rates

    def getUsersOfItem(self, itemIdx):
        if itemIdx in self.item_users:
            return self.item_users[itemIdx]
        else:
            return []

    def getUserRatesVector(self, userIdx):
        rates_vec = np.zeros(shape=self.numItem, dtype=np.short)
        items = self.user_items[userIdx]

        for itemIdx in items:
            rates_vec[itemIdx] = self.getRating(userIdx, itemIdx)
        return rates_vec

    def next_batch(self, batch_num):
        """
        :param batch_num:
        :return: user idxs in next batch
        """
        start = batch_num * self.batch_size
        end = start + self.batch_size
        if end > self.numUser:
            end = self.numUser
        return self.userIdxs[start:end]

    def binarize(self, threshold):
        for (userIdx, itemIdx) in self.userItem_rating:
            if self.userItem_rating[userIdx, itemIdx] >= threshold:
                self.userItem_rating[userIdx, itemIdx] = 1
            else:
                self.userItem_rating[userIdx, itemIdx] = 0

    def __iter__(self):
        return self.userItem_rating.__iter__()

    def __str__(self):
        res = \
        'numRatings: '+str(len(self.userItem_rating)) + '\n' + \
        'numUser: ' + str(self.numUser) + '\n' + \
        'numItem: ' + str(self.numItem)
        return res

if __name__ == '__main__':
    batch_size = 30
    mat = RatingMatrix(batch_size=batch_size)
    for i in range(100):
        mat.put(i, 1, 0)
        mat.put(i, 2, 5)


    for iter in range(10):
        for batch in range(mat.numUser//batch_size+1):
            print(batch)
            print(mat.next_batch(batch))
    # for (userIdx, itemIdx) in mat.userItem_rating:
    #    print(mat.userItem_rating[userIdx, itemIdx])