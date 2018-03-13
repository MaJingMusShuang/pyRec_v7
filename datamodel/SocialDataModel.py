from .DataModel import DataModel


class SocialDataModel(DataModel):
    def __init__(self, config):
        super().__init__(config)
        self.socialInputPath = './data' + '/' + self.dataDirectory+ '/trusts.txt'
        self.wholeTrustDataSet = []
        self.userIdx_FriendIndicies = {} # {1:[1, 2, 3, 4,...]} {userIdx: friendIndices}

        self.userIdxs_CICT = [] # Cold Item Cold Trust
        self.userIdxs_CIWT = []
        self.userIdxs_WICT = []
        self.userIdxs_WIWT = []

    def buildDataModel(self, **crossValidation):
        super().buildDataModel(**crossValidation)
        self.readConvertSocialData()

    def readConvertSocialData(self):
        self.logger.info('Reading and converting social data...')
        trusterSet = set()
        trusteeSet = set()

        # initilize every user only trust himself/herself at the very begining
        # for userIdx in self.trainMatrix.userIdxs:
        #     self.userIdx_FriendIndicies[userIdx] = [userIdx]
        #     self.wholeTrustDataSet.append([userIdx, userIdx])

        with open(self.socialInputPath) as inputFile:
            for line in inputFile:
                thisUserId, thatUserId, trust = line.strip().split(' ')
                if thisUserId in self.userId_userIdx and thatUserId in self.userId_userIdx:
                    thisUserIdx = self.userId_userIdx[thisUserId]
                    thatUserIdx = self.userId_userIdx[thatUserId]
                    if thisUserIdx not in self.trainMatrix.userIdxs or thatUserIdx not in self.trainMatrix.userIdxs:
                        continue #filter out users who are not in train set
                    self.wholeTrustDataSet.append([thisUserIdx, thatUserIdx])
                    if thisUserIdx not in self.userIdx_FriendIndicies:
                        self.userIdx_FriendIndicies[thisUserIdx] = []
                        self.wholeTrustDataSet.append([thisUserIdx, thisUserIdx])
                    self.userIdx_FriendIndicies[thisUserIdx].append(thatUserIdx)
                    trusterSet.add(thisUserIdx)
                    trusteeSet.add(thatUserIdx)
        self.logger.info('num Trust: ' + str(len(self.wholeTrustDataSet)))
        self.logger.info('num Trusters: ' + str(len(trusterSet)))
        self.logger.info('num Trustees: ' + str(len(trusteeSet)))
        self.logger.info('Density: ' + str(len(self.wholeTrustDataSet) / (self.numUser * self.numUser)))


    def divideUsers(self):
        sum_user_items = 0
        sum_trust_items = 0
        for userIdx in self.preferenceMatrix.userIdxs:
            sum_user_items += self.preferenceMatrix.getNumItems(userIdx)
            sum_trust_items += len(self.getTrustItemsSet(userIdx))
        avg_user_items = sum_user_items / self.preferenceMatrix.numUser
        avg_trust_items = sum_trust_items / self.preferenceMatrix.numUser
        self.logger.info('avg user items: ' + str(avg_user_items))
        self.logger.info('avg trust items: ' + str(avg_trust_items))
        splitter = avg_user_items / 4 # users who have ratings less than spliiter are cold item users

        for userIdx in self.preferenceMatrix.userIdxs:
            len_items = self.preferenceMatrix.getNumItems(userIdx)
            len_friends = len(self.getTrustUserIdxs(userIdx))

            if len_items < splitter and len_friends == 0:
                self.userIdxs_CICT.append(userIdx)
            elif len_items < splitter and len_friends != 0:
                self.userIdxs_CIWT.append(userIdx)
            elif len_items >= splitter and len_friends == 0:
                self.userIdxs_WICT.append(userIdx)
            elif len_items >= splitter and len_friends != 0:
                self.userIdxs_WIWT.append(userIdx)
        self.logger.info('num of CIWT user: ' + str(len(self.userIdxs_CIWT)))

        userIdxs_CIWT = []
        for userIdx in self.userIdxs_CIWT:
            if userIdx  in self.testMatrix.userIdxs:
                userIdxs_CIWT.append(userIdx)
        self.userIdxs_CIWT = userIdxs_CIWT
        self.logger.info('num of CIWT user in testSet: ' + str(len(self.userIdxs_CIWT)))



    def getTrustItemsSet(self, userIdx):
        trustItemSet = set()
        if userIdx in self.userIdx_FriendIndicies.keys():
            trustUserIdxs = self.userIdx_FriendIndicies[userIdx]
            for trustUserIdx in trustUserIdxs:
                trustUserItems = self.preferenceMatrix.getItemsOfUser(trustUserIdx)
                trustItemSet.update(trustUserItems)
        return trustItemSet

    def getTrustUserIdxs(self, userIdx):
        userIdx_friends = []
        if userIdx in self.userIdx_FriendIndicies.keys():
            userIdx_friends = self.userIdx_FriendIndicies[userIdx]
        return userIdx_friends

