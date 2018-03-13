from datamodel.SocialDataModel import SocialDataModel
from recommender.CAttCoTDAE_V2 import CAttCoTDAE
from evaluator.RankingEvaluator import RankingEvaluator
import time

if __name__=='__main__':
    batchSize = 1000
    dataConfig = {
        'dataDirectory': 'ciao',
        'outputPath': 'ciao',
        'trainRatio': 0.8,
        'batch_size': 500
    }
    dataModel = SocialDataModel(dataConfig)
    dataModel.buildDataModel()

    config = {
        'numFactor_L1': 20,
        'numFactor_L2': 10,
        'learnRate': 0.01,
        'maxIter': 1000,
        'alpha': 0.2,
        'lam': 0.1,
        'beta': 1,
        'optiType': 'adam',
        'outputType': 'rating',
        'lossType': 'mse',
        'trainType': 'train',
    }

    recommender = CAttCoTDAE(config, dataModel)
    recommender.buildModel()
    recommender.trainModel()
    print('trainRatio', dataConfig['trainRatio'])



        # numFactor_L2_List = [20, 50, 100, 200]
        #
        # for numFactor_L2 in numFactor_L2_List:
        #     dataConfig = {
        #         'dataDirectory': 'filmtrust',
        #         'outputPath': 'filmtrust',
        #         'trainRatio': 0.7,
        #         'validRatio': 0.1,
        #     }
        #     dataModel = SocialDataModel(dataConfig)
        #     dataModel.buildDataModel()
        #
        #     config = {
        #         'numFactor_L1': 100,
        #         'numFactor_L2': 10,
        #         'learnRate': 0.001,
        #         'maxIter': 1000,
        #         'numFactor': 10,
        #         'batchSize': 1000,
        #         'alpha': 0.2,
        #         'lam': 0.1,
        #         'beta': 0.1,
        #         'optiType': 'gd',
        #         'outputType': 'rating',
        #         'trainType': 'test',
        #         'lossType': 'mse',
        #         'seed': 1,
        #         'precisionK': 10,
        #     }
        #     recommender = TDAE(config, dataModel)
        #     recommender.buildModel()
        #     recommender.trainModel()

