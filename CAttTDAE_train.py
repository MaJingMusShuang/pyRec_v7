from datamodel.SocialDataModel import SocialDataModel
from recommender.CAttTDAE import CAttTDAE
import time

if __name__=='__main__':
    batchSize = 1000
    dataConfig = {
        'dataDirectory': 'filmtrust',
        'outputPath': 'filmtrust',
        'trainRatio': 0.8,
        'batch_size': 3600
    }
    dataModel = SocialDataModel(dataConfig)
    dataModel.buildDataModel()

    config = {
        'numFactor_L1': 20,
        'numFactor_L2': 10,
        'learnRate': 0.01,
        'maxIter': 3000,
        'alpha': 0.2,
        'lam': 0.1,
        'beta': 1,
        'optiType': 'adam',
        'outputType': 'rating',
        'lossType': 'mse',
        'trainType': 'train',
    }

    recommender = CAttTDAE(config, dataModel)
    recommender.buildModel()
    recommender.trainModel()
    print('trainRatio', dataConfig['trainRatio'])


