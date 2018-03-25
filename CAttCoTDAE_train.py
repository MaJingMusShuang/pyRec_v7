from datamodel.SocialDataModel import SocialDataModel
from recommender.CAttCoTDAE import CAttCoTDAE
import time

if __name__=='__main__':
    batchSize = 1000
    dataConfig = {
        'dataDirectory': 'ciao',
        'outputPath': 'ciao',
        'trainRatio': 0.8,
        'batch_size': 3700
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
        'max_num_items_trust': 3000
    }

    recommender = CAttCoTDAE(config, dataModel)
    recommender.buildModel()
    recommender.trainModel()
    print('trainRatio', dataConfig['trainRatio'])


