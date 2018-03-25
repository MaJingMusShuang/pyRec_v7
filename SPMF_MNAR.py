from datamodel.SocialDataModel import SocialDataModel

if __name__=='__main__':
    batchSize = 1000
    dataConfig = {
        'dataDirectory': 'ciao',
        'trainRatio': 1,
        'batch_size': 500
    }
    dataModel = SocialDataModel(dataConfig)
    dataModel.readRatingData()
    dataModel.readSocialData()
    dataModel.pccSimilarity()
    dataModel.jaccardSimilarity()
    # dataModel.figure2b()

