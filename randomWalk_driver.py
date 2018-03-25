from datamodel.SocialDataModel import SocialDataModel
from datamodel.DataModel import DataModel
from recommender.RandomWalk import RandomWalk
import pandas as pd
import logging
logger = logging.getLogger(__name__)

if __name__=='__main__':

    dataConfig = {
        'dataDirectory': 'ciao',
        'outputPath': 'ciao',
        'trainRatio': 0.8
    }



    dataModel = SocialDataModel(dataConfig)
    dataModel.buildDataModel()

    config = {
        'topK': 5
    }
    logger.info('\n\n topK{}\n\n\n\n\n'.format(config['topK']))

    numWalks_list = [numWalks for numWalks in range(20, 305, 40)]
    walk_length_list = [walk_length for walk_length in range(5, 305, 20)]
    results = []

    recommender = RandomWalk(config, dataModel)
    recommender.buildGraph()

    rowNum = 0
    max_val = 0
    max_indexs = (-1, -1)
    for numWalks in numWalks_list:
        results.append([])
        for walkLength in walk_length_list:
            walks = recommender.build_random_walks(numWalks=numWalks, walkLength=walkLength)
            recommender.predict(walks)
            avgPrec = recommender.evaluate()
            if avgPrec > max_val:
                max_val = avgPrec
                max_indexs = (numWalks, walkLength)
            results[rowNum].append(avgPrec)
        rowNum += 1
    df = pd.DataFrame(results, index=numWalks_list, columns=walk_length_list)

    logger.info('max:{}, indexs:{}'.format(max_val, max_indexs))
    df.to_csv('Social_results_top5_ciao_fast.csv')