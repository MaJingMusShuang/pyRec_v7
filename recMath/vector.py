import numpy as np
def noneZeroEntryIdxs(vector):
    """
    :param vector: one hot vector [0, 0, 0, 1, 0]
    :return: [4] idx list of none-zero entry
    """
    labels_len = len(vector)
    noneZeroEntryIdxsList = []
    for idx in range(labels_len):
        if vector[idx] == 1:
            noneZeroEntryIdxsList.append(idx)
    return noneZeroEntryIdxsList

def softmax(vector):
    """
    Compute softmax values for entry in vector
    :param vector:[3.0,1.0, 0.2]
    :return:[ 0.8360188 0.11314284 0.05083836]
    """
    return np.exp(vector)/np.sum(np.exp(vector), axis=0)