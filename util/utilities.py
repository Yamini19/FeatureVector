def label(data, column):
    unique = data[column].unique()
    k = 0
    for str in unique:
        data.loc[data[column] == str, column] = k
        k += 1


def replace(data, weights, features, drop=True):
    for feature in features:
        data = data.merge(weights[feature], how='left', on=[feature])
        if drop == True:
            data = data.drop([feature], axis=1)
    return data
