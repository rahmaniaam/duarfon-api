import pandas as pd
import numpy as np

from random import seed
from random import randrange

def custom_one_hot_encoder(df, test_df, target, cat=[]):

    for col in cat:
        target_0 = set(df[df[target] == 0][col].unique())
        target_1 = set(df[df[target] == 1][col].unique())
        targeted = target_1

        if len(target_0) < len(target_1):
            targeted = target_0

        for t in targeted:
            df[col + '_{}'.format(t)] = 0
            test_df[col + '_{}'.format(t)] = 0

        for i, row in df.iterrows():
            if row[col] in targeted:
                df.set_value(i,col + '_{}'.format(row[col]),1)

        for i, row in test_df.iterrows():
            if row[col] in targeted:
                test_df.set_value(i,col + '_{}'.format(row[col]),1)

    return df, test_df

def fillNaN(dataset):
    for row_idx in range(dataset.shape[0]):
        for col_idx in range(dataset.shape[1]):
            if np.isnan(dataset[row_idx][col_idx]):
                dataset[row_idx][col_idx] = 0.0
    return dataset

def giniIndex(groups, classes):
    gini = 0.0
    totalSamples = float(sum([len(group) for group in groups]))
    for group in groups:
        score = 0.0
        groupSize = float(len(group))
        if groupSize == 0:
            continue
        for classValue in classes:
            proportion = [row[-1] for row in group].count(classValue) / groupSize
            score += proportion * proportion
        gini += (1.0 - score) * (groupSize / totalSamples)
    return gini

def testSplit(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def getSplit(dataset):
    bIndex, bValue, bScore, bGroups = 999, 999, 999, None
    classValues = list(set(row[-1] for row in dataset))
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = testSplit(index, row[index], dataset)
            gini = giniIndex(groups, classValues)
            if gini < bScore:
                bIndex, bValue, bScore, bGroups = index, row[index], gini, groups
    return {'index':bIndex, 'value':bValue, 'groups':bGroups}

def finalNode(group):
    res = [row[-1] for row in group]
    return max(set(res), key=res.count)

def split(node, maxDepth, minSize, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = finalNode(left + right)
        return
    if depth >= maxDepth:
        node['left'], node['right'] = finalNode(left), finalNode(right)
        return
    if len(left) <= minSize:
        node['left'] = finalNode(left)
    else:
        node['left'] = getSplit(left)
        split(node['left'], maxDepth, minSize, depth+1)
    if len(right) <= minSize:
        node['right'] = finalNode(right)
    else:
        node['right'] = getSplit(right)
        split(node['right'], maxDepth, minSize, depth+1)

def buildTree(train, maxDepth, minSize):
    root = getSplit(train)
    split(root, maxDepth, minSize, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def decisionTree(train, test, maxDepth, minSize):
    tree = buildTree(train, maxDepth, minSize)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

def removearray(dataset,arr):
    index = 0
    size = len(dataset)
    while index != size and not np.array_equal(dataset[index],arr):
        index += 1
    if index != size:
        dataset.pop(index)
    else:
        raise ValueError('array not found in list.')

def accuracy(actual, predicted):
    correct = 0
    for i in range (len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual)) * 100.0

def run():
    train = pd.read_csv("fraud_train.csv")
    train = train.drop(["flag_transaksi_finansial"], axis=1)
    train = train.drop(["status_transaksi"], axis=1)
    train = train.drop(["bank_pemilik_kartu"], axis=1)

    test = pd.read_csv("fraud_test.csv")
    test = test.drop(["flag_transaksi_finansial"], axis=1)
    test = test.drop(["status_transaksi"], axis=1)
    test = test.drop(["bank_pemilik_kartu"], axis=1)
    target = "flag_transaksi_fraud"

    cat = [
        "tipe_kartu",
        "id_merchant",
        "nama_merchant",
        "tipe_mesin",
        "tipe_transaksi",
        "nama_negara",
        "nama_kota",
        "lokasi_mesin",
        "pemilik_mesin",
        "kepemilikan_kartu",
        "nama_channel",
        "id_channel"
    ]

    df = train.copy()[:100]
    test_df = test.copy()[:100]
    df, test_df = custom_one_hot_encoder(df, test_df, target, cat)

    dataTrain = fillNaN(df.values)
    dataTest = fillNaN(test_df.values)

    max_depth = 500
    min_size = 1
    datatest = train[100:13125].values
    fortest = list()
    for row in datatest:
        fortest.append(row[-1])

    # actual = pd.read_csv("csvFile.csv")
    pred = decisionTree(dataTrain, datatest, max_depth, min_size)
    score = accuracy(fortest, pred)
    # print(score)

    result = list()
    for x in range(datatest.shape[0]):
        result.append([datatest[x][0], pred[x]])
    # print(len(result))

    # print("target: {}\n".format(str(target)))
    count0 = 0
    count1 = 0
    for row in result:
        if row[1]==0.0:
            count0 += 1
        if row[1]==1.0:
            count1 += 1
    # print("0:\t {}".format(str(count0)))
    # print("1:\t {}".format(str(count1)))

    results = pd.DataFrame(result, columns=['X', target])
    return results
    # results[target].hist()
    # plt.xlabel(target)
    # plt.ylabel('jumlah_transaksi')
    # plt.show()

    # labels = 'X', 'flag_transaksi_fraud'
    # xSize = count0/(count0+count1)
    # flagSize = count1/(count0+count1)
    # sizes = [xSize, flagSize]
    # explode = (0, 0.1)

    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    # ax1.axis('equal')

    # plt.show()

    # results.to_csv('predictionResult.csv')
    # print('suds')