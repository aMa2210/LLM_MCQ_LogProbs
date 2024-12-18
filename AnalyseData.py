import pandas as pd
import matplotlib.pyplot as plt

def getDf(filename_dir, filename_think):

    df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
    df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')

    return df_dir, df_think

def getWrongDf(filename_dir, filename_think):

    df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
    df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_think[['a', 'b', 'c', 'd']] = df_think[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_think['predicted'] = df_think[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_dir_wrong = df_dir[df_dir['predicted'] != df_dir['answer'].map(answer_map)]
    df_think_wrong = df_think[df_think['predicted'] != df_think['answer'].map(answer_map)]

    return df_dir_wrong, df_think_wrong

def getCorrectDf(filename_dir, filename_think):

    df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
    df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_think[['a', 'b', 'c', 'd']] = df_think[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_think['predicted'] = df_think[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_dir_correct = df_dir[df_dir['predicted'] == df_dir['answer'].map(answer_map)]
    df_think_correct = df_think[df_think['predicted'] == df_think['answer'].map(answer_map)]


    return df_dir_correct, df_think_correct


def getBothWrongDf(filename_dir, filename_think):

    df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
    df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_think[['a', 'b', 'c', 'd']] = df_think[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_think['predicted'] = df_think[['a', 'b', 'c', 'd']].idxmax(axis=1)

    # 计算错误行：predicted != answer
    df_dir_wrong = df_dir[df_dir['predicted'] != df_dir['answer'].map(answer_map)]
    df_think_wrong = df_think[df_think['predicted'] != df_think['answer'].map(answer_map)]

    # 交集：同时错误的行
    df_dir_both_wrong = df_dir_wrong[df_dir_wrong.index.isin(df_think_wrong.index)]
    df_think_both_wrong = df_think_wrong[df_think_wrong.index.isin(df_dir_wrong.index)]

    return df_dir_both_wrong, df_think_both_wrong

def getAgreedProb(filename_dir, filename_think):

    df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
    df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_think[['a', 'b', 'c', 'd']] = df_think[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_think['predicted'] = df_think[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_dir_consistent = df_dir[df_dir['predicted'] == df_think['predicted']]
    df_think_consistent = df_think[df_dir['predicted'] == df_think['predicted']]

    return df_dir_consistent, df_think_consistent

def getAccuracy(filename):

    df = pd.read_csv(filename, encoding='ISO-8859-1')

    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    df[['a', 'b', 'c', 'd']] = df[['a', 'b', 'c', 'd']].fillna(0) # fill null value

    df['predicted'] = df[['a', 'b', 'c', 'd']].idxmax(axis=1)

    # calculate accuracy
    correct_predictions = (df['predicted'] == df['answer'].map(answer_map)).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions
    return accuracy
