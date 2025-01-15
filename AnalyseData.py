import pandas as pd
import matplotlib.pyplot as plt


def getAverageProb(df):
    df['max_value'] = df[['a', 'b', 'c', 'd']].max(axis=1)
    average_max_value = df['max_value'].mean()
    return average_max_value
def getDf(filename_dir, filename_think = None):
    if filename_think is not None:
        if isinstance(filename_dir, str) & isinstance(filename_think, str):
            df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
            df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
        if isinstance(filename_dir, list) & isinstance(filename_think, list):
            df_dir = []
            for filename_1 in filename_dir:
                df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
                df_tmp = deleteOutliers(df_tmp)
                df_dir.append(df_tmp)
            df_dir = pd.concat(df_dir, ignore_index=True)

            df_think = []
            for filename_1 in filename_think:
                df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
                df_tmp = deleteOutliers(df_tmp)
                df_think.append(df_tmp)
            df_think = pd.concat(df_think, ignore_index=True)

        df_dir = deleteOutliers(df_dir)
        df_think = deleteOutliers(df_think)
        return df_dir, df_think
    else:
        if isinstance(filename_dir, str):
            df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
        if isinstance(filename_dir, list):
            df_dir = []
            for filename_1 in filename_dir:
                df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
                df_tmp = deleteOutliers(df_tmp)
                df_dir.append(df_tmp)
            df_dir = pd.concat(df_dir, ignore_index=True)
        df_dir = deleteOutliers(df_dir)

        return df_dir

def getWrongDf(filename_dir, filename_think = None):
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    if filename_think is not None:
        if isinstance(filename_dir, str) & isinstance(filename_think, str):
            df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
            df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
        if isinstance(filename_dir, list) & isinstance(filename_think, list):
            df_dir = []
            for filename_1 in filename_dir:
                df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
                df_tmp = deleteOutliers(df_tmp)
                df_dir.append(df_tmp)
            df_dir = pd.concat(df_dir, ignore_index=True)

            df_think = []
            for filename_1 in filename_think:
                df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
                df_tmp = deleteOutliers(df_tmp)
                df_think.append(df_tmp)
            df_think = pd.concat(df_think, ignore_index=True)

        df_dir = deleteOutliers(df_dir)
        df_think = deleteOutliers(df_think)


        df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0) # fill null value
        df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)

        df_think[['a', 'b', 'c', 'd']] = df_think[['a', 'b', 'c', 'd']].fillna(0) # fill null value
        df_think['predicted'] = df_think[['a', 'b', 'c', 'd']].idxmax(axis=1)

        df_dir_wrong = df_dir[df_dir['predicted'] != df_dir['answer'].map(answer_map)]
        df_think_wrong = df_think[df_think['predicted'] != df_think['answer'].map(answer_map)]

        return df_dir_wrong, df_think_wrong
    else:
        if isinstance(filename_dir, str):
            df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
        if isinstance(filename_dir, list):
            df_dir = []
            for filename_1 in filename_dir:
                df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
                df_tmp = deleteOutliers(df_tmp)
                df_dir.append(df_tmp)
            df_dir = pd.concat(df_dir, ignore_index=True)
        df_dir = deleteOutliers(df_dir)
        answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0)  # fill null value
        df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)
        df_dir_wrong = df_dir[df_dir['predicted'] != df_dir['answer'].map(answer_map)]
        return df_dir_wrong


def getCorrectDf(filename_dir, filename_think = None):
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    if filename_think is not None:
        df_dir,df_think = getDf(filename_dir,filename_think)
        # if isinstance(filename_dir, str) & isinstance(filename_think, str):
        #     df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
        #     df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
        # if isinstance(filename_dir, list) & isinstance(filename_think, list):
        #     df_dir = []
        #     for filename_1 in filename_dir:
        #         df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
        #         df_tmp = deleteOutliers(df_tmp)
        #         df_dir.append(df_tmp)
        #     df_dir = pd.concat(df_dir, ignore_index=True)
        #
        #     df_think = []
        #     for filename_1 in filename_think:
        #         df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
        #         df_tmp = deleteOutliers(df_tmp)
        #         df_think.append(df_tmp)
        #     df_think = pd.concat(df_think, ignore_index=True)

        df_dir = deleteOutliers(df_dir)
        df_think = deleteOutliers(df_think)


        df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0) # fill null value
        df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)

        df_think[['a', 'b', 'c', 'd']] = df_think[['a', 'b', 'c', 'd']].fillna(0) # fill null value
        df_think['predicted'] = df_think[['a', 'b', 'c', 'd']].idxmax(axis=1)

        df_dir_correct = df_dir[df_dir['predicted'] == df_dir['answer'].map(answer_map)]
        df_think_correct = df_think[df_think['predicted'] == df_think['answer'].map(answer_map)]


        return df_dir_correct, df_think_correct

    else:
        df_dir = getDf(filename_dir)
        df_dir = deleteOutliers(df_dir)
        df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0)  # fill null value
        df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)
        df_dir_correct = df_dir[df_dir['predicted'] == df_dir['answer'].map(answer_map)]
        return df_dir_correct


def getBothWrongDf(filename_dir, filename_think):

    df_dir,df_think = getDf(filename_dir,filename_think)
    # pd.read_csv(filename_dir, encoding='ISO-8859-1')
    # df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')

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
    df_dir_both_wrong = deleteOutliers(df_dir_both_wrong)
    df_think_both_wrong = deleteOutliers(df_think_both_wrong)
    return df_dir_both_wrong, df_think_both_wrong

def getWrong2CorrectDf(filename_dir, filename_think):

    # df_dir = pd.read_csv(filename_dir, encoding='ISO-8859-1')
    # df_think = pd.read_csv(filename_think, encoding='ISO-8859-1')
    df_dir,df_think = getDf(filename_dir,filename_think)
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    df_dir[['a', 'b', 'c', 'd']] = df_dir[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_dir['predicted'] = df_dir[['a', 'b', 'c', 'd']].idxmax(axis=1)

    df_think[['a', 'b', 'c', 'd']] = df_think[['a', 'b', 'c', 'd']].fillna(0) # fill null value
    df_think['predicted'] = df_think[['a', 'b', 'c', 'd']].idxmax(axis=1)

    # 计算错误行：predicted != answer
    df_dir_wrong = df_dir[df_dir['predicted'] != df_dir['answer'].map(answer_map)]
    df_think_correct = df_think[df_think['predicted'] == df_think['answer'].map(answer_map)]

    # 交集：同时错误的行
    df_dir_result = df_dir_wrong[df_dir_wrong.index.isin(df_think_correct.index)]
    df_think_result = df_think_correct[df_think_correct.index.isin(df_dir_wrong.index)]
    df_dir_result = deleteOutliers(df_dir_result)
    df_think_result = deleteOutliers(df_think_result)
    return df_dir_result, df_think_result


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
    df_dir_consistent = deleteOutliers(df_dir_consistent)
    df_think_consistent = deleteOutliers(df_think_consistent)
    return df_dir_consistent, df_think_consistent

def getAccuracy(filename):
    if isinstance(filename, list):
        df = []
        for filename_1 in filename:
            df_tmp = pd.read_csv(filename_1, encoding='ISO-8859-1')
            df_tmp = deleteOutliers(df_tmp)
            df.append(df_tmp)
        df = pd.concat(df, ignore_index=True)


    elif isinstance(filename, str):
        df = pd.read_csv(filename, encoding='ISO-8859-1')
        df = deleteOutliers(df)
    elif isinstance(filename, pd.DataFrame):
        df = filename
    else:
        raise TypeError("filename is not a list neither a str")
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    df[['a', 'b', 'c', 'd']] = df[['a', 'b', 'c', 'd']].fillna(0) # fill null value

    df['predicted'] = df[['a', 'b', 'c', 'd']].idxmax(axis=1)

    # calculate accuracy
    correct_predictions = (df['predicted'] == df['answer'].map(answer_map)).sum()
    total_predictions = len(df)
    # print(total_predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy

def deleteOutliers(df_ori): # delete rows where the sum of four options is less than 0.5
    df = df_ori.copy()
    df.loc[:, 'sum'] = df[['a', 'b', 'c', 'd']].sum(axis=1)
    df = df[df['sum'] >= 0.5]
    df = df.drop(columns=['sum'])

    return df
