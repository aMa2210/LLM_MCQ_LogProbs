import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AnalyseData import getBothWrongDf,getAgreedProb,getAccuracy,getWrongDf,getDf,getCorrectDf
import itertools

def main():

    filenames_direct = ['abstract_algebra_LogProbs_Direct.csv',
                        'anatomy_LogProbs_Direct.csv',
                        'college_biology_LogProbs_Direct.csv']
    filenames_think = ['abstract_algebra_LogProbs_afterThinking.csv',
                       'anatomy_LogProbs_afterThinking.csv',
                       'college_biology_LogProbs_afterThinking.csv']
    # prefix = 'Results/gpt4o-mini/'
    # prefix = 'Results/llama3.1-8B/'
    # prefix = 'Results/llama3.2-11B-vision-instruct/'
    # prefix = 'Results/gemma2-9b-it/'
    # prefix = 'Results/Mistral-7B-Instruct-v0.3/'
    prefix = 'Results/Yi-1.5-9B-Chat/'
    filenames_direct = [prefix + name for name in filenames_direct]
    filenames_think = [prefix + name for name in filenames_think]

    plotAccuracy(filenames_direct,filenames_think)

    plotChosenProb(filenames_direct,filenames_think)
    plotNotChosenProb(filenames_direct,filenames_think)

    plotWrongAnswerProbChange(filenames_direct, filenames_think)
    plotWrongAnswerRemainingProbChange(filenames_direct, filenames_think)
    plotHistogramsAll(filenames_direct, filenames_think)
    plotHistogramsCorrect(filenames_direct, filenames_think)
    plotHistogramsWrong(filenames_direct, filenames_think)

def plotHistogramsAll(filenames_direct,filenames_think):
    logprob_dir = []
    logprob_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        df_dir, df_think = getDf(filename_dir, filename_think)
        logprob_dir.append(df_dir[['a', 'b', 'c', 'd']].max(axis=1))
        logprob_think.append(df_think[['a', 'b', 'c', 'd']].max(axis=1))
    logprob_dir = list(itertools.chain(*logprob_dir))
    logprob_think = list(itertools.chain(*logprob_think))
    plotHistograms(logprob_dir,'Probability Distribution of Answers for All Questions(Direct)')
    plotHistograms(logprob_think,'Probability Distribution of Answers for All Questions(After Thinking)')

def plotHistogramsCorrect(filenames_direct,filenames_think):
    logprob_dir = []
    logprob_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        df_dir, df_think = getCorrectDf(filename_dir, filename_think)
        logprob_dir.append(df_dir[['a', 'b', 'c', 'd']].max(axis=1))
        logprob_think.append(df_think[['a', 'b', 'c', 'd']].max(axis=1))
    logprob_dir = list(itertools.chain(*logprob_dir))
    logprob_think = list(itertools.chain(*logprob_think))
    plotHistograms(logprob_dir,'Distribution of Probabilities for Correct Answers(Direct)')
    plotHistograms(logprob_think,'Distribution of Probabilities for Correct Answers(After Thinking)')

def plotHistogramsWrong(filenames_direct,filenames_think):
    logprob_dir = []
    logprob_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        df_dir, df_think = getWrongDf(filename_dir, filename_think)
        logprob_dir.append(df_dir[['a', 'b', 'c', 'd']].max(axis=1))
        logprob_think.append(df_think[['a', 'b', 'c', 'd']].max(axis=1))
    logprob_dir = list(itertools.chain(*logprob_dir))
    logprob_think = list(itertools.chain(*logprob_think))
    plotHistograms(logprob_dir,'Distribution of Probabilities for Wrong Answers(Direct)')
    plotHistograms(logprob_think,'Distribution of Probabilities for Wrong Answers(After Thinking)')

def plotWrongAnswerProbChange(filenames_direct,filenames_think):
    average_logprob_dir = []
    average_logprob_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        df_dir, df_think = getBothWrongDf(filename_dir, filename_think)
        df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value = df_dir['max_value'].mean()
        average_logprob_dir.append(average_max_value)

        df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value2 = df_think['max_value'].mean()
        average_logprob_think.append(average_max_value2)

    plotAccuracyComparison(average_logprob_dir, average_logprob_think, xlabel_name='Dataset',
                           ylabel_name='Average Probability',
                           title='Comparison of Average Probability for Chosen Option in Wrong Answer Only')

def plotWrongAnswerRemainingProbChange(filenames_direct,filenames_think):
    average_logprob_dir = []
    average_logprob_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        df_dir, df_think = getBothWrongDf(filename_dir, filename_think)

        df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
        df_dir['sum_remaining'] = df_dir[['a', 'b', 'c', 'd']].sum(axis=1) - df_dir['max_value']
        average_sum_remaining = df_dir['sum_remaining'].mean()
        average_logprob_dir.append(average_sum_remaining)

        df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
        df_think['sum_remaining'] = df_think[['a', 'b', 'c', 'd']].sum(axis=1) - df_think['max_value']
        average_sum_remaining2 = df_think['sum_remaining'].mean()
        average_logprob_think.append(average_sum_remaining2)

    print(average_logprob_think)
    plotAccuracyComparison(average_logprob_dir, average_logprob_think, xlabel_name='Dataset',
                           ylabel_name='Average Probability',
                           title='Comparison of Average Probability for Remaining Option in Wrong Answer Only', pos='upper right')



def plotNotChosenProb(filenames_direct,filenames_think):
    # plot logprob when two results agree on the option
    average_logprob_dir = []
    average_logprob_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        df_dir,df_think = getAgreedProb(filename_dir,filename_think)

        df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
        df_dir['sum_remaining'] = df_dir[['a', 'b', 'c', 'd']].sum(axis=1) - df_dir['max_value']
        average_sum_remaining = df_dir['sum_remaining'].mean()
        average_logprob_dir.append(average_sum_remaining)

        df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
        df_think['sum_remaining'] = df_think[['a', 'b', 'c', 'd']].sum(axis=1) - df_think['max_value']
        average_sum_remaining2 = df_think['sum_remaining'].mean()
        average_logprob_think.append(average_sum_remaining2)

    plotAccuracyComparison(average_logprob_dir, average_logprob_think, xlabel_name='Dataset', ylabel_name='Average Probability',
                           title='Comparison of Average Sum of Probabilities for Remaining Options', pos='upper right')

def plotChosenProb(filenames_direct,filenames_think):
    # plot logprob when two results agree on the option
    average_logprob_dir = []
    average_logprob_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        df_dir,df_think = getAgreedProb(filename_dir,filename_think)

        df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value = df_dir['max_value'].mean()
        average_logprob_dir.append(average_max_value)

        df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value2 = df_think['max_value'].mean()
        average_logprob_think.append(average_max_value2)

    plotAccuracyComparison(average_logprob_dir, average_logprob_think, xlabel_name='Dataset', ylabel_name='Average Probability',
                           title='Comparison of Average Probability for Chosen Option')

def plotAccuracy(filenames_direct, filenames_think):

    # plot accuracy comparison barchart
    accuracies_direct = []
    accuracies_think = []
    for filename_dir, filename_think in zip(filenames_direct, filenames_think):
        acc_dir = getAccuracy(filename_dir)
        acc_think = getAccuracy(filename_think)
        accuracies_direct.append(acc_dir)
        accuracies_think.append(acc_think)
    plotAccuracyComparison(accuracies_direct, accuracies_think)



def plotAccuracyComparison(accuracies_direct, accuracies_think, xlabel_name='Dataset', ylabel_name='Accuracy',
                           title='Comparison of Accuracy for Different Datasets',pos = 'upper left'):

    labels = ['abstract_algebra', 'anatomy', 'college_biology']

    fontsize = 16
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(x, accuracies_direct, width, label='Direct', color='blue')
    ax.bar([i + width for i in x], accuracies_think, width, label='After Thinking', color='orange')

    ax.set_xlabel(xlabel_name, fontsize=fontsize)
    ax.set_ylabel(ylabel_name, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    ax.legend(fontsize=fontsize,loc=pos)

    plt.show()


def plotHistograms(data, title):
    bins = [i / 10 for i in range(11)]
    plt.hist(data, bins=bins, density=True, alpha=0.75, color='blue', edgecolor='black')

    fontsize = 16
    # 设置图表标题和标签
    plt.title(title,fontsize=fontsize)
    plt.xlabel('Value Range',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.xticks([i / 10 for i in range(11)])
    plt.tick_params(axis='both', labelsize=fontsize)
    # 显示图表
    plt.show()


if __name__ == '__main__':
    main()
