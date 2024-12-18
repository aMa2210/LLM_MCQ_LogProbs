import pandas as pd
import matplotlib.pyplot as plt
from AnalyseData import getBothWrongDf,getAgreedProb,getAccuracy

def main():
    filenames_direct = ['Results/abstract_algebra_LogProbs_Direct.csv',
                        'Results/anatomy_LogProbs_Direct.csv',
                        'Results/college_biology_LogProbs_Direct.csv']
    filenames_think = ['Results/abstract_algebra_LogProbs_afterThinking.csv',
                       'Results/anatomy_LogProbs_afterThinking.csv',
                       'Results/college_biology_LogProbs_afterThinking.csv']
    # plotChosenProb(filenames_direct,filenames_think)
    # plotNotChosenProb(filenames_direct,filenames_think)
    # plotAccuracy(filenames_direct,filenames_think)
    plotWrongAnswerProbChange(filenames_direct, filenames_think)
    # plotWrongAnswerRemainingProbChange(filenames_direct, filenames_think)

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
                           title='Comparison of Average Probability for Remaining Option in Wrong Answer Only')



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
                           title='Comparison of Average Sum of Probabilities for Remaining Options')

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
                           title='Comparison of Accuracy for Different Datasets'):

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

    ax.legend(fontsize=fontsize,loc='upper left')

    plt.show()


if __name__ == '__main__':
    main()
