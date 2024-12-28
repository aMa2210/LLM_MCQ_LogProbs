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

## ******************************
    # filenames_direct = ['abstract_algebra_LogProbs_Direct.csv',
    #                     'anatomy_LogProbs_Direct.csv',
    #                     'college_biology_LogProbs_Direct.csv']
    # filenames_think = ['abstract_algebra_LogProbs_afterThinking.csv',
    #                    'anatomy_LogProbs_afterThinking.csv',
    #                    'college_biology_LogProbs_afterThinking.csv']
    #
    # # filenames_direct = ['anatomy_LogProbs_Direct.csv']
    # # filenames_think = ['anatomy_LogProbs_afterThinking.csv']
    #
    # prefixs = ['Results/gpt4o-mini/','Results/llama3.1-8B/','Results/llama3.2-11B-vision-instruct/',
    #            'Results/gemma2-9b-it/','Results/Mistral-7B-Instruct-v0.3/','Results/Yi-1.5-9B-Chat/']
    # model_names = ['GPT-4o-mini','Llama-3.1-8B-Instruct','Llama-3.2-11B-Vision-Instruct','gemma-2-9b-it','Mistral-7B-Instruct-v0.3','Yi-1.5-9B-Chat']
    # filenames_direct = [[prefix + name for name in filenames_direct] for prefix in prefixs]
    # filenames_think = [[prefix + name for name in filenames_think] for prefix in prefixs]
    #
    # plotHistogramsCorrectAllModels(filenames_direct,filenames_think,model_names)
    # plotHistogramsWrongAllModels(filenames_direct,filenames_think,model_names)

## **************************
    # model_names = ['GPT-4o-mini', 'Llama-3.1-8B-Instruct', 'Llama-3.2-11B-Vision-Instruct', 'gemma-2-9b-it',
    #                'Mistral-7B-Instruct-v0.3', 'Yi-1.5-9B-Chat']
    #
    # prefixs = ['Results/gpt4o-mini/','Results/llama3.1-8B/','Results/llama3.2-11B-vision-instruct/',
    #            'Results/gemma2-9b-it/','Results/Mistral-7B-Instruct-v0.3/','Results/Yi-1.5-9B-Chat/']
    #
    # filenames_direct = ['abstract_algebra_LogProbs_Direct.csv']
    # filenames_think = ['abstract_algebra_LogProbs_afterThinking.csv']
    # filenames_direct = [[prefix + name for name in filenames_direct] for prefix in prefixs]
    # filenames_think = [[prefix + name for name in filenames_think] for prefix in prefixs]
    # plotHistogramsCorrectAllModels(filenames_direct,filenames_think,model_names)
    # plotHistogramsWrongAllModels(filenames_direct,filenames_think,model_names)
    # filenames_direct = ['anatomy_LogProbs_Direct.csv']
    # filenames_think = ['anatomy_LogProbs_afterThinking.csv']
    # filenames_direct = [[prefix + name for name in filenames_direct] for prefix in prefixs]
    # filenames_think = [[prefix + name for name in filenames_think] for prefix in prefixs]
    # plotHistogramsCorrectAllModels(filenames_direct,filenames_think,model_names)
    # plotHistogramsWrongAllModels(filenames_direct,filenames_think,model_names)
    # filenames_direct = ['college_biology_LogProbs_Direct.csv']
    # filenames_think = ['college_biology_LogProbs_afterThinking.csv']
    # filenames_direct = [[prefix + name for name in filenames_direct] for prefix in prefixs]
    # filenames_think = [[prefix + name for name in filenames_think] for prefix in prefixs]
    # plotHistogramsCorrectAllModels(filenames_direct,filenames_think,model_names)
    # plotHistogramsWrongAllModels(filenames_direct,filenames_think,model_names)



def plotHistogramsCorrectAllModels(filenames_direct,filenames_think,model_names):
    data = []
    labels = []
    for f_dir,f_think,model_name in zip(filenames_direct, filenames_think, model_names):
        logprob_dir = []
        logprob_think = []
        for filename_dir, filename_think in zip(f_dir, f_think):
            df_dir, df_think = getCorrectDf(filename_dir, filename_think)
            logprob_dir.append(df_dir[['a', 'b', 'c', 'd']].max(axis=1))
            logprob_think.append(df_think[['a', 'b', 'c', 'd']].max(axis=1))
        logprob_dir = list(itertools.chain(*logprob_dir))
        logprob_think = list(itertools.chain(*logprob_think))

        print(model_name+'Correct_dir:'+str(np.std([x - 1 for x in logprob_dir])))
        print(model_name+'Correct_think:'+str(np.std([x - 1 for x in logprob_think])))
        data.append(logprob_dir)
        data.append(logprob_think)
        labels.append(f"{model_name} - Direct")
        labels.append(f"{model_name} - Think")
    PlotComparisonHistogram_subplots(filenames_direct,data,labels)


def plotHistogramsWrongAllModels(filenames_direct,filenames_think,model_names):
    data = []
    labels = []
    for f_dir,f_think,model_name  in zip(filenames_direct, filenames_think, model_names):
        logprob_dir = []
        logprob_think = []
        for filename_dir, filename_think in zip(f_dir, f_think):
            df_dir, df_think = getWrongDf(filename_dir, filename_think)
            logprob_dir.append(df_dir[['a', 'b', 'c', 'd']].max(axis=1))
            logprob_think.append(df_think[['a', 'b', 'c', 'd']].max(axis=1))
        logprob_dir = list(itertools.chain(*logprob_dir))
        logprob_think = list(itertools.chain(*logprob_think))
        print(model_name+'Wrong_dir:'+str(np.std([x - 1 for x in logprob_dir])))
        print(model_name+'Wrong_think:'+str(np.std([x - 1 for x in logprob_think])))
        data.append(logprob_dir)
        data.append(logprob_think)
        labels.append(f"{model_name} - Direct")
        labels.append(f"{model_name} - Think")
    PlotComparisonHistogram_subplots(filenames_direct,data,labels)



def PlotComparisonHistogram_subplots(filenames_direct, data, labels):
    # 确保数据格式正确
    if len(data) % 2 != 0:
        raise ValueError("Data length must be even as it assumes pairs of comparison groups.")

    # 计算总组数
    num_groups = len(data) // 2

    # 设置 bins 和样式
    bins = [i / 10 for i in range(11)]
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange'] * 2
    hatches = ['', '/'] * len(filenames_direct)

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for i, ax in enumerate(axes):
        group_data = data[i::2]
        group_colors = colors[0:len(filenames_direct)]
        group_hatches = hatches[i::2]

        # 绘制直方图
        n, bins, patches = ax.hist(group_data, bins=bins, density=True, alpha=0.75,
                                   color=group_colors, edgecolor='black')


        for patch_set, hatch in zip(patches, group_hatches):
            for patch in patch_set.patches:
                patch.set_hatch(hatch)

        ax.set_xlabel('Value Range', fontsize=16)
        if i == 0:
            ax.set_ylabel('Density', fontsize=16)
        ax.set_xticks([i / 10 for i in range(11)])
        ax.set_xlim(left=0, right=1)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(labels[i::2], fontsize=12)
        ax.set_title(f'{"Answer Directly" if i == 0 else "Answer After Thinking"}', fontsize=16)

    plt.tight_layout()
    plt.show()

def PlotComparisonHistogram(filenames_direct,data,labels):
    bins = [i / 10 for i in range(11)]
    colors = ['red','red', 'green','green', 'blue', 'blue','yellow', 'yellow', 'purple', 'purple','orange','orange']
    hatches = ['', '/']
    hatches = hatches*len(filenames_direct)
    n, bins, patches = plt.hist(data, bins=bins, density=True, alpha=0.75, color=colors,edgecolor='black')
    for patch_set, hatch in zip(patches, hatches):
        for patch in patch_set.patches:
            patch.set_hatch(hatch)

    fontsize = 16

    # plt.title(title,fontsize=fontsize)
    plt.xlabel('Value Range',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.xticks([i / 10 for i in range(11)])
    plt.xlim(left=0, right=1)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.legend(labels, fontsize=12)
    # plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()



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

    plt.title(title,fontsize=fontsize)
    plt.xlabel('Value Range',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.xticks([i / 10 for i in range(11)])
    plt.tick_params(axis='both', labelsize=fontsize)

    plt.show()


if __name__ == '__main__':
    main()
