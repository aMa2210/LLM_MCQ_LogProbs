import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AnalyseData import getBothWrongDf,getAgreedProb,getAccuracy,getWrongDf,getDf,getCorrectDf,getAverageProb,getWrong2CorrectDf
import itertools
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle
def main():

    file_names = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology',
        'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics',
        'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
        'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics',
        'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
        'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
        'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
        'nutrition',
        'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
        'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'virology', 'world_religions'
    ]
    filenames_direct = [f"{file}_LogProbs_Direct.csv" for file in file_names]

    filenames_think = [f"{file}_LogProbs_afterThinking.csv" for file in file_names]

    # prefix = 'Results/gpt4o-mini/'
    # # prefix = 'Results/llama3.1-8B/'
    # # prefix = 'Results/llama3.2-11B-vision-instruct/'
    # # prefix = 'Results/gemma2-9b-it/'
    # # prefix = 'Results/Mistral-7B-Instruct-v0.3/'
    # # prefix = 'Results/Yi-1.5-9B-Chat/'
    #
    # filenames_direct = [prefix + name for name in filenames_direct]
    # filenames_think = [prefix + name for name in filenames_think]
    #
    # # plotAccuracy(filenames_direct,filenames_think)
    #
    # # plotChosenProb(filenames_direct,filenames_think)
    # # plotNotChosenProb(filenames_direct,filenames_think)
    #
    # # plotWrongAnswerProbChange(filenames_direct, filenames_think)
    # # plotWrongAnswerRemainingProbChange(filenames_direct, filenames_think)
    # plotHistogramsAll(filenames_direct, filenames_think)
    # plotHistogramsCorrect(filenames_direct, filenames_think)
    # plotHistogramsWrong(filenames_direct, filenames_think)

## ******************************

    prefixs = ['Results/gpt4o-mini/','Results/gpt4o/','Results/llama3.1-8B/','Results/llama3.2-11B-vision-instruct/',
               'Results/gemma2-9b-it/','Results/Mistral-7B-Instruct-v0.3/','Results/Yi-1.5-9B-Chat/']
    model_names = ['GPT-4o-mini','GPT-4o','Llama-3.1-8B-Instruct','Llama-3.2-11B-Vision-Instruct','gemma-2-9b-it','Mistral-7B-Instruct-v0.3','Yi-1.5-9B-Chat']
    filenames_direct = [[prefix + name for name in filenames_direct] for prefix in prefixs]
    filenames_think = [[prefix + name for name in filenames_think] for prefix in prefixs]

    # plotHistogramsCorrectAllModels(filenames_direct,filenames_think,model_names)
    # plotHistogramsWrongAllModels(filenames_direct,filenames_think,model_names)
    model_names = ['GPT-4o-mini', 'GPT-4o', 'Llama-3.1-8B', 'Llama-3.2-11B', 'gemma-2-9b',
                   'Mistral-7B', 'Yi-1.5-9B']

    # plotAccuracy_AllDataset(filenames_direct=filenames_direct,filenames_think=filenames_think,model_names=model_names)
    # plotAverageProb_Selected_All(filenames_direct=filenames_direct,filenames_think=filenames_think,model_names=model_names)
    # plotAverageProb_Selected_Correct(filenames_direct=filenames_direct,filenames_think=filenames_think,model_names=model_names)
    # plotAverageProb_Selected_Wrong(filenames_direct=filenames_direct,filenames_think=filenames_think,model_names=model_names)
    # plotAverageProb_VS_GainAccuracy(filenames_direct=filenames_direct,filenames_think=filenames_think,model_names=model_names)
    # plotHeatMap(filenames_direct=filenames_direct,filenames_think=filenames_think,model_names=model_names,category_names=file_names)
    plotAverageProb_BothWrong_vs_Wrong2Correct(filenames_direct=filenames_direct,filenames_think=filenames_think,model_names=model_names,category_names=file_names)

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


def plotHeatMap(filenames_direct,filenames_think,model_names,category_names):
    data = {model_name: [] for model_name in model_names}

    data_filename = 'heatmap_data_cache.pkl'
    try:
        # Try reading existing data
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)
        print("Data loaded from pickle file.")
    except FileNotFoundError:
        for filename_direct, filename_think,model_name in zip(filenames_direct, filenames_think, model_names):
            for category_direct,category_think,category_name in zip(filename_direct,filename_think,category_names):
                df_direct, df_think = getDf(category_direct,category_think)
                prob_increase = getAverageProb(df_think) - getAverageProb(df_direct)
                acc_increase = getAccuracy(df_think)-getAccuracy(df_direct)

                df_direct_correct, df_think_correct = getCorrectDf(category_direct,category_think)
                prob_increase_correct = getAverageProb(df_think_correct) - getAverageProb(df_direct_correct)

                df_direct_wrong, df_think_wrong = getWrongDf(category_direct,category_think)
                prob_increase_wrong = getAverageProb(df_think_wrong) - getAverageProb(df_direct_wrong)

                data[model_name].append([category_name,acc_increase,prob_increase,prob_increase_correct,prob_increase_wrong])
        save_data_as_pickle(data, data_filename)
    # plotGainAccuracyVsProbIncrease(data,model_names)
    plot_heatmap(data,model_names,sort_column=3)

def plotAverageProb_BothWrong_vs_Wrong2Correct(filenames_direct, filenames_think, model_names, category_names):

    data = {model_name: [] for model_name in model_names}

    data_filename = 'heatmap_data_cache_bothVSWrong2Correct.pkl'
    try:
        # Try reading existing data
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)
        print("Data loaded from pickle file.")
    except FileNotFoundError:
        for filename_direct,filename_think,model_name in zip(filenames_direct,filenames_think,model_names):
            for category_direct, category_think, category_name in zip(filename_direct, filename_think, category_names):
                df_dir,df_think = getBothWrongDf(category_direct,category_think)

                df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
                Both_average_max_value = df_dir['max_value'].mean()

                df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
                Both_average_max_value2 = df_think['max_value'].mean()

                df_dir,df_think = getWrong2CorrectDf(category_direct,category_think)

                df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
                In2C_average_max_value = df_dir['max_value'].mean()

                df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
                In2C_average_max_value2 = df_think['max_value'].mean()
                increase_both = Both_average_max_value2-Both_average_max_value
                Increase_in2Co = In2C_average_max_value2-In2C_average_max_value

                data[model_name].append([category_name, increase_both, Increase_in2Co, Increase_in2Co-increase_both])
        save_data_as_pickle(data, data_filename)

    plot_heatmap(data,model_names,vmin =0,vmax =0.5,sort_column = 3, xlabels = ['Prob. inc. (Both Incorr.)', 'Prob. inc. (Incorr. to Corr.)', 'Difference'])


def plot_heatmap(data,model_names,vmin = 0,vmax = 0.5, sort_column = 1,xlabels = ['Acc. inc.','Prob. inc. (all)','Prob. inc. (corr.)','Prob. inc. (incorr.)']):

    # heatmapcolor
    custom_cmap_1 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "blue"])
    custom_cmap_2 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "orange"])
    custom_cmap_3 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "green"])
    custom_cmap_4 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "pink"])
    custom_cmap_5 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "purple"])
    custom_cmap_6 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "brown"])
    custom_cmap_7 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "black"])
    custom_cmap = [custom_cmap_1, custom_cmap_2, custom_cmap_3, custom_cmap_4, custom_cmap_5, custom_cmap_6,
                   custom_cmap_7]
    start_indexs = [0,19,38]
    end_indexs = [19,38,None]


    # plt.ioff()
    # vmin, vmax = 0, 0.5
    cbar_ticks = np.linspace(vmin, vmax, num=int((vmax - vmin) / 0.1) + 1)
    label = []
    sorted_indices = None
    for start_index, end_index, in zip(start_indexs,end_indexs):
        fig, axes = plt.subplots(1, 7, figsize=(20, 16))
        fig.subplots_adjust(hspace=0.2, wspace=-0.3)
        for i, ax in enumerate(axes.flat):  # use flat to get every subplot
            row, col = divmod(i, len(model_names))  # calculate row col

            # data_plot = np.array([lst[1:] for lst in data[model_names[col]][start_index:end_index]])
            if sorted_indices is None:
                result = np.zeros_like(np.array(data[model_names[0]]), dtype=np.float64)
                datas_tmp = []
                for modelIndex in range(len(model_names)):
                    data_plot_i = np.array(data[model_names[modelIndex]])[:, 1:].astype(np.float64)
                    data_plot_i = data_plot_i / np.mean(data_plot_i, axis=0)
                    datas_tmp.append(data_plot_i)
                datas_tmp_array = np.array(datas_tmp)
                sorted_datas = np.sort(datas_tmp_array, axis=0)
                middle_5_data = sorted_datas[1:6, :, :]
                final_result = np.sum(middle_5_data, axis=0)
                result[:, 1:] += final_result

                sorted_indices = np.argsort(result[:, sort_column])  # sort by this column


            data_plot = np.array(data[model_names[col]])

            data_plot = data_plot[sorted_indices]

            label = data_plot[start_index:end_index][:, 0].tolist()
            # label = data_plot[:, 0].tolist()
            data_plot = data_plot[:, 1:].astype(np.float64)
            # data_plot = data_plot / np.mean(data_plot, axis=0)
            # data_plot = data_plot[:, 1:]
            data_plot = data_plot[start_index:end_index]
            # data_plot = data_plot.astype(float)
            sns.heatmap(
                data_plot,
                fmt=".2f",
                annot=True,
                cmap=custom_cmap[col],
                cbar=False,  # 隐藏颜色条
                square=True,
                ax=ax,
                annot_kws={"size": 8},  # number font size
                vmin=vmin,
                vmax=vmax,
            )
            if col != 0:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(label, fontsize=8, rotation=0)
                if end_index is None:
                    ax.set_xticklabels(xlabels,fontsize=8, rotation=90)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
            cbar = ax.figure.colorbar(ax.collections[0], ax=ax, location="top",orientation='horizontal', fraction=0.03, pad=0.035, shrink=0.55)
            cbar.set_label(model_names[i], fontsize=12)
            cbar.outline.set_visible(False)
            cbar.set_ticks(cbar_ticks)
            cbar.ax.tick_params(labelsize=8)
            # cbar.ax.xaxis.set_label_position('bottom')
            cbar.ax.xaxis.set_ticks_position('bottom')

        # # 设置整体标题
        # fig.suptitle("4x7 Heatmap Grid", fontsize=16, y=0.95)
        # plt.tight_layout(pad=1.0)
        # 显示图像
        plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.2)
        plt.show()

# deprecated
def plotHeatMap_deprecated(filenames_direct,filenames_think,model_names,category_names):
    data = {model_name: [] for model_name in model_names}

    data_filename = 'heatmap_data_cache.pkl'
    try:
        # Try reading existing data
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)
        print("Data loaded from pickle file.")
    except FileNotFoundError:
        for filename_direct, filename_think,model_name in zip(filenames_direct, filenames_think, model_names):
            for category_direct,category_think,category_name in zip(filename_direct,filename_think,category_names):
                df_direct, df_think = getDf(category_direct,category_think)
                prob_increase = getAverageProb(df_think) - getAverageProb(df_direct)
                acc_increase = getAccuracy(df_think)-getAccuracy(df_direct)

                df_direct_correct, df_think_correct = getCorrectDf(category_direct,category_think)
                prob_increase_correct = getAverageProb(df_think_correct) - getAverageProb(df_direct_correct)

                df_direct_wrong, df_think_wrong = getCorrectDf(category_direct,category_think)
                prob_increase_wrong = getAverageProb(df_think_wrong) - getAverageProb(df_direct_wrong)

                data[model_name].append([category_name,acc_increase,prob_increase,prob_increase_correct,prob_increase_wrong])
        save_data_as_pickle(data, data_filename)
    # plotGainAccuracyVsProbIncrease(data,model_names)

    # heatmapcolor
    custom_cmap_1 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "blue"])
    custom_cmap_2 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "orange"])
    custom_cmap_3 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "green"])
    custom_cmap_4 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "pink"])
    custom_cmap_5 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "purple"])
    custom_cmap_6 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "brown"])
    custom_cmap_7 = LinearSegmentedColormap.from_list("lightgray_to_green", ["#f0f0f5", "black"])
    custom_cmap = [custom_cmap_1, custom_cmap_2, custom_cmap_3, custom_cmap_4, custom_cmap_5, custom_cmap_6,
                   custom_cmap_7]
    # set size
    lengthPergraph = 20
    # fig, axes = plt.subplots(len(category_names), len(model_names), figsize=(200, 120))
    fig, axes = plt.subplots(1, 7, figsize=(20, 12))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)  # 调整子图之间的间距
    plt.ioff()
    for i, ax in enumerate(axes.flat):  # use flat to get every subplot
        row, col = divmod(i, len(model_names))  # calculate row col

        # print('***************')
        # print(i)
        # print(row)
        # print(col)
        # print('***************')
        data_plot = np.array(data[model_names[col]][row][1:])

        data_plot = data_plot.reshape(1, -1)
        print(data_plot)

        sns.heatmap(
            data_plot,
            fmt=".2f",
            annot=True,
            cmap=custom_cmap[col],
            cbar=False,  # 隐藏颜色条
            square=True,
            ax=ax,
            annot_kws={"size": 8}  # number font size
        )
        if col != 0:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_xticklabels(['Acc. inc.','Prob. inc. (all)','Prob. inc. (corr.)','Prob. inc. (incorr.)'])
            ax.set_yticklabels([category_names[row]])
            # if row == 0:
        #     ax.set_title(f"Heatmap {col + 1}", fontsize=10)
        # if row == 2:
        #     break

    # # 设置整体标题
    # fig.suptitle("4x7 Heatmap Grid", fontsize=16, y=0.95)

    # 显示图像
    plt.show()

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

        # print(model_name+'Correct_dir:'+str(np.std([x - 1 for x in logprob_dir])))
        # print(model_name+'Correct_think:'+str(np.std([x - 1 for x in logprob_think])))
        data.append(logprob_dir)
        data.append(logprob_think)
        labels.append(f"{model_name} - Direct")
        labels.append(f"{model_name} - Thinking")
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
        labels.append(f"{model_name} - Thinking")
    PlotComparisonHistogram_subplots(filenames_direct,data,labels)



def PlotComparisonHistogram_subplots(filenames_direct, data, labels):

    if len(data) % 2 != 0:
        raise ValueError("Data length must be even as it assumes pairs of comparison groups.")

    # 计算总组数
    num_groups = len(data) // 2

    # 设置 bins 和样式
    bins = [i / 10 for i in range(11)]
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown']* 2
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


def plotAccuracy_AllDataset(filenames_direct, filenames_think, model_names):
    acc_dir = []
    acc_think = []
    for filename_direct in filenames_direct:
        acc_dir.append(getAccuracy(filename_direct))
    for filename_think in filenames_think:
        acc_think.append(getAccuracy(filename_think))
    plotBarComparison_all(data1=acc_dir,data2=acc_think,labels=model_names)






def plotAverageProb_Selected_All(filenames_direct, filenames_think, model_names):
    average_logprob_dir = []
    average_logprob_think = []
    for filename_direct,filename_think in zip(filenames_direct,filenames_think):
        df_dir,df_think = getDf(filename_direct,filename_think)

        df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value = df_dir['max_value'].mean()
        average_logprob_dir.append(average_max_value)

        df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value2 = df_think['max_value'].mean()
        average_logprob_think.append(average_max_value2)
    average_logprob_dir = [x * 100 for x in average_logprob_dir]
    average_logprob_think = [x * 100 for x in average_logprob_think]
    plotBarComparison_all(data1=average_logprob_dir,data2=average_logprob_think,ylabel_name='Average Probability of Selected Options (%)',labels=model_names,legendOut=True,yrange=10)


def plotAverageProb_Selected_Correct(filenames_direct, filenames_think, model_names):
    average_logprob_dir = []
    average_logprob_think = []
    for filename_direct,filename_think in zip(filenames_direct,filenames_think):
        df_dir,df_think = getCorrectDf(filename_direct,filename_think)

        df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value = df_dir['max_value'].mean()
        average_logprob_dir.append(average_max_value)

        df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value2 = df_think['max_value'].mean()
        average_logprob_think.append(average_max_value2)
    average_logprob_dir = [x * 100 for x in average_logprob_dir]
    average_logprob_think = [x * 100 for x in average_logprob_think]
    plotBarComparison_all(data1=average_logprob_dir,data2=average_logprob_think,ylabel_name='Average Probability of Selected Options (%)',labels=model_names,legendOut=True,yrange=10)


def plotAverageProb_Selected_Wrong(filenames_direct, filenames_think, model_names):
    average_logprob_dir = []
    average_logprob_think = []
    for filename_direct,filename_think in zip(filenames_direct,filenames_think):
        df_dir,df_think = getWrongDf(filename_direct,filename_think)

        df_dir['max_value'] = df_dir[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value = df_dir['max_value'].mean()
        average_logprob_dir.append(average_max_value)

        df_think['max_value'] = df_think[['a', 'b', 'c', 'd']].max(axis=1)
        average_max_value2 = df_think['max_value'].mean()
        average_logprob_think.append(average_max_value2)
    average_logprob_dir = [x * 100 for x in average_logprob_dir]
    average_logprob_think = [x * 100 for x in average_logprob_think]
    plotBarComparison_all(data1=average_logprob_dir,data2=average_logprob_think,ylabel_name='Average Probability of Selected Options (%)',labels=model_names,legendOut=True,yrange=10)



def plotAverageProb_VS_GainAccuracy(filenames_direct, filenames_think, model_names):
    data = {model_name: [] for model_name in model_names}
    for filename_direct, filename_think,model_name in zip(filenames_direct, filenames_think, model_names):
        for catogory_direct,catogory_think in zip(filename_direct,filename_think):
            df_direct, df_think = getDf(catogory_direct,catogory_think)
            prob_increase = getAverageProb(df_think) - getAverageProb(df_direct)
            acc_increase = getAccuracy(df_think)-getAccuracy(df_direct)
            data[model_name].append([acc_increase,prob_increase])
    plotGainAccuracyVsProbIncrease(data,model_names)

def plotGainAccuracyVsProbIncrease(data, model_names):

    all_data = []
    for model_name in model_names:
        for i, (acc_increase, prob_increase) in enumerate(data[model_name]):
            all_data.append([acc_increase, prob_increase, model_name])

    # 将 all_data 转换为 DataFrame
    df = pd.DataFrame(all_data, columns=['Accuracy Gain', 'Probability Increase', 'Model'])
    fontsize = 16
    # palette = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown']
    palette = sns.color_palette('Set1')
    # palette = ['#8ca5c0','#6280a5','#56648a','#8d7e95','#ca9a96','#facaa9','#544b6d']
    palette[2] = '#1d3039'
    palette[5] = '#f2c0c6'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Accuracy Gain', y='Probability Increase', hue='Model', data=df, palette=palette, s=100)

    # 设置标题和标签
    # plt.title('Accuracy Gain vs. Average Probability Increase', fontsize=fontsize)
    plt.xlabel('Gain in Accuracy', fontsize=fontsize)
    plt.ylabel('Increase in Average Probability', fontsize=fontsize)
    # plt.xticks(ticks=[-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=fontsize)
    # plt.yticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xlim(-0.15, 0.5)
    # plt.ylim(-0.05, 0.5)
    plt.legend(title='Model', loc='upper left',fontsize=fontsize)

    # 显示图形
    plt.tight_layout()
    plt.show()


def plotBarComparison_all(data1, data2, labels, xlabel_name='Model Name', ylabel_name='Accuracy',pos = 'best', legendOut: bool = False,yrange=0.1, legends = ['Direct','After Thinking']):

    # Plot barchart

    fontsize = 16
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))  # 增大图表尺寸

    ax.bar(x, data1, width, label=legends[0], color='blue')
    ax.bar([i + width for i in x], data2, width, label=legends[1], color='orange')

    ax.set_xlabel(xlabel_name, fontsize=fontsize)
    ax.set_ylabel(ylabel_name, fontsize=fontsize)
    # ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_yticks([i * yrange for i in range(11)])
    frameon = True
    if legendOut:
        fontsize -= 4
        frameon = False
    ax.legend(fontsize=fontsize,loc=pos, frameon=frameon)
    # else:
    #     ax.legend(fontsize=fontsize, loc=pos, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


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


# Deprecated
def plotAccuracyComparison(accuracies_direct, accuracies_think, xlabel_name='Dataset', ylabel_name='Accuracy',
                           title='Comparison of Accuracy for Different Datasets',pos = 'upper left'):
    # Deprecated
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

def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()
