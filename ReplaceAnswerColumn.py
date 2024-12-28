import pandas as pd

def main():

    # model_name = 'Yi-1.5-9B-Chat/'
    # model_name = 'Mistral-7B-instruct-v0.3/'
    model_name = 'llama3.1-8B/'
    model_name = 'gemma2-9b-it/'
    model_name = 'llama3.2-11B-vision-instruct/'
    ori_files = ['abstract_algebra.xlsx','anatomy.xlsx','college_biology.xlsx']
    ori_files = ['MMLU/'+ file for file in ori_files]
    result_files = [['abstract_algebra_LogProbs_afterThinking.csv','abstract_algebra_LogProbs_Direct.csv'],
                    ['anatomy_LogProbs_afterThinking.csv','anatomy_LogProbs_Direct.csv'],
                    ['college_biology_LogProbs_afterThinking.csv','college_biology_LogProbs_Direct.csv']]
    # result_files = [['abstract_algebra_LogProbs_Direct.csv'],
    #                 ['anatomy_LogProbs_Direct.csv'],
    #                 ['college_biology_LogProbs_Direct.csv']]
    result_files = [['Results/' + model_name + file for file in sublist] for sublist in result_files]
    for ori_file, result_files_2 in zip(ori_files,result_files):
        for result_file in result_files_2:
            replaceAnswer(result_file,ori_file)


def replaceAnswer(result_file, ori_file):
    # 读取xlsx文件
    xlsx_data = pd.read_excel(ori_file)

    # 读取csv文件
    csv_data = pd.read_csv(result_file, encoding='ISO-8859-1')

    # 替换csv文件中的answer列
    csv_data['answer'] = xlsx_data['answer']

    # 保存修改后的csv文件
    csv_data.to_csv(result_file, index=False)

if __name__ == '__main__':
    main()