import pandas as pd

def main():

    model_names = ['Mistral-7B-instruct-v0.3/','llama3.1-8B/','gemma2-9b-it/','llama3.2-11B-vision-instruct/','Yi-1.5-9B-Chat/','gpt4o-mini/','gpt4o/']
    # model_name = 'Mistral-7B-instruct-v0.3/'
    # model_name = 'llama3.1-8B/'
    # model_name = 'gemma2-9b-it/'
    # model_name = 'llama3.2-11B-vision-instruct/'
    # # model_name = 'Yi-1.5-9B-Chat/'
    # model_name = 'gpt4o-mini/'
    # model_name = 'gpt4o/'
    # ori_files = ['abstract_algebra.xlsx','anatomy.xlsx','college_biology.xlsx']

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
    ori_files = [file+'.xlsx' for file in file_names]
    ori_files = ['MMLU/'+ file for file in ori_files]
    for model_name in model_names:
        result_files = [
            [f"{file}_LogProbs_afterThinking.csv", f"{file}_LogProbs_Direct.csv"]
            for file in file_names
        ]
        #
        # result_files = [['abstract_algebra_LogProbs_afterThinking.csv','abstract_algebra_LogProbs_Direct.csv'],
        #                 ['anatomy_LogProbs_afterThinking.csv','anatomy_LogProbs_Direct.csv'],
        #                 ['college_biology_LogProbs_afterThinking.csv','college_biology_LogProbs_Direct.csv']]
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