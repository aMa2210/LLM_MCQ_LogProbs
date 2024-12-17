import os
import pandas as pd
import re
from openai import OpenAI
import tiktoken
import math

def main():
    # file_name = 'abstract_algebra'
    # file_name = 'anatomy'
    file_name = 'college_biology'
    answerDirect(file_name)
    answerAfterThinking(file_name)


def answerDirect(file_name):
    client = OpenAI()

    mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

    message_system = "Please respond with only the letter of the solution, in the format {'sol': 'solution'}."
    model_names = ['gpt-4o-mini-2024-07-18']
    tokenizer = tiktoken.get_encoding('o200k_base')

    # file_name = 'abstract_algebra'

    data_name = 'MMLU/' + file_name + '.xlsx'
    result_name = 'Results/' + file_name + '_LogProbs_Direct.csv'
    df = pd.read_excel(data_name)
    print(message_system)
    # assistant_content = "{'sol': '"
    try:
        result_dict = {}
        for model_name in model_names:
            for index, row in df.iterrows():
                try:
                    question = row['question']
                    choices = row['choices'].replace('[', '').replace(']', '')
                    # choices_list = re.findall(r"'(\d+)'", choices)

                    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", choices)
                    choices_list = [match[0] if match[0] else match[1] for match in matches]

                    letters = ['a) ', 'b) ', 'c) ', 'd) ']
                    choices_with_letters = zip(letters, choices_list)
                    labeled_choices = [f"{letter}{choice.strip()}" for letter, choice in choices_with_letters]
                    choices = " ".join(labeled_choices)
                    message_content = f"{question} Choices: {choices}."
                    print(message_content)
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": message_system},
                            {"role": "user", "content": message_content},
                            # {"role": "assistant", "content": assistant_content}
                        ],
                        temperature=0,
                        logprobs=True,
                        top_logprobs=10,
                        # max_tokens=1
                    )
                    # match = re.search(r"([a-d])", completion.choices[0].message.content)
                    # print(str(index) + completion.choices[0].message.content)
                    tokens = tokenizer.encode(completion.choices[0].message.content)
                    target_word = "sol"

                    answer_index = 0

                    for idx, token in enumerate(tokens):
                        decoded_token = tokenizer.decode([token])
                        if target_word in decoded_token:
                            answer_index = idx

                    choice = completion.choices[0]
                    logprobs = choice.logprobs.content[answer_index + 3].top_logprobs
                    probabilities = [math.exp(logprob.logprob) for logprob in logprobs if logprob is not None]
                    text = [logprob.token.strip().lower() for logprob in logprobs if logprob is not None]
                    a = sum(probabilities)
                    probabilities = [probability / a for probability in probabilities]

                    merged = {}
                    for t, p in zip(text, probabilities):
                        if t in merged:
                            merged[t] += p
                        else:
                            merged[t] = p

                    for option, total_probability in merged.items():
                        print(f"{option}  probability  {total_probability}")
                        if index not in result_dict:
                            result_dict[index] = {}
                        result_dict[index][option] = str(total_probability)
                except Exception as row_e:
                    print(f"Error processing index {index} with model {model_name}: {row_e}")
                    if index not in result_dict:
                        result_dict[index] = {'processing_error': str(row_e)}
        for index, results in result_dict.items():
            for option, value in results.items():
                df.at[index, option] = value

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        df.to_csv(result_name, index=False)
        print(f"Results have been saved to {result_name}")

def answerAfterThinking(file_name):
    client = OpenAI()

    mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

    message_system = "Please think step by step before providing an answer, once you have the solution end the respond only with the letter of the solution, in the format {'sol': 'solution'}."
    model_names = ['gpt-4o-mini-2024-07-18']
    tokenizer = tiktoken.get_encoding('o200k_base')

    # file_name = 'abstract_algebra'

    data_name = 'MMLU/'+file_name+'.xlsx'
    result_name = 'Results/'+file_name+'_LogProbs_afterThinking.csv'
    df = pd.read_excel(data_name)
    print(message_system)
    # assistant_content = "{'sol': '"
    try:
        result_dict = {}
        for model_name in model_names:
            for index, row in df.iterrows():
                try:
                    question = row['question']
                    choices = row['choices'].replace('[', '').replace(']', '')
                    # choices_list = re.findall(r"'(\d+)'", choices)

                    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", choices)
                    choices_list = [match[0] if match[0] else match[1] for match in matches]

                    letters = ['a) ', 'b) ', 'c) ', 'd) ']
                    choices_with_letters = zip(letters, choices_list)
                    labeled_choices = [f"{letter}{choice.strip()}" for letter, choice in choices_with_letters]
                    choices = " ".join(labeled_choices)
                    message_content = f"{question} Choices: {choices}."
                    print(message_content)
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": message_system},
                            {"role": "user", "content": message_content},
                            # {"role": "assistant", "content": assistant_content}
                        ],
                        temperature=0,
                        logprobs=True,
                        top_logprobs=10,
                        # max_tokens=1
                    )
                    # match = re.search(r"([a-d])", completion.choices[0].message.content)
                    # print(str(index) + completion.choices[0].message.content)
                    tokens = tokenizer.encode(completion.choices[0].message.content)
                    target_word = "sol"

                    answer_index = 0

                    for idx, token in enumerate(tokens):
                        decoded_token = tokenizer.decode([token])
                        if target_word in decoded_token:
                            answer_index = idx

                    choice = completion.choices[0]
                    logprobs = choice.logprobs.content[answer_index+3].top_logprobs
                    probabilities = [math.exp(logprob.logprob) for logprob in logprobs if logprob is not None]
                    text = [logprob.token.strip().lower() for logprob in logprobs if logprob is not None]
                    a = sum(probabilities)
                    probabilities = [probability/a for probability in probabilities]

                    merged = {}
                    for t, p in zip(text, probabilities):
                        if t in merged:
                            merged[t] += p
                        else:
                            merged[t] = p

                    for option, total_probability in merged.items():
                        print(f"{option}  probability  {total_probability}")
                        if index not in result_dict:
                            result_dict[index] = {}
                        result_dict[index][option] = str(total_probability)
                except Exception as row_e:
                    print(f"Error processing index {index} with model {model_name}: {row_e}")
                    if index not in result_dict:
                        result_dict[index] = {'processing_error': str(row_e)}
        for index, results in result_dict.items():
            for option, value in results.items():
                df.at[index, option] = value

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        df.to_csv(result_name, index=False)
        print(f"Results have been saved to {result_name}")


if __name__ == '__main__':
    main()