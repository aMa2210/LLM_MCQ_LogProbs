import pandas as pd

splits = {'test': 'college_biology/test-00000-of-00001.parquet', 'validation': 'college_biology/validation-00000-of-00001.parquet', 'dev': 'college_biology/dev-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
df.to_excel('MMLU/college_biology.xlsx', index=False)# college_biology anatomy