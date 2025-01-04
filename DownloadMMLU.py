import pandas as pd

# List of all categories in the dataset
categories = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology',
    'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics',
    'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics',
    'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
    'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    'virology', 'world_religions'
]
# Base URL for the dataset
base_url = "hf://datasets/cais/mmlu/"

# Loop through categories to download and save test datasets
for category in categories:
    try:
        # Construct the path to the test file
        test_file_path = f"{base_url}{category}/test-00000-of-00001.parquet"

        # Read the dataset
        df = pd.read_parquet(test_file_path)

        # Save to Excel
        output_file = f"MMLU/{category}.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Saved {output_file}")

    except Exception as e:
        print(f"Error downloading or saving {category}: {e}")
