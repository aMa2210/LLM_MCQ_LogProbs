This repo provides the data and scripts for the paper: Multiple Choice Questions: Thinking Makes Large Language Models (LLMs) More Self-Confident Even When They are Wrong

Repository Contents
- MMLU: Contains the MMLU datasets from https://huggingface.co/datasets/cais/mmlu
- Results: Include results under two types of prompts: one where the LLM directly generates an answer and another where the answer is generated after thinking. Each prompt contains two CSV files for each dataset, one for the original response and the other for the probabilities of the response options.
- Figure: Outputs and visualizations derived from the analysis.

- The Python script in the root directory is used for analyzing and plotting the data in the Results folder.
