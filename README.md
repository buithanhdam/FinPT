# FinPT: Financial Risk Prediction with Profile Tuning on Pretrained Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2308.00065-b31b1b.svg)](https://arxiv.org/abs/2308.00065)
## Author **YuweiYin - FinPT**
[![github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcAYGe2T6dS37PYqaZuF3cPR7ajxntKY4Otw&s)](https://github.com/YuweiYin/FinPT/tree/master)

![picture](https://yuweiyin.com/files/img/2023-07-22-FinPT.png)

* **Abstract**:

```text
Financial risk prediction plays a crucial role in the financial sector. 
Machine learning methods have been widely applied for automatically 
detecting potential risks and thus saving the cost of labor.
However, the development in this field is lagging behind in recent years 
by the following two facts: 1) the algorithms used are somewhat outdated, 
especially in the context of the fast advance of generative AI and 
large language models (LLMs); 2) the lack of a unified and open-sourced 
financial benchmark has impeded the related research for years.
To tackle these issues, we propose FinPT and FinBench: the former is a 
novel approach for financial risk prediction that conduct Profile Tuning 
on large pretrained foundation models, and the latter is a set of 
high-quality datasets on financial risks such as default, fraud, and churn.
In FinPT, we fill the financial tabular data into the pre-defined instruction 
template, obtain natural-language customer profiles by prompting LLMs, and 
fine-tune large foundation models with the profile text to make predictions.
We demonstrate the effectiveness of the proposed FinPT by experimenting with 
a range of representative strong baselines on FinBench. The analytical studies 
further deepen the understanding of LLMs for financial risk prediction.
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

- **Data time-series** : (xlsx or csv) with columns and rows data type format in ./data folder.

## Experiments

The instructions obtained in Step 1 and customer profiles generated in Step 2
are provided as `instruction_for_profile_{dataname}` and `{dataname}_profile` in FinBench.

### Step 1
- **Prepare instructions profile** for each of rows in data
```bash
python run_step1_get_instruction.py --data_dir ./data --output_dir ./output --dataset_names products_finpt_data
```
### Step 2
- **Profile** with pre-trained foundation models (LLM) such as: GPT, Gemini,..
- Im using Gemini from GoogleGenerative for this step
- You need to setup `GOOGLE_API_KEY` in your `.env`
```bash
python run_step2_profile.py --ds_name products_finpt_data --model_name gemini-1.5-flash --train_ratio 0.7 --val_ratio 0.15
```
### Step 3 Run FinPT

```bash
MODELS=("bert" "finbert" "gpt2" "t5-base" "flan-t5-base" "t5-xxl" "flan-t5-xxl" "llama-7b" "llama-13b")
```
