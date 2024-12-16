# emotion_analysis
Using NLP technology for sentiment analysis of sentences

## Installation
Install transformers and the packages required for development.
```bash
conda create -n entity python=3.10
conda activate emotion_analysis
git clone https://github.com/Hzzz123rfefd/emotion_analysis.git
cd emotion_analysis
pip install -r requirements.txt
```

## Usage
### Dataset
Firstly, you can download the jobstreet dataset, [conll2003](https://www.kaggle.com/datasets/azraimohamad/jobstreet-all-job-dataset),put them into `datasets/jobstreet`
your directory structure should be:
- emotion_analysis/
  - datasets/
    - jobstreet/
      - jobstreet_all_job_dataset.csv

Then, you can process jobstreet data with following script:
```bash
python datasets/jobstreet/process.py
```

No matter what dataset you use, please convert it to the required dataset format for this project, as follows (you can also view it in data/train. json)
```jsonl
{"text": "your text", "label": 1}   
```

### Bert Model
If you don't have the BERT model on your computer, you can download the model through the following script
```bash
python download_model
```

### Trainning
An examplary training script with a Cross Entropy loss is provided in train.py.
You can adjust the model parameters in config/entity_extraction_base_bert.yml
```bash
python train.py --model_config_path config/emotion_analysis_base_bert.yml
```

### Inference
Once you have trained your model, you can use the following script to perform entity recognition on the data
you can set your text in inference.py
```bash
python inference.py --model_config_path config/emotion_analysis_base_bert.yml
```
You can see the following input and output:
```text
text = "South Korea made virtually certain of an Asian Cup quarter-final spot with a 4-2 win over Indonesia in a Group A match on Saturday . "

label = 2
```
