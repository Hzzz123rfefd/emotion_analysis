import sys
import os
sys.path.append(os.getcwd())
import csv
import json
import csv
import json
import random

file_path = "datasets/jobstreet/jobstreet_all_job_dataset.csv"  
train_file = "jobstreet_train/train.jsonl"  
test_file = "jobstreet_train/test.jsonl"  
mapping_file = "jobstreet_train/category_to_id.json"  

if not os.path.isdir("jobstreet_train"):
    os.makedirs("jobstreet_train")

categories = set()
data = []

with open(file_path, "r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        description = row["descriptions"]
        category = row["category"]
        data.append({"text": description, "category": category})
        categories.add(category)

category_to_id = {category: idx for idx, category in enumerate(sorted(categories))}

with open(mapping_file, "w", encoding="utf-8") as f:
    json.dump(category_to_id, f, ensure_ascii=False, indent=4)


random.seed(42)  
random.shuffle(data)

split_idx = int(len(data) * 0.7)
train_data = data[:split_idx]
test_data = data[split_idx:]

def save_to_jsonl(file_path, dataset):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in dataset:
            json_obj = {
                "text": item["text"],
                "label": category_to_id[item["category"]]
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

save_to_jsonl(train_file, train_data)
save_to_jsonl(test_file, test_data)