import csv
import json
import os

_dir = os.path.dirname(__file__)

# Load class labels
csv_file_path = os.path.join(_dir, "class_labels_indices.csv")

with open(csv_file_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    lines = list(reader)

labels = []
ids = []  # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

num_classes = len(labels)

# Load high-level category mapping (Music, Speech, SFX)
category_mapping_path = os.path.join(_dir, "category_mapping.json")

with open(category_mapping_path, "r") as f:
    _category_data = json.load(f)

# List where index i contains the high-level category for class i
index_to_category: list[str] = _category_data["index_to_category"]

# High-level category names
high_level_categories: list[str] = _category_data["categories"]

# Number of high-level categories
num_high_level_categories = len(high_level_categories)

# Mapping from category name to index
category_to_index: dict[str, int] = {cat: i for i, cat in enumerate(high_level_categories)}
