import os
import json

data_dir = 'data/PCR_dataset'
data = {'train': {'data': []}, 'val': {'data': []}, 'test': {'data': []}}

for dataset_type in data.keys():
    dataset_path = os.path.join(data_dir, dataset_type)

    classes = sorted(os.listdir(dataset_path))

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)

        images = os.listdir(class_path)

        for image_name in images:
            image_path = os.path.join(class_path, image_name)
            image_info = {
                "impath": image_path,
                "label": class_idx,
                "classname": class_name
            }
            data[dataset_type]['data'].append(image_info)

json_file_path = 'indices/PCR/PCR.json'
json_file_path2 = 'data/PCR_dataset/PCR.json'
if not os.path.exists(json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"JSON file created at {json_file_path}")
else:
    print(f"JSON file already exists at {json_file_path}")
data_dir = 'data/PCR_dataset'
data = {'train': [],'val': [], 'test': []}

for dataset_type in data.keys():
    dataset_path = os.path.join(data_dir, dataset_type)

    classes = sorted(os.listdir(dataset_path))

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)

        images = os.listdir(class_path)

        for image_name in images:
            image_path = os.path.join(class_name, image_name)
            data[dataset_type].append((image_path, class_idx, class_name))


with open(json_file_path2, 'w') as json_file:
    json.dump(data, json_file, indent=4)