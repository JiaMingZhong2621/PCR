import os
import re

from engine.datasets.benchmark import Benchmark, read_split, split_trainval, save_split


class PEEL(Benchmark):

    dataset_name = "Aged_tangerine_peel"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "peel")
        self.split_path = os.path.join(self.dataset_dir, "peel.json")

        train, val, test = read_split(self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)
                item = {"impath": impath, "label": label, "classname": renamed_action}
                items.append(item)

        return items

class PCR(Benchmark):

    dataset_name = "PCR_dataset"

    def __init__(self, data_dir):
        root = data_dir
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "PCR")
        self.split_path = os.path.join(self.dataset_dir, "PCR.json")

        train, val, test = read_split(self.split_path, self.image_dir)

        super().__init__(train=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = cname2lab[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                renamed_action = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                impath = os.path.join(self.image_dir, renamed_action, filename)
                item = {"impath": impath, "label": label, "classname": renamed_action}
                items.append(item)

        return items