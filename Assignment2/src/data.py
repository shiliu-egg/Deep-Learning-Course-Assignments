import os
import torch
import requests
import zipfile
from torch.utils.data import Dataset, random_split
from typing import List, Dict, Tuple, Literal
import cv2
from torchvision import transforms
import random


class TinyImagenet(Dataset):
    dataRecord: Dict = dict()
    trainRadio: float = -1
    validRadio: float = -1

    def __init__(
        self,
        trainRadio: float = 0.8,
        validRadio: float = 0.2,
        type: Literal["train", "valid", "test"] = "train",
    ) -> None:
        super().__init__()
        if type not in {"train", "valid", "test"}:
            raise ValueError("'%s' is not a supported type" % type)

        assert abs(trainRadio + validRadio - 1) < 1e-5
        _trainRadio = TinyImagenet.trainRadio
        _validRadio = TinyImagenet.validRadio
        if _trainRadio != -1 and abs(trainRadio - _trainRadio) > 1e-5:
            raise ValueError(
                f"There is already a dataset with trainRadio {_trainRadio}"
            )
        if _validRadio != -1 and abs(validRadio - _validRadio) > 1e-5:
            raise ValueError(
                f"There is already a dataset with validRadio {_validRadio}"
            )

        self.dataDir: str = None
        self.labels: List[str] = None
        self.label2id: Dict[str, int] = None
        self.trainData: List[Tuple[torch.Tensor, int]] = []
        self.validData: List[Tuple[torch.Tensor, int]] = []
        self.transform = transforms.ToTensor()
        TinyImagenet.trainRadio = trainRadio
        TinyImagenet.validRadio = validRadio

        self.getData()
        self.getLabels()
        if type == "test":
            self.dealValid()
            self.data = self.validData
        else:
            if not TinyImagenet.dataRecord:
                self.dealTrain()
                stateOri = random.getstate()
                random.seed(0)
                random.shuffle(self.trainData)
                random.setstate(stateOri)
                N = len(self.trainData)
                trainNum = int(N * trainRadio)
                TinyImagenet.dataRecord["train"] = self.trainData[:trainNum]
                TinyImagenet.dataRecord["valid"] = self.trainData[trainNum:]
            self.data = TinyImagenet.dataRecord[type]
        print(f"finish {type} dataset")

    def getData(self):
        """获取数据集文件"""
        self.root = os.path.join("..", "data")
        self.dataDir = os.path.join(self.root, "tiny-imagenet-200")
        if not os.path.exists(self.dataDir):
            os.makedirs(self.root,exist_ok=True)
            dataFile = os.path.join(self.root, "tiny-imagenet-200.zip")
            if not os.path.exists(dataFile):
                url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
                print(f"download data from {url}")
                content = requests.get(url, allow_redirects=True).content
                with open(dataFile, "wb") as f:
                    f.write(content)
            print("extract zip file")
            with zipfile.ZipFile(dataFile) as f:
                f.extractall(self.root)

    def getLabels(self):
        """获取数据集的所有 label"""
        filename = os.path.join(self.dataDir, "wnids.txt")
        assert os.path.exists(filename)

        with open(filename, "r", encoding="utf8") as f:
            self.labels = [label.strip() for label in f.readlines()]
            self.label2id = {label: index for index, label in enumerate(self.labels)}

    def dealTrain(self):
        """处理 train 数据集"""
        self.trainData = []
        for label in self.labels:
            statFile = os.path.join(self.dataDir, "train", label, f"{label}_boxes.txt")
            with open(statFile, "r", encoding="utf8") as f:
                for line in f:
                    line = line.strip().split()
                    picName = line[0]
                    picFile = os.path.join(
                        self.dataDir, "train", label, "images", picName
                    )
                    self.trainData.append((picFile, self.label2id[label]))
    def dealValid(self):
        """处理 valid 数据集"""
        statFile = os.path.join(self.dataDir, "val", "val_annotations.txt")
        self.validData = []
        with open(statFile, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip().split()
                picName = line[0]
                label = line[1]
                picFile = os.path.join(self.dataDir, "val", "images", picName)
                self.validData.append((picFile, self.label2id[label]))

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        picFile, label = self.data[index]
        img = cv2.imread(picFile).reshape(64, 64, 3)
        img = self.transform(img)
        return (img, label)

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    trainDataset = TinyImagenet(type="train")
    validDataset = TinyImagenet(type="valid")
    testDataset = TinyImagenet(type="test")
    print("finish all")
