import os
import time
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BipartiteGraphDataset(Dataset):
    def __init__(self, dataset):
        super(BipartiteGraphDataset, self).__init__()
        self.dataset = dataset
        self.trainData, self.allPos, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0

        with open(os.path.join(self.dataset, "train.txt"), "r") as f:
            for line in f:
                line = line.strip().split(" ")
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.allPos[user] = items
                for item in items:
                    self.trainData.append([user, item])
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        with open(os.path.join(self.dataset, "test.txt"), "r") as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.testData[user] = items
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

    def __getitem__(self, idx):
        user, item = self.trainData[idx]
        return user, self.allPos[user], item

    def __len__(self):
        return len(self.trainData)

@dataclass
class BipartiteGraphCollator:
    def __call__(self, batch) -> dict:
            """
            배치 데이터를 처리하여 모델에 입력으로 제공하기 위한 딕셔너리를 반환합니다.

            Args:
                batch (list): 배치 데이터. 각 요소는 (user, items, labels)로 구성됩니다.

            Returns:
                dict: 모델에 입력으로 제공하기 위한 딕셔너리. 다음 키를 포함합니다.
                    - "inputs": 모델의 입력으로 사용될 텐서
                    - "inputs_mask": 입력 텐서의 마스크
                    - "labels": 레이블 텐서
            """
            user, items, labels = zip(*batch)
            bs = len(user)
            max_len = max([len(item) for item in items])
            inputs = [[user[i]] + items[i] + [0] * (max_len - len(items[i])) for i in range(bs)] # user + items + padding
            inputs_mask = [[1] + [1] * len(items[i]) + [0] * (max_len - len(items[i])) for i in range(bs)]
            labels = [[label] for label in labels]
            inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

            return {
                "inputs": inputs,
                "inputs_mask": inputs_mask,
                "labels": labels
            }
