import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class Dataset_Custom(Dataset):
    def __init__(
        self,
        seq_len,
        label_len,
        pred_len,
        flag,
        used_col_prefixes,
        root_path="data/",
        data_path="fintexts/NVDA.parquet",
    ):
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.root_path = root_path
        self.data_path = data_path
        self.used_col_prefixes = used_col_prefixes

        self.__read_data__()
        self.tot_len = len(self.data_price_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_parquet(os.path.join(self.root_path, self.data_path))
        if "fnspid" in self.data_path:
            df_raw = df_raw.ffill()
        
        df_raw = df_raw[
            ("2019-01-01" <= df_raw["date"]) & (df_raw["date"] < "2023-12-19")
        ].reset_index(drop=True)
        ddf_raw = df_raw[["date"]]
        pdf_raw = df_raw[["open", "high", "low", "close"]]
        tdf_raw = []
        for prefix in self.used_col_prefixes:
            tmp = df_raw[[f"{prefix}_emb{i}" for i in range(384)]].reset_index()
            col_name = {"index": "stan"}
            for i in range(384):
                col_name.update({f"{prefix}_emb{i}": f"emb{i}"})
            tmp.rename(columns=col_name, inplace=True)
            tdf_raw.append(tmp)
        # tdf_raw
        tdf_raw = pd.concat(tdf_raw, axis=0).groupby("stan").apply(lambda x: x.mean())
        tdf_raw = tdf_raw[[f"emb{i}" for i in range(384)]]

        # Train/val/test split

        num_train = len(ddf_raw[ddf_raw["date"] < "2022-01-01"])
        num_vali = len(
            ddf_raw[
                ("2022-01-01" <= ddf_raw["date"]) & (ddf_raw["date"] < "2023-01-01")
            ]
        )
        num_test = len(
            ddf_raw[
                ("2023-01-01" <= ddf_raw["date"]) & (ddf_raw["date"] < "2024-01-01")
            ]
        )
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_pdate = pdf_raw[border1:border2]
        self.scaler.fit(pdf_raw.values)
        pdate = self.scaler.transform(pdf_raw.values)

        df_stamp = ddf_raw[["date"]][border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp = df_stamp.drop(["date"], axis=1).values

        self.data_price_x = pdate[border1:border2]
        self.data_text_x = tdf_raw[border1:border2].fillna(0)
        self.data_price_y = pdate[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Multivariate: use all features, Univariate: use single feature (channel-independence)
        seq_price_x = self.data_price_x[s_begin:s_end]
        seq_price_y = self.data_price_y[r_begin:r_end]
        seq_text_x = self.data_text_x[s_begin:s_end].values

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_price_x, seq_price_y, seq_text_x, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return len(self.data_price_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def get_dataset_dataloader(
    seq_len,
    label_len,
    pred_len,
    flag,
    used_col_prefixes,
    data_path,
    root_path,
    batch_size,
):
    dataset = Dataset_Custom(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        flag=flag,
        used_col_prefixes=used_col_prefixes,
        data_path=data_path,
        root_path=root_path,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if flag == "train" else False,
        num_workers=1,
        drop_last=True if flag == "train" else False,
    )
    return dataset, data_loader