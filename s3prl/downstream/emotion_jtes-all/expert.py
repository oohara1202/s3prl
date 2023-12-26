import os
import math
import torch
import random
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .model import *
from .dataset import JTESDataset, collate_fn


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        DATA_ROOT = self.datarc['root']
        meta_data = self.datarc["meta_data"]

        print(f"[Expert] - Using predefined data: train, val, and test")

        train_path = os.path.join(meta_data, 'train_meta_data.json')
        assert os.path.exists(train_path)
        print(f'[Expert] - Training path: {train_path}')

        val_path = os.path.join(meta_data, 'val_meta_data.json')
        assert os.path.exists(val_path)
        print(f'[Expert] - Validating path: {val_path}')

        ### testはvalと同じに ###
        # test_path = os.path.join(meta_data, 'test_meta_data.json')
        # assert os.path.exists(test_path)
        # print(f'[Expert] - Testing path: {test_path}')
        
        train_dump = train_path + '.pkl'
        if os.path.exists(train_dump):
            with open(train_dump, 'rb') as f:
                self.train_dataset = pickle.load(f)
                print(f'[Expert] - Loaded: {train_dump}')
        else:
            self.train_dataset = JTESDataset(DATA_ROOT, train_path, self.datarc['pre_load'])
            with open(train_dump, 'wb') as f:
                pickle.dump(self.train_dataset, f)
            print(f'[Expert] - Saved: {train_dump}')

        dev_dump = val_path + '.pkl'  # 名前をval --> dev
        if os.path.exists(dev_dump):
            with open(dev_dump, 'rb') as f:
                self.dev_dataset = pickle.load(f)
                print(f'[Expert] - Loaded: {dev_dump}')
        else:
            self.dev_dataset = JTESDataset(DATA_ROOT, val_path, self.datarc['pre_load'])  
            with open(dev_dump, 'wb') as f:
                pickle.dump(self.dev_dataset, f)
            print(f'[Expert] - Saved: {dev_dump}')

        self.test_dataset = self.dev_dataset

        print(f'[Expert] - Emotion class: {self.train_dataset.class_num}')

        model_cls = eval(self.modelrc['select'])  # <class 's3prl.downstream.model.UtteranceLevel'>
        model_conf = self.modelrc.get(self.modelrc['select'], {})  # {'pooling': 'MeanPooling'}
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])  # 768 --> 256
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = self.train_dataset.class_num,
            **model_conf,
        )
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

        self.softmax = nn.Softmax(dim=1)  # for feature extraction


    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        features = pad_sequence(features, batch_first=True)  # バッチ（list）内の最大フレームにあわせてゼロ埋め
        features = self.projector(features)  # [B, Len, 256]
        predicted, _, hidden_state = self.model(features, features_len)  # add hidden_state

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records["filename"] += filenames
        records["predict"] += [self.test_dataset.idx2emotion[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.test_dataset.idx2emotion[idx] for idx in labels.cpu().tolist()]

        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'emotion/{mode}-{key}',  # modified
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:  # modified
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:  # modified
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)

        return save_names

    def extract(self, feature):
        device = feature[0].device
        feature_len = torch.IntTensor([len(feat) for feat in feature]).to(device=device)
        feature = pad_sequence(feature, batch_first=True)  # 1ファイルごとに見るが一応ゼロ埋め処理を埋める
        feature = self.projector(feature)  # [B, Len, 768] --> [B, Len, 256]
        predicted, _, hidden_state = self.model(feature, feature_len)  # add hidden_state

        predicted_classid = predicted.max(dim=-1).indices
        pp = self.softmax(predicted)

        predicted_classid = predicted_classid.cpu()  # 識別クラス
        pp = pp.cpu()                                # 事後確率
        hidden_state = hidden_state.cpu()            # 表現ベクトル

        return predicted_classid, pp, hidden_state
