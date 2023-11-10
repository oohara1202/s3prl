import os
from os.path import basename, splitext, join as path_join
import sys
import re
import json
import random
from librosa.util import find_files

FOLD_NUM = 5
DATASET_PATH = '/abelab/DB4/JTES/jtes_v1.1/wav'
METADATA_PTAH = 'downstream/emotion_jtes'

LABEL_DIR_PATH = 'dialog/EmoEvaluation'
WAV_DIR_PATH = 'sentences/wav'


def get_wav_paths(data_dirs):
    wav_paths = find_files(data_dirs)
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        start = wav_path.find('Session')
        wav_path = wav_path[start:]
        wav_dict[wav_name] = wav_path

    return wav_dict

# コメントはidxが1のとき
def preprocess(idx, path):
    train_meta_data = []
    test_meta_data = []
    train_out_path = os.path.join(path, 'train_meta_data.json')
    test_out_path = os.path.join(path, 'test_meta_data.json')
    
    all_idx = list(range(1, 51))  # 1~50のリスト
    chunk_size = 10
    train_spk = [i for i in all_idx if i not in range((idx-1)*chunk_size+1, idx*chunk_size+1)]  # 11~50のリスト
    test_text = list(range((idx-1)*chunk_size+1, idx*chunk_size+1))  # 1~10のリスト
    test_spk = random.sample(test_text, 5)  # 1~10のリストからランダムに5抽出
    for spk_id in train_spk:
        spk_m = f'm{str(spk_id).zfill(2)}'  # m01
        spk_f = f'f{str(spk_id).zfill(2)}'
        for emotion in ['neu', 'joy', 'sad', 'ang']:
            train_text = random.sample(train_spk, 30)  # 11~50のリストからランダムに30抽出
            for text_id in train_text:
                wav_m = os.path.join(DATASET_PATH, spk_m, emotion, f'{spk_m}_{emotion}_{str(text_id).zfill(2)}.wav')
                assert os.path.isfile(wav_m)
                wav_f = os.path.join(DATASET_PATH, spk_f, emotion, f'{spk_f}_{emotion}_{str(text_id).zfill(2)}.wav')
                assert os.path.isfile(wav_f)

                train_meta_data.append({
                    'path': wav_f,
                    'label': emotion,
                    'speaker': spk_m
                })
                train_meta_data.append({
                    'path': wav_m,
                    'label': emotion,
                    'speaker': spk_f
                })

    train_data = {
        'labels': {'neu': 0, 'joy': 1, 'sad': 2, 'ang': 3},
        'meta_data': train_meta_data
    }

    with open(train_out_path, 'w') as f:
        json.dump(train_data, f)

    for spk_id in test_spk:
        spk_m = f'm{str(spk_id).zfill(2)}'  # m01
        spk_f = f'f{str(spk_id).zfill(2)}'
        for emotion in ['neu', 'joy', 'sad', 'ang']:
            for text_id in test_text:
                wav_m = os.path.join(DATASET_PATH, spk_m, emotion, f'{spk_m}_{emotion}_{str(text_id).zfill(2)}.wav')
                assert os.path.isfile(wav_m)
                wav_f = os.path.join(DATASET_PATH, spk_f, emotion, f'{spk_f}_{emotion}_{str(text_id).zfill(2)}.wav')
                assert os.path.isfile(wav_f)

                test_meta_data.append({
                    'path': wav_f,
                    'label': emotion,
                    'speaker': spk_m
                })
                test_meta_data.append({
                    'path': wav_m,
                    'label': emotion,
                    'speaker': spk_f
                })

    test_data = {
        'labels': {'neu': 0, 'joy': 1, 'sad': 2, 'ang': 3},
        'meta_data': test_meta_data
    }

    with open(test_out_path, 'w') as f:
        json.dump(test_data, f)

    print(f'Train data: {len(train_meta_data)}')
    print(f'Test data:  {len(test_meta_data)}')

def main():
    """Main function."""
    out_dir = os.path.join(METADATA_PTAH, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    for idx in range(1, FOLD_NUM+1):
        os.makedirs(f"{out_dir}/Session{idx}", exist_ok=True)
        
        print(f"Session{idx}")
        preprocess(idx, path_join(f"{out_dir}/Session{idx}"))


if __name__ == "__main__":
    main()
