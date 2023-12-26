# python downstream/emotion_jtes-all/jtes-all_preprocess.py
import os
import json

DATASET_PATH = '/abelab/DB4/JTES/jtes_v1.1/wav'
METADATA_PTAH = 'downstream/emotion_jtes-all'

def main():
    """Main function."""

    emotion_conv = {'neu':'Neutral', 'joy':'Happy', 'sad':'Sad'}

    out_dir = os.path.join(METADATA_PTAH, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)

    train_meta_data = []
    train_out_path = os.path.join(out_dir, 'train_meta_data.json')

    # 話者1 ~ 47をtrainに
    for spk_idx in range(1, 48):
        spk_m = f'm{str(spk_idx).zfill(2)}'  # m01
        spk_f = f'f{str(spk_idx).zfill(2)}'  # f01
        # for emotion in ['neu', 'joy', 'sad', 'ang']:
        for emotion in ['neu', 'joy', 'sad']:  # angを抜く
            for text_idx in range(1, 51):
                wav_m = os.path.join(DATASET_PATH, spk_m, emotion, f'{spk_m}_{emotion}_{str(text_idx).zfill(2)}.wav')
                assert os.path.exists(wav_m)
                wav_f = os.path.join(DATASET_PATH, spk_f, emotion, f'{spk_f}_{emotion}_{str(text_idx).zfill(2)}.wav')
                assert os.path.exists(wav_f)

                train_meta_data.append({
                    'path': wav_f,
                    'label': emotion_conv[emotion],
                    'speaker': spk_m
                })
                train_meta_data.append({
                    'path': wav_m,
                    'label': emotion_conv[emotion],
                    'speaker': spk_f
                })

    train_data = {
        'labels': {'Neutral': 0, 'Happy': 1, 'Sad': 2},
        'meta_data': train_meta_data
    }

    with open(train_out_path, 'w') as f:
        json.dump(train_data, f)

    print(f'Train data: {len(train_meta_data)}')

    val_meta_data = []
    val_out_path = os.path.join(out_dir, 'val_meta_data.json')

    # 話者48 ~ 50をvalに
    for spk_idx in range(48, 51):
        spk_m = f'm{str(spk_idx).zfill(2)}'  # m48
        spk_f = f'f{str(spk_idx).zfill(2)}'  # f48
        # for emotion in ['neu', 'joy', 'sad', 'ang']:
        for emotion in ['neu', 'joy', 'sad']:  # angを抜く
            for text_idx in range(1, 51):
                wav_m = os.path.join(DATASET_PATH, spk_m, emotion, f'{spk_m}_{emotion}_{str(text_idx).zfill(2)}.wav')
                assert os.path.exists(wav_m)
                wav_f = os.path.join(DATASET_PATH, spk_f, emotion, f'{spk_f}_{emotion}_{str(text_idx).zfill(2)}.wav')
                assert os.path.exists(wav_f)

                val_meta_data.append({
                    'path': wav_f,
                    'label': emotion_conv[emotion],
                    'speaker': spk_m
                })
                val_meta_data.append({
                    'path': wav_m,
                    'label': emotion_conv[emotion],
                    'speaker': spk_f
                })

    val_data = {
        'labels': {'Neutral': 0, 'Happy': 1, 'Sad': 2},
        'meta_data': val_meta_data
    }

    with open(val_out_path, 'w') as f:
        json.dump(val_data, f)

    print(f'val data: {len(val_meta_data)}')

if __name__ == "__main__":
    main()
