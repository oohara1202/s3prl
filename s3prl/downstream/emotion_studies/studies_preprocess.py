import os
import glob
import json

DATASET_ROOT = '/abelab/DB4'
FILELISTS_ROOT = '/work/abelab4/s_koha/vits/filelists/studies'
METADATA_PTAH = 'downstream/emotion_studies'

enSpk2jpSpk = {'Teacher':'講師', 'FStudent':'女子生徒', 'MStudent':'男子生徒'}
enEmo2jpEmo = {'平静': 'Neutral', '喜び': 'Happy', '悲しみ': 'Sad', '怒り': 'Angry'}

def _get_filename2emotion(studies_dir):
    filename2emotion = dict()

    for type_name in ['ITA', 'Long_dialogue', 'Short_dialogue']:
        type_dir = os.path.join(studies_dir, type_name)
        
        # Emotion100-Angry, LD01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]
        for dname in dir_list:
            d = os.path.join(type_dir, dname)

            for spk in ['Teacher', 'MStudent', 'FStudent']:
                txt_files = sorted(glob.glob(os.path.join(d, '**/txt/*.txt'), recursive=True))
                wav_files = sorted(glob.glob(os.path.join(d, f'**/wav/*{spk}*.wav'), recursive=True))
                i = 0
                for txt_file in txt_files:
                    with open(txt_file, mode='r', encoding='utf-8') as f:
                        lines = f.readlines()
                    lines = [s for s in lines if s.split('|')[0]==enSpk2jpSpk[spk]]
                    for line in lines:
                        emotion = line.split('|')[1]  # 感情

                        filepath = wav_files[i]

                        filename2emotion[filepath] = enEmo2jpEmo[emotion]

                        i+=1

    return filename2emotion

def main():
    studies_dir = os.path.join(DATASET_ROOT, 'STUDIES')
    assert os.path.isdir(studies_dir)

    # ファイルパスをkeyに感情を記録
    filename2emotion = _get_filename2emotion(studies_dir)

    out_dir = os.path.join(METADATA_PTAH, 'meta_data')
    os.makedirs(os.path.join(out_dir), exist_ok=True)

    # 学習データの分割ごとにファイル作成
    filelists = glob.glob(os.path.join(FILELISTS_ROOT, '*.txt'))
    for filelist in filelists:
        tr_type = os.path.splitext(os.path.basename(filelist))[0].split('_')[4]
        
        meta_data = list()
        train_out_path = os.path.join(out_dir, f'{tr_type}_meta_data.json')

        with open(filelist, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            filepath_list = [s.rstrip().split('|')[0].replace('dataset/', DATASET_ROOT+'/') for s in lines]
            for filepath in filepath_list:
                assert os.path.exists(filepath)

                meta_data.append({
                    'path': filepath,
                    'label': filename2emotion[filepath]
                })
                
        train_data = {
            'labels': {'Neutral': 0, 'Happy': 1, 'Sad': 2},
            'meta_data': meta_data
        }

        with open(train_out_path, 'w') as f:
            json.dump(train_data, f)
            print(f'{tr_type} data: {len(meta_data)}')

if __name__ == '__main__':
    main()
