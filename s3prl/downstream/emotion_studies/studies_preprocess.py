import os
import glob
import json

DATASET_ROOT = '/abelab/DB4'
STUDIES_FILELISTS_ROOT = 'downstream/emotion_studies/filelists'
METADATA_PTAH = 'downstream/emotion_studies'

enSpk2jpSpk = {'Teacher':'講師', 'FStudent':'女子生徒', 'MStudent':'男子生徒'}
enEmo2jpEmo = {'平静': 'Neutral', '喜び': 'Happy', '悲しみ': 'Sad', '怒り': 'Angry'}

def _get_filename2emotion_studies(studies_dir):
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

# /work/abelab4/s_koha/vits/prep_dataset/05_prep_studies.pyに倣う
def make_filelists(studies_dir):
    ############################################
    # ここを変える
    # validationとtestのディレクトリ名
    VAL_DIRS = ['LD04']
    TEST_DIRS = ['LD01', 'LD02', 'LD03', 'SD01', 'SD06', 'SD07', 'SD12']
    dst_filename = 'studies_audio_sid_text'  # 出力ファイル名（一部）
    speakers = ['Teacher', 'MStudent', 'FStudent']
    enSpk2jpSpk = {'Teacher':'講師', 'FStudent':'女子生徒', 'MStudent':'男子生徒'}
    speaker2id = {'講師':0, '男子生徒':1, '女子生徒':2}  # 話者-->ID
    ############################################

    os.makedirs(STUDIES_FILELISTS_ROOT, exist_ok=True)

    # 出力するファイルリスト
    filelist = dict()
    filelist['val'] = list()
    filelist['test'] = list()
    filelist['train'] = list()

    for type_name in ['ITA', 'Long_dialogue', 'Short_dialogue']:
        type_dir = os.path.join(studies_dir, type_name)
        # Emotion100-Angry, LD01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]

        for dname in dir_list:
            if dname in VAL_DIRS:
                tr_name = 'val'
            elif dname in TEST_DIRS:
                tr_name = 'test'
            else:
                tr_name = 'train'
            
            d = os.path.join(type_dir, dname)

            for spk in speakers:
                files = list()  # 一旦保存
                txt_files = sorted(glob.glob(os.path.join(d, '**/txt/*.txt'), recursive=True))
                wav_files = sorted(glob.glob(os.path.join(d, f'**/wav/*{spk}*.wav'), recursive=True))
                i = 0
                for txt_file in txt_files:
                    with open(txt_file, mode='r', encoding='utf-8') as f:
                        lines = f.readlines()
                    lines = [s for s in lines if s.split('|')[0]==enSpk2jpSpk[spk]]
                    for line in lines:
                        speaker = line.split('|')[0]  # 話者
                        emotion = line.split('|')[1]  # 感情
                        
                        # 怒り（Angry）は除外，しない！
                        # if emotion == '怒り':
                        #     i+=1
                        #     continue
                        
                        text = line.split('|',)[2]  # 平文
                        text = text.replace('\u3000', '')  # 全角スペースを削除
                        filepath = wav_files[i]

                        newline = f'{filepath}|{speaker2id[speaker]}|{text}\n'
                        files.append(newline)

                        i+=1

                filelist[tr_name].extend(files)

    for t in ['val', 'test', 'train']:
        savename = os.path.join(STUDIES_FILELISTS_ROOT, f'{dst_filename}_{t}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])

def main():
    studies_dir = os.path.join(DATASET_ROOT, 'STUDIES')
    assert os.path.isdir(studies_dir)

    # ファイルパスをkeyに感情を記録
    filename2emotion = _get_filename2emotion_studies(studies_dir)

    make_filelists(studies_dir)

    out_dir = os.path.join(METADATA_PTAH, 'meta_data')
    os.makedirs(os.path.join(out_dir), exist_ok=True)

    # 学習データの分割ごとにファイル作成
    studies_filelists = sorted(glob.glob(os.path.join(STUDIES_FILELISTS_ROOT, '*.txt')))

    for studies_filelist in studies_filelists:
        tr_type = os.path.splitext(os.path.basename(studies_filelist))[0].split('_')[4]
        
        meta_data = list()
        train_out_path = os.path.join(out_dir, f'{tr_type}_meta_data.json')

        with open(studies_filelist, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            filepath_list = [s.rstrip().split('|')[0] for s in lines]
            for filepath in filepath_list:
                assert os.path.exists(filepath)

                meta_data.append({
                    'path': filepath,
                    'label': filename2emotion[filepath]
                })
                
        train_data = {
            'labels': {'Neutral': 0, 'Happy': 1, 'Sad': 2, 'Angry': 3},
            'meta_data': meta_data
        }

        with open(train_out_path, 'w') as f:
            json.dump(train_data, f)
            print(f'{tr_type} data: {len(meta_data)}')

if __name__ == '__main__':
    main()
