# memo

## STUDIES-PLUS SER model

The data consists of studies (three speakers) and calls (one speaker).

Training procedure:

```bash
python run_downstream.py -n emotion_studiesplus_reazon -m train -u hf_hubert_custom -k rinna/japanese-hubert-base -d emotion_studiesplus -c downstream/emotion_studiesplus/config.yaml
```

Extract and save features from "studies-calls" filelists:

```bash
python run_downstream.py -m extract -e result/downstream/emotion_studiesplus_reazon/dev-best.ckpt --extract_file downstream/emotion_studiesplus/meta_data/train_meta_data.json

python run_downstream.py -m extract -e result/downstream/emotion_studiesplus_reazon/dev-best.ckpt --extract_file downstream/emotion_studiesplus/meta_data/val_meta_data.json

python run_downstream.py -m extract -e result/downstream/emotion_studiesplus_reazon/dev-best.ckpt --extract_file downstream/emotion_studiesplus/meta_data/test_meta_data.json
```

## STUDIES-Teacher SER model

This model is used for objective evaluation experiments.
The data is only studies-teacher.

Training procedure:

```bash
python run_downstream.py -n emotion_teacher_reazon -m train -u hf_hubert_custom -k rinna/japanese-hubert-base -d emotion_teacher -c downstream/emotion_teacher/config.yaml
```

## STUDIES SER model

The data consists of studies (three speakers) with FOUR EMOTIONS including ANGRY.

```bash
python run_downstream.py -n emotion_studies_reazon -m train -u hf_hubert_custom -k rinna/japanese-hubert-base -d emotion_studies -c downstream/emotion_studies/config.yaml
```

## JTES SER model

The data consists of ALL data of jtes.

```bash
python run_downstream.py -n emotion_jtes-all_reazon -m train -u hf_hubert_custom -k rinna/japanese-hubert-base -d emotion_jtes-all -c downstream/emotion_jtes-all/config.yaml
```

Extract and save features from "studies-calls" filelists:

```bash
python run_downstream.py -m extract -e result/downstream/emotion_jtes-all_reazon/dev-best.ckpt --extract_file downstream/emotion_studiesplus/meta_data/train_meta_data.json

python run_downstream.py -m extract -e result/downstream/emotion_jtes-all_reazon/dev-best.ckpt --extract_file downstream/emotion_studiesplus/meta_data/val_meta_data.json

python run_downstream.py -m extract -e result/downstream/emotion_jtes-all_reazon/dev-best.ckpt --extract_file downstream/emotion_studiesplus/meta_data/test_meta_data.json
```

## How to inference

```bash
python run_downstream.py -m infer -e result/downstream/emotion_teacher_reazon/dev-best.ckpt --inference_dir /work/abelab4/s_koha/vits/wav/atr503/studies-teacher_base_ft
python run_downstream.py -m infer -e result/downstream/emotion_teacher_reazon/dev-best.ckpt --inference_dir /work/abelab4/s_koha/vits/wav/atr503/studies-teacher_xvector_ft
```
