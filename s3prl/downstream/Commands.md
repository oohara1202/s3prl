# Commands

## IEMOCAP

### HuBERT LibriSpeech model

```bash
nohup ./emotion_libri.sh > emotion_libri.out
```

```bash
for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    python3 run_downstream.py -n emotion_libri_$test_fold -m train -u hubert -d emotion_libri -c downstream/emotion_libri/config.yaml -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/emotion_libri_$test_fold/dev-best.ckpt
done
```

### HuBERT ReazonSpeech model

```bash
nohup ./emotion_reazon.sh > emotion_reazon.out
```

```bash
for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    python3 run_downstream.py -n emotion_reazon_$test_fold -m train -u hf_hubert_custom -k rinna/japanese-hubert-base -d emotion_reazon -c downstream/emotion_reazon/config.yaml -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/emotion_reazon_$test_fold/dev-best.ckpt
done
```
