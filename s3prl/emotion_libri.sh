for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    python3 run_downstream.py -n emotion_libri_$test_fold -m train -u hubert -d emotion_libri -c downstream/emotion_libri/config.yaml -o "config.downstream_expert.datarc.test_fold='$test_fold'"
    python3 run_downstream.py -m evaluate -e result/downstream/emotion_libri_$test_fold/dev-best.ckpt
done