# python calc_acc.py emotion_libri
import os
import argparse
from sklearn import metrics

def main():
    FOLD_NUM = 5
    ROOT_PATH = 'result/downstream'
    
    args = _get_parser().parse_args()

    # クラス一覧を取得
    class_list = _get_class_dict(os.path.join(ROOT_PATH, f'{args.d}_fold1'))
    class_dict = dict()
    for i, c in enumerate(class_list):
        class_dict[c] = i

    acc_list = list()
    war_list = list()
    for i in range(1, FOLD_NUM+1):
        dname = os.path.join(ROOT_PATH, f'{args.d}_fold{i}')
        assert os.path.isdir(dname)
        
        truth_file = os.path.join(dname, f'test_fold{i}_truth.txt')
        predict_file = os.path.join(dname, f'test_fold{i}_predict.txt')

        with open(truth_file, mode='r', encoding='utf-8') as tf:
            lines = tf.readlines()
            truth_list = [class_dict[s.rstrip().split(' ')[1]] for s in lines]

        with open(predict_file, mode='r', encoding='utf-8') as pf:
            lines = pf.readlines()
            predict_list = [class_dict[s.rstrip().split(' ')[1]] for s in lines]

        acc = metrics.accuracy_score(truth_list, predict_list)
        acc_list.append(acc)
        war = metrics.recall_score(truth_list, predict_list, average='weighted')
        war_list.append(war)

    print('Accuracy:')
    print(' '.join([str(round(acc*100, 2)) for acc in acc_list]))
    print(round(sum(acc_list*100)/len(acc_list), 2))

    print('\nWeighted Average Recall')
    print(' '.join([str(round(war*100, 2)) for war in war_list]))
    print(round(sum(war_list*100)/len(war_list), 2))

def _get_class_dict(dpath):
   with open(os.path.join(dpath, 'test_fold1_truth.txt'), mode='r', encoding='utf-8') as pf:
        lines = pf.readlines()
        classes = [s.rstrip().split(' ')[1] for s in lines]
        
        # 重複を消して返す
        return list(dict.fromkeys(classes))

def _get_parser():
    parser = argparse.ArgumentParser(description='Calculate accuracy.')
    parser.add_argument(
        'd',
        type=str,
        help='FULL Path of Directory',
    )
    return parser

if __name__ == "__main__":
  main()
