import numpy as np
import pandas as pd
import yaml
from time import time
import warnings
from metrics import spot_eval

warnings.filterwarnings('ignore')


def anomaly_score_example(source: np.array, reconstructed: np.array):
    """
    Calculate anomaly score
    :param source: original data
    :param reconstructed: reconstructed data
    :return:
    """
    n, d = source.shape
    d_dis = np.zeros((d,))
    for i in range(d):
        dis = np.abs(source[:, i] - reconstructed[:, i])
        dis = dis - np.mean(dis)
        d_dis[i] = np.percentile(dis, anomaly_score_example_percentage)
    if d <= anomaly_distance_topn:
        return d / np.sum(1 / d_dis)
    topn = 1 / d_dis[np.argsort(d_dis)][-1 * anomaly_distance_topn:]
    return anomaly_distance_topn / np.sum(topn)


def p_normalize(x: np.array):
    """
    Normalization
    :param: data
    :return:
    """
    p_min = 0.05
    x_max, x_min = np.max(x), np.min(x)
    x_min *= 1 - p_min
    return (x - x_min) / (x_max - x_min)


def read_config(config: dict):
    """
    init global parameters  
    :param config: config dictionary, please refer to detector-config.yml
    :return:
    """
    global random_state, \
        workers, \
        rec_window, rec_stride, \
        det_window, det_stride, \
        anomaly_scoring, \
        without_grouping, \
        data_path, rb, re, cb, ce, header, rec_windows_per_cycle, \
        label_path, save_path, \
        anomaly_score_example_percentage, anomaly_distance_topn
    if 'anomaly_scoring' in config.keys():
        anomaly_scoring_config = config['anomaly_scoring']
        if 'anomaly_score_example' in anomaly_scoring_config.keys():
            anomaly_score_example_percentage = \
                int(anomaly_scoring_config['anomaly_score_example']
                    ['percentage'])
            if anomaly_score_example_percentage > 100 \
                    or anomaly_score_example_percentage < 0:
                raise Exception('percentage must be between 0 and 100')
            anomaly_distance_topn = \
                int(anomaly_scoring_config['anomaly_score_example']['topn'])

    if 'global' in config.keys():
        global_config = config['global']
        if 'random_state' in global_config.keys():
            random_state = int(global_config['random_state'])

    data_config = config['data']

    if 'reconstruct' in data_config.keys():
        data_rec_config = data_config['reconstruct']
        rec_window = int(data_rec_config['window'])
        rec_stride = int(data_rec_config['stride'])

    if 'detect' in data_config.keys():
        data_det_config = data_config['detect']
        det_window = int(data_det_config['window'])
        det_stride = int(data_det_config['stride'])

    data_path = data_config['path']
    label_path = data_config['label_path']
    save_path = data_config['save_path']

    header = data_config['header']
    rb, re = data_config['row_begin'], data_config['row_end']
    cb, ce = data_config['col_begin'], data_config['col_end']
    rec_windows_per_cycle = data_config['rec_windows_per_cycle']

    detector_config = config['detector_arguments']
    if detector_config['anomaly_scoring'] == 'anomaly_score_example':
        anomaly_scoring = anomaly_score_example
    else:
        raise Exception(
            'unknown config[detector][anomaly_scoring]: %s',
            detector_config['anomaly_scoring']
        )
    
    workers = int(detector_config['workers'])
    without_grouping = detector_config['without_grouping']

def smooth_data(df, window=3):
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = df[col].rolling(window=window).mean().bfill().values
    return df


def run(data: np.array, label, filename):
    """
    :param data input
    """

    n, d = data.shape
    if n < rec_window * rec_windows_per_cycle:
        raise Exception('data point count less than 1 cycle')

    # data = data.values
    for i in range(d):
        data[:, i] = normalization(data[:, i])

    detector = CSAnomalyDetector(
        workers=workers,
        distance=anomaly_scoring,
        random_state=random_state,
        without_grouping=without_grouping,
    )
    rec = detector.reconstruct(
        data, rec_window, rec_windows_per_cycle, rec_stride
    )
    score = detector.predict(
        data, rec, det_window, det_stride
    )
    np.savetxt(save_path + filename.split('.')[0] + '_rec.txt', rec, '%.6f', ',')
    np.savetxt(save_path + filename.split('.')[0] + '_score.txt', score, '%.6f', ',')

    proba = spot_eval(score[:1200], score)
    # Best F-score
    precision, recall, f1score, _ = evaluation(label[rb:re], proba)

    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1_score: ', f1score)

    return precision, recall, f1score


if __name__ == '__main__':
    import argparse
    import sys
    import os

    sys.path.append(os.path.abspath('..'))

    from detector import CSAnomalyDetector
    from utils import normalization
    from utils.metrics import evaluation

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config path')
    args = parser.parse_args()
    config = 'detector-config.yml'
    if args.config:
        config = args.config
    with open(config, 'r', encoding='utf8') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    read_config(config_dict)  # init global parameters

    single_t = 0
    total_t = 0
    file_list = os.listdir(data_path)
    precision_list = []
    recall_list = []
    f1score_list = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in file_list:
        if filename.endswith('.csv'):
            t = time()
            test_data = pd.read_csv(os.path.join(data_path, filename), header=header).iloc[rb:re, cb:ce]
            label_pth = os.path.join(label_path, filename)
            print('\n\n******** current file: ', os.path.join(data_path, filename), '********')
            label = np.loadtxt(label_pth, int, delimiter=',')
            test_data = smooth_data(test_data, window=18)
            precision, recall, f1score = run(test_data.values, label, filename)
            precision_list.append(precision)
            recall_list.append(recall)
            f1score_list.append(f1score)
            single_t = time() - t
            print('Single_time: %f' % (single_t))
            total_t += single_t
    
    # print('Total_t: %f, precision: %f, recall: %f, f1score: %f' % (total_t, np.mean(precision_list), np.mean(recall_list), np.mean(f1score_list)))
