import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch
import math


# 获取标签的函数（保持不变）
def get_2_labels_list(behavior_data_dir, behavior_file):
    behavior_df = pd.read_excel(os.path.join(behavior_data_dir, behavior_file))
    event_dict = {
        1: 'img_F/rep_F',
        2: 'img_F/rep_G',
        3: 'img_F/rep_H',
        4: 'img_G/rep_F',
        5: 'img_G/rep_G',
        6: 'img_G/rep_H',
        7: 'img_H/rep_F',
        8: 'img_H/rep_G',
        9: 'img_H/rep_H',
        10: 'img_nan/rep_NAN',
    }
    event_dict_inv = {v: k for k, v in event_dict.items()}
    CRESP_img = behavior_df['WordStim.CRESP']
    RESP_rep = behavior_df['WordStim.RESP']
    labels_str_list = ['img_' + str(CRESP_img[i]) + '/rep_' + str(RESP_rep[i]).upper() for i in range(len(CRESP_img))]
    labels_list = [event_dict_inv[i] for i in labels_str_list]
    return labels_list, event_dict

def get_img_or_resp_labels(labels, label_option='img'):
    labels = np.array([label for label in labels if label != 10])
    img_label = (labels - 1) // 3
    resp_label = (labels - 1) % 3
    if label_option == 'img':
        return img_label
    else:
        return resp_label

def load_labels_for_subject(subject_id, behavior_data_dir,label_option):
    behavior_file = f'{subject_id}.xlsx'  # 假设标签文件名为 subject_id.xlsx
    labels, events_dict = get_2_labels_list(behavior_data_dir, behavior_file)
    labels = get_img_or_resp_labels(labels, label_option=label_option)
    return labels


# 特征计算的辅助函数
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def compute_PSD_scalar(signal, fs, nperseg=256):
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    return np.mean(Pxx)


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# 处理每个被试的EEG数据和提取特征
def decompose_for_subject(subject_id, behavior_data_dir, eeg_data_dir, fs=200):
    # 获取该被试的标签
    labels = load_labels_for_subject(subject_id, behavior_data_dir,"img")

    # 创建存储特征和标签的容器
    decomposed_features = np.empty([0, 64, 10])  # 每个通道10个特征（5个DE + 5个PSD）
    all_labels = np.array([])

    for trial in range(124):  # 每个被试有126个片段
        eeg_file = os.path.join(eeg_data_dir, f'eeg_{subject_id}_{trial + 1}.npy')  # 假设数据存储为 .npy 文件
        tmp_trial_signal = np.load(eeg_file)  # 加载该片段的EEG数据

        num_sample = int(tmp_trial_signal.shape[1] / fs)  # 根据采样频率计算片段数

        temp_features = np.empty([num_sample, 64, 10])  # 创建一个空数组来存放特征
        all_labels = np.append(all_labels, [labels[trial]] * num_sample)

        for channel in range(64):  # 62个通道
            trial_signal = tmp_trial_signal[channel]

            # 计算不同频带的滤波信号
            delta = butter_bandpass_filter(trial_signal, 1, 4, fs, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, fs, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, fs, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, fs, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 51, fs, order=3)

            for index in range(num_sample):
                # 计算每个频带的DE特征
                de_delta = compute_DE(delta[index * fs:(index + 1) * fs])
                de_theta = compute_DE(theta[index * fs:(index + 1) * fs])
                de_alpha = compute_DE(alpha[index * fs:(index + 1) * fs])
                de_beta = compute_DE(beta[index * fs:(index + 1) * fs])
                de_gamma = compute_DE(gamma[index * fs:(index + 1) * fs])

                # 计算每个频带的PSD标量特征
                psd_delta = compute_PSD_scalar(delta[index * fs:(index + 1) * fs], fs)
                psd_theta = compute_PSD_scalar(theta[index * fs:(index + 1) * fs], fs)
                psd_alpha = compute_PSD_scalar(alpha[index * fs:(index + 1) * fs], fs)
                psd_beta = compute_PSD_scalar(beta[index * fs:(index + 1) * fs], fs)
                psd_gamma = compute_PSD_scalar(gamma[index * fs:(index + 1) * fs], fs)

                # 将DE和PSD特征组合
                temp_features[index, channel] = np.array([
                    de_delta, de_theta, de_alpha, de_beta, de_gamma,
                    psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma
                ])

        decomposed_features = np.vstack([decomposed_features, temp_features])

    return decomposed_features, all_labels


# 运行并保存特征
behavior_data_dir = r'F:\实验室\代码\pythonProject1\data\behavior-excel'
eeg_data_dir = r'F:\实验室\数据集\IAPS\eeg-preprocessed\python-data\origin-eeg-data'

# for subject_id in range(1, 23):  # 处理22个被试
#     features, labels = decompose_for_subject(subject_id, behavior_data_dir, eeg_data_dir)
#
#     # 保存特征和标签
#     np.save(f'features_subject_{subject_id}.npy', features)
#     np.save(f'labels_subject_{subject_id}.npy', labels)
#
#     print(f"Features and labels for subject {subject_id} saved.")

features, labels = decompose_for_subject(1, behavior_data_dir, eeg_data_dir)

# 保存特征和标签
np.save(f'features_subject_{1}.npy', features)
np.save(f'labels_subject_{1}.npy', labels)
