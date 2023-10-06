import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter
import math


# PSD计算函数
def compute_PSD(signal, fs, nperseg=256):
    """
    计算功率谱密度（PSD）
    :param signal: 输入信号
    :param fs: 采样率
    :param nperseg: 每段的长度（可以根据需要调整）
    :return: f: 频率轴，Pxx: 功率谱密度
    """
    from scipy.signal import welch
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    return f, Pxx


def decompose(file, name):
    # trial*channel*sample
    data = loadmat(file)
    frequency = 200  # seed数据集的采样率下采样到200了

    decomposed_de = np.empty([0, 62, 5])
    decomposed_psd = np.empty([0, 62, 5])  # 存储PSDs
    label = np.array([])
    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    for trial in range(15):
        tmp_trial_signal = data[name + '_eeg' + str(trial + 1)]
        num_sample = int(len(tmp_trial_signal[0]) / 100)  # 每个样本为100个采样点
        print('{}-{}'.format(trial + 1, num_sample))

        temp_de = np.empty([0, num_sample])
        temp_psd = np.empty([0, num_sample])  # 每个trial对应的PSDs
        label = np.append(label, [all_label[trial]] * num_sample)

        for channel in range(62):
            trial_signal = tmp_trial_signal[channel]

            # 滤波
            delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)

            # 计算每个频段的DE
            DE_delta = np.zeros(shape=[0], dtype=float)
            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)
            
            PSD_delta = np.zeros(shape=[0], dtype=float)
            PSD_theta = np.zeros(shape=[0], dtype=float)
            PSD_alpha = np.zeros(shape=[0], dtype=float)
            PSD_beta = np.zeros(shape=[0], dtype=float)
            PSD_gamma = np.zeros(shape=[0], dtype=float)
            
            for index in range(num_sample):
                DE_delta = np.append(DE_delta, compute_DE(delta[index * 100:(index + 1) * 100]))
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 100:(index + 1) * 100]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 100:(index + 1) * 100]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 100:(index + 1) * 100]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 100:(index + 1) * 100]))

                # 计算PSD
                _, psd_delta = compute_PSD(delta[index * 100:(index + 1) * 100], frequency)
                _, psd_theta = compute_PSD(theta[index * 100:(index + 1) * 100], frequency)
                _, psd_alpha = compute_PSD(alpha[index * 100:(index + 1) * 100], frequency)
                _, psd_beta = compute_PSD(beta[index * 100:(index + 1) * 100], frequency)
                _, psd_gamma = compute_PSD(gamma[index * 100:(index + 1) * 100], frequency)

                PSD_delta = np.append(PSD_delta, psd_delta)
                PSD_theta = np.append(PSD_theta, psd_theta)
                PSD_alpha = np.append(PSD_alpha, psd_alpha)
                PSD_beta = np.append(PSD_beta, psd_beta)
                PSD_gamma = np.append(PSD_gamma, psd_gamma)

            temp_de = np.vstack([temp_de, DE_delta])
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

            # 将PSD也合并到temp_psd中
            temp_psd = np.vstack([temp_psd, PSD_delta])
            temp_psd = np.vstack([temp_psd, PSD_theta])
            temp_psd = np.vstack([temp_psd, PSD_alpha])
            temp_psd = np.vstack([temp_psd, PSD_beta])
            temp_psd = np.vstack([temp_psd, PSD_gamma])

        # 重构数据
        temp_trial_de = temp_de.reshape(-1, 5, num_sample)
        temp_trial_de = temp_trial_de.transpose([2, 0, 1])
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

        temp_trial_psd = temp_psd.reshape(-1, 5, num_sample)
        temp_trial_psd = temp_trial_psd.transpose([2, 0, 1])
        decomposed_psd = np.vstack([decomposed_psd, temp_trial_psd])

    print("trial_DE shape:", decomposed_de.shape)
    print("trial_PSD shape:", decomposed_psd.shape)
    return decomposed_de, decomposed_psd, label


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# 究极整合版
import os
import numpy as np

file_path = 'F:\实验室\数据集\SEED\Preprocessed_EEG\\'

people_name = ['1_20131027', '1_20131030', '1_20131107',
               '6_20130712', '6_20131016', '6_20131113',
               '7_20131027', '7_20131030', '7_20131106',
               '15_20130709', '15_20131016', '15_20131105',
               '12_20131127', '12_20131201', '12_20131207',
               '10_20131130', '10_20131204', '10_20131211',
               '2_20140404', '2_20140413', '2_20140419',
               '5_20140411', '5_20140418', '5_20140506',
               '8_20140511', '8_20140514', '8_20140521',
               '13_20140527', '13_20140603', '13_20140610',
               '3_20140603', '3_20140611', '3_20140629',
               '14_20140601', '14_20140615', '14_20140627',
               '11_20140618', '11_20140625', '11_20140630',
               '9_20140620', '9_20140627', '9_20140704',
               '4_20140621', '4_20140702', '4_20140705']

short_name = ['djc', 'djc', 'djc', 'mhw', 'mhw', 'mhw', 'phl', 'phl', 'phl',
              'zjy', 'zjy', 'zjy', 'wyw', 'wyw', 'wyw', 'ww', 'ww', 'ww',
              'jl', 'jl', 'jl', 'ly', 'ly', 'ly', 'sxy', 'sxy', 'sxy',
              'xyl', 'xyl', 'xyl', 'jj', 'jj', 'jj', 'ys', 'ys', 'ys',
              'wsf', 'wsf', 'wsf', 'wk', 'wk', 'wk', 'lqj', 'lqj', 'lqj']


# 在主代码中使用 `decompose` 函数
X = np.empty([0, 62, 5])
y = np.empty([0, 1])
PSD_X = np.empty([0, 62, 5])  # 用于存储PSD结果

for i in range(len(people_name)):
    file_name = file_path + people_name[i]
    print('processing {}'.format(people_name[i]))
    decomposed_de, decomposed_psd, label = decompose(file_name, short_name[i])
    X = np.vstack([X, decomposed_de])
    PSD_X = np.vstack([PSD_X, decomposed_psd])
    y = np.append(y, label)

print("Final X shape:", X.shape)
print("Final PSD_X shape:", PSD_X.shape)
