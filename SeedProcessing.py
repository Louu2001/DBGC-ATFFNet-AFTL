import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter, welch
import math

def compute_PSD_scalar(signal, fs, nperseg=256):
    """
    计算功率谱密度（PSD）并返回一个标量值
    :param signal: 输入信号
    :param fs: 采样率
    :param nperseg: 每段的长度
    :return: PSD的标量值（这里使用平均功率）
    """
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    # 计算平均功率作为标量值
    return np.mean(Pxx)

def decompose(file, name):
    data = loadmat(file)
    frequency = 200

    decomposed_features = np.empty([0, 62, 10])  # 修改为10，因为每个通道现在有10个特征(5个DE + 5个PSD)
    label = np.array([])
    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    for trial in range(15):
        tmp_trial_signal = data[name + '_eeg' + str(trial + 1)]
        num_sample = int(len(tmp_trial_signal[0]) / 200)
        print('{}-{}'.format(trial + 1, num_sample))

        temp_features = np.empty([num_sample, 62, 10])  # 修改为10个特征
        label = np.append(label, [all_label[trial]] * num_sample)

        for channel in range(62):
            trial_signal = tmp_trial_signal[channel]

            # 滤波
            delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)

            for index in range(num_sample):
                # 计算DE特征
                de_delta = compute_DE(delta[index * 200:(index + 1) * 200])
                de_theta = compute_DE(theta[index * 200:(index + 1) * 200])
                de_alpha = compute_DE(alpha[index * 200:(index + 1) * 200])
                de_beta = compute_DE(beta[index * 200:(index + 1) * 200])
                de_gamma = compute_DE(gamma[index * 200:(index + 1) * 200])

                # 计算PSD标量特征
                psd_delta = compute_PSD_scalar(delta[index * 200:(index + 1) * 200], frequency)
                psd_theta = compute_PSD_scalar(theta[index * 200:(index + 1) * 200], frequency)
                psd_alpha = compute_PSD_scalar(alpha[index * 200:(index + 1) * 200], frequency)
                psd_beta = compute_PSD_scalar(beta[index * 200:(index + 1) * 200], frequency)
                psd_gamma = compute_PSD_scalar(gamma[index * 200:(index + 1) * 200], frequency)

                # 将DE和PSD特征组合在一起
                temp_features[index, channel] = np.array([
                    de_delta, de_theta, de_alpha, de_beta, de_gamma,
                    psd_delta, psd_theta, psd_alpha, psd_beta, psd_gamma
                ])

        decomposed_features = np.vstack([decomposed_features, temp_features])

    print("Features shape:", decomposed_features.shape)
    return decomposed_features, label

# 其他辅助函数保持不变
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

# 调用 decompose 函数
features, labels = decompose('1_20131027.mat', 'djc')

# 保存特征和标签到文件
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Features and labels saved to 'features.npy' and 'labels.npy'.")