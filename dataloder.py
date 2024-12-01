import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from SeedProcessing import butter_bandpass_filter, compute_DE, compute_PSD_scalar


class SEED(Dataset):
    def __init__(self, root_path=r'F:\实验室\数据集\SEED\ExtractedFeatures',
                 subject_ids=list(range(1, 16)), train_test='TRAIN', window_size=200):
        super(SEED, self).__init__()
        self.root_path = root_path
        self.window_size = window_size
        self.load_path = r"F:\实验室\数据集\SEED\Preprocessed_EEG"
        sub_id = subject_ids[0]

        data, labels = self.decompose('1_20131027.mat', 'djc', train_test)

        # 将 numpy.ndarray 转换为 PyTorch Tensor
        labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

        self.labels = torch.Tensor(labels)
        self.data = torch.Tensor(data)
        self.class_names = torch.unique(self.labels)
        # self.max_seq_len = self.feature.shape[1]
        # self.feature_df = pd.DataFrame(self.feature.numpy()[0])
        # self.max_space_seq_len = self.feature.shape[1]
        # self.max_time_seq_len = 200

    def decompose(self, file, name, flag):
        data = loadmat(file)
        frequency = 200

        decomposed_features = np.empty([0, 62, 10])  # 修改为10，因为每个通道现在有10个特征(5个DE + 5个PSD)
        label = np.array([])
        all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

        trials = np.arange(1, 16)

        if flag == 'TRAIN':
            trials = trials[:9]
        elif flag == 'TEST':
            trials = trials[9:]

        for trial in trials:
            tmp_trial_signal = data[name + '_eeg' + str(trial)]
            num_sample = int(len(tmp_trial_signal[0]) / 200)
            print('{}-{}'.format(trial, num_sample))

            temp_features = np.empty([num_sample, 62, 10])  # 修改为10个特征
            label = np.append(label, [all_label[trial-1]] * num_sample)

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

    def __getitem__(self, index):
        # 获取数据

        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)