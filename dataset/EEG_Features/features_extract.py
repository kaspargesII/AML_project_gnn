import numpy as np
import mne
import pickle as pkl
import os
from scipy.signal.windows import hann


n_vids = 28
freq = 250
nsec = 30
nchn = 32
#
freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]

def _get_relative_psd(relative_energy_graph, freq_bands, sample_freq, stft_n=256):
    start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
    end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
    # print(start_index, end_index)
    psd = np.mean(relative_energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
    # print('psd:', psd.shape)
    return psd

def extract_psd_feature(data, window_size, freq_bands, stft_n=256):
    sample_freq = freq
    # Ptr operation
    if len(data.shape) > 2:
        data = np.squeeze(data)
    n_channels, n_samples = data.shape
    point_per_window = int(sample_freq * window_size)
    window_num = int(n_samples // point_per_window)
    psd_feature = np.zeros((window_num, len(freq_bands), n_channels))
    # print('psd feature shape:', psd_feature.shape)

    for window_index in range(window_num):
        start_index, end_index = point_per_window * window_index, point_per_window * (window_index + 1)
        window_data = data[:, start_index:end_index]
        hdata = window_data * hann(point_per_window)
        fft_data = np.fft.fft(hdata, n=stft_n)
        # print('fft_data shape:',fft_data.shape)
        energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])
        # print('energy_graph.shape:', energy_graph.shape)
        relative_energy_graph = energy_graph/np.sum(energy_graph)

        for band_index, band in enumerate(freq_bands):
            # band_ave_psd = _get_average_psd(energy_graph, band, sample_freq, stft_n)
            band_relative_psd = _get_relative_psd(relative_energy_graph, band, sample_freq, stft_n)
            psd_feature[window_index, band_index, :] = band_relative_psd

    return psd_feature


if __name__ == '__main__':

    datapath = '../Processed_data'

    datapaths = os.listdir(datapath)

    datapaths.sort()

    PSD_savepath = '../EEG_Features/PSD'
    if not os.path.exists(PSD_savepath):
        # 如果不存在，创建文件夹
        os.makedirs(PSD_savepath)

    DE_savepath = '../EEG_Features/DE'
    if not os.path.exists(DE_savepath):
        # 如果不存在，创建文件夹
        os.makedirs(DE_savepath)

    # (28,32,30,4)
    subs_psd = np.zeros((n_vids, nchn, nsec, len(freq_bands)))
    subs_de = np.zeros((n_vids, nchn, nsec, len(freq_bands)))

    for idx, sub in enumerate(datapaths):

        f = open(os.path.join(datapath, sub), 'rb')

        data_sub = pkl.load(f)
        # print(data_sub.shape)        # (28, 32, 7500)

        # PSD features
        for video in range(0, n_vids):
            # (30, 5, 32)
            psd_data = extract_psd_feature(data_sub[video,:,:], 1, freq_bands)
            # (32, 30, 5)
            psd_data = np.transpose(psd_data, (2, 0, 1))
            subs_psd[video,:,:,:] = psd_data

        # save the PSD and DE features
        f = open(PSD_savepath + '/' + str(sub) + '.pkl', 'wb')
        pkl.dump(subs_psd, f)
        f.close()

        # DE features
        for i in range(len(freq_bands)):
            for video in range(0, n_vids):
                data_video = data_sub[video,:,:]
                low_freq = freq_bands[i][0]
                high_freq = freq_bands[i][1]
                # band pass filter for the specific data band
                # shape (32, 7500)
                data_video_filt = mne.filter.filter_data(data_video, freq, l_freq=low_freq, h_freq=high_freq)
                # shape (32, 30, 250)
                data_video_filt = data_video_filt.reshape(nchn, -1, freq)
                print('data filtered :', data_video_filt.shape)
                de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(data_video_filt, 2)))
                print(de_one.shape)
                # n_subs, video, channel,  second, frequency
                subs_de[video, : , :, i] = de_one

        g = open(DE_savepath + '/' + str(sub) + '.pkl', 'wb')
        pkl.dump(subs_de, g)
        g.close()















