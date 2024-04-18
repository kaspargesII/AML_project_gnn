import os
import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
from Behavioural_Data_validation import *
from mne.viz.topomap import _add_colorbar, plot_topomap, _hide_frame
import sys
import pickle
from Behavioural_Data_validation import *
sys.path.append("../../..")
from Code.Preprocessing.Preprocessing import *

n_chns = 30
freq = 250

def plot_the_topo_corr(y_axis,x_axis,psd_corr,filename,fontx,fonty):
    # y: emotions
    # x: bands
    # data: [bands, topics, chns]
    pos = np.load('./pos.npy')
    new_order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,17,16]
    pos = pos[new_order][:-2]
    y_axes = len(y_axis)
    x_axes = len(x_axis)
    fig, axes = plt.subplots(y_axes, x_axes+1, figsize=(x_axes * y_axes, 50))
    cmap = 'RdBu_r'
    rows, columns = axes.shape
    for column in range(columns):
        for row in range(rows):
            _hide_frame(axes[row][column])
    for i, y in enumerate(y_axis):# emotion
        for j, x in enumerate(x_axis):# band
            # im = _plot_topomap_multi_cbar(np.squeeze(emotion_psd[emotion][band][0,:]),
            #               pos,axes[i][j],vmax=1,vmin=-1, title=None,colorbar=False)
            if i == 0:
                axes[i][j].set_title(x, fontsize=fontx, y=1.3)
            if j == 0:
                axes[i][j].set_title(y, fontsize=fonty, x=-1.8, y=0.5, loc='left')
            vmin = np.min(psd_corr[:,:,:]); vmax = np.max(psd_corr[:,:,:])
            # print(vmin, vmax)
            # corr: [bands, topics, chns]
            im, _ = plot_topomap(psd_corr[j,i,:], pos, vmin=vmin, vmax=vmax, axes=axes[i][j], cmap=cmap, show=False)
    # Plot the colorbar
    a = plt.subplot(3, x_axes+2, 2*x_axes+4, frameon=False)
    _add_colorbar(a, im, cmap=cmap, side='left',size='16%')
    _hide_frame(a)
    # plt.show()
    plt.gcf().subplots_adjust(left=0.2)
    plt.savefig('./'+ filename + '.png');plt.clf();

# def _get_average_psd(energy_graph, freq_bands, sample_freq, stft_n=256):
#     start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
#     end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
#     ave_psd = np.mean(energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
#     return ave_psd

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

    # read the large npy data
    score_paths = '../../../Data'
    # Score
    files = os.listdir(score_paths)
    files.sort()
    score_cal = Score_cal(score_paths, files)
    # Cal the correlation between score of emotion video and its corresponding eeg power
    subject_scores = score_cal.subject_scores
    # The subject list
    subject_list = list(subject_scores.keys())
    # print(subject_scores)

    # EEG DATA
    data_path = '../../../Processed_data'
    data_paths = os.listdir(data_path)
    data_paths.sort()
    n_vids = 28; chn = 30; fs = 250; sec = 30;
    data = np.zeros((len(data_paths),n_vids,chn, fs * sec))
    for idx, path in enumerate(data_paths):
        f = open(os.path.join(data_path,path), 'rb')
        data_sub = pickle.load(f)
        data[idx,:,:,:] = data_sub[:,:-2,:]
    # # data shape :(sub, vid, chn, fs * sec)
    # # print(data.shape)
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
    videos = list(np.arange(1,29))
    # # calculate the psd feature for each subject
    corr_array = np.zeros((len(subject_list), len(bands), len(original_topic_themes), n_chns))
    # subject_psd = {str(video): np.zeros((5,30)) for video in range(1,29)}
    subject_psd = np.zeros((len(bands), len(videos), n_chns))
    for idx in range(0,len(subject_list)):
        # (28,12)
        subject_score = subject_scores[subject_list[idx]]
        subject_score = np.squeeze([subject_score[str(video)] for video in videos])
        subject_data = data[idx,:,:,:]
        for video in range(1,29):
            # (30,5,30)
            psd_features = extract_psd_feature(subject_data[video-1,:,:], 1, freq_bands)
            # subject_psd (5,28,30)
            subject_psd[:,video-1,:] = np.median(psd_features, axis=0)
        for j, band in enumerate(bands):
            for i,emotion in enumerate(original_topic_themes):
                for ch in range(0, n_chns):
                    # cal the corr inside a subject then cal the fdr pvalue
                    corr, p = stats.pearsonr(subject_psd[j,:,ch], subject_score[:, i])
                    # corr_array [subjects, bands, topics, chns]
                    corr_array[idx, j, i, ch] = corr

    # print(corr_array)
    corr_array_ave = np.mean(corr_array,axis=0)
    # corr : [bands, topics, chns]
    emotion_psd_corr = corr_array_ave[:,:8,:]
    dim_psd_corr = corr_array_ave[:,8:,:]
    # print(emotion_psd_corr.shape)

    # Plot the topo
    emotion_axis.remove('Neutral')
    plot_the_topo_corr(emotion_axis, bands, emotion_psd_corr, 'emotion_corr',70,70)
    plot_the_topo_corr(dimension_axis, bands, dim_psd_corr, 'dim_emotion_corr',50,50)

