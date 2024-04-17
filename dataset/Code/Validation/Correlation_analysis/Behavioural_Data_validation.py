# !/usr/bin/env python3
# -*- coding:utf-8 -*-
# VERSION1: DATE:2022.8.30 FOR ANALYZE THE SCORE OF PER TRIAL

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
# import h5py
import hdf5storage
# import matlab.engine
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import fdrcorrection


# Experiment part
topic = 12; videos_num = 28
topic_themes = ["Joy","Tenderness","Inspiration","Amusement","Anger","Disgust","Fear","Sadness"]
original_topic_themes = ["Joy","Tenderness","Inspiration","Amusement","Anger","Disgust","Fear","Sadness","Arousal","Valence","Familiarity","Liking"]

# Analyze part
emotion_axis = ["Joy","Tenderness","Inspiration","Amusement","Anger","Disgust","Fear","Sadness","Neutral"]
dimension_axis = ["Arousal","Valence","Familiarity","Liking"]
emotions = ["Joy","Tenderness","Inspiration","Amusement","Anger","Disgust","Fear","Sadness","Neutral"]

# Use key-value to make sure the score of each trial don't be massive
class Score_cal():

    def __init__(self, path,files):
        '''
        2022-11
        Since the mat is '-v7.3' version, it cannot be opened by scipy
        What's more,I found that I failed to open some mat files with h5py lib
        Finally,I found hdf5storage works.

        2023-7
        I found that when I change the python interpreter into 3.10, sio.loadmat
        work again for loading the mat file. As I think, the difference mainly comes
        from the file you load. When the size of file is larger than 2GB,
        hdf5storage is needed.
        '''

        self.path = path
        # subject_scores {subject:{videos:scores}}
        self.subject_list = []
        # subject num
        self.ppl = 0
        self.subject_scores = {}
        self.video_scores = np.zeros((len(files), videos_num, topic))
        self.emotions_score = []
        self.emotion_topic_score = {}

        for idx,sub in enumerate(files):
            subject = sub
            file = os.path.join(path,sub,'After_remarks.mat')
            self.subject_list.append(subject)
            scoreData = sio.loadmat(file)['After_remark']
            '''
            The new rows consist of [score, trial, vid, Accuracy, Response Time]
            '''
            dictionary = {}

            for index in range(0, scoreData.shape[0]):
                # each columns (xx,1,1)
                video_num = int(scoreData[index][0][2])
                score = scoreData[index][0][0]
                self.video_scores[idx,video_num-1,:] = score
                dictionary.setdefault(str(video_num), score)
            # subject_scores: {'subject' :{'video':score}}
            self.subject_scores.setdefault(subject, dictionary)
            # ppl can not used as a local variable
            self.ppl = len(self.subject_list)

        for emotion in emotions:
            self.emotions_score.append(np.zeros((self.ppl, topic)))
        self.emotion_topic_score = dict(zip(emotions, self.emotions_score))

        # Here we provide 3 processions
        # One is subject_scores consists of videos scores
        # One is emotion_topic_score so that we can use t-test to verify the difference
        # Another is the mean value for plot the bar picture
        sub_ = 0;
        for subject in self.subject_list:
            videos_score = self.subject_scores[subject]
            # Sum up the video score of the same emotion
            for video, scores in videos_score.items():
                emotion = video2label(int(video))
                self.emotion_topic_score[emotion][sub_, :] = self.emotion_topic_score[emotion][sub_, :] + scores
            sub_ += 1

        # each emotion is a (subject,topic) matrix
        for emotion, subject_score in self.emotion_topic_score.items():
            if emotion == 'Neutral':
                self.emotion_topic_score[emotion] = subject_score / 4
            else:
                self.emotion_topic_score[emotion] = subject_score / 3

    # Plot the bar
    def bar_plot(self):

        f_result = anova(self.emotion_topic_score.copy(), emotion_axis)
        mean_se_result_dic = mean_se_cal(self.emotion_topic_score.copy(), emotion_axis)
        mean_result = [mean for [mean, se] in mean_se_result_dic.values()]
        labels = ['Non-target emotion', 'Target emotion',
                  'NOT significantly lower than the target emotion']
        emotion_axis_ = emotion_axis.copy()
        emotion_axis_.remove('Neutral')
        print('mean_result :', np.array(mean_result).shape)
        draw_ratio(mean_se_result_dic, f_result, emotion_axis, emotion_axis_, labels,'e')
        # plt.show()
        plt.savefig('./' + 'Emotion_Assessment.eps',format='eps');
        plt.savefig('./' + 'Emotion_Assessment.png');
        plt.clf()

        # labels = ['Emotion Dimension Bars']
        labels = ['Positive video clip','Negative video clip','Neutral video clip']
        f_result = []
        std_result = [std for [mean, std] in mean_se_result_dic.values()]
        mean_result = np.array(mean_result)[:,-len(dimension_axis):].T
        std_result = np.array(std_result)[:,-len(dimension_axis):].T
        mean_se_dim = [[mean_result[i,:], std_result[i,:]] for i in range(0,len(dimension_axis))]
        mean_se_dim_dic = dict(zip(dimension_axis,mean_se_dim))
        # plot the bar
        draw_ratio(mean_se_dim_dic, f_result, dimension_axis, emotion_axis, labels,'d')

        plt.savefig('./' + 'Dimension_Assessment.eps',format='eps');
        plt.savefig('./' + 'Dimension_Assessment.png');
        plt.clf()

    # cal the corr matrix
    def ratio_data(self):
        subs_videos_data = self.video_scores
        videos_data = np.mean(subs_videos_data, axis=0)
        videos_std = np.std(subs_videos_data, axis=0)
        pd.DataFrame(data=videos_data,columns=original_topic_themes).to_csv('./videos_mean.csv',float_format='%.2f')
        pd.DataFrame(data=videos_std, columns=original_topic_themes).to_csv('./videos_std.csv',float_format='%.2f')


def video2label(video):
    label = None
    if video in [1, 2, 3]:
        label = 'Anger'  # anger
    elif video in [4, 5, 6]:
        label = 'Disgust'  # disgust
    elif video in [7, 8, 9]:
        label = 'Fear'  # fear
    elif video in [10, 11, 12]:
        label = 'Sadness' # sadness
    elif video in [13, 14, 15, 16]:
        label = 'Neutral'  # Neutral
    elif video in [17, 18, 19]:
        label = 'Amusement'  # amusement
    elif video in [20, 21, 22]:
        label = 'Inspiration'  # inspiration
    elif video in [23, 24, 25]:
        label = 'Joy'  # joy
    elif video in [26, 27, 28]:
        label = 'Tenderness'  # tenderness
    else:
        pass
    return label


# Draw the pictures
def draw_ratio(mean_se_data,t_data,haxis,vaxis,labels,e_or_d):
    X = list(range(1,len(haxis)+1))
    Y_bool = list(np.in1d(original_topic_themes,vaxis))
    if 'Neutral' in vaxis:
        Y_bool.insert(8,True)
    plt.figure(figsize=(30, 25))
    t1 = None; t2 = None; t3 = None;
    # The main idea of drawing is draw step by step
    for index,emotion in enumerate(haxis):
        # y1 (Mean of each original_topic,)
        y1 = mean_se_data[emotion][0].squeeze()
        # Pick up the topic required
        y1 = y1[[i for i,val in enumerate(Y_bool) if val]]
        arange_length = 0.1 * len(vaxis)/2
        x1 = np.arange(X[index] - arange_length, X[index] + arange_length - 0.01, arange_length*2/(len(vaxis)))

        if e_or_d == 'e':
            # If plot the emotion
            t1 = plt.bar(x1, y1, facecolor='#FFFFFF', edgecolor='black', width=0.1)
            plt.xlabel('Types of video clips', fontsize=55, labelpad=50)
            # High light the target emotion
            if emotion in vaxis:
                target_index = topic_themes.index(emotion)
                t2 = plt.bar(x1[target_index], y1[target_index], facecolor='gray', edgecolor='black', width=0.1)
                # Darken the insignificant bar, if there exits
                if (t_data != []) & (len(t_data[emotion]) != 0):
                    for rows in t_data[emotion]:
                        t3 = plt.bar(x1[rows], y1[rows], facecolor='#D3D3D3', edgecolor='black', width=0.1)
            else:
                # t2 = None;
                t3 = None
        # emotion or dimension
        elif e_or_d == 'd':
            t1 = plt.bar(x1, y1, facecolor='#FFFFFF', edgecolor='black', width=0.1, hatch='\\')
            plt.xlabel('Rating items', fontsize=55)
            target_index = [topic_themes.index(emotion) for emotion in ["Anger","Disgust","Fear","Sadness"]]
            t2 = plt.bar(x1[target_index], y1[target_index], facecolor='#FFFFFF', edgecolor='black', width=0.1,hatch='/')
            neu_index = vaxis.index('Neutral')
            t3 = plt.bar(x1[8], y1[8], facecolor='#FFFFFF', edgecolor='black', width=0.1, hatch='-')

        # Draw the error bar
        # mean_se_data [0]=mean [1]=se
        y_se = mean_se_data[emotion][1][[i for i,val in enumerate(Y_bool) if val]]
        plt.errorbar(x1,y1,yerr=y_se,fmt='o',ecolor='black',color='black',elinewidth=3,capsize=4,capthick=2)

    plt.xticks(X, haxis,fontsize=55,rotation=40)
    plt.yticks(fontsize=55);
    plt.ylabel('Self-report ratings',fontsize=55, labelpad=50)
    if e_or_d == 'e':
        plt.legend(handles=[t1, t2, t3], labels=labels, loc='upper right', fontsize=40, edgecolor='black')
    elif e_or_d == 'd':
        plt.legend(handles=[t1, t2, t3], labels=labels, loc='lower right', fontsize=40,edgecolor='black')
    plt.subplots_adjust(bottom=0.25)
    # plt.show()


# Use anova to verify whether the second bar is statistically different
def anova(data,haxis):
    # data: {emotion:[subjects, original_topics}
    haxis_ = haxis.copy()
    # emotions_.remove('Neutral')
    # Do the rmanova first
    data_array = np.squeeze([data[h] for h in haxis_])
    if haxis_ == emotion_axis:
        haxis_.remove('Neutral')
    p_array = np.zeros((len(haxis_)))
    F_array = np.zeros((len(haxis_)))
    print('haxis_:', haxis_)
    for idx in range(0,len(haxis_)):
        data_anova = data_array[idx,:,:len(haxis_)].T
        # data_anova [haxis_, subjects]
        F,p = stats.f_oneway(*[d.ravel() for d in data_anova])
        F_array[idx] = round(F, 3)
        p_array[idx] = p
    p_array_fdr = fdrcorrection(p_array, alpha=0.05, method='indep')[1]
    # p_array_fdr = [round(p, 3) for p in p_array_fdr]
    print('F_Array:', F_array,'\n', 'p_Array:', p_array_fdr)
    # t-test
    # data key:emotion, value:(subject_nums,topic
    emotion_f_value = []
    for i in range(0,len(haxis_)):
        emotion_f_value.append([])
    emotion_t_dic = dict(zip(haxis, emotion_f_value))
    for index,emotion in enumerate(haxis_):
        # Skip the target step for neutral
        if emotion != 'Neutral':
            target_index = topic_themes.index(emotion)
        else:
            target_index = None
        # Every rows are done the multicomp to the target rows
        p_values = []; groups = []
        if (target_index != None) & (p_array_fdr[index] < 0.05):
            print('Target emotion:', haxis_[index])
            data_value = data[emotion][:, :len(haxis_)]
            # data_value = data[emotion][:,:len(haxis_)].reshape(-1,1)
            # for sub in range(0, data[emotion].shape[0]):
            #     groups.extend(haxis_)
            # tukeyHSD = pairwise_tukeyhsd(data_value,groups, alpha=0.05)
            p_matrix = []
            data_value_target = data_value[:,target_index]
            non_target = list(np.arange(0,len(haxis_)))
            non_target.remove(target_index)
            for non in non_target:
                _,p = stats.ttest_rel(data_value_target, data_value[:,non])
                p_matrix.append(p)
            # FDR, do it if you need.
            fdr = fdrcorrection(p_matrix, alpha=0.05, method='indep')[1]
            # results = np.array(tukeyHSD.summary().data)
            # t_pos = np.where((results[:,0]==haxis_[index])|(results[:,1]==haxis_[index]))[0]
            # correspond_result = fdr[t_pos-1]
            # corr_ = tukeyHSD.pvalues[t_pos-1]
            correspond_result = fdr
            print(correspond_result)
            t_not_sig = np.where(correspond_result > 0.05)[0]
            print(emotion, "not significant emotion index: ", t_not_sig)
            if t_not_sig.size == 0:
                continue
            else:
                # Collect the no significant emotion
                no_sig_emotion = non_target[t_not_sig]
                emotion_t_dic[emotion].append([topic_themes.index(no_sig) for no_sig in no_sig_emotion])
        else:
            continue
    return emotion_t_dic

def mean_se_cal(data,haxis):
    # data key:emotion, value:(subject_nums,topic)
    emotion_mean_ste_value = []
    for idx in range(0,len(haxis)):
        emotion_mean_ste_value.append([])
    emotion_mean_dic = dict(zip(haxis,emotion_mean_ste_value))
    for index,emotion in enumerate(haxis):
        emotion_array = np.transpose(data[emotion])
        # (topic,subject_nums)
        emotion_mean = np.mean(emotion_array,axis=1)
        emotion_se = np.std(emotion_array,axis=1)/np.sqrt(emotion_array.shape[1])
        # mean:(topic,)
        emotion_mean_dic[emotion].append(emotion_mean)
        emotion_mean_dic[emotion].append(emotion_se)
    return emotion_mean_dic


# Read through the directory
if __name__ == '__main__':
    # Loading the score file
    filePath = ['../../../Data']
    for idx, path in enumerate(filePath):
        # filesPath = glob(path + '/**', recursive=True)
        # condition = lambda t:(str(t).endswith('.mat'))
        # filtered_files = list(filter(condition,filesPath))
        files = os.listdir(path)
        files.sort()
        score_cal = Score_cal(path,files)
        score_cal.bar_plot()
        score_cal.ratio_data()
