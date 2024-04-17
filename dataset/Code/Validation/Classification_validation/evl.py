import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

n_sub = 123

# DRAW THE BAR
def draw_ratio(path,name,cls):

    pd_Data = pd.read_csv(path)
    acc_list = np.array(pd_Data[['0']]) * 100
    print('shape of list:', acc_list.shape)
    acc_mean = np.mean(acc_list)
    se = np.std(acc_list)/np.sqrt(acc_list.shape[0])
    print(name + ' mean: %.1f' %acc_mean,' standard error:%.1f'%se)
    # plt.figure(figsize=(13, 10))
    plt.figure(figsize=(20, 10))
    x_haxis = [str(num) for num in range(1,n_sub+1+1)]
    y = np.vstack((acc_list,acc_mean)).flatten()
    y[:-1] = np.sort(y[:-1])
    x = np.arange(0,len(x_haxis))
    plt.ylim(0,100);
    # plt.xlabel('subjects',fontsize=20);
    plt.ylabel('Accuracy (%)',fontsize=30);
    plt.yticks(fontsize=25);plt.xticks(fontsize=25)

    plt.bar(x[:-1], y[:-1], facecolor='#D3D3D3', edgecolor='black', width=0.9,label='accuacy for each subject')

    plt.bar(x[-1]+5,y[-1],facecolor='#f5fffa', edgecolor='black', width=2.5, label='averaged accuracy')
    plt.errorbar(x[-1]+5, y[-1], yerr=se, fmt='o', ecolor='black', color='#000000', elinewidth=1, capsize=2, capthick=1)

    # chance level
    y_ = np.ones((y.shape[0]+0)) * 1/int(cls) * 100
    x_ = np.arange(0,y_.shape[0])
    plt.plot(x_,y_,linestyle='dashed',color='#fe0202')
    plt.legend(loc='upper left',fontsize=20,edgecolor='black',ncol=2)
    # legend = plt.legend(fontsize=20)

    plt.savefig('./' + name + '.png');
    plt.savefig('./' + name + '.eps',format='eps');
    plt.clf()


def confusionMat(subs_data, name):
    # (n_sub, n_vids, secs)
    import seaborn as sns
    n_vids = subs_data.shape[1];n_subs = subs_data.shape[0]
    if n_vids == 28:
        cls = 9
    else:
        cls = 2
    confusion = np.zeros((cls,cls))
    for sub in range(0,subs_data.shape[0]):
        data = subs_data[sub,:,:]
        emotions = []
        for c in range(0,cls):
            emotions.append([])
            if cls == 2:
                emotions[c] = data[12*c:12*c+12,:].reshape(1,-1)
            if (cls == 9)&(c<4):
                emotions[c] = data[3*c:3*c+3,:].reshape(1,-1)
            elif (cls == 9)&(c==4):
                emotions[c] = data[12:16,:].reshape(1,-1)
            elif  (cls == 9)&(c>4):
                emotions[c] = data[16+(c-5)*3:16+(c-5)*3+3,:].reshape(1,-1)

        for real in range(0,cls):
            for predict in range(0,cls):
                # real=2,predict=1，即真实是0，但被模型预测成real的
                confusion[real,predict] += (np.sum(emotions[real] == predict)/emotions[real].shape[1])/n_subs

    if cls == 9:
        index = ['Anger','Disgust','Fear','Sadness','Neutral','Amusement','Inspiration','Joy','Tenderness']
        font_size = 20
    elif cls == 2:
        index = ['Negative', 'Positive']
        font_size = 20
    confusion = pd.DataFrame(confusion * 100,index=index,columns=index)
    annot_kws = {"fontsize": font_size};
    plt.figure(figsize=(8, 7));
    # move the pic to the top
    plt.subplots_adjust(bottom=0.25,left=0.25)
    figure = sns.heatmap(confusion,annot=False,fmt=".1f", cmap='Blues',
                         xticklabels=index,annot_kws=annot_kws,square=True)
    plt.xlabel('Predicted label',fontsize=font_size);
    plt.ylabel('True label',fontsize=font_size)
    plt.xticks(rotation=35,fontsize=font_size)
    plt.yticks(rotation=0,fontsize=font_size)
    figure.get_figure().savefig('./' + name + '.png')
    figure.get_figure().savefig('./' + name + '.eps', format='eps')

# SVM
svm_cross_28_10_folds = 'subject_cross_vids_28_valid_10-folds.csv'
svm_intra_28_10_folds = 'subject_intra_vids_28_valid_10-folds.csv'
svm_cross_24_10_folds = 'subject_cross_vids_24_valid_10-folds.csv'
svm_intra_24_10_folds = 'subject_intra_vids_24_valid_10-folds.csv'

draw_ratio(svm_cross_28_10_folds,'svm_acc_cross_28',cls=9)
draw_ratio(svm_intra_28_10_folds,'svm_acc_intra_28',cls=9)
draw_ratio(svm_cross_24_10_folds,'svm_acc_cross_24',cls=2)
draw_ratio(svm_intra_24_10_folds,'svm_acc_intra_24',cls=2)


# CLISA

# score Bar
clisa_score_10_cls9 = 'Clisa_score_cls9_10_folds.csv'
clisa_score_loo_cls9 = 'Clisa_score_cls9_loo.csv'
clisa_score_10_cls2 = 'Clisa_score_cls2_10_folds.csv'
clisa_score_loo_cls2 = 'Clisa_score_cls2_loo.csv'

draw_ratio(clisa_score_10_cls9,'clisa_10folds_cls9', cls=9)
draw_ratio(clisa_score_loo_cls9,'clisa_loo_cls9', cls=9)
draw_ratio(clisa_score_10_cls2,'clisa_10folds_cls2', cls=2)
draw_ratio(clisa_score_loo_cls2,'clisa_loo_cls2', cls=2)

# confusion Mat
result_data = sio.loadmat('Clisa_results_cls9_10_folds.mat')['mlp']
confusionMat(result_data,'clisa_confusion_cls9_10folds')
result_data_ = sio.loadmat('Clisa_results_cls9_loo.mat')['mlp']
confusionMat(result_data_,'clisa_confusion_cls9_loo')

result_data = sio.loadmat('Clisa_results_cls2_10_folds.mat')['mlp']
confusionMat(result_data,'clisa_confusion_cls2_10folds')
result_data_ = sio.loadmat('Clisa_results_cls2_loo.mat')['mlp']
confusionMat(result_data_,'clisa_confusion_cls2_loo')




