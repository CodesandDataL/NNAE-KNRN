from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sys
from time import localtime,strftime
from collections import Counter

## This is AUROC
## Shows the trade-off between TPR & FPR when decision threshold changes.
def cal_auc(gt, anomaly_score):
    fpr, tpr, thresholds = metrics.roc_curve(gt, anomaly_score)
    auc = metrics.auc(fpr, tpr)
    return auc

def log_saving(info, pred_label, gt, anomaly_score=[], file_path='result.txt'):
    original = sys.stdout
    sys.stdout = open(file_path, 'a+')
    if anomaly_score == []:
        auc = cal_auc(gt, pred_label)
    else:
        auc = cal_auc(gt, anomaly_score)
    tn, fp, fn, tp = confusion_matrix(gt,pred_label).ravel()
    print("==========================================================================\n\n")
    print(strftime("%a, %d %b %Y %H:%M:%S", localtime()))
    print(info+'\n')
    print(classification_report(gt,pred_label,target_names=['Inlier','Outlier']))
    print("precision: " + str(tp/(tp+fp)) + "\n")
    print("recall: " + str(tp/(fn+tp)) + "\n")
    print("accuracy: " + str((tp+tn)/(tp+tn+fp+fn)) + "\n")
    print("AUC: " + str(auc)+"\n")
    print("tn, fp, fn, tp: " + str(tn)+ "," +str(fp)+ "," +str(fn)+ "," +str(tp) + "\n")
    print("\n==========================================================================")
    sys.stdout = original
    print("tn, fp, fn, tp: " + str(tn)+ "," +str(fp)+ "," +str(fn)+ "," +str(tp) + "\n")
    return auc

if __name__ == '__main__':
	# log_saving('AutoEncoder Model...', [0,1,0,0,0],[0,1,0,0,0],'results.txt')
    average_auc([123])
    average_auc(123)