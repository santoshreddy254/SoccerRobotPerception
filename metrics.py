import numpy as np
import cv2
import torch

def get_centroids(image):
    image = image*255
    ret,thresh = cv2.threshold(image,127,255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00']!=0.0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx,cy))
    return centroids

def get_detection_metrics(predictions,targets):
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)
    tn = np.zeros(3)
    targets = targets.numpy()
    predictions = predictions.detach().numpy()
    for sample in range(targets.shape[0]):
        target = targets[sample]
        prediction = predictions[sample]
        for i in range(3):
            gt_centroids = get_centroids(target[i])
            pred_centroids = get_centroids(prediction[i])
            if len(pred_centroids) == len(gt_centroids) == 0:
                tn[i]+=1
            fp[i]+=len(pred_centroids)
            for gt_point in gt_centroids:
                match_found = False
                for pred_point in pred_centroids:
                    if np.linalg.norm(np.subtract(gt_point,pred_point)) <= 1:
                        match_found = True
                        tp[i]+=1
                        fp[i]-=1
                        break
                if not match_found:
                    fn[i]+=1

    return tp,fp,fn,tn


def evaluate_detection(model,dataloader,device):
    model.eval()
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)
    tn = np.zeros(3)
    accuracy = [0,0,0]
    precision = [0,0,0]
    recall = [0,0,0]
    f1_score = [0,0,0]
    fdr = [0,0,0]
    count = 0
    for img, label in dataloader:
        prediction,_ = model(img.to(device))
        prediction = prediction.cpu()
        temp_tp,temp_fp,temp_fn,temp_tn = get_detection_metrics(prediction,label)
        tp+=temp_tp
        fp+=temp_fp
        fn+=temp_fn
        tn+=temp_tn
        count+=1
        # if count>5:
        #     break
    for i in range(3):
        recall[i] = tp[i]/(tp[i]+fn[i])
        precision[i] = tp[i]/(tp[i]+fp[i])
        accuracy[i] = (tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])
        f1_score[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])
        fdr[i] = 1-precision[i]
    return f1_score, accuracy, recall, precision, fdr

def evaluate_segmentation(model,dataloader,device):
    model.eval()
    gt_counts = [0,0,0]
    correct_counts = [0,0,0]
    pred_counts = [0,0,0]
    acc = [0,0,0]
    iou = [0,0,0]
    count = 0
    for img,label in dataloader:
        _, segmentation = model(img.to(device))
        segmentation = segmentation.cpu()
        _, pred_segmentation = torch.max(segmentation, dim=1)
        for i in range(3):
            gt_counts[i]+=(label == i).sum().item()
            correct_counts[i]+=((label == i) & (pred_segmentation == i)).sum().item()
            pred_counts[i]+=(pred_segmentation == i).sum().item()
        count+=1
        # if count>5:
        #     break
    for i in range(3):
        acc[i] = correct_counts[i]/gt_counts[i]
        iou[i] = correct_counts[i]/(gt_counts[i]+pred_counts[i]-correct_counts[i])
    return acc, iou
