from losses import total_variation, detection_loss, segmentation_loss
import torch
from metrics import evaluate_detection, evaluate_segmentation



def train_model(model, num_epochs, train_det_loader, train_seg_loader,
                val_detection_loader, val_seg_loader, optimizer, writer,
                device=torch.device("cpu")):

    train_loss_blob = 0.0

    train_loss_seg = 0.0
    loss_blob = 0
    loss_seg = 0
    train_loss = 0.0
    num_batches_lim = len(train_seg_loader)
    model.to(device)
    model.train(True)
    for epoch in range(num_epochs):
        print('#### Epoch_{}'.format(epoch))
        for i_batch, (image, label) in enumerate(train_det_loader):
            print("Batch:{}".format(i_batch))

            if (i_batch < 5):
                image = image.to(device)
                label = label.to(device)
                blob_output, seg_output = model(image)

                crit_loss_blob = detection_loss(blob_output, label) + total_variation(blob_output)

                writer.add_scalar("Det_Loss/train", crit_loss_blob.item(), (epoch+1)*i_batch)


                print(crit_loss_blob.item())
                optimizer.zero_grad()

                crit_loss_blob.backward()

                optimizer.zero_grad()

                seg_input, seg_target = next(iter(train_seg_loader))
                seg_input = seg_input.to(device)
                # seg_target = seg_target.to(device)
                seg_target = torch.as_tensor(seg_target, dtype=torch.long, device=device)


                optimizer.zero_grad()


                blob_output, seg_output = model(seg_input)

                crit_loss_seg = segmentation_loss(seg_output, seg_target)

                print(crit_loss_seg.item())
                seg_variance_loss = total_variation(seg_output)
                print(seg_variance_loss)
                loss_seg = crit_loss_seg + seg_variance_loss
                print(loss_seg.item())
                writer.add_scalar("Seg_Loss/train", loss_seg.item(), (epoch+1)*i_batch)
                loss_seg.backward()

                train_loss_seg += loss_seg.item()

                train_loss += (loss_blob + loss_seg)


                optimizer.step()
            else:
                break
        print("Evaluation after epoch: ",epoch+1)
        f1_score, accuracy, recall, precision, fdr = evaluate_detection(model,val_detection_loader)
        print("Ball detection metrics: \n F1 score: %.3f, Accuracy: %.3f, Recall: %.3f, Precision: %.3f, FDR: %.3f"%(f1_score[0],accuracy[0],recall[0],precision[0],fdr[0]))
        writer.add_scalar("Ball_Detection/eval", accuracy[0], epoch)
        print("Goal Post detection metrics: \n F1 score: %.3f, Accuracy: %.3f, Recall: %.3f, Precision: %.3f, FDR: %.3f"%(f1_score[1],accuracy[1],recall[1],precision[1],fdr[1]))
        writer.add_scalar("Goal_post_Detection/eval", accuracy[1], epoch)
        print("Robot detection metrics: \n F1 score: %.3f, Accuracy: %.3f, Recall: %.3f, Precision: %.3f, FDR: %.3f"%(f1_score[2],accuracy[2],recall[2],precision[2],fdr[2]))
        writer.add_scalar("Robot_Detection/eval", accuracy[2], epoch)
        acc, iou = evaluate_segmentation(model,val_seg_loader)
        print("Background: Accuracy: %.3f, IoU: %.3f"%(acc[0],iou[0]))
        writer.add_scalar("Background_Segmentation/eval", acc[0], epoch)
        print("Field: Accuracy: %.3f, IoU: %.3f"%(acc[1],iou[1]))
        writer.add_scalar("Field_Segmentation/eval", acc[1], epoch)
        print("Line: Accuracy: %.3f, IoU: %.3f"%(acc[2],iou[2]))
        writer.add_scalar("Line_Segmentation/eval", acc[2], epoch)


    train_losses = [train_loss_blob, train_loss_seg]


    return model
