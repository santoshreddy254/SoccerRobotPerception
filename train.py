




def train_model(model, num_epochs, train_det_loader, train_seg_loader, criterion_blob, criterion_seg, 
                optimizer, device):

    train_loss_blob = 0.0
    train_loss_seg = 0.0
    loss_blob = 0
    loss_seg = 0
    train_loss = 0.0
    num_batches_lim = len(train_seg_loader)
    
    model.train(True)
    for epoch in range(num_epochs):
        print('#### Epoch_{}'.format(epoch))
        for i_batch, (image, label) in enumerate(train_det_loader):
            print("Batch:{}".format(i_batch))
            if (i_batch < num_batches_lim):
                
                blob_output, seg_output = model(image)
            
                crit_loss_blob = criterion_blob(blob_output, label) + total_variation(blob_output)
                
                
                print(crit_loss_blob.item())
                optimizer.zero_grad()
                
                crit_loss_blob.backward()
                
                optimizer.zero_grad()

                seg_input, seg_target = next(iter(train_seg_loader))

                seg_target = torch.as_tensor(seg_target, dtype=torch.long, device=device)


                optimizer.zero_grad()
                
                
                lob_output, seg_output = model(image)

                crit_loss_seg = criterion_seg(seg_output, seg_target)
                print(crit_loss_seg.item())
                seg_variance_loss = total_variation(seg_output)
                print(seg_variance_loss)
                loss_seg = crit_loss_seg + seg_variance_loss
                print(loss_seg.item())
                loss_seg.backward()

                train_loss_seg += loss_seg.item()

                train_loss += (loss_blob + loss_seg)


                optimizer.step()
            else:
                break

    train_losses = [train_loss_blob, train_loss_seg]
    
    return train_losses, model


    