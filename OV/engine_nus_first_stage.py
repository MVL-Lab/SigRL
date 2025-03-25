import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import util.lr_sched as lrs
from models.rank_loss import ranking_lossT
from util.misc import compute_F1, compute_AP

def train(model, clip_model, args, optimizer, dataloader, logger, label_emb, epoch, loss_func):

    logger.info("TRAINING MODE")

    mean_dist_loss = 0.
    mean_rank_loss = 0.
    mean_constrast_loss = 0.

    t = tqdm(dataloader)
    ds=args.data_path.split("/")[-1]
    for i, (train_inputs, train_labels) in enumerate(t):

        lrs.adjust_learning_rate(optimizer, i / len(dataloader) + epoch, args)

        # import pdb; pdb.set_trace() print(torch.nonzero(1+train_labels[2]))
        optimizer.zero_grad()
        
        ### remove empty label images while training ###
        if ds == 'NUS_WIDE':
            temp_label = torch.clamp(train_labels,0,1)
            temp_seen_labels = temp_label.sum(1)
            temp_label = temp_label[temp_seen_labels>0]
            train_labels   = train_labels[temp_seen_labels>0]
            train_inputs   = train_inputs[temp_seen_labels>0]
        if train_inputs.shape[0] == 0:
            continue

        train_inputs = train_inputs.cuda()
        train_labels = train_labels.cuda()
        if ds == 'NUS_WIDE':
            label_embed = label_emb[:925]
        else:
            label_embed = label_emb
        logits, pred_feat, dist_feat = model(train_inputs, label_embed)
        
        rank_loss = ranking_lossT(logits, train_labels.float())
        
        with torch.no_grad():
             _, tea_dist_feat = clip_model.encode_image(train_inputs)
        # 2 VLP image encoder

        dist_loss = F.l1_loss(dist_feat, tea_dist_feat.float())
        loss = dist_loss + rank_loss #+ constrast_loss

        mean_dist_loss += dist_loss.item()
        mean_rank_loss += rank_loss.item()

        loss.requires_grad_()
        loss.backward()
        t.set_postfix(total_loss=float(loss.data), dist_loss=float(dist_loss), rank_loss=float(rank_loss))#, constrast_loss=float(constrast_loss))

        optimizer.step()

    mean_dist_loss /= len(dataloader)
    mean_rank_loss /= len(dataloader)
    mean_constrast_loss /= len(dataloader)

    learning_rate = optimizer.param_groups[-1]['lr']
    
    logger.info("------------------------------------------------------------------")
    logger.info("FINETUNING Epoch: {}/{} \tRankLoss: {:.6f}\tDistLoss: {:.6f}\tLearningRate {}".format(epoch, args.epochs, mean_rank_loss, mean_dist_loss, learning_rate))
    logger.info("------------------------------------------------------------------")

    torch.save(model.state_dict(), os.path.join(args.record_path, "model_epoch_{}.pth".format(epoch)))


########### TEST FUNC ###########
def test(model, args, dataloader, logger, label_emb, len_testdataset, writer, epoch=-1):

    logger.info("=======================EVALUATION MODE=======================")
    
    # NUS-WIDE
    prediction_81 = torch.empty(len_testdataset,81)
    prediction_1006 = torch.empty(len_testdataset,1006)
    lab_81 = torch.empty(len_testdataset,81)
    lab_1006 = torch.empty(len_testdataset,1006)
    
    test_batch_size = args.test_batch_size
    
    cnt = 0
    for features, labels_1006, labels_81, _ in tqdm(dataloader):
        strt = cnt
        endt = min(cnt + test_batch_size, len_testdataset)
        cnt += test_batch_size

        with torch.no_grad():

            logits_81, _, _ = model(features.cuda(), label_emb[925:])
            logits_1006, _, _= model(features.cuda(), label_emb)
        
    
        prediction_81[strt:endt,:] = logits_81
        prediction_1006[strt:endt,:] = logits_1006
        lab_81[strt:endt,:] = labels_81
        lab_1006[strt:endt,:] = labels_1006
    

    logger.info("completed calculating predictions over all images")

    logits_81_5 = prediction_81.clone()
    logits_81_3 = prediction_81.clone()

    ap_81 = compute_AP(prediction_81.cuda(), lab_81.cuda())
    F1_3_81,P_3_81,R_3_81 = compute_F1(logits_81_3.cuda(), lab_81.cuda(), 'overall', k_val=3)
    F1_5_81,P_5_81,R_5_81 = compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)

    logger.info('ZSL AP: %.4f',torch.mean(ap_81))
    logger.info('k=3: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_3_81),torch.mean(P_3_81),torch.mean(R_3_81))
    logger.info('k=5: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_5_81),torch.mean(P_5_81),torch.mean(R_5_81))

    logits_1006_5 = prediction_1006.clone()
    logits_1006_3 = prediction_1006.clone()

    ap_1006 = compute_AP(prediction_1006.cuda(), lab_1006.cuda())
    F1_3_1006,P_3_1006,R_3_1006 = compute_F1(logits_1006_3.cuda(), lab_1006.cuda(), 'overall', k_val=3)
    F1_5_1006,P_5_1006,R_5_1006 = compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)

    logger.info('GZSL AP:%.4f',torch.mean(ap_1006))
    logger.info('g_k=3: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_3_1006), torch.mean(P_3_1006), torch.mean(R_3_1006))
    logger.info('g_k=5: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_5_1006), torch.mean(P_5_1006), torch.mean(R_5_1006))
    
def eval(model, args, dataloader, label_emb, logger, len_testdataset):

    logger.info("EVAL MODE")

    txt_feat = label_emb

    # NUS-WIDE
    prediction_81 = torch.empty(len_testdataset,81)
    prediction_1006 = torch.empty(len_testdataset,1006)
    lab_81 = torch.empty(len_testdataset,81)
    lab_1006 = torch.empty(len_testdataset,1006)
    
    test_batch_size = args.test_batch_size
    
    cnt = 0
    for features, labels_1006, labels_81, _ in tqdm(dataloader):
        strt = cnt
        endt = min(cnt + test_batch_size, len_testdataset)
        cnt += test_batch_size

        with torch.no_grad():
        
            pred_feat, dist_feat = model.encode_img(features.cuda())
            score1 = torch.topk(pred_feat @ label_emb[925:].t(),k=model.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ label_emb[925:].t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits_81 = (score1 + score2) / 2
        
            score1 = torch.topk(pred_feat @ label_emb.t(),k=model.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ label_emb.t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits_1006 = (score1 + score2) / 2
    
        prediction_81[strt:endt,:] = logits_81
        prediction_1006[strt:endt,:] = logits_1006
        lab_81[strt:endt,:] = labels_81
        lab_1006[strt:endt,:] = labels_1006
   
    logits_81_5 = prediction_81.clone()
    logits_81_3 = prediction_81.clone()

    ap_81 = compute_AP(prediction_81.cuda(), lab_81.cuda())
    F1_3_81,P_3_81,R_3_81 = compute_F1(logits_81_3.cuda(), lab_81.cuda(), 'overall', k_val=3)
    F1_5_81,P_5_81,R_5_81 = compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)

    logger.info('ZSL AP: %.4f',torch.mean(ap_81))
    logger.info('k=3: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_3_81),torch.mean(P_3_81),torch.mean(R_3_81))
    logger.info('k=5: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_5_81),torch.mean(P_5_81),torch.mean(R_5_81))

    logits_1006_5 = prediction_1006.clone()
    logits_1006_3 = prediction_1006.clone()

    ap_1006 = compute_AP(prediction_1006.cuda(), lab_1006.cuda())
    F1_3_1006,P_3_1006,R_3_1006 = compute_F1(logits_1006_3.cuda(), lab_1006.cuda(), 'overall', k_val=3)
    F1_5_1006,P_5_1006,R_5_1006 = compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)

    logger.info('GZSL AP:%.4f',torch.mean(ap_1006))
    logger.info('g_k=3: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_3_1006), torch.mean(P_3_1006), torch.mean(R_3_1006))
    logger.info('g_k=5: F1:%.4f,P:%.4f,R:%.4f',torch.mean(F1_5_1006), torch.mean(P_5_1006), torch.mean(R_5_1006))
    