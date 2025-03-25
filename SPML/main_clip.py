# -*- coding: utf-8 -*-
import os
import copy
import time
import json
import numpy as np
import torch
import datasets
import datetime
import models
import argparse
from tqdm import tqdm
from enum import Enum
from losses import compute_batch_loss
import datetime
from instrumentation import train_logger
import warnings
import torchvision.transforms as transforms
import utils
import logging
import codecs
# import clip
from models import clip

import pickle

from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from models import read_clip

warnings.filterwarnings("ignore")

def write_description_to_folder(file_name, config):
    with codecs.open(file_name, 'w') as desc_f:
        desc_f.write("- Training Parameters: \n")
        for key, value in config.__dict__.items():
            desc_f.write("  - {}: {}\n".format(key, value))

def init_log(args, record_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    log_path = os.path.join(record_path, f'{time}_recording_log.txt')
    fh = logging.FileHandler(log_path, mode='w') 
    fh.setLevel(logging.DEBUG)  
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def read_label(args):
    label_emd_path = os.path.join('data',args.dataset,'label_emb.pt')
    with open(label_emd_path, 'rb+') as f:
        label_emb = pickle.load(f)
    return label_emb


def run_train_phase(model, P, Z, logger, epoch, phase, parallel):
    
    '''
    Run one training phase.

    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training.
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase == 'train'
    
    clip_model, _ = clip.load(P['clip_path'], device=Z['device'], jit=False)

    model.train()
    if epoch == 1:
        P['txt_features_train'] = P['txt_features']
        P['txt_features_train'] = P['txt_features_train'].to(Z['device'])

    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))
    for batch in tqdm(Z['dataloaders'][phase], desc=desc):#, mininterval=1800):

        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)

        if not P['model'] == "clip_vision":
            desired_size = (336, 336)
        else:
            desired_size = (224, 224)

        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']])
        resized_batch.to(Z['device'], non_blocking=True)
        if P['obs']:
            batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
            batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass:
        Z['optimizer'].zero_grad()

        with torch.set_grad_enabled(True):
            if parallel: 
                if not P['model'] == "clip_vision":
                    batch['logits'], batch['logits_pl'], batch['similarity'] = model.module.f(batch['image'], resized_batch, P['txt_features_train'])
                else:
                    batch['logits'], batch['feat'] = model.module.f(batch['image'], resized_batch, P['txt_features_train'])
            else:
                if not P['model'] == "clip_vision":
                    batch['logits'], batch['logits_pl'], batch['similarity'] = model.f(batch['image'], resized_batch, P['txt_features_train'])
                else:
                    batch['logits'], batch['feat'] = model.f(batch['image'], resized_batch, P['txt_features_train'])
            
            batch['preds'] = torch.sigmoid(batch['logits'])

            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
             
            
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            if not P['model'] == "clip_vision":
                batch = compute_batch_loss(batch, P, Z)
            else:
                batch = compute_batch_loss(batch, P, Z)

        # backward pass:
        batch['loss_tensor'].backward()
        Z['optimizer'].step()
        # save current batch data:
        logger.update_phase_data(batch)


def run_eval_phase_val(model, P, Z, logger, epoch, phase, parallel):

    '''
    Run one evaluation phase.

    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training.
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''
    
    clip_model, _ = clip.load(P['clip_path'], device=Z['device'], jit=False)
    
    assert phase in ['val', 'test']
    model.eval()
    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))

    if epoch == 1:
        P['txt_features_val'] = P['txt_features']
        P['txt_features_val'] = P['txt_features_val'].to(Z['device'])

    for batch in tqdm(Z['dataloaders'][phase], desc=desc, mininterval=1800):
        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        # if not model_name == "CLIPVIT":

        if not P['model'] == "clip_vision":
            desired_size = (336, 336)
        else:
            desired_size = (224, 224)

        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']])
        resized_batch.to(Z['device'], non_blocking=True)
        
        if P['obs']:
            batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
            batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)

        
        # forward pass:
        with torch.set_grad_enabled(False):
            
            if parallel: 
                if not P['model'] == "clip_vision":
                    batch['logits'], batch['logits_pl'], batch['similarity'] = model.f(batch['image'], resized_batch, P['txt_features_val'])
                else:
                    batch['logits'], batch['feat']= model.f(batch['image'], resized_batch, P['txt_features_val'])
            else:
                model_name = model.f.vision_extractor.__class__.__name__
                if not P['model'] == "clip_vision":
                    batch['logits'], batch['logits_pl'], batch['similarity'] = model.f(batch['image'], resized_batch, P['txt_features_val'])
                else:
                    batch['logits'], batch['feat']= model.f(batch['image'], resized_batch, P['txt_features_val'])
            
            
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)

            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics

            if not P['model'] == "clip_vision":
                batch = compute_batch_loss(batch, P, Z)
            else:
                batch = compute_batch_loss(batch, P, Z)

        # save current batch data:
        logger.update_phase_data(batch)


def train(model, P, Z, parallel):
    '''
    Train the model.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    '''
    if parallel:
        best_weights_f = copy.deepcopy(model.module.f.state_dict())
    else:
        best_weights_f = copy.deepcopy(model.f.state_dict())
    logger = train_logger(P, logg) # initialize logger

    if_early_stop = False

    for epoch_idx in range(0, P['num_epochs']):
        print('start epoch [{}/{}] ...'.format(epoch_idx + 1, P['num_epochs']))
        logg.info("start epoch [%.4f}/{%.4f}] ...", epoch_idx + 1, P['num_epochs'])
        P['epoch'] = epoch_idx + 1
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()

            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, P, Z, logger, P['epoch'], phase, parallel)
                #if P['epoch'] >= P['warmup_epoch'] and P['loss'] == 'EM_APL':
                    #aysmmetric_pseudo_labeling(model, P, Z, logger, P['epoch'], phase)
                # save end-of-phase metrics:
                logger.compute_phase_metrics(phase, P['epoch'])
                # print epoch status:
                logger.report(t_init, time.time(), phase, P['epoch'])
                
            else:
                run_eval_phase_val(model, P, Z, logger, P['epoch'], phase, parallel)
                # save end-of-phase metrics:
                logger.compute_phase_metrics(phase, P['epoch'])
                # print epoch status:
                logger.report(t_init, time.time(), phase, P['epoch'])
            
            # # save end-of-phase metrics:
            # logger.compute_phase_metrics(phase, P['epoch'])

            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, P['epoch'], P['val_set_variant'])
            if new_best:
                print('*** new best weights ***')
                if parallel:
                    best_weights_f = copy.deepcopy(model.module.f.state_dict())
                else:
                    best_weights_f = copy.deepcopy(model.f.state_dict())
                # print('\nSaving best weights for f to {}/best_model_state.pt'.format(P['save_path']))
                # torch.save(best_weights_f, os.path.join(P['save_path'], '_best_model_state.pt'))
                
            '''
            elif (not new_best) and (phase == 'val'):
                print('*** early stop ***')
                if_early_stop = True
                break
            '''
        if if_early_stop:
            break

    print('')
    print('*** TRAINING COMPLETE ***')
    print('Best epoch: {}'.format(logger.best_epoch))
    print('Best epoch validation score: {:.2f}'.format(logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    print('Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))
    
    logg.info('*** TRAINING COMPLETE ***')
    logg.info('Best epoch: {%.4f}',logger.best_epoch)
    logg.info('Best epoch validation score: {%.4f}',logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant']))
    logg.info('Best epoch test score:       {%.4f}',logger.get_stop_metric('test', logger.best_epoch, 'clean'))
  
    return P, model, logger, best_weights_f


def initialize_training_run(P, feature_extractor, linear_classifier, parallel):

    '''
    Set up for model training.
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    
    np.random.seed(P['seed'])

    Z = {}

    # accelerator:
    #GPU=1
    #device = torch.device('cuda:'+str(GPU) if torch.cuda.is_available() else 'cpu')
    #text_features = np.load('VOC20text_feature.npy')
    
    Z['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P['txt_features'] = torch.from_numpy(P['txt_features'])

    # data:
    Z['datasets'] = datasets.get_data(P)

    # observed label matrix:
    label_matrix = Z['datasets']['train'].label_matrix
    num_examples = int(np.shape(label_matrix)[0])
    mtx = np.array(label_matrix).astype(np.int8)
    total_pos = np.sum(mtx == 1)
    total_neg = np.sum(mtx == 0)
    print('training samples: {} total'.format(num_examples))
    print('true positives: {} total, {:.2f} per example on average.'.format(total_pos, total_pos / num_examples))
    print('true negatives: {} total, {:.2f} per example on average.'.format(total_neg, total_neg / num_examples))
    
    logg.info('*** START TRANING ***')
    logg.info('training samples: {%.4f} total',num_examples)
    logg.info('true positives: {%.4f} total, {%.4f} per example on average.',total_pos, total_pos / num_examples)
    logg.info('true negatives: {%.4f} total, {%.4f} per example on average.',total_neg, total_neg / num_examples)
    
    if P['obs']:
        observed_label_matrix = Z['datasets']['train'].label_matrix_obs
        num_examples = int(np.shape(observed_label_matrix)[0])
        obs_mtx = np.array(observed_label_matrix).astype(np.int8)
        obs_total_pos = np.sum(obs_mtx == 1)
        obs_total_neg = np.sum(obs_mtx == -1)
        print('observed positives: {} total, {:.2f} per example on average.'.format(obs_total_pos, obs_total_pos / num_examples))
        print('observed negatives: {} total, {:.2f} per example on average.'.format(obs_total_neg, obs_total_neg / num_examples))
        
        logg.info('observed positives: {%.4f} total, {%.4f} per example on average.',obs_total_pos, obs_total_pos / num_examples)
        logg.info('observed negatives: {%.4f} total, {%.4f} per example on average.',obs_total_neg, obs_total_neg / num_examples)

    # save dataset-specific parameters:
    P['num_classes'] = Z['datasets']['train'].num_classes


    
    
    # dataloaders:
    Z['dataloaders'] = {}
    if parallel:
        for phase in ['train', 'val', 'test']:
            if phase == "train":
                sampler = torch.utils.data.DistributedSampler(
                    Z['datasets'][phase],
                    num_replicas = utils.get_world_size(),
                    rank = utils.get_rank(),
                    shuffle = True
                )
            else:
                sampler = None
            Z['dataloaders'][phase] = torch.utils.data.DataLoader(
                Z['datasets'][phase],
                batch_size = P['bsize'],
                shuffle = False,
                sampler = sampler,
                num_workers = P['num_workers'],
                drop_last = False  # FIXME
            )
    else:
        for phase in ['train', 'val', 'test']:
            Z['dataloaders'][phase] = torch.utils.data.DataLoader(
                Z['datasets'][phase],
                batch_size = P['bsize'],
                shuffle = phase == 'train',
                sampler = None,
                num_workers = P['num_workers'],
                drop_last = False  # FIXME
            )

    # pseudo-labeling data:
    P['unlabel_num'] = []

    if P['obs']:
        for i in range(observed_label_matrix.shape[1]):
            P['unlabel_num'].append(np.sum(observed_label_matrix[:, i] == 0))

    # model:
    model = models.MultilabelModel(P, Z, feature_extractor, linear_classifier)
    # model = models.MultilabelModel_baseline(P, Z, feature_extractor, linear_classifier)

    # optimization objects:
    f_params = [param for param in list(model.f.parameters()) if param.requires_grad]

    Z['optimizer'] = torch.optim.Adam(
        f_params,
        lr=P['lr']
    )

    return P, Z, model


def execute_training_run(P, feature_extractor, linear_classifier, parallel=False):

    '''
    Initialize, run the training process, and save the results.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    P, Z, model = initialize_training_run(P, feature_extractor, linear_classifier, parallel)
    model.to(Z['device'])
    if parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        # clip_model = torch.nn.parallel.DataParallel(clip_model)
        print("Dataparallel has been launched!")

    P, model, logger, best_weights_f = train(model, P, Z, parallel)
    
    

    final_logs = logger.get_logs()
    if parallel:
        model.module.f.load_state_dict(best_weights_f)
    else:
        model.f.load_state_dict(best_weights_f)

    # return model.module.f.feature_extractor if parallel else model.f.feature_extractor, model.f.linear_classifier, final_logs
    return model.module.f.feature_extractor if parallel else model.f.feature_extractor, model.module.f.linear_classifier if parallel else model.f.linear_classifier, final_logs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SPML_CLIP')
    # parser.add_argument('-g', '--gpu', default='2,3', choices=['0', '1', '2', '3'], type=str)
    parser.add_argument("--local-rank", type=int,   default=0)
    parser.add_argument('-d', '--dataset', default='pascal', choices=['pascal', 'coco'], type=str)
    parser.add_argument('-l', '--loss', default='EM_APL', choices=['pr','wan','bce', 'iun', 'an', 'EM', 'EM_APL', 'EM_PL', 'EM_PL_ASL'], type=str)
    parser.add_argument('-m', '--model', default='clip_vision', choices=['clip_vision_baseline', 'clip_vision','resnet50', 'convnext_xlarge_22k', 'convnext_xlarge_1k'], type=str)
    parser.add_argument('-t', '--temp', default=0.03, type=float)
    parser.add_argument('-th', '--threshold', default=0.3, type=float)

    parser.add_argument('-p', '--partial', default=0.0, type=float)
    parser.add_argument('-o', '--obs', default=True, type=bool)
    # parser.add_argument('-f', '--flag', default=0, type=int)

    parser.add_argument('-s', '--pytorch_seed', default=0, type=int)  # try 0, 1, 8
    parser.add_argument("--clip_path",              type=str,   default='./pretrain/ViT-B-16.pt')
    parser.add_argument("--topk",                   type=int,   default=16     )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    

    args = parser.parse_args()
    

    P = {}
    parallel = False
    if torch.cuda.device_count() > 1:
        utils.init_distributed_mode(args)
        parallel = True

    # Top-level parameters:
    # P['GPU'] = args.gpu
    P['dataset'] = args.dataset
    P['loss'] = args.loss
    P['val_set_variant'] = 'clean'  # clean, observed
    P['test_set_variant'] = 'clean' # clean, observed
    # System parameters:
    # os.environ["CUDA_VISIBLE_DEVICES"] = P['GPU']
    P['pytorch_seed'] = args.pytorch_seed
    torch.manual_seed(P['pytorch_seed'])
    torch.cuda.manual_seed(P['pytorch_seed'])

    P['clip_path'] = args.clip_path
    P['topk'] = args.topk

    P['obs'] = args.obs
    
    # Optimization parameters:
    if P['dataset'] == 'pascal':
        P['bsize'] = 1 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-5 
        P['warmup_epoch'] = 0
        P['alpha'] = 0.2
        P['beta_pos'] = 0.7  #0.7
        P['beta_neg'] = 0.0 
        P['unknown']  = 4.0  #P['alpha'] = 0.2
        P['positive'] = 2.0  #P['beta_neg'] = 0.0 
        P['negative'] = 4.0  #P['beta_pos'] = 0.2 
        P['txt_features'] = np.load('VOC20text_feature_labelonly.npy')
        P['partial'] = args.partial #[0.1, 0.2, 0.3, 0.4]
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

        P['threshold'] = args.threshold #[0.1, 0.15, 0.2, 0.25, 0.3]
        
    elif P['dataset'] == 'coco':
        P['bsize'] = 1 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-5 
        P['warmup_epoch'] = 0
        P['unknown']  = 4  #P['alpha'] = 0.2
        P['positive'] = 2  #P['beta_neg'] = 0.0 
        P['negative'] = 4  #P['beta_pos'] = 0.2 
        P['alpha'] = 0.1
        P['beta_pos'] = 0.7
        P['beta_neg'] = 0.0 
        P['partial'] = args.partial
        P['txt_features'] = np.load('CoCo80text_feature_labelonly.npy')

        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold ##[0.1, 0.15, 0.2, 0.25, 0.3]

    # Additional parameters:
    P['seed'] = 1200  # overall numpy seed
    P['use_pretrained'] = True  # True, False
    P['num_workers'] = 8

    P['stop_metric'] = 'map'  # metric used to select the best epoch

    # P['stop_metric'] = ['map', 'rec_at_1', 'prec_at_1', 'top_1', 'rec_at_3', 'prec_at_3', 'top_3', 'rec_at_5', 'prec_at_5', 'top_5']
    P['metric'] = {}
    P['metric']= dict({'map' : '0', 'rec_at_1' : '0', 'prec_at_1' : '0', 'top_1' : '0', 'rec_at_3' : '0', 'prec_at_3' : '0', 'top_3' : '0', 'rec_at_5' : '0', 'prec_at_5' : '0', 'top_5' : '0'})

    # Dataset parameters:
    P['split_seed'] = 1200  # seed for train/val splitting
    P['val_frac'] = 0.2  # fraction of train set to split off for val
    P['ss_seed'] = 999  # seed for subsampling
    P['ss_frac_train'] = 1.0  # fraction of training set to subsample
    P['ss_frac_val'] = 1.0  # fraction of val set to subsample

    # # Dependent parameters:
    # if P['loss'] == 'bce':
    #     P['train_set_variant'] = 'clean'
    # else:
    #     P['train_set_variant'] = 'observed'

    P['train_set_variant'] = 'observed'

    # training parameters:
    P['num_epochs'] = 10
    P['freeze_feature_extractor'] = False
    P['use_feats'] = False
    P['arch'] = args.model #{'clip_vision','resnet50', 'convnext_xlarge_22k', 'convnext_xlarge_1k','clip_vision_1k+12k'}
    #P['feature_extractor_arch'] = 'resnet50'

    # baseline ResNet50 
    # P['feat_dim'] = 2048 

    P['save_path'] = './results/' + P['dataset'] + '/' + P['arch']
    P['record_path'] = './results/' + P['dataset']

    record_name = datetime.datetime.now().strftime('%y-%m-%d-%H')
    args.record_path = os.path.join(P['record_path'],record_name)
    os.makedirs(args.record_path, exist_ok=True)
    logg = init_log(args, args.record_path)
    write_description_to_folder(os.path.join(args.record_path, "configs.txt"), args)
    logg.info("------------------------------------------------------------------")

    # run training process:
    P['model'] = args.model
    P['partial'] = args.partial
    print('[{} + {}+ {}+ {}] start exp ...'.format(P['dataset'],P['model'], P['loss'], P['partial']))
    print("P is: ", P)
    logg.info('[{%s} + {%s}+ {%s}+ {%.4f}] start exp ...',P['dataset'], P['model'], P['loss'], P['partial'])
    logg.info('SPML is: {%d}',P['obs'])
    logg.info('P is: ', P)
   
    (feature_extractor, linear_classifier, logs) = execute_training_run(P, parallel=parallel, feature_extractor=None, linear_classifier=None)
