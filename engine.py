# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import torch
import preprocess_data
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils

from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from pathlib import Path

from torch import nn

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, use_softlabel=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    optimizer.zero_grad()

    # for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = batch[0]
        targets = batch[-1]

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output, outvect = model(samples, onlyfc=False)
            loss = criterion(output, targets)
  
        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            if use_softlabel:
                targets = torch.tensor([0 if i==2 or i==0 else 1 for i in targets]).to(device)
            class_acc = (output.max(-1)[-1] == targets).float().mean()*100
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion=torch.nn.CrossEntropyLoss(), use_amp=False, use_softlabel=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header): # 학습할 때는 data가 data_loader_val임 
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)
        if use_softlabel:
            target = torch.tensor([0 if i==2 or i==0 else 1 for i in target]).to(device)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)
        
        acc1, acc2 = accuracy(output, target, topk=(1, 2)) # top5는 의미 없어 2로 변경

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)

        for class_name, class_id in data_loader.dataset.class_to_idx.items():
            if use_softlabel:
                class_id = 0 if class_id==2 or class_id==0 else 1 
                class_name = 'negative' if class_name == 'amb_neg' else class_name
                class_name = 'positive' if class_name == 'amb_pos' else class_name

            mask = (target == class_id)
            target_class = torch.masked_select(target, mask)
            data_size = target_class.shape[0]
            if data_size > 0:
                mask = mask.unsqueeze(1).expand_as(output)
                output_class = torch.masked_select(output, mask)
                if use_softlabel:
                    output_class = output_class.view(-1, 2)
                else:
                    output_class = output_class.view(-1, len(data_loader.dataset.class_to_idx))
                acc1_class, acc2_class = accuracy(output_class, target_class, topk=(1, 2)) # top5는 의미 없어 2로 변경
                metric_logger.meters[f'acc1_{class_name}'].update(acc1_class.item(), n=data_size)
                metric_logger.meters[f'acc2_{class_name}'].update(acc2_class.item(), n=data_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def prediction(args, device):
    from datasets import PotholeDataset, get_split_data
    from sklearn.metrics import precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay
    import random

    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    totorch = transforms.ToTensor()

    # 모델 생성 train한 모델과 같은 모델을 생성해야 함.
    model = create_model(
        args.model, 
        pretrained=False, 
        num_classes=args.nb_classes, 
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
    )
    model.to(device)

    # Trained Model
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model,
        optimizer=None, loss_scaler=None, model_ema=None)
    model.eval()

    # Data laod     
    data_list = []
    result = []
    sets = get_split_data(data_root=Path(args.eval_data_path), 
                                  test_r=args.test_val_ratio[0], 
                                  val_r=args.test_val_ratio[1], 
                                  file_write=args.split_file_write,
                                  label_list = args.label_list) 
        
    data_list = sets['test'] if len(sets['test']) > 0 else sets['val']

    random.shuffle(data_list)  # Data list shuffle
    tonorm = transforms.Normalize(mean, std)  # Transform 생성
    for data in tqdm(data_list, desc='Image Cropping... '):
        crop_img = preprocess_data.crop_image(
            image_path = data[0] / data.image_path, 
            bbox = data.bbox, 
            padding = args.padding, 
            padding_size = args.padding_size, 
            use_shift = args.use_shift, 
            use_bbox = args.use_bbox, 
            imsave = args.imsave
        )

        # File 이름에 label이 있는지 확인
        spltnm = str(data[1]).split('_')
        target = int(spltnm[0][1]) if spltnm[0][0] == 't' else -1

        # label이 따로 있는 경우 아래 4가지 label로 지정
        if target == -1:
            if data[1] == 'amb_neg':
                target = 0 # amb_neg
            elif data[1] == 'amb_pos':
                target = 1 # amb_pos
            elif data[1] == 'negative':
                target = 2 # neg
            elif data[1] == 'positive':
                target = 3 # pos
            else:
                target =-1

        crop_img = cv2.resize(crop_img, (args.input_size, args.input_size))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(crop_img)
        input_tensor = totorch(pil_image).to(device)
        input_tensor = input_tensor.unsqueeze(dim=0)
        input_tensor = tonorm(input_tensor)
        
        # model output 
        output_tensor = model(input_tensor) 
        pred, conf = int(torch.argmax(output_tensor).detach().cpu().numpy()), float((torch.max(output_tensor)).detach().cpu().numpy())
        # softmax = nn.Softmax()
        # probs = softmax(output_tensor)

        # output = np.squeeze(output_tensor)
        probs = softmax(output_tensor) # softmax 통과

        probs_max = ((torch.max(probs)).detach().cpu().numpy())*100
        result.append((pred, probs_max, target, data[0] / data.image_path, data.label))
        
    ##################################### save result image & anno #####################################

    if args.pred_save:
        import os
        os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'images', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'annotations', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'images', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'annotations', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'negative' / 'images', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'negative' / 'annotations', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'positive' / 'images', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'positive' / 'annotations', exist_ok=True)
        if args.pred_save_with_conf:
            os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'negative' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'inference', exist_ok=True)

        amb_neg = [(x[-2], 'amb_neg', x[1], x[-1]) for x in result if x[0]==0]
        amb_pos = [(x[-2], 'amb_pos', x[1], x[-1]) for x in result if x[0]==1]
        neg = [(x[-2], 'negative', x[1], x[-1]) for x in result if x[0]==2]
        pos = [(x[-2], 'positive', x[1], x[-1]) for x in result if x[0]==3]

        for an in tqdm(amb_neg, desc='Ambiguous Negative images copying... '):
            img_path = str(an[0])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(an[0], Path(args.pred_save_path) /'amb_neg' / 'images')
            shutil.copy(annot_path, Path(args.pred_save_path) / 'amb_neg' / 'annotations')
            if args.pred_save_with_conf:
                img_plt = plt.imread(img_path)
                plt.imshow(img_plt)
                # plt.axis('off')
                plt.title(f"{an[1]},  {an[2]:.2f}%")
                plt.xlabel(f"target: {an[-1]}")
                fn = os.path.basename(an[0])
                plt.savefig(Path(args.pred_save_path) / 'amb_neg' / 'inference' / fn, dpi=200)


        for ap in tqdm(amb_pos, desc='Ambiguous Positive images copying... '):
            img_path = str(ap[0])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(ap[0], Path(args.pred_save_path) / 'amb_pos' / 'images')
            shutil.copy(annot_path, Path(args.pred_save_path) / 'amb_pos' / 'annotations')
            if args.pred_save_with_conf:
                img_plt = plt.imread(img_path)
                plt.imshow(img_plt)
                # plt.axis('off')
                plt.title(f"{ap[1]}, {ap[2]:.2f}%")
                plt.xlabel(f"target: {ap[-1]}")
                fn = os.path.basename(ap[0])
                plt.savefig(Path(args.pred_save_path) / 'amb_pos' / 'inference' / fn, dpi=200)

        for n in tqdm(neg, desc='Negative images copying... '):
            img_path = str(n[0])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(n[0], Path(args.pred_save_path) /'negative' / 'images')
            shutil.copy(annot_path, Path(args.pred_save_path) / 'negative' / 'annotations')
            if args.pred_save_with_conf:
                img_plt = plt.imread(img_path)
                plt.imshow(img_plt)
                # plt.axis('off')
                plt.title(f"{n[1]}, {n[2]:.2f}%")
                plt.xlabel(f"target: {n[-1]}")
                fn = os.path.basename(n[0])
                plt.savefig(Path(args.pred_save_path) / 'negative' / 'inference' / fn, dpi=200)

        for p in tqdm(pos, desc='Positive images copying... '):
            img_path = str(p[0])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(p[0], Path(args.pred_save_path) / 'positive' / 'images')
            shutil.copy(annot_path, Path(args.pred_save_path) / 'positive' / 'annotations')
            if args.pred_save_with_conf:
                img_plt = plt.imread(img_path)
                plt.imshow(img_plt)
                # plt.axis('off')
                plt.title(f"{p[1]}, {p[2]:.2f}%")
                plt.xlabel(f"target: {p[-1]}")
                fn = os.path.basename(p[0])
                plt.savefig(Path(args.pred_save_path) / 'positive' / 'inference' / fn, dpi=200)
    ##################################### save result image & anno #####################################

    ##################################### save evalutations #####################################
    if args.pred_eval:
        if np.sum(np.array(result)[...,2]) < 0:
            conf_TN = [x[1] for x in result if (x[0]==0)]
            conf_TP = [x[1] for x in result if (x[0]==1)]
            conf_FN = []
            conf_FP = []

            # index set    
            itn = [i for i in range(len(result)) if (result[i][0]==0)]
            itp = [i for i in range(len(result)) if (result[i][0]==1)]

            # histogram P-N 
            plt.hist((conf_TN, conf_TP), label=('Negative', 'Positive'),histtype='bar', bins=50)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            plt.legend(loc='upper left')
            plt.savefig(args.pred_eval_name+'hist_PN.png')
            plt.close()

        else:
            y_pred = [i[0] for i in result]
            y_target = [i[2] for i in result]
            pos_val = 3

            # 4class to 2class 변경
            if args.use_softlabel:
                y_pred = [0 if i==2 or i==0 else 1 for i in y_pred]
                y_target = [0 if i==2 or i==0 else 1 for i in y_target]
                pos_val = 1

            # 4class to 3class 변경
            # org_y_pred = [i[0] for i in result]
            # org_y_target = [i[2] for i in result]

            # y_pred = []
            # y_target = []

            # if args.use_softlabel:
            #     for i in org_y_pred:
            #         if i == 0 or i == 1:
            #             y_pred.append(0)
            #         elif i == 2:
            #             y_pred.append(1)
            #         else:
            #             y_pred.append(2)
            # 
            #     for i in org_y_target:
            #         if i == 0 or i == 1:
            #             y_target.append(0)
            #         elif i == 2:
            #             y_target.append(1)
            #         else:
            #             y_target.append(2)
            #     pos_val = 2

            # precision recall 계산
            precision = precision_score(y_target, y_pred, average= "macro")
            recall = recall_score(y_target, y_pred, average= "macro")
            cm = confusion_matrix(y_target, y_pred)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            plt.title('Precision: {0:.4f}, Recall: {1:.4f}'.format(precision, recall))
            plt.savefig('image/'+args.pred_eval_name+'cm.png')
            plt.close()
            print(cm)
            print('정밀도(Precision): {0:.4f}, 재현율(Recall): {1:.4f}'.format(precision, recall))
            print('F1-score : {0:.4f}'.format(2 * (precision * recall) / (precision + recall)))

            # collect data 
            conf_TN = [x[1] for p, t, x in zip(y_pred, y_target,result) if p==t and p!=pos_val] 
            conf_TP = [x[1] for p, t, x in zip(y_pred, y_target,result) if p==t and p==pos_val] 
            conf_FN = [x[1] for p, t, x in zip(y_pred, y_target,result) if p!=t and p!=pos_val] 
            conf_FP = [x[1] for p, t, x in zip(y_pred, y_target,result) if p!=t and p==pos_val] 
            
            # get index 
            itn = [i for i in range(len(result)) if (y_pred[i]==y_target[i] and y_pred[i]!=pos_val)]
            itp = [i for i in range(len(result)) if (y_pred[i]==y_target[i] and y_pred[i]==pos_val)]
            ifn = [i for i in range(len(result)) if (y_pred[i]!=y_target[i] and y_pred[i]!=pos_val)]
            ifp = [i for i in range(len(result)) if (y_pred[i]!=y_target[i] and y_pred[i]==pos_val)]
            
            # histogram T-F 
            plt.hist(((conf_TN+conf_TP),(conf_FN+conf_FP)), label=('True', 'False'),histtype='bar', bins=50)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            plt.legend(loc='best')
            plt.savefig('image/'+args.pred_eval_name+'hist_tf.png')
            plt.close()

            # histogram TN TP FN FP
            plt.hist((conf_TN,conf_TP,conf_FN,conf_FP), label=('TN', 'TP','FN','FP'),histtype='bar', bins=30)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            plt.legend(loc='best')
            plt.savefig('image/'+args.pred_eval_name+'hist_4.png')
            plt.close()
            
        # scatter graph
        if len(conf_TN):
            plt.scatter(conf_TN, itn, alpha=0.4, color='tab:blue', label='TN', s=20)
        if len(conf_TP):
            plt.scatter(conf_TP, itp, alpha=0.4, color='tab:orange', label='TP', s=20)
        if len(conf_FN):
            plt.scatter(conf_FN, ifn, alpha=0.4, color='tab:green', marker='x', label='FN', s=20)
        if len(conf_FP):
            plt.scatter(conf_FP, ifp, alpha=0.4, color='tab:red', marker='x', label='FT', s=20)
        plt.legend(loc='best')
        plt.xlabel('Confidence')
        plt.ylabel('Image Index')
        plt.savefig('image/'+args.pred_eval_name+'scater.png')
        plt.close()

        # histogram 
        plt.hist(((conf_TN+conf_TP+conf_FN+conf_FP)), histtype='bar', bins=50)
        plt.xlabel('Confidence')
        plt.ylabel('Conunt')
        plt.savefig('image/'+args.pred_eval_name+'hist.png')
        plt.close()

    ##################################### save evalutations #####################################

