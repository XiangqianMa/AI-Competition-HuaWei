from torch import optim
import torch
import tqdm
from config import get_classify_config
from solver import Solver
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import codecs, json
import time
import yaml

from utils.set_seed import seed_torch
from models.build_model import PrepareModel
from utils.meter import AverageMeter, accuracy
from datasets.datasets import GetDataloader
import pickle
import random


class TrainVal():
    def __init__(self, config, fold):
        self.config = config

        prepare_model = PrepareModel()
        self.model = prepare_model.create_model(
            model_type=config.model_type,
            classes_num=config.classes_num,
            pretrained_path=config.pretrained_path
        )
        self.optimizer = prepare_model.create_optimizer(config.model_type, self.model, config)
        self.exp_lr_scheduler = prepare_model.create_lr_scheduler(
            config.lr_scheduler,
            self.optimizer, 
            step_size=config.lr_step_size,
            epoch=config.epoch,
            )
        self.criterion = prepare_model.create_criterion(config, config.classes_num)
        
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.solver = Solver(self.model)
        self.fold = fold
        # log初始化
        self.writer, self.time_stamp = self.init_log()
        self.model_path = os.path.join(self.config.save_path, self.config.model_type, self.time_stamp)

    def train(self, train_loader, valid_loader):
        ''' 完成模型的训练，保存模型与日志
        Args:
            train_loader: 训练数据的DataLoader
            valid_loader: 验证数据的Dataloader
            fold: 当前跑的是第几折
        '''
        global_step = 0
        max_accuracy_valid = 0
        for epoch in range(self.config.epoch):
            self.model.train()
            
            epoch += 1
            epoch_loss = 0
            tbar = tqdm.tqdm(train_loader)
            for i, (images, labels) in enumerate(tbar):
                # 网络的前向传播与反向传播
                labels_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
                epoch_loss += loss.item()
                self.solver.backword(self.optimizer, loss)

                # 保存到tensorboard，每一步存储一个
                self.writer.add_scalar('train_loss', loss.item(), global_step+i)
                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (param_group['lr'])
                descript = "Fold: %d, Train Loss: %.7f, lr: %s" % (self.fold, loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)

            # 每一个epoch完毕之后，执行学习率衰减
            self.exp_lr_scheduler.step()
            global_step += len(train_loader)

            # Print the log info
            print('Finish Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.config.epoch, epoch_loss/len(tbar)))

            # 验证模型
            val_loss, val_accuracy = \
                self.validation(valid_loader, self.model, self.criterion, self.optimizer, True)

            if val_accuracy > max_accuracy_valid: 
                is_best = True
                self.max_accuracy_valid = accuracy
            else:
                is_best = False
            
            state = self.model.module.state_dict(),
            self.solver.save_checkpoint(os.path.join(self.model_path, '%s_classify_fold%d.pth' % (self.config.model_type, self.fold)), state, is_best)
            self.writer.add_scalar('valid_loss', val_loss, epoch)
            self.writer.add_scalar('valid_accuracy', val_accuracy, epoch)

    def validation(self, data_iterator, model, criterion, optimizer, use_cuda):
        tqdm_iterator = tqdm.tqdm(data_iterator)
        model.eval()
        with torch.no_grad(): 
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()

            for inputs, targets in tqdm_iterator:
                # measure data loading time
                data_time.update(time.time() - end)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                # compute output
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                info = '(Usage:{usage} | Data: {data:.3f}s | Batch: {bt:.3f}s |  Loss: {loss:.4f} | top1: {top1: .4f} | top5: ' \
                    '{top5: .4f}'.format(
                    usage='val',
                    data=data_time.val,
                    bt=batch_time.val,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )

                tqdm_iterator.set_description(info)

        return losses.avg, top1.avg

    def init_log(self):
        # 保存配置信息和初始化tensorboard
        TIMESTAMP = "log-{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
        log_dir = os.path.join(self.config.save_path, self.config.model_type, TIMESTAMP)
        writer = SummaryWriter(log_dir=log_dir)
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as config_yaml:
            print('Writing config to %s' % os.path.join(log_dir, 'config.yaml'))
            yaml.dump(self.config, config_yaml)
        
        return writer, TIMESTAMP


if __name__ == "__main__":
    data_root = 'data/huawei_data/train_data'
    folds_split = 1
    test_size = 0.2
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    config = get_classify_config()
    get_dataloader = GetDataloader(data_root, folds_split=1, test_size=test_size)
    train_dataloaders, val_dataloaders = get_dataloader.get_dataloader(config.batch_size, config.image_size,  mean, std, transforms=None)
    
    for fold_index, [train_loader, valid_loader] in enumerate(zip(train_dataloaders, val_dataloaders)):
        train_val = TrainVal(config, fold_index)
        train_val.train(train_loader, valid_loader)