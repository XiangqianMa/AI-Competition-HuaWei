import torch
import tqdm
import datetime
import os
import pickle
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import json
import codecs

from config import get_classify_config
from solver import Solver
from utils.set_seed import seed_torch
from models.build_model import PrepareModel
from datasets.create_dataset import GetDataloader
from losses.get_loss import Loss
from utils.classification_metric import ClassificationMetric
from datasets.data_augmentation import DataAugmentation


class TrainVal:
    def __init__(self, config, fold):
        """
        Args:
            config: 配置参数
            fold: 当前为第几折
        """
        self.config = config
        self.fold = fold
        self.epoch = config.epoch
        self.num_classes = config.num_classes
        self.lr_scheduler = config.lr_scheduler
        print('USE LOSS: {}'.format(config.loss_name))

        # 加载模型
        prepare_model = PrepareModel()
        self.model = prepare_model.create_local_attention_model(
            model_type=config.model_type,
            classes_num=self.num_classes,
            last_stride=config.last_stride,
            droprate=config.droprate
        )

        # 得到最新产生的权重文件
        weight_path = os.path.join('checkpoints', config.model_type)
        lists = os.listdir(weight_path)  # 获得文件夹内所有文件
        lists.sort(key=lambda fn: os.path.getmtime(weight_path + '/' + fn))  # 排序
        weight_path = os.path.join(weight_path, lists[-1], 'model_best.pth')

        # 加载之前训练的权重
        pretrained_dict = torch.load(weight_path)['state_dict']
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # filter out unnecessary keys
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('Successfully Loaded from %s' % weight_path)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        # 加载优化器
        self.optimizer = prepare_model.create_optimizer(config.model_type, self.model, config)

        # 加载衰减策略
        self.exp_lr_scheduler = prepare_model.create_lr_scheduler(
            self.lr_scheduler,
            self.optimizer,
            step_size=config.lr_step_size,
            restart_step=config.restart_step,
            multi_step=config.multi_step
        )

        # 加载损失函数
        self.criterion = Loss(config.model_type, config.loss_name, self.num_classes)

        # 实例化实现各种子函数的 solver 类
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.solver = Solver(self.model, self.device)

        # log初始化
        self.writer, self.time_stamp = self.init_log()
        self.model_path = os.path.join(self.config.save_path, self.config.model_type, self.time_stamp)

        # 初始化分类度量准则类
        with open("online-service/model/label_id_name.json", 'r', encoding='utf-8') as json_file:
            self.class_names = list(json.load(json_file).values())
        self.classification_metric = ClassificationMetric(self.class_names, self.model_path)

        self.max_accuracy_valid = 0

    def train(self, train_loader, valid_loader):
        """ 完成模型的训练，保存模型与日志
        Args:
            train_loader: 训练数据的DataLoader
            valid_loader: 验证数据的Dataloader
        """
        global_step = 0
        max_accuracy_valid = 0
        for epoch in range(self.epoch):
            self.model.train()
            epoch += 1
            images_number, epoch_corrects = 0, 0

            tbar = tqdm.tqdm(train_loader)
            for i, (images, labels) in enumerate(tbar):
                # 网络的前向传播与反向传播
                labels_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(labels_predict, labels, self.criterion)
                self.solver.backword(self.optimizer, loss)

                images_number += images.size(0)
                epoch_corrects += self.model.module.get_classify_result(labels_predict, labels, self.device).sum()
                train_acc_iteration = self.model.module.get_classify_result(labels_predict, labels, self.device).mean()

                # 保存到tensorboard，每一步存储一个
                descript = self.criterion.record_loss_iteration(self.writer.add_scalar, global_step + i)
                self.writer.add_scalar('TrainAccIteration', train_acc_iteration, global_step + i)

                params_groups_lr = str()
                for group_ind, param_group in enumerate(self.optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % group_ind + ': %.12f, ' % param_group[
                        'lr']

                descript = '[Train Fold {}][epoch: {}/{}][Lr :{}][Acc: {:.4f}]'.format(self.fold, epoch, self.epoch,
                                                                                       params_groups_lr,
                                                                                       train_acc_iteration) + descript

                tbar.set_description(desc=descript)

            # 写到tensorboard中
            epoch_acc = epoch_corrects / images_number
            self.writer.add_scalar('TrainAccEpoch', epoch_acc, epoch)
            self.writer.add_scalar('Lr', self.optimizer.param_groups[0]['lr'], epoch)
            descript = self.criterion.record_loss_epoch(len(train_loader), self.writer.add_scalar, epoch)

            # Print the log info
            print('[Finish epoch: {}/{}][Average Acc: {:.4}]'.format(epoch, self.epoch, epoch_acc) + descript)

            # 验证模型
            val_accuracy, val_loss, is_best = self.validation(valid_loader)

            # 保存参数
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'max_score': max_accuracy_valid
            }
            self.solver.save_checkpoint(
                os.path.join(
                    self.model_path,
                    '%s_fold%d.pth' % (self.config.model_type, self.fold)
                ),
                state,
                is_best
            )

            # 写到tensorboard中
            self.writer.add_scalar('ValidLoss', val_loss, epoch)
            self.writer.add_scalar('ValidAccuracy', val_accuracy, epoch)

            # 每一个epoch完毕之后，执行学习率衰减
            if self.lr_scheduler == 'ReduceLR':
                self.exp_lr_scheduler.step(val_loss)
            else:
                self.exp_lr_scheduler.step()
            global_step += len(train_loader)

    def validation(self, valid_loader):
        tbar = tqdm.tqdm(valid_loader)
        self.model.eval()
        labels_predict_all, labels_all = np.empty(shape=(0,)), np.empty(shape=(0,))
        epoch_loss = 0
        with torch.no_grad():
            for i, (_, images, labels) in enumerate(tbar):
                # 网络的前向传播
                labels_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(labels_predict, labels, self.criterion)

                epoch_loss += loss

                # 先经过softmax函数，再经过argmax函数
                labels_predict = F.softmax(labels_predict, dim=1)
                labels_predict = torch.argmax(labels_predict, dim=1).detach().cpu().numpy()

                labels_predict_all = np.concatenate((labels_predict_all, labels_predict))
                labels_all = np.concatenate((labels_all, labels))

                descript = '[Valid][Loss: {:.4f}]'.format(loss)
                tbar.set_description(desc=descript)

            classify_report, my_confusion_matrix, acc_for_each_class, oa, average_accuracy, kappa = \
                self.classification_metric.get_metric(
                    labels_all,
                    labels_predict_all
                )

            if oa > self.max_accuracy_valid:
                is_best = True
                self.max_accuracy_valid = oa
                self.classification_metric.draw_cm_and_save_result(
                    classify_report,
                    my_confusion_matrix,
                    acc_for_each_class,
                    oa,
                    average_accuracy,
                    kappa
                )
            else:
                is_best = False

            print('OA:{}, AA:{}, Kappa:{}'.format(oa, average_accuracy, kappa))

            return oa, epoch_loss / len(tbar), is_best

    def init_log(self):
        # 保存配置信息和初始化tensorboard
        TIMESTAMP = "log-{0:%Y-%m-%dT%H-%M-%S}-localAtt".format(datetime.datetime.now())
        log_dir = os.path.join(self.config.save_path, self.config.model_type, TIMESTAMP)
        writer = SummaryWriter(log_dir=log_dir)
        with codecs.open(os.path.join(log_dir, 'config.json'), 'w', "utf-8") as json_file:
            json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

        seed = int(time.time())
        seed_torch(seed)
        with open(os.path.join(log_dir, 'seed.pkl'), 'wb') as f:
            pickle.dump({'seed': seed}, f, -1)

        return writer, TIMESTAMP


if __name__ == "__main__":
    config = get_classify_config()
    config.lr = 3e-4  # 重新设置学习率
    data_root = config.dataset_root
    folds_split = config.n_splits
    test_size = config.val_size
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if config.augmentation_flag:
        transforms = DataAugmentation(config.erase_prob, full_aug=True, gray_prob=config.gray_prob)
    else:
        transforms = None
    get_dataloader = GetDataloader(data_root, folds_split=folds_split, test_size=test_size)
    train_dataloaders, val_dataloaders = get_dataloader.get_dataloader(config.batch_size, config.image_size, mean, std,
                                                                       transforms=transforms)

    for fold_index, [train_loader, valid_loader] in enumerate(zip(train_dataloaders, val_dataloaders)):
        if fold_index in config.selected_fold:
            train_val = TrainVal(config, fold_index)
            train_val.train(train_loader, valid_loader)
