"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm


class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            # running_loss = 0.0
            # train_loss = []
            running_loss_bvp = 0.0
            running_loss_spo2 = 0.0
            train_loss_bvp = []
            train_loss_spo2 = []


            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                # spo2 predc
                rPPG, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))
                                # 打印数据集的内容和形状
                if torch.isnan(batch[0]).any() or torch.isnan(batch[1]).any() or torch.isnan(batch[2]).any():
                    continue  # Skip this batch if any input contains nan


                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                spo2_label = batch[2].to(torch.float32).to(self.device)
                # 删除spo2_label的最后一个维度
                spo2_label = spo2_label.squeeze(-1)
                # print(f"spo2_label_shape: {spo2_label.shape} spo2_pred_shape: {spo2_pred.shape}")

                rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)

                # print(f"rPPG.shape: {rPPG.shape}\n BVP_label.shape: {BVP_label.shape}")
                loss_bvp = self.loss_model(rPPG, BVP_label)
                # loss_spo2 使用MAE函数
                loss_spo2 = torch.nn.L1Loss()(spo2_pred, spo2_label)
                loss = loss_bvp

                # print(f"loss_spo2: {loss_spo2}")

                loss = loss_bvp + loss_spo2  # 合并损失用于反向传播
                loss.backward()
                running_loss_bvp += loss_bvp.item()
                running_loss_spo2 += loss_spo2.item()
                if idx % 100 == 99:  # 每100个小批量打印一次损失
                    print(f'[{epoch}, {idx + 1:5d}] BVP loss: {running_loss_bvp / 100:.3f} SpO2 loss: {running_loss_spo2 / 100:.3f}')
                    running_loss_bvp = 0.0
                    running_loss_spo2 = 0.0
                train_loss_bvp.append(loss_bvp.item())
                train_loss_spo2.append(loss_spo2.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss_bvp=loss_bvp.item(), loss_spo2=loss_spo2.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append((np.mean(train_loss_bvp), np.mean(train_loss_spo2)))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)


    def train222(self, data_loader):
        """Training routine for model including rPPG and spo2"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # Forward pass
                rPPG, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))

                BVP_label = batch[1].to(torch.float32).to(self.device)
                spo2_label = batch[2].to(torch.float32).to(self.device)
                print(f"spo2_label: {spo2_label}")
                rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)

                # Calculate losses
                loss_rPPG = self.loss_model(rPPG, BVP_label)
                loss_spo2 = self.loss_model(spo2_pred, spo2_label)
                loss = loss_rPPG + loss_spo2  # Combine losses if needed, or handle differently

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item(), loss_rPPG=f"{loss_rPPG.item():.4f}", loss_spo2=f"{loss_spo2.item():.4f}")

                # Print every 100 mini-batches
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] total loss: {running_loss / 100:.3f}, avg loss_rPPG: {np.mean([l.item() for l in train_loss_rPPG]):.4f}, avg loss_spo2: {np.mean([l.item() for l in train_loss_spo2]):.4f}')
                    running_loss = 0.0

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())
                self.scheduler.step()

            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('Validation loss: ', valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("Best trained epoch: {}, Min validation loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)



    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test222(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        rppg_predictions = dict()
        spo2_predictions = dict()
        rppg_label = dict()
        spo2_label = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, rppg_label, spo2_label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE), test_batch[2].to(self.config.DEVICE) # 3, 4
                pred_ppg_test, pred_spo2_test, _, _, _ = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    rppg_label = rppg_label.cpu()
                    spo2_label = spo2_label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    pred_spo2_test = pred_spo2_test.cpu()
                print("test_batch.shape: ", len(test_batch))
                print("test_batch： ", test_batch)
                for idx in range(batch_size):
                    subj_index = test_batch[3][idx]
                    sort_index = int(test_batch[4][idx])
                    if subj_index not in rppg_predictions.keys():
                        rppg_predictions[subj_index] = dict()
                        rppg_label[subj_index] = dict()
                    rppg_predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    rppg_label[subj_index][sort_index] = rppg_label[idx]
                    
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in spo2_predictions.keys():
                        spo2_predictions[subj_index] = dict()
                        spo2_label[subj_index] = dict()
                    spo2_predictions[subj_index][sort_index] = pred_spo2_test[idx]
                    spo2_label[subj_index][sort_index] = spo2_label[idx]

        print('')
        calculate_metrics(rppg_predictions, rppg_label, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(rppg_predictions, rppg_label, self.config)


    def test(self, data_loader):
        """Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print("\n===Testing===")
        rppg_predictions = dict()
        spo2_predictions = dict()
        rppg_labels = dict()
        spo2_labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, rppg_label, spo2_label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE), test_batch[2].to(self.config.DEVICE) # 3, 4
                pred_ppg_test, pred_spo2_test, _, _, _ = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    rppg_label = rppg_label.cpu()
                    spo2_label = spo2_label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    pred_spo2_test = pred_spo2_test.cpu()
                    
                # print("test_batch.shape: ", len(test_batch))
                # print("test_batch： ", test_batch)
                
                for idx in range(batch_size):
                    subj_index = test_batch[3][idx]
                    sort_index = int(test_batch[4][idx])
                    if subj_index not in rppg_predictions:
                        rppg_predictions[subj_index] = dict()
                        rppg_labels[subj_index] = dict()
                    rppg_predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    rppg_labels[subj_index][sort_index] = rppg_label[idx]
                        
                for idx in range(batch_size):
                    subj_index = test_batch[3][idx]
                    sort_index = int(test_batch[4][idx])
                    if subj_index not in spo2_predictions:
                        spo2_predictions[subj_index] = dict()
                        spo2_labels[subj_index] = dict()
                    spo2_predictions[subj_index][sort_index] = pred_spo2_test[idx]
                    spo2_labels[subj_index][sort_index] = spo2_label[idx]

        print('')
        calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
        calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
        
        if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs 
            self.save_test_outputs(rppg_predictions, rppg_labels, self.config)
            self.save_test_outputs(spo2_predictions, spo2_labels, self.config)



    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)