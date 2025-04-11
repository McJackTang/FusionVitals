"""PhysNet Trainer."""
import os
from collections import OrderedDict
import sys
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import pdb
import csv

class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.model_name = config.MODEL.NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.task = config.TASK
        self.dataset_type = config.DATASET_TYPE # 接收数据集类型

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
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
        lrs = []  # 用于记录学习率的列表
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            running_loss_bvp = 0.0
            running_loss_spo2 = 0.0
            train_loss = []
            train_loss_bvp = []
            train_loss_spo2 = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=120)
            
            # first_batches = iter(tbar)

            # # Print the first few batches to check their structure and dimensions
            # try:
            #     for i in range(3):  # You can adjust the number based on your batch size and needs
            #         batch = next(first_batches)
            #         # print(f"Batch {i} Type: {type(batch)}")
            #         if isinstance(batch, (list, tuple)):
            #             for j, item in enumerate(batch):
            #                 # print(f"  Item {j} shape: {item.shape if hasattr(item, 'shape') else 'Not a tensor'}")
            #         else:
            #             pass
            #             # print(f"Batch {i} shape: {batch.shape if hasattr(batch, 'shape') else 'Not a tensor'}")
            # except StopIteration:
            #     print("No more batches to display.")
            # except Exception as e:
            #     print(f"Error during batch processing: {e}")
            
            for idx, batch in enumerate(tbar):
                
                # print(f"batch: {batch}")
                
                tbar.set_description("Train epoch %s" % epoch)
                loss_bvp = torch.tensor(0.0)  # Initialize to avoid UnboundLocalError
                loss_spo2 = torch.tensor(0.0)  # Initialize to avoid UnboundLocalError
                # self.dataset_type
                if self.dataset_type != "both":
                    if self.task == "bvp":
                        rPPG, _, x_visual, x_visual3232, x_visual1616 = self.model(
                            batch[0].to(torch.float32).to(self.device))
                        BVP_label = batch[1].to(torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        loss = self.loss_model(rPPG, BVP_label)
                        running_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        _, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(
                            batch[0].to(torch.float32).to(self.device))
                        spo2_label = batch[2].to(torch.float32).to(self.device).squeeze(-1)
                        loss = torch.nn.L1Loss()(spo2_pred, spo2_label) 
                        running_loss_spo2 += loss.item()
                    elif self.task == "both":
                        rPPG, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(
                            batch[0].to(torch.float32).to(self.device))
                        BVP_label = batch[2].to(torch.float32).to(self.device)
                        spo2_label = batch[3].to(torch.float32).to(self.device).squeeze(-1)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        # print(rPPG.shape, BVP_label.shape)
                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        loss_spo2 = torch.nn.L1Loss()(spo2_pred, spo2_label)
                        loss = loss_bvp + loss_spo2
                        running_loss_bvp += loss_bvp.item()
                        running_loss_spo2 += loss_spo2.item()
                    else:
                        raise ValueError(f"Unknown task: {self.task}")
                
                    
                    
                else:  # both face and finger
                    face_data = batch[0].to(torch.float32).to(self.device)
                    finger_data = batch[1].to(torch.float32).to(self.device)
                    combined_data = torch.cat((face_data, finger_data), dim=1)  # 在通道维度拼接
                    
                    # 在使用卷积层之前，将其移动到相同的设备 #kernel_size
                    conv = torch.nn.Conv3d(in_channels=6, out_channels=3, kernel_size=1).to(self.device)
                    combined_reduced = conv(combined_data)
                    print(f"combined_reduced: {combined_reduced.shape}")
                    if self.task == "bvp":
                        rPPG, _, x_visual, x_visual3232, x_visual1616 = self.model(combined_reduced)
                        BVP_label = batch[2].to(torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        loss = self.loss_model(rPPG, BVP_label)
                        running_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        _, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(combined_reduced)
                        spo2_label = batch[3].to(torch.float32).to(self.device).squeeze(-1)
                        loss = torch.nn.L1Loss()(spo2_pred, spo2_label)
                        running_loss_spo2 += loss.item()
                    elif self.task == "both":
                        rPPG, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(combined_reduced)
                        BVP_label = batch[2].to(torch.float32).to(self.device)
                        spo2_label = batch[3].to(torch.float32).to(self.device).squeeze(-1)
                        rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / (torch.std(BVP_label) + 1e-8)
                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        loss_spo2 = torch.nn.L1Loss()(spo2_pred, spo2_label)
                        loss = loss_bvp + loss_spo2
                        running_loss_bvp += loss_bvp.item()
                        running_loss_spo2 += loss_spo2.item()
                    else:
                        raise ValueError(f"Unknown task: {self.task}")
                loss.backward()
                running_loss += loss.item()
                train_loss.append(loss.item())
                if self.task in ["bvp", "both"]:
                    train_loss_bvp.append(running_loss_bvp)
                if self.task in ["spo2", "both"]:
                    train_loss_spo2.append(running_loss_spo2)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                lrs.append(self.scheduler.get_last_lr()[0])  # 记录当前学习率
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item(), loss_bvp=running_loss_bvp, loss_spo2=running_loss_spo2)

            mean_training_losses.append(np.mean(train_loss))
            if self.task in ["bvp", "both"]:
                mean_training_losses.append(np.mean(train_loss_bvp))
            if self.task in ["spo2", "both"]:
                mean_training_losses.append(np.mean(train_loss_spo2))
                
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

    def valid(self, data_loader):
        """Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        valid_loss_bvp = 0.0
        valid_loss_spo2 = 0.0
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                if self.task == "bvp":
                    BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                    rPPG, _, x_visual, x_visual3232, x_visual1616 = self.model(
                        valid_batch[0].to(torch.float32).to(self.device))
                    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                    BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
                    loss = self.loss_model(rPPG, BVP_label)
                    valid_loss_bvp += loss.item()
                elif self.task == "spo2":
                    spo2_label = valid_batch[2].to(torch.float32).to(self.device).squeeze(-1)
                    _, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(
                        valid_batch[0].to(torch.float32).to(self.device))
                    loss = torch.nn.L1Loss()(spo2_pred, spo2_label)
                    valid_loss_spo2 += loss.item()
                elif self.task == "both":
                    BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                    spo2_label = valid_batch[2].to(torch.float32).to(self.device).squeeze(-1)
                    rPPG, spo2_pred, x_visual, x_visual3232, x_visual1616 = self.model(
                        valid_batch[0].to(torch.float32).to(self.device))
                    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                    BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
                    loss_bvp = self.loss_model(rPPG, BVP_label)
                    loss_spo2 = torch.nn.L1Loss()(spo2_pred, spo2_label)
                    loss = loss_bvp + loss_spo2
                    valid_loss_bvp += loss_bvp.item()
                    valid_loss_spo2 += loss_spo2.item()
                else:
                    raise ValueError(f"Unknown task: {self.task}")

                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item(), loss_bvp=valid_loss_bvp, loss_spo2=valid_loss_spo2)

            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test222(self, data_loader):
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
        if self.task == "bvp":
            calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
        elif self.task == "spo2":
            calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
        elif self.task == "both":
            calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
            calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")

        if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs 
            self.save_test_outputs(rppg_predictions, rppg_labels, self.config)
            self.save_test_outputs(spo2_predictions, spo2_labels, self.config)

    def test(self, data_loader):
        """Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print("\n===Testing===")
        rppg_predictions = dict()
        spo2_predictions = dict()
        rppg_labels = dict()
        spo2_labels = dict()
        
        # 定义列名
        header = [
            'V_TYPE', 'TASK', 'Epoch Number', 'HR_MAE', 'HR_MAE_STD', 'HR_RMSE', 'HR_RMSE_STD',
            'HR_MAPE', 'HR_MAPE_STD', 'HR_Pearson', 'HR_Pearson_STD', 'HR_SNR','HR_SNR_STD',
            'SPO2_MAE', 'SPO2_MAE_STD', 'SPO2_RMSE', 'SPO2_RMSE_STD', 'SPO2_MAPE',
            'SPO2_MAPE_STD', 'SPO2_Pearson', 'SPO2_Pearson_STD', 'SPO2_SNR','SPO2_SNR_STD',
            'Model'
        ] 
        
        

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
                
                if self.dataset_type == "both":
                    # For multi-modal input, combine face and finger data
                    face_data = test_batch[0].to(self.config.DEVICE)
                    finger_data = test_batch[1].to(self.config.DEVICE)
                    combined_data = torch.cat((face_data, finger_data), dim=1)
                    
                    rppg_label = test_batch[2].to(self.config.DEVICE)
                    spo2_label = test_batch[3].to(self.config.DEVICE)
                    
                    # Apply 1x1 conv to reduce channel dimension
                    conv = torch.nn.Conv3d(in_channels=6, out_channels=3, kernel_size=1).to(self.config.DEVICE)
                    combined_reduced = conv(combined_data)
                    
                    pred_ppg_test, pred_spo2_test, _, _, _ = self.model(combined_reduced)
                else:
                    data, rppg_label, spo2_label = test_batch[0].to(
                        self.config.DEVICE), test_batch[1].to(self.config.DEVICE), test_batch[2].to(self.config.DEVICE)
                    pred_ppg_test, pred_spo2_test, _, _, _ = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    rppg_label = rppg_label.cpu()
                    spo2_label = spo2_label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    pred_spo2_test = pred_spo2_test.cpu()
                
                if self.dataset_type == "both":
                    
                    for idx in range(batch_size):
                        subj_index = test_batch[4][idx]
                        sort_index = int(test_batch[5][idx])
                        if subj_index not in rppg_predictions:
                            rppg_predictions[subj_index] = dict()
                            rppg_labels[subj_index] = dict()
                        rppg_predictions[subj_index][sort_index] = pred_ppg_test[idx]
                        rppg_labels[subj_index][sort_index] = rppg_label[idx]
                            
                    for idx in range(batch_size):
                        subj_index = test_batch[4][idx]
                        sort_index = int(test_batch[5][idx])
                        if subj_index not in spo2_predictions:
                            spo2_predictions[subj_index] = dict()
                            spo2_labels[subj_index] = dict()
                        spo2_predictions[subj_index][sort_index] = pred_spo2_test[idx]
                        spo2_labels[subj_index][sort_index] = spo2_label[idx]
                else:  
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
        file_exists = os.path.isfile('/data2/lk/rppg-toolbox/result.csv')
        with open('/data2/lk/rppg-toolbox/result.csv', 'a', newline='') as csvfile:
            # 推理 怎么更鲁邦呢
            #epoch_num = int(self.config.INFERENCE.MODEL_PATH.split('/')[-1].split('.')[0].split('_')[-1][5:]) + 1
            epoch_num = self.max_epoch_num #train
            csv_writer = csv.writer(csvfile)

            # 如果文件不存在，则写入列名
            if not file_exists:
                csv_writer.writerow(header)
            if self.task == "bvp":
                result = calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
                metrics = result["metrics"]
                # 提取 HR 相关指标
                HR_MAE, HR_MAE_STD = metrics.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics.get("FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics.get("FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics.get("FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics.get("FFT_SNR", (None, None)) if "FFT_SNR" in metrics else (None, None)

                
                data_to_add = [
                    self.dataset_type, self.task, epoch_num, HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    "/", "/", "/", "/", "/", "/",
                    "/", "/", "/","/",
                    self.model_name
                ]
            elif self.task == "spo2":
                result = calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
                metrics = result["metrics"]
                # 提取 SPO2 相关指标
                SPO2_MAE, SPO2_MAE_STD = metrics.get("FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics.get("FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics.get("FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics.get("FFT_Pearson", (None, None))          
                data_to_add = [
                    self.dataset_type, self.task, epoch_num, "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/",
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/",
                    self.model_name
                ]       
            
            
            elif self.task == "both":
                result_rppg = calculate_metrics(rppg_predictions, rppg_labels, self.config, "rppg")
                result_spo2 = calculate_metrics(spo2_predictions, spo2_labels, self.config, "spo2")
                metrics_rppg = result_rppg["metrics"]
                HR_MAE, HR_MAE_STD = metrics_rppg.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics_rppg.get("FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics_rppg.get("FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics_rppg.get("FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics_rppg.get("FFT_SNR", (None, None))
                metrics_spo2 = result_spo2["metrics"]
                SPO2_MAE, SPO2_MAE_STD = metrics_spo2.get("FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics_spo2.get("FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics_spo2.get("FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics_spo2.get("FFT_Pearson", (None, None))
                data_to_add = [
                    self.dataset_type, self.task, epoch_num, HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/",
                    self.model_name
                ]                   
        
           # 写入数据行
            csv_writer.writerow(data_to_add)
       
        # 这里把指标都保存下
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

    def save_test_outputs(self, predictions, labels, config):
        if not os.path.exists(config.TEST.OUTPUT_SAVE_DIR):
            os.makedirs(config.TEST.OUTPUT_SAVE_DIR)
        output_file = os.path.join(config.TEST.OUTPUT_SAVE_DIR, f"{self.model_file_name}_test_outputs.npz")
        np.savez(output_file, predictions=predictions, labels=labels)
        print(f"Saved test outputs to: {output_file}")

