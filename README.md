# LADH
Non-Contact Health Monitoring During Daily Personal Care Routines

## 📖 Abstract
Here is LADH dataset collected by Qinghai University. The dataset collected 240 synchronized non-contact facial videos (including both RGB and IR modalities)across five scenarios(including sitting, sitting while brushing teeth and combing hair, standing, standing while brushing teeth and combing hair, and post-exercise), with 11 participants taking part continuously over 10 days. This dataset captures PPG, respiration rate (RR), and SpO2, and is designed to validate the accuracy and superiority of rPPG in daily personal care scenarios.

TABLE(DATASET COMPARISON)
| Dataset | Videos | Camera-Position | Vitals | Long-term | Obscured |
|---------|----------|-----------------|--------|-----------|----------|
| PURE | 40 | Face | PPG/SpO₂ | ✗ | ✗ |
| UBFC-PPG | 42 | Face | PPG | ✗ | ✗ |
| MMPD | 660 | Face | PPG | ✗ | ✗ |
| SUMS | 80 | Face+Finger | PPG/SpO₂/RR | ✗ | ✗ |
| LADH | 240 | Face(RGB+IR) | PPG/SpO₂/RR | ✓ | ✓ |

## 🔍 Experiment Setup
We recruited 21 participants to collect data under daily five scenarios. Data collection utilized a camera module to capture facial videos of participants’ both RGB and IR modalities, physiological ground-truth signals were recorded using a CMS50E pulse oximeter for PPG and SpO2, and an HKH-11C respiratory sensor to monitor breathing patterns. Video recordings were acquired at a resolution of 640×480 pixels and a frame rate of 30 frames per second (FPS). The PPG signals were recorded at a frequency of 20 Hz, while respiratory waves were captured at 50 Hz. The experiment setup is shown as follows.

![device](./rppg_tool_LADH/images/collection.png)

(Schematic illustration of the experimental setup of data collection while participants are brushing teeth.)

## ⚙️ Experiment Procedure
This study divided data collection into two groups: the first dataset was collected from 10 subjects performing five scenarios in a single day, while the second dataset was obtained from 11 subjects who conducted the same five scenarios daily over 10 consecutive days. During the seated resting condition (station 1), subjects wore an HKH-11C respiratory sensor on their abdomen and a CMS50E pulse oximeter on their left index finger while sitting upright facing the camera. They were instructed to remain motionless, maintain a fixed gaze at the camera, and undergo two minutes of physiological data recording. Subsequently, while maintaining the same equipment setup in a seated position, subjects performed toothbrushing and hair-combing actions for two minutes, which was labeled as station 2. Station 3 represented a standing resting state, where subjects stood in front of the camera with the same sensor configuration as station 1, and data were collected for two minutes. Station 4 repeated the actions of station 2 but in a standing posture, also lasting two minutes. Following the completion of the first four conditions, subjects engaged in physical exercise (e.g., squats, high knees, or breath-holding) to induce physiological changes. Post-exercise, while maintaining the same sensor setup as station 1, subjects underwent an additional two-minute recording period, designated as station 5.

![Experiment process](./rppg_tool_LADH/images/stations.png)

(A visual illustration of our daily data collection protocol.)

## 🔍 Samples
|                           |state-1|state-2|state-3|state-4|state-5|
|:-------------------------:|:-----:|:------:|:----------:|:----:|:----:|
|face-rgb|![](./rppg_tool_LADH/images/v01.gif)|![](./rppg_tool_LADH/images/v02_B.gif) ![](./rppg_tool_LADH/images/v02_C.gif)|![](./rppg_tool_LADH/images/v03.gif)|![](./rppg_tool_LADH/images/v04_B.gif) ![](./rppg_tool_LADH/images/v04_C.gif) |![](./rppg_tool_LADH/images/v05.gif)|
|face-ir|![](./rppg_tool_LADH/images/v01_IR.gif)|![](./rppg_tool_LADH/images/v02_IR_B.gif) ![](./rppg_tool_LADH/images/v02_IR_C.gif)|![](./rppg_tool_LADH/images/v03_IR.gif)|![](./rppg_tool_LADH/images/v04_IR_B.gif) ![](./rppg_tool_LADH/images/v04_IR_C.gif) |![](./rppg_tool_LADH/images/v05_IR.gif)|

## :notebook: Neural Network Model
 We introduce a novel design in the FusionNet module by incorporating a modality-aware fusion mechanism. Specifically, a gated feature selection strategy is employed to adaptively modulate the contribution of each modality based on its global contextual representation, thereby effectively integrating information from both facial RGB and facial IR video streams. This design enables the model to dynamically emphasize the more informative modality under varying environmental conditions (e.g., changes in illumination), significantly enhancing the robustness and generalizability of the physiological signal estimation framework.
 ![Model](./rppg_tool_LADH/images/model.png)
 ( FusionPhys Model with Input frames of facial RGB and facial IR. PPG, RR and SpO2 estimation tasks are trained simultaneously with a combined loss.)

## :wrench: Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate rppg-toolbox` 

STEP3: `pip install -r requirements.txt` 

## ⚙️ Example of neural network training

Please use config files under `./configs/train_configs/LADH_PHYSNET_*`

## :notebook: Train on LADH, valid on LADH and test on LADH with FusionPhysNet 

STEP1: Download the LADH raw data by asking the [paper authors]().

STEP2: Modify `./configs/train_configs/LADH_PHYSNET_face_RGB_IR_both.yaml` 

STEP4: Run `python main.py --config_file ./configs/train_configs/LADH_PHYSNET_face_RGB_IR_both.yaml --r_lr 9e-3 --epochs 30 --path res_30_9e-3/face_RGB_IR_both` 

Note1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time. 

Note2: The example yaml setting will allow 70% of LADH(state 1, 2, 3, 4, 5) to train, 20% of LADH to valid and 20% of LADH to test. After training, it will use the best model(with the least validation loss) to test on LADH.（This is the day-wise partitioning experiment）

Note3: You can set the learning rate, epochs and save path

## :scroll: Yaml File Setting
The rPPG-Toolbox uses yaml file to control all parameters for training and evaluation. 
You can modify the existing yaml files to meet your own training and testing requirements.

Here are some explanation of parameters:
* #### TOOLBOX_MODE: 
  * `train_and_test`: train on the dataset and use the newly trained model to test.
  * `only_test`: you need to set INFERENCE-MODEL_PATH, and it will use pre-trained model initialized with the MODEL_PATH to test.
* #### TASK:
  * `bvp`: only bvp => hr.
  * `spo2`: only spo2.
  * `rr`: only rr.
  * `both`: bvp => hr and spo2 and rr.
* #### DATASET_TYPE:
  * `face`: only RGB video.
  * `face_IR`: only IR video.
  * `both`:  RGB and IR video.
* #### TRAIN / VALID / TEST: 
  * `DATA.INFO.STATE`: Filter the dataset by 5 states, like [1, 2, 3, 4, 5]
  * `DATA.INFO.TYPE`: 1 stands for face, 2 stands for face_IR. like [1, 2]
  * `DATA.DATASET_TYPE`: face,  face_IR or both, the type of dataset
  * `DATA_PATH`: The input path of raw data
  * `CACHED_PATH`: The output path to preprocessed data. This path also houses a directory of .csv files containing data paths to files loaded by the dataloader. This filelist (found in default at CACHED_PATH/DataFileLists). These can be viewed for users to understand which files are used in each data split (train/val/test)

  * `EXP_DATA_NAME` If it is "", the toolbox generates a EXP_DATA_NAME based on other defined parameters. Otherwise, it uses the user-defined EXP_DATA_NAME.  
  * `BEGIN" & "END`: The portion of the dataset used for training/validation/testing. For example, if the `DATASET` is PURE, `BEGIN` is 0.0 and `END` is 0.8 under the TRAIN, the first 80% PURE is used for training the network. If the `DATASET` is PURE, `BEGIN` is 0.8 and `END` is 1.0 under the VALID, the last 20% PURE is used as the validation set. It is worth noting that validation and training sets don't have overlapping subjects.  
  * `DATA_TYPE`: How to preprocess the video data
  * `LABEL_TYPE`: How to preprocess the label data
  * `DO_CHUNK`: Whether to split the raw data into smaller chunks
  * `CHUNK_LENGTH`: The length of each chunk (number of frames)
  * `CROP_FACE`: Whether to perform face detection
  * `DYNAMIC_DETECTION`: If False, face detection is only performed at the first frame and the detected box is used to crop the video for all of the subsequent frames. If True, face detection is performed at a specific frequency which is defined by `DYNAMIC_DETECTION_FREQUENCY`. 
  * `DYNAMIC_DETECTION_FREQUENCY`: The frequency of face detection (number of frames) if DYNAMIC_DETECTION is True
  * `LARGE_FACE_BOX`: Whether to enlarge the rectangle of the detected face region in case the detected box is not large enough for some special cases (e.g., motion videos)
  * `LARGE_BOX_COEF`: The coefficient of enlarging. See more details at `https://github.com/ubicomplab/rPPG-Toolbox/blob/main/dataset/data_loader/BaseLoader.py#L162-L165`. 

  
* #### MODEL : Set used model FusionPhysnet right now and their parameters.
* #### METRICS: Set used metrics. Example: ['MAE','RMSE','MAPE','Pearson']

## :open_file_folder: Dataset
The toolbox supports the LADH dataset. Cite corresponding papers when using.

* [LADH](https://github.com/McJackTang/FusionVitals)
    * In order to use this dataset in a deep model, you should organize the files as follows:
    
    -----------------
         data/LADH/
         |   |-- 12_05/
         |       |-- p_12_05_caip
         |           |-- v01
         |               |-- BVP.csv
         |               |-- HR.csv
         |               |-- RR.csv
         |               |-- SpO2.csv
         |               |-- frames_timestamp_IR.csv
         |               |-- frames_timestamp_RGB.csv
         |               |-- video_RGB_H264.avi
         |               |-- video_IR_H264.avi
         |           |-- v02
         |           |-- v03
         |           |-- v04
         |           |-- v05
         |       |-- p_12_05_huangxj
         |           |-- v01
         |               |-- ...
         |           |-- v02
         |           |-- v03
         |           |-- v04
         |           |-- v05
         |       |-- p_12_05_liutj
         |       |-- p_12_05_lujg
         |       |-- ...
         |   |-- 12_06/
         |       |-- p_12_06_caip
         |       |-- p_12_06_huangxj
         |       |-- p_12_06_liutj
         |       |-- p_12_06_lujg
         |       |...
         |   |-- ...
         |   
         |     
    -----------------

## :eyes: Experiment Results
 In the subject-wise partitioning experiment, multimodal fusion with joint training outperforms single-modality and single-task approaches, particularly for HR and RR estimation. The dataset was partitioned such that data from 8 subjects were used for training, 3 subjects for validation, and an additional dataset from 10 individuals was reserved for testing. The results indicated significant improvements in the MAE for HR, which decreased from 9.02 to 7.12, reflecting a 21.06% error reduction, and for RR, which decreased from 2.25 to 1.43, reflecting a 36.44% error reduction. This suggests that multimodal fusion and joint training are more effective for periodic tasks likeHRandRR,whileSpO2doesnot exhibit clear periodic fluctuations and is inferred through indirect signals.

TABLE 1 ：RESULTS OF HR-SpO₂-RR MULTI-TASK TRAINING BY SUBJECT
<table> 
    <tr>               
      <th>Modality</th>
      <th colspan="2">HR TASK</th>
      <th colspan="2">SpO2 TASK</th>
      <th colspan="2">RR TASK</th>
    </tr>
    <tr>      
      <th> </th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      </tr>
     <tr>      
      <th>Both(Single Task)</th>
      <td>9.02</td>
      <td>10.99</td>
      <td>1.10</td>
      <td>1.19</td>
      <td>2.25</td>
      <td>10.16</td>
     </tr>
     <tr>      
      <th>RGB(Multi Task)</th>
      <td>9.34</td>
      <td>12.08</td>
      <td>1.29</td>
      <td>1.39</td>
      <td>3.08</td>
      <td>13.78</td>
     </tr>
     <tr>      
      <th>IR(Multi Task)</th>
      <td>12.99</td>
      <td>15.73</td>
      <td>1.23</td>
      <td>1.33</td>
      <td>2.41</td>
      <td>11.20</td>
     </tr>
     <tr>      
      <th>Both(Multi Task)</th>
      <td>7.12</td>
      <td>8.93</td>
      <td>1.14</td>
      <td>1.23</td>
      <td>1.43</td>
      <td>6.53</td>
     </tr>
</table>

In the day-wise partitioning experiment, multimodal fusionwith joint training improvesHRestimation, and multitask learningbenefits SpO2 andRRestimation. In thisexperiment,datacollectedover10daysweresplit into 7 days for training, 2 days forvalidation, and 1 dayfor testing. The results showed that for HR estimation, multimodal fusion with joint training outperformed single-modality and single-task approaches, reducing MAE from 5.23 to 4.99 (a 4.59% error reduction). In IR-based joint training, errors for SpO2 and RR were reduced by 2.29% and 41.25%, respectively. This highlights the effectiveness of multimodal fusion for HR and multitask learning for SpO2 and RR. 

TABLE 2 ：RESULTS OF HR-SpO₂-RR MULTI-TASK TRAINING BY DAY
<table> 
    <tr>               
      <th>Modality</th>
      <th colspan="2">HR TASK</th>
      <th colspan="2">SpO2 TASK</th>
      <th colspan="2">RR TASK</th>
    </tr>
    <tr>      
      <th> </th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      </tr>
     <tr>      
      <th>Both(Single Task)</th>
      <td>5.23</td>
      <td>5.44</td>
      <td>1.31</td>
      <td>1.38</td>
      <td>2.57</td>
      <td>13.45</td>
     </tr>
     <tr>      
      <th>RGB(Multi Task)</th>
      <td>5.73</td>
      <td>5.77</td>
      <td>1.35</td>
      <td>1.43</td>
      <td>1.99</td>
      <td>9.12</td>
     </tr>
     <tr>      
      <th>IR(Multi Task)</th>
      <td>8.35</td>
      <td>8.98</td>
      <td>1.28</td>
      <td>1.36</td>
      <td>1.51</td>
      <td>6.74</td>
     </tr>
     <tr>      
      <th>Both(Multi Task)</th>
      <td>4.99</td>
      <td>5.21</td>
      <td>1.29</td>
      <td>1.37</td>
      <td>2.24</td>
      <td>11.38</td>
     </tr>
 
</table>

Comparison of the subject-wise and day-wise experiments illustrates how day-wise analysis can improve the adaptability of models to individual user data. While the subject-wise experiment shows strong performance for periodic tasks through multimodal fusion and joint training, the day-wise experiment emphasizes the ability of the model to adapt more closely to individual data. This could indicate that, in future personalized health monitoring systems, such as a health mirror, models can better accommodate daily variations and offer more tailored results to users, enhancing the accuracyof HR, RR, and SpO2 estimation on an individual level.
## :eyes: Comparative Experiment
The table1 shows the Mean Absolute Error (MAE) and Mean Absolute Percent Error (MAPE) performance of the LADH dataset under unsupervised algorithms.
<table> 
    <tr>               
      <th>Test Set</th>
      <th colspan="7">LADH</th>
    </tr>
    <tr>      
      <th>Method</th> 
      <th>ICA</th>
      <th>POS</th>
      <th>CHROM</th>
      <th>GREEN</th>
      <th>LGI</th>
      <th>PBV</th>
      <th>OMIT</th>
      </tr>
     <tr>      
      <th>MAE↓</th> 
      <td>22.09</td>
      <td>11.27</td>
      <td>12.34</td>
      <td>26.67</td>
      <td>19.54</td>
      <td>21.73</td>
      <td>19.52</td>
     </tr>
     <tr>      
      <th>MAPE↓</th> 
      <td>23.99</td>
      <td>12.17</td>
      <td>13.34</td>
      <td>29.21</td>
      <td>21.21</td>
      <td>23.71</td>
      <td>21.19</td>
     </tr>
</table>

Table 2 shows the cross-dataset experimental results of the LADH, SUMS, and PURE datasets on the PhysNet model.
<table> 
    <tr>  
      <th rowspan="2"></th> 
      <th>Train Set</th>
      <th colspan="2">LADH</th>
      <th colspan="2">SUMS</th>
      <th colspan="2">PURE</th>
    </tr>
    <tr> 
      <th>Test Set</th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
      <th>MAE↓</th>
      <th>MAPE↓</th>
    </tr>
    <tr>      
      <th rowspan="3">PhysNet</th> 
      <th>LADH</th>
      <td>8.15</td>
      <td>9.19</td>
      <td>16.93</td>
      <td>18.2</td>
      <td>17</td>
      <td>18.78</td>
     </tr>
     <tr>      
      <th>SUMS</th>
      <td>11.23</td>
      <td>15.45</td>
      <td>3.36</td>
      <td>3.84</td>
      <td>14.95</td>
      <td>17.11</td>
     </tr>
     <tr>      
      <th>PURE</th>
      <td>8.1</td>
      <td>8.83</td>
      <td>7.97</td>
      <td>8.87</td>
      <td>0.59</td>
      <td>0.77</td>
     </tr>
</table>

## 🗝️ Access and Usage

The dataset included 240 videos from 21 subjects. Dataset size is 133.22 GB.  
There are two ways for downloads： OneDrive and Baidu Netdisk. 

To access the dataset, you are supposed to download this [data release agreement](https://github.com/McJackTang/FusionVitals/blob/main/LADH_Release_Agreement.pdf).  
Please scan and dispatch the completed agreement via your institutional email to <tjk24@mails.tsinghua.edu.cn> and cc <yuntaowang@tsinghua.edu.cn>. The email should have the subject line 'LADH Access Request -  your institution.' In the email,  outline your institution's **website** and **publications** for seeking access to the LADH, including its intended application in your specific research project. The email should be sent by a **faculty** rather than a student.   

## 📄 Citation
Title: [Non-Contact Health Monitoring During Daily Personal Care Routines](https://www.arxiv.org/abs/2506.09718)  
Xulin Ma, Jiankai Tang, Zhang Jiang, Songqin Cheng, Yuanchun Shi, Dong LI, Xin Liu, Daniel McDuff, Xiaojing Liu, Yuntao Wang, "Non-Contact Health Monitoring During Daily Personal Care Routines", IEEE BSN, 2025  
```
@inproceedings{ma2025non,
  title={Non-Contact Health Monitoring During Daily Personal Care Routines},
  author={Ma*, Xulin and Tang*, Jiankai and Jiang, Zhang and Cheng, Songqin and Shi, Yuanchun and Li, Dong and Liu, Xin and McDuff, Daniel and Liu, Xiaojing and Wang, Yuntao},
  booktitle={IEEE BSN 2025},
  year={2025}
}

@inproceedings{tang2023mmpd,
  title={MMPD: Multi-Domain Mobile Video Physiology Dataset},
  author={Tang, Jiankai and Chen, Kequan and Wang, Yuntao and Shi, Yuanchun and Patel, Shwetak and McDuff, Daniel and Liu, Xin},
  booktitle={2023 45th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@inproceedings{liu2024rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Sengupta, Roni and Patel, Shwetak and Wang, Yuntao and McDuff, Daniel},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@inproceedings{liu2024summit,
  title={Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High Altitudes},
  author={Liu*, Ke and Tang*, Jiankai and Jiang, Zhang and Wang, Yuntao and Liu, Xiaojing and Li, Dong and Shi, Yuanchun},
  booktitle={2024 IEEE Smart World Congress (SWC)},
  pages={284--291},
  year={2024}
}


```
