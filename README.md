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
We recruited 21 participants to collect data under daily five scenarios. Data collection utilized a camera module to capture facial videos of participants’ both RGB and IR modalities, aphysiological ground-truth signals were recorded using a CMS50E pulse oximeter for PPG and SpO2, and an HKH-11C respiratory sensor to monitor breathing patterns. Video recordings were acquired at a resolution of 640×480 pixels and a frame rate of 30 frames per second (FPS). The PPG signals were recorded at a frequency of 20 Hz, while respiratory waves were captured at 50 Hz. The experiment setup is shown as follows.

![device](./rppg_tool_LADH/images/collection.png)

(Schematic illustration of the experimental setup of data collection while participants are brushing teeth.)

## ⚙️ Experiment Procedure
This study divided data collection into two groups: the first dataset was collected from 10 subjects performing five scenarios in a single day, while the second dataset was obtained from 11 subjects who conducted the same five scenarios daily over 10 consecutive days. During the seated resting condition (station 1), subjects wore an HKH-11C respiratory sensor on their abdomen and a CMS50E pulse oximeter on their left index finger while sitting upright facing the camera. They were instructed to remain motionless, maintain a fixed gaze at the camera, and undergo two minutes of physiological data recording. Subsequently, while maintaining the same equipment setup in a seated position, subjects performed toothbrushing and hair-combing actions for two minutes, which was labeled as station 2. Station 3 represented a standing resting state, where subjects stood in front of the camera with the same sensor configuration as station 1, and data were collected for two minutes. Station 4 repeated the actions of station 2 but in a standing posture, also lasting two minutes. Following the completion of the first four conditions, subjects engaged in physical exercise (e.g., squats, high knees, or breath-holding) to induce physiological changes. Post-exercise, while maintaining the same sensor setup as station 1, subjects underwent an additional two-minute recording period, designated as station 5.

![Experiment process](./rppg_tool_LADH/images/stations.png)

(A visual illustration of our daily data collection protocol.)
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
The toolbox supports SUMS dataset, Cite corresponding papers when using.

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

