# LADH
Non-Contact Health Monitoring During Daily Personal Care Routines

## 📖 Abstract
Here is LADH dataset collected by Qinghai University. The dataset collected 240 synchronized non-contact facial videos (including both RGB and IR modalities)across five scenarios(including sitting, sitting while brushing teeth and combing hair, standing, standing while brushing teeth and combing hair, and post-exercise), with 11 participants taking part continuously over 10 days. This dataset captures PPG, respiration rate (RR), and SpO2, and is designed to validate the accuracy and superiority of rPPG in daily personal care scenarios.

## 🔍 Experiment Setup
We recruited 21 participants to collect data under daily five scenarios. Data collection utilized a camera module to capture facial videos of participants’ both RGB and IR modalities, aphysiological ground-truth signals were recorded using a CMS50E pulse oximeter for PPG and SpO2, and an HKH-11C respiratory sensor to monitor breathing patterns. Video recordings were acquired at a resolution of 640×480 pixels and a frame rate of 30 frames per second (FPS). The PPG signals were recorded at a frequency of 20 Hz, while respiratory waves were captured at 50 Hz. The experiment setup is shown as follows.

![device](./rppg_tool_LADH/images/collection.png)

(Schematic illustration of the experimental setup of data collection while participants are brushing teeth.)

## ⚙️ Experiment Procedure
This study divided data collection into two groups: the first dataset was collected from 10 subjects performing five scenarios in a single day, while the second dataset was obtained from 11 subjects who conducted the same five scenarios daily over 10 consecutive days. During the seated resting condition (station 1), subjects wore an HKH-11C respiratory sensor on their abdomen and a CMS50E pulse oximeter on their left index finger while sitting upright facing the camera. They were instructed to remain motionless, maintain a fixed gaze at the camera, and undergo two minutes of physiological data recording. Subsequently, while maintaining the same equipment setup in a seated position, subjects performed toothbrushing and hair-combing actions for two minutes, which was labeled as station 2. Station 3 represented a standing resting state, where subjects stood in front of the camera with the same sensor configuration as station 1, and data were collected for two minutes. Station 4 repeated the actions of station 2 but in a standing posture, also lasting two minutes. Following the completion of the first four conditions, subjects engaged in physical exercise (e.g., squats, high knees, or breath-holding) to induce physiological changes. Post-exercise, while maintaining the same sensor setup as station 1, subjects underwent an additional two-minute recording period, designated as station 5.

![Experiment process](./rppg_tool_LADH/images/stations.png)

(A visual illustration of our daily data collection protocol.)
