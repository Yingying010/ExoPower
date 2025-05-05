# ExoPower: EMG-Controlled Upper Limb Exoskeleton

This repository contains all code related to the ExoPower system — a wearable, single-arm exoskeleton driven by surface electromyography (sEMG) signals for real-time intent recognition and motor-assisted elbow flexion. The system supports live classification, actuator control, and cloud-based data synchronization.

## 📁 Folder Structure

### `AnalyzeData/`
Scripts for EMG data preprocessing, feature extraction, model training, and evaluation.

- `A_RecordDataset.py`: Interactive script for collecting and labeling raw EMG data.
- `B_trainModel_RF.py`: Train a Random Forest classifier on extracted features.
- `B_trainModel_LSTM.py`: Train an LSTM-based model for temporal classification.
- `B_trainModel_xgbModel.py`: Train an XGBoost model for EMG classification.
- `C_CompareModelTraining.py`: Compare performance across the three classifiers.
- `C_simRealTime_RF.py`: Simulate real-time prediction using pre-recorded EMG data and RF model.
- `D_realTime_RF.py`: Real-time prediction using EMG sensor input and Random Forest model.
- `D_realTime_LSTM.py`, `D_realTime_xgbModel.py`: Real-time pipelines for other models.

### `model/`
Saved models trained with the scripts above.

- `rfModel.pkl`: Trained Random Forest model (used in real-time deployment).
- `lstm_emg_model.pt`: Trained LSTM model (PyTorch).
- `xgbModel.pkl`: Trained XGBoost model.

### `motor/`
Scripts for real-time actuator control using Dynamixel servos.

- `A_EMGMotor_RealControl.py`: Core script for EMG-driven motor actuation.
- `B_EMGMotor_simControl.py`: Simulated control logic without hardware.
- Other files: Additional PID-based experiments and motor control modes.

### `upload/`
Scripts for MQTT-based communication and cloud integration with AWS IoT Core.

- `mqttUpload.py`: Reads EMG sensor data and publishes via MQTT to another device.
- `aws_subscribe.py`: Subscribes to MQTT topic and forwards data to AWS.
- `cloud.py`, `read.py`: Auxiliary cloud communication and debug scripts.
- `emg02.cert.pem`, `emg02.private.key`, `root-CA.crt`: AWS IoT security credentials.
- `signal_1.csv`: Sample labeled EMG dataset used for testing.

## 🚀 Features

- Real-time EMG acquisition, filtering, and classification
- Support for multiple classifiers (RF, LSTM, XGBoost)
- Live actuator control using Dynamixel motor (XM430-W210-T)
- Low-latency multithreaded system design
- MQTT-based edge-to-cloud communication
- AWS IoT Core integration for remote monitoring

## 📦 Dependencies

- Python 3.8+
- `scikit-learn`, `numpy`, `pandas`, `joblib`, `seaborn`, `matplotlib`
- `dynamixel_sdk` (for motor control)
- Arduino Uno + EMG AFE sensor (e.g., Sichiray)
- MQTT broker (e.g., Mosquitto)
- AWS IoT Core (for cloud sync)

## 📄 License

This project is for educational and research purposes only.
