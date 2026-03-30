#  AI Breast Cancer Detection System

An end-to-end AI-powered web application for detecting breast cancer from mammogram images using Deep Learning and Hybrid Machine Learning models.

---

## 📸 Project Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/2430563e-5b6c-45af-8ebc-13c3615b8e45" width="900">
</p>
 Upload a mammogram → Get prediction → Visualize with Grad-CAM

---

## 🚀 Features

- 📤 Upload mammogram images (Drag & Drop / Click / Paste)
- 🧠 Multi-model prediction:
  - CNN Model
  - ResNet Model
  - Hybrid Model (CNN + SVM)
- 🎯 Final prediction (Normal / Benign / Malignant)
- 🔥 Grad-CAM visualization (Explainable AI)
- 📊 Model comparison graph
- 🖥️ Modern responsive web interface (Flask)

---

## 🧠 Models Used

| Model | Description |
|------|------------|
| CNN | Custom Convolutional Neural Network |
| ResNet50 | Transfer Learning using pretrained model |
| Hybrid Model | CNN feature extraction + SVM classifier |

---

## 📂 Project Structure
```bash
breast-cancer-detection/
│
├── app.py
├── final_predict.py
├── predict.py
├── train.py
├── train_resnet.py
├── hybrid_model.py
├── gradcam.py
│
├── models/
│ ├── cnn_model.py
│ └── resnet_model.py
│
├── utils/
│ ├── dataset.py
│ ├── metrics.py
│ └── organise_dataset.py
│
├── templates/
│ └── index.html
│
├── static/
│ └── uploads/
│
├── model.pth
├── resnet_model.pth
├── svm_model.pkl
└── README.md
```
---


## ⚙️ Installation

```bash
git clone https://github.com/Karthiktelagi/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
pip install -r requirements.txt
python app.py
Then open:
http://127.0.0.1:5000/
```
---
📊 Dataset
```bash
Mammogram dataset (BIRADS categories)
Classes:
Normal
Benign
Malignant

⚠️ Dataset not included due to size limitations.
```
---
🔬 Workflow
```bash
Data preprocessing
Image resizing & normalization
CNN model training
ResNet model training
Feature extraction from CNN
Hybrid classification using SVM
Prediction via web app
Grad-CAM visualization
```
---
📈 Results
```bash
Accuracy: ~95%
Robust multi-class classification
Explainable predictions using Grad-CAM
```
---
🛠️ Tech Stack
```bash
Python 🐍
Flask 🌐
PyTorch 🔥
OpenCV 📷
Scikit-learn 🤖
HTML + CSS + JavaScript 🎨
```
---
👨‍💻 Author

Karthik TS
🎓 Engineering Student | 🔐 Cybersecurity Enthusiast | 🤖 AI Developer

⭐ Future Improvements
Cloud deployment (AWS / Render)
User authentication system
Mobile application
Real-time hospital integration

🏁 Conclusion

This project demonstrates the integration of deep learning and machine learning for real-world medical applications with explainable AI, helping in early breast cancer detection.


---

