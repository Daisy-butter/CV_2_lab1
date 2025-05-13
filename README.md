# FineTuning Pre-Trained CNN on Caltech-101

### 🚀 **Fine-tuning Pretrained Convolutional Neural Networks on Caltech-101**

This repository contains an implementation of fine-tuning convolutional neural networks (CNNs) that have been pretrained on ImageNet to perform object classification on the Caltech-101 dataset. Whether you're looking to run this project on an Alibaba Cloud server, your local machine, or any compatible computing resource, we've got you covered.

---

### 👩‍🔬 **Author**
**Ruihan Wu**  
Fudan University 

---

### 📌 **Project Features**
- **Dataset**: [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) with 101 object categories.
- **Architecture**: ResNet18 pretrained on ImageNet.
- **Capabilities**: 
  - Data preprocessing with augmentation and reproducibility.
  - Fine-tuning pretrained models for improved accuracy.
  - Side-by-side comparison with training from scratch.
- **Evaluation**: Analysis of top-1 and top-5 accuracy with visualizations.

---

### 🛠️ **Getting Started**

#### 1. Install Dependencies
Ensure you have Python, PyTorch, TensorBoard, and all necessary packages installed before proceeding. If your machine doesn't meet the configuration requirements, consider using a cloud server like Alibaba Cloud.

#### 2. Data Preprocessing
1. Download the dataset: [Caltech-101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02)  
   Save it to a local directory or upload it to your server.
   
2. Modify the file path in `data_preprocess.py` to point to the dataset location.  

3. Run `data_preprocess.py` to complete the following steps:
   - Data augmentation
   - Standardization
   - Reproducibility through a fixed random seed
   - Splitting the dataset into training, validation, and test sets

---

### 🏋️ **Model Training**

#### A. Pretraining on ImageNet
Use the pretrained ResNet18 model from `pretrain.py`.  
- Configure the parameter `data_dir` to match your dataset location.  
- Run the script to begin training, and monitor:
  - Training and validation accuracy for each epoch.
  - Final test set accuracy.
  - Loss curves and accuracy curves generated for analysis.

#### B. Fine-tuning
Open `pretrain_finetuning.ipynb` to experiment with fine-tuning.  
- Run all cells, and explore accuracy under various hyperparameter combinations.  
- This stage may take more time due to the optimization involved in fine-tuning.  
- Visualizations for top-1 and top-5 accuracy will be generated for insight into model performance.

#### C. Comparison with Training from Scratch
To compare the performance of pretrained models with models trained from scratch:  
- Open `train.py`.  
- Modify `data_dir` to match your dataset's location.  
- Run the script to train a ResNet18 from scratch using only Caltech-101 data.  

By comparing the results, you can evaluate the performance boost provided by pretraining.

---

### 📊 **Results**
- Pretrained models typically demonstrate a significant improvement in classification accuracy compared to models trained from scratch.
- Visualizations (loss curves, accuracy curves, top-1 and top-5 accuracy plots) are provided for in-depth analysis.

---

### 🌟 **Flexibility**
You can run this project on:
1. **Alibaba Cloud**: Upload scripts and data to your server.
2. **Local Machine**: Ensure that your machine meets the system requirements.
3. **Other Platforms**: Platforms like Google Colab or HPC clusters may also be configured to handle the workload.

---

### 🔗 **References**
- **Dataset**: [Caltech-101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02)  
- **Pretrained Weights**: ResNet18 pretrained on ImageNet via PyTorch.

---

### 📫 **Contact**
Have questions? Want to collaborate? Feel free to reach out!  

**Email**: [22307140084@m.fudan.edu.cn](mailto:22307140084@m.fudan.edu.cn)

I'm happy to hear your feedback, ideas, or anything related to this work!

---

### 📂 **Project Structure**
```plaintext
📦 project_directory
 ┣ 📂 data_preprocessing/
 ┃ ┗ 📜 data_preprocess.py // Preprocesses and splits the dataset
 ┣ 📂 training/
 ┃ ┣ 📜 pretrain.py        // Fine-tunes pretrained ResNet18
 ┃ ┣ 📜 train.py           // Trains ResNet18 from scratch
 ┃ ┗ 📜 pretrain_finetuning.ipynb // Notebook for fine-tuning experiments
 ┣ 📂 results/
 ┃ ┗ 📜 visualizations/    // Stores loss curves, accuracy plots, etc.
 ┗ 📜 README.md            // You are here!
