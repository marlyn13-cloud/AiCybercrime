# AI CyberCrime Detection Dashboard
-------------------------------------------------------------------------------------------------------------------------------------------
This project utilizes machine learning and computer vision to analyze cybersecurity datasets and generate AI-based summaries. Additionally, it features automated image captioning and summarization for uploaded images.

-------------------------------------------------------------------------------------------------------------------------------------------
**TABLE OF CONTENTS**

-Overview

-Features

-Technologies Used

-Example Use Cases

-Sample Usage

-------------------------------------------------------------------------------------------------------------------------------------------
**FEATURES**
-------------------------------------------------------------------------------------------------------------------------------------------

**Data & Model Dashboard**
- Upload cybersecurity-related CSV files. (Datasets can be found on Kaggle.com)
- Automatically detects the outcome/label column (e.g., `outcome`, `label`, `target`, etc.).
- Handles missing values using appropriate strategies.
- Encodes categorical variables and scales numeric features.
- Performs Principal Component Analysis (PCA) and visualizes explained variance.
- Trains a **Random Forest Classifier** and reports:
  - Accuracy
  - Classification Report
- Uses a **transformer-based AI model** to summarize model performance.

**Image Summarization**
- Automatically generates a caption using a transformer vision-language model.
- Summarizes the caption using a text summarization model.

-------------------------------------------------------------------------------------------------------------------------------------------
**Technologies Used**
-------------------------------------------------------------------------------------------------------------------------------------------
- **Streamlit** – Web dashboard
- **Scikit-learn** – Machine learning (PCA, Random Forest, etc.)
- **Pandas / NumPy** – Data manipulation
- **Matplotlib** – Visualization
- **Transformers (Hugging Face)**:
  - `nlpconnect/vit-gpt2-image-captioning` for image-to-text
  - `facebook/bart-large-cnn` (default) for summarization

-------------------------------------------------------------------------------------------------------------------------------------------
**Example Use Cases**

-Analyzing cybersecurity datasets for phishing or intrusion detection.

-Teaching ML concepts with a user-friendly interface.

-------------------------------------------------------------------------------------------------------------------------------------------
**SAMPLE USAGE**
-------------------------------------------------------------------------------------------------------------------------------------------
EXAMPLE 1:

<img width="1299" height="510" alt="cyber1" src="https://github.com/user-attachments/assets/b890a349-58c0-4f32-970e-34adb47c4ee4" />

<img width="728" height="545" alt="cyber2" src="https://github.com/user-attachments/assets/344cf0f1-7923-42df-8a09-4ab2fdc85ed8" />

<img width="1232" height="348" alt="cyber3" src="https://github.com/user-attachments/assets/6aceadcd-4f74-40dd-96a4-d69032d6efbe" />

<img width="1262" height="456" alt="cyber4" src="https://github.com/user-attachments/assets/787db570-4471-4bae-96b2-455b5cf01a88" />

EXAMPLE 2:

<img width="1301" height="556" alt="cyber5" src="https://github.com/user-attachments/assets/b3e840b4-d0e1-4647-8f7a-3e8fa9417405" />

<img width="739" height="570" alt="cyber6" src="https://github.com/user-attachments/assets/98981995-9ad1-492e-8c1a-6622dbe52933" />

<img width="1290" height="555" alt="cyber7" src="https://github.com/user-attachments/assets/4cc9b4e1-d34d-4e19-8f4e-785b7ff8b523" />
