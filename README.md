
# Double ConvLSTM Model for Cryptocurrency Candle Direction Prediction

This repository contains the implementation of a **Double ConvLSTM** deep learning model for predicting cryptocurrency price movement direction, using engineered binary features derived from price action. This code supports our paper submitted to the *Soft Computing* journal and includes data preprocessing, model training, evaluation, and visualization.

## 🔍 Overview

The model predicts whether the next candle in a crypto time series (e.g., ETH/USDT) will close higher or lower than it opens. The process involves:

- Building a custom dataset from raw price data,
- Engineering directional features using **Differential Propagation Thresholding (DPT)**,
- Training a **Conv1D + LSTM + Conv1D + LSTM** architecture,
- Evaluating performance via accuracy, confusion matrix, and precision/recall metrics.

## 📂 Files

- `DoubleConvLSTM_DPT.py`: Main code for preprocessing, model training, and evaluation.
- `README.md`: Documentation.

## ⚙️ Requirements

- Python 3.7+
- TensorFlow (>=2.x)
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Model Architecture

The model consists of the following layers:

1. `Conv1D` (for local pattern extraction),
2. `LSTM` with 128 units (for short-term temporal dependencies),
3. `Conv1D` (for refined feature selection),
4. `LSTM` with 64 units (for long-term temporal abstraction),
5. `Dropout` for regularization,
6. Dense layers for binary classification.

## 📊 Results

After training, the model achieves over **80% accuracy** on the ETH/USDT candle direction task using BTC-ETH features. Evaluation metrics include:

- Test accuracy
- Confusion matrix
- Precision, Recall, and F1-score

## 📈 Example Output

- Training & validation accuracy/loss over epochs
- Confusion matrix heatmap
- Precision/recall statistics

## 📁 Dataset

The mentioned `BTC_ETH_1d.csv` file contains:
- BTC and ETH price features (open, close, high, low, volume)
- Time-aligned rows for input into the model

For full dataset reproduction or other altcoins, follow the same preprocessing approach described in the paper.

## 🔬 Citation

If you use this code or dataset, please cite our paper (link forthcoming upon publication).

## 📜 License

This project is provided for research and educational use only.
