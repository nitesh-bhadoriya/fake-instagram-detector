# 📸 Fake Instagram Profile Detector

A deep learning-based project to detect **fake Instagram accounts** using account metadata like followers, follows, and post counts. This project applies data preprocessing, feature engineering, and a neural network model for binary classification.

---

## 📊 Dataset Overview

* **Features**: `#followers`, `#follows`, `#posts`
* **Label**: `fake` (1 = Fake, 0 = Real)
* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/jembovski/fake-istagram-detection-dataset)

The dataset simulates real and fake account behaviors. It's ideal for building models that can distinguish suspicious patterns based on public metadata.

---

## ⚙️ Project Workflow

1. **Data Cleaning**:

   * Handled zeros before log transformations
   * Normalized features using StandardScaler

2. **Feature Engineering**:

   * Applied `log10` transform to reduce skewness
   * Removed original skewed columns to improve model learning

3. **Modeling**:

   * Built a Deep Neural Network using TensorFlow/Keras
   * Structure: 3 layers (ReLU activation, Dropout, Sigmoid output)

4. **Evaluation**:

   * Used metrics like Accuracy, Precision, Recall, F1-Score
   * Plotted training/validation accuracy & loss
   * Generated Confusion Matrix for classification insights

---

## 🧠 Model Architecture

```text
Input Layer: 6 features
→ Dense (32 neurons, ReLU)
→ Dropout (20%)
→ Dense (16 neurons, ReLU)
→ Output (1 neuron, Sigmoid)
```

* **Optimizer**: Adam
* **Loss Function**: Binary Crossentropy
* **Epochs**: 20
* **Batch Size**: 16

---

## 📈 Results & Insights

* Achieved strong classification performance on validation data
* Visualized learning curves to monitor overfitting
* Key features influencing classification: follower/following ratios, log-transformed metadata

---

## 📚 Libraries Used

* `TensorFlow`, `Keras`
* `Scikit-learn`
* `Pandas`, `NumPy`
* `Seaborn`, `Matplotlib`

---

## 🔮 Future Improvements

* Add NLP features from bio/captions
* Incorporate image-based profile analysis
* Use SHAP/LIME for interpretability
* Deploy model with a web interface or API

---

## 🛡️ Disclaimer

This project is for **educational purposes**. It uses synthetic/aggregated data to demonstrate the use of deep learning for fake profile detection.
