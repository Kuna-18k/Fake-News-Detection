# 📰 Fake News Detection using Machine Learning

> An intelligent system to automatically detect whether a news article is **real** or **fake**, built using machine learning and NLP techniques.

![Fake News Banner](https://img.freepik.com/premium-vector/fake-news-banner_118339-59.jpg)  
<sup><i>Image Source: Freepik</i></sup>

---

## 🚀 Project Highlights

- ✅ Classifies news into **REAL** or **FAKE**
- 🔠 Uses **TF-IDF Vectorization** for text features
- 🤖 Implements **PassiveAggressiveClassifier**
- 📈 Achieves high accuracy on benchmark datasets
- 💡 Can be extended for real-time applications

---

## 🧠 Model Pipeline

```mermaid
graph LR
A[🗂️ Load Dataset] --> B[🧹 Clean & Preprocess Text]
B --> C[🧾 TF-IDF Vectorization]
C --> D[📊 Train/Test Split]
D --> E[🧠 Train Classifier (PAC)]
E --> F[📈 Evaluate Model]
F --> G[✅ Predict New Samples]
```

---

## 📊 Dataset Info

| Feature      | Description                                 |
|--------------|---------------------------------------------|
| 📁 Source     | [Kaggle - Fake and Real News](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) |
| 📄 Samples    | ~44,000 total news articles                |
| 🏷️ Labels     | `FAKE`, `REAL`                              |
| 📝 Columns    | `title`, `text`, `label`                   |

---

## 📦 Tech Stack

| Tool         | Purpose                    |
|--------------|----------------------------|
| Python 🐍     | Programming language        |
| Pandas       | Data handling               |
| NumPy        | Numerical computing         |
| Scikit-learn | ML model & evaluation       |
| NLTK         | Text preprocessing (NLP)    |
| TfidfVectorizer | Text feature extraction |
| Matplotlib / Seaborn | Visualization      |

---

## 📈 Model Performance

| Metric       | Score (Example) |
|--------------|-----------------|
| Accuracy     | 93.2%           |
| Precision    | 0.93            |
| Recall       | 0.93            |
| F1 Score     | 0.93            |

🧪 Confusion matrix and classification report generated using test data.

---

## 🖥️ How to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Jupyter Notebook**
   ```bash
   jupyter notebook Fake_news_detection.ipynb
   ```

---

## 🧾 Sample Prediction Output

```plaintext
📰 Input: "Government launches new economic policy to fight inflation"
✅ Prediction: REAL

📰 Input: "NASA discovers aliens on the moon"
❌ Prediction: FAKE
```

---

## 🔮 Future Scope

- 🔁 Real-time news detection via APIs
- 📱 Convert to web app using Streamlit
- 🌐 Support for multilingual text
- 🧠 Use transformer models like BERT for better accuracy

---

## 🙌 Acknowledgements

- [Kaggle Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Scikit-learn, NLTK, and the open-source ML community

---

## ⭐ Show Your Support

If you found this helpful, **give it a star ⭐**, fork the repo, and share it with others!  
Feel free to open issues or contribute to make it better.

---

> Made with 💻 and ☕ by **Me**
