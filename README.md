# 🥊 UFC Fight Predictor – Machine Learning Project

This is a terminal-based UFC fight outcome predictor built using Python and machine learning (Random Forest Classifier). It uses fighter statistics to simulate and predict the winner of a fight between two chosen fighters.

## ⚙️ Features
- Predict fight winner based on stats
- Trained using Random Forest on historical fight data
- Command-line interface with ASCII UI
- Input validation and simple prime score logic

## 📁 Dataset
- `fighter_stats.csv`: General fighter attributes
- `large_dataset.csv`: Fight-by-fight historical data

## 🧠 Model
- Model: RandomForestClassifier (scikit-learn)
- Features: striking, grappling, height, weight, reach, and custom "prime score"
- Accuracy: ~74%

## 🚀 How to Run
1. Clone the repo
2. Make sure you have Python 3 installed
3. Install requirements:  
   `pip install pandas scikit-learn`
4. Place both CSVs in the same folder as `main.py`
5. Run the app:
   ```bash
   python main.py
