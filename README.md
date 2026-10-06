# SpamShield — SMS Spam Detector

SpamShield is a machine learning project that classifies SMS messages as **Spam** or **Not Spam** using natural language processing and machine learning.

## Features

- SMS spam detection through an interactive Streamlit interface
- Text preprocessing and TF-IDF feature extraction
- Machine learning using Multinomial Naive Bayes and Logistic Regression
- Model comparison using accuracy, precision, recall, and F1-score
- Confusion matrix and performance dashboard
- Prediction history within the application session

## Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Streamlit
- Joblib

## Project Structure

```text
SpamShield/
├── app.py
├── train_model.py
├── requirements.txt
├── model_comparison.csv
├── confusion_matrix.csv
├── .gitignore
└── README.md
```

## Installation and Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Download the SMS Spam Collection dataset and place it at `data/SMSSpamCollection`.
5. Train the models using `python train_model.py`.
6. Launch the application using `python -m streamlit run app.py`.

## Model Evaluation

The project compares Multinomial Naive Bayes and Logistic Regression using accuracy, spam precision, spam recall, and spam F1-score.

See `model_comparison.csv` and `confusion_matrix.csv` for the recorded evaluation results.

## Future Improvements

- Improve spam recall and reduce missed spam messages.
- Experiment with classification thresholds and additional models.
- Test on new, unseen SMS messages.

## Author

Aditya Narwade