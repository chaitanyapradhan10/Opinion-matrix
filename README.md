# OpinionMetrix AI

AI-powered customer review sentiment analysis using a fine-tuned **DistilBERT Transformer model** with an interactive **Streamlit web application**.

---

## 📌 Project Overview

OpinionMetrix AI is an end-to-end sentiment analysis system designed to analyze customer reviews and classify them into three sentiment categories:

- Positive
- Neutral
- Negative

The project includes data preprocessing, exploratory data analysis, Transformer model training, prediction, evaluation, Streamlit deployment, and MySQL database integration.

---

## 🚀 Features

- Real-time customer review sentiment analysis
- Positive, Neutral, and Negative sentiment classification
- Fine-tuned DistilBERT Transformer model
- Prediction confidence score
- Probability distribution for each sentiment
- Interactive Streamlit dashboard
- MySQL database integration
- Prediction history storage
- Exploratory Data Analysis (EDA)
- Text preprocessing and normalization
- Model evaluation
- Git LFS support for the trained model

---

## 🧠 Machine Learning Model

The project uses:

**DistilBERT (`distilbert-base-uncased`)**

The model is fine-tuned for customer review sentiment classification.

### Sentiment Classes

```text
Positive
Neutral
Negative

The trained model is stored in:

models/transformer/

The large model file is managed using Git LFS.

📊 Dataset

The final dataset contains 7,583 customer reviews.

Dataset Distribution
Sentiment	Number of Reviews
Positive	3,256
Negative	2,289
Neutral	2,038
Total	7,583

Duplicate reviews were removed during preprocessing, and rows with missing sentiment labels were excluded.

🔍 Exploratory Data Analysis

The project includes an EDA notebook:

notebook/EDA.ipynb

The EDA covers:

Dataset shape and columns
Missing value analysis
Duplicate review analysis
Sentiment distribution
Review length analysis
Sample reviews
URL detection
HTML content detection
Noisy text analysis
🏗️ Project Structure
Opinion-matrix/
│
├── data/
│   ├── reviews.csv
│   └── database.py
│
├── models/
│   └── transformer/
│       ├── config.json
│       ├── evaluation_report.txt
│       ├── label_encoder.pkl
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── notebook/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── logging_config.py
│   ├── predict_transformer.py
│   └── train_transformer.py
│
├── app.py
├── db.py
├── download_models.py
├── requirements.txt
├── styles.py
├── .gitignore
└── .gitattributes
📂 Project Components
data/

Contains the project dataset and database-related files.

reviews.csv

Contains the customer reviews and their sentiment labels.

database.py

Contains database-related functionality.

notebook/

Contains the exploratory data analysis notebook.

EDA.ipynb

The notebook is used to analyze:

Dataset structure
Missing values
Duplicate values
Sentiment distribution
Review length
Text quality
Noisy data
src/

Contains the core machine learning components.

data_preprocessing.py

Responsible for:

Loading the dataset
Checking missing values
Removing duplicate reviews
Cleaning text
Tokenization
Stopword removal
Lemmatization/stemming
Preparing the dataset
train_transformer.py

Responsible for:

Loading the dataset
Preparing sentiment labels
Splitting the dataset
Loading the DistilBERT tokenizer
Loading the DistilBERT model
Training the Transformer
Evaluating the model
Saving the trained model
Saving the tokenizer
Saving the label encoder
Generating the evaluation report
predict_transformer.py

Responsible for:

Loading the trained Transformer model
Loading the tokenizer
Loading the label encoder
Processing user input
Generating sentiment predictions
Calculating confidence
Returning sentiment probabilities
logging_config.py

Provides logging configuration for the project.

🌐 Streamlit Application
app.py

app.py is the main application file.

It provides the user interface for:

Entering customer reviews
Generating sentiment predictions
Displaying prediction confidence
Displaying sentiment probabilities
Visualizing prediction results
Saving predictions to MySQL
🗄️ Database

The project uses MySQL to store prediction results.

The database stores:

Review text
Predicted sentiment
Positive probability
Neutral probability
Negative probability
Correct sentiment
Prediction timestamp
⚙️ Technologies Used
Programming
Python
Machine Learning & NLP
PyTorch
Hugging Face Transformers
DistilBERT
Scikit-learn
NLTK
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Altair
Web Application
Streamlit
Database
MySQL
PyMySQL
Development Tools
Git
GitHub
Git LFS
VS Code
📦 Installation
Step 1: Clone the Repository
git clone https://github.com/maheshwari1394/Opinion-matrix.git

Move into the project directory:

cd Opinion-matrix
Step 2: Create a Virtual Environment
python -m venv opinionmetrixai

Activate the environment on Windows:

.\opinionmetrixai\Scripts\Activate.ps1
Step 3: Install Dependencies
pip install -r requirements.txt
🔐 Streamlit Configuration

Create the following directory:

.streamlit/

Inside it create:

secrets.toml

Add your MySQL configuration:

[connections.mysql]
type = "mysql"
host = "localhost"
port = 3306
database = "opinionmetrixai"
username = "root"
password = "YOUR_MYSQL_PASSWORD"

Important: Never upload .streamlit/secrets.toml to GitHub because it contains sensitive credentials.

🗄️ MySQL Setup
Step 1: Create Database

Open MySQL Workbench and run:

CREATE DATABASE IF NOT EXISTS opinionmetrixai;
Step 2: Select Database
USE opinionmetrixai;
Step 3: Create Prediction Table
CREATE TABLE prediction (
    id INT AUTO_INCREMENT PRIMARY KEY,
    review TEXT NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    prob_positive FLOAT NOT NULL,
    prob_neutral FLOAT NOT NULL,
    prob_negative FLOAT NOT NULL,
    correct_sentiment VARCHAR(20) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
🧪 Model Training

The Transformer model can be trained using:

src/train_transformer.py

Run:

python src/train_transformer.py

The training process:

Loads the review dataset
Cleans the review text
Encodes sentiment labels
Splits the dataset
Loads the DistilBERT tokenizer
Loads the DistilBERT classification model
Fine-tunes the model
Evaluates the model
Saves the trained model
Saves the tokenizer
Saves the label encoder
Generates an evaluation report
🔮 Model Prediction

Predictions can be generated using:

src/predict_transformer.py

Example:

from src.predict_transformer import TransformerSentimentPredictor

predictor = TransformerSentimentPredictor(
    model_dir="models/transformer"
)

result = predictor.predict(
    "The product quality is excellent."
)

print(result)

Example output:

Sentiment: positive
Confidence: 99.76%

The prediction also provides probabilities for:

Positive
Neutral
Negative
📈 Model Evaluation

The evaluation report is available at:

models/transformer/evaluation_report.txt

This file contains the evaluation results generated after model training.

▶️ Running the Application

After completing the environment and MySQL setup, start the Streamlit application:

streamlit run app.py

The application will open in your browser.

🔮 Example
Input
The product quality is excellent and I am very satisfied.
Output
Sentiment: Positive

The application also displays:

Prediction confidence
Positive probability
Neutral probability
Negative probability
🔄 Complete System Workflow
                  Customer Review
                         │
                         ▼
                  Streamlit Input
                         │
                         ▼
                  Text Processing
                         │
                         ▼
                 DistilBERT Tokenizer
                         │
                         ▼
              Fine-Tuned DistilBERT
                         │
                         ▼
                Sentiment Prediction
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Positive     Neutral    Negative
              │          │          │
              └──────────┼──────────┘
                         ▼
                Confidence & Scores
                         │
                         ▼
                 Streamlit Dashboard
                         │
                         ▼
                  MySQL Database
