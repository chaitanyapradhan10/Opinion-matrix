# OpinionMetrix AI

AI-powered customer review sentiment analysis using a fine-tuned **DistilBERT Transformer model** with an interactive **Streamlit web application** and **MySQL database integration**.

---

## 📌 Project Overview

**OpinionMetrix AI** is an end-to-end Natural Language Processing (NLP) system designed to analyze customer reviews and classify them into three sentiment categories:

- 🟢 Positive
- 🟡 Neutral
- 🔴 Negative

The project covers the complete machine learning workflow, including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Transformer model training
- Model evaluation
- Sentiment prediction
- Streamlit deployment
- MySQL database integration
- Prediction history storage
- Git LFS support for the trained model

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

Positive, Neutral, Negative

The trained model is stored in:

`models/transformer/`

Large model files are managed using **Git LFS**.

---

## 📊 Dataset

The final dataset contains **7,583 customer reviews**.

### Dataset Distribution

| Sentiment | Number of Reviews |
|-----------|------------------:|
| Positive  | 3,256 |
| Negative  | 2,289 |
| Neutral   | 2,038 |
| **Total** | **7,583** |

During preprocessing:

- Duplicate reviews were removed.
- Rows with missing sentiment labels were excluded.
- Review text was cleaned and normalized.

---

## 🔍 Exploratory Data Analysis

The project includes an EDA notebook:

`notebook/EDA.ipynb`

The EDA covers:

- Dataset shape and columns
- Missing value analysis
- Duplicate review analysis
- Sentiment distribution
- Review length analysis
- Sample reviews
- URL detection
- HTML content detection
- Noisy text analysis
- Text quality analysis

---

# 🏗️ Project Structure

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

---

# 📂 Project Components

## `data/`

Contains the project dataset and database-related files.

### `reviews.csv`

Contains customer reviews and their corresponding sentiment labels.

### `database.py`

Contains database-related functionality.

---

## `models/`

Contains the trained Transformer model and associated files.

### `models/transformer/`

Contains:

- Trained DistilBERT model
- Tokenizer
- Model configuration
- Label encoder
- Training arguments
- Evaluation report

---

## `notebook/`

Contains the exploratory data analysis notebook.

### `EDA.ipynb`

The notebook analyzes:

- Dataset structure
- Missing values
- Duplicate values
- Sentiment distribution
- Review length
- Text quality
- Noisy data
- URLs
- HTML content

---

## `src/`

Contains the core machine learning and NLP components.

### `data_preprocessing.py`

Responsible for:

- Loading the dataset
- Checking missing values
- Removing duplicate reviews
- Cleaning text
- Text normalization
- Tokenization
- Stopword removal
- Lemmatization/stemming
- Preparing the dataset

### `train_transformer.py`

Responsible for:

- Loading the dataset
- Preparing sentiment labels
- Splitting the dataset
- Loading the DistilBERT tokenizer
- Loading the DistilBERT model
- Training the Transformer
- Evaluating the model
- Saving the trained model
- Saving the tokenizer
- Saving the label encoder
- Generating the evaluation report

### `predict_transformer.py`

Responsible for:

- Loading the trained Transformer model
- Loading the tokenizer
- Loading the label encoder
- Processing user input
- Generating sentiment predictions
- Calculating confidence
- Returning sentiment probabilities

### `logging_config.py`

Provides logging configuration for the project.

---

# 🌐 Streamlit Application

## `app.py`

`app.py` is the main Streamlit application.

It provides an interactive interface for:

- Entering customer reviews
- Generating sentiment predictions
- Displaying prediction confidence
- Displaying sentiment probabilities
- Visualizing prediction results
- Saving predictions to MySQL

---

# 🗄️ Database

The project uses **MySQL** to store prediction results.

The database stores:

| Field | Description |
|-------|-------------|
| `id` | Unique prediction ID |
| `review` | Customer review text |
| `sentiment` | Predicted sentiment |
| `prob_positive` | Positive probability |
| `prob_neutral` | Neutral probability |
| `prob_negative` | Negative probability |
| `correct_sentiment` | Actual/correct sentiment |
| `created_at` | Prediction timestamp |

---

# ⚙️ Technologies Used

## Programming

- Python

## Machine Learning & NLP

- PyTorch
- Hugging Face Transformers
- DistilBERT
- Scikit-learn
- NLTK

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Altair

## Web Application

- Streamlit

## Database

- MySQL
- PyMySQL

## Development Tools

- Git
- GitHub
- Git LFS
- VS Code

---

# 📦 Installation

## Step 1: Clone the Repository

    git clone https://github.com/maheshwari1394/Opinion-matrix.git

Move into the project directory:

    cd Opinion-matrix

---

## Step 2: Create a Virtual Environment

    python -m venv opinionmetrixai

### Windows PowerShell

    .\opinionmetrixai\Scripts\Activate.ps1

### Windows CMD

    opinionmetrixai\Scripts\activate

### Linux / macOS

    source opinionmetrixai/bin/activate

---

## Step 3: Install Dependencies

    pip install -r requirements.txt

---

# 🔐 Streamlit Configuration

Create the following directory:

`.streamlit/`

Inside it, create:

`secrets.toml`

Add your MySQL configuration:

    [connections.mysql]
    type = "mysql"
    host = "localhost"
    port = 3306
    database = "opinionmetrixai"
    username = "root"
    password = "YOUR_MYSQL_PASSWORD"

Replace `YOUR_MYSQL_PASSWORD` with your local MySQL password.

> ⚠️ **Important:** Never upload `.streamlit/secrets.toml` to GitHub because it contains sensitive credentials.

Add the following to `.gitignore`:

    .streamlit/secrets.toml

---

# 🗄️ MySQL Setup

## Step 1: Create Database

Open **MySQL Workbench** and run:

    CREATE DATABASE IF NOT EXISTS opinionmetrixai;

## Step 2: Select Database

    USE opinionmetrixai;

## Step 3: Create Prediction Table

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

---

# 🧪 Model Training

The Transformer model can be trained using:

`src/train_transformer.py`

Run:

    python src/train_transformer.py

The training process:

1. Loads the review dataset
2. Cleans the review text
3. Encodes sentiment labels
4. Splits the dataset
5. Loads the DistilBERT tokenizer
6. Loads the DistilBERT classification model
7. Fine-tunes the Transformer
8. Evaluates the model
9. Saves the trained model
10. Saves the tokenizer
11. Saves the label encoder
12. Generates the evaluation report

---

# 🔮 Model Prediction

Predictions can be generated using:

`src/predict_transformer.py`

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

- Positive
- Neutral
- Negative

---

# 📈 Model Evaluation

The evaluation report is available at:

`models/transformer/evaluation_report.txt`

This file contains the evaluation results generated after model training.

---

# ▶️ Running the Application

After completing the Python environment and MySQL setup, start the Streamlit application:

    streamlit run app.py

The application will open in your browser.

---

# 🔮 Example

### Input

    The product quality is excellent and I am very satisfied.

### Output

    Sentiment: Positive

The application also displays:

- Prediction confidence
- Positive probability
- Neutral probability
- Negative probability

---

# 🔄 Complete System Workflow

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
      ┌────┼────┐
      ▼    ▼    ▼
    Positive Neutral Negative
      │    │    │
      └────┼────┘
           ▼
    Confidence & Scores
           │
           ▼
    Streamlit Dashboard
           │
           ▼
    MySQL Database

---

# 🔁 End-to-End Pipeline

    Dataset
       │
       ▼
    Data Preprocessing
       │
       ▼
    Exploratory Data Analysis
       │
       ▼
    Train / Validation Split
       │
       ▼
    DistilBERT Tokenization
       │
       ▼
    Transformer Fine-Tuning
       │
       ▼
    Model Evaluation
       │
       ▼
    Trained Model
       │
       ▼
    Prediction Module
       │
       ▼
    Streamlit Application
       │
       ▼
    Sentiment + Confidence + Probabilities
       │
       ▼
    MySQL Prediction History

---
