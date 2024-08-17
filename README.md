# AMR-GENOMIC-ANALYSIS - ResistAI
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pages Overview](#pages-overview)
  - [Home](#home)
  - [Analysis](#analysis)
  - [Train Model](#train-model)
  - [Make a Forecast](#make-a-forecast)
  - [Make Prediction](#make-prediction)
  - [About](#about)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

- ## Introduction

ResistAI is a web application designed to analyze, train models, forecast, and make predictions on antimicrobial resistance (AMR) data. This tool aims to support healthcare professionals, researchers, and public health authorities in making informed decisions about antibiotic usage and resistance patterns. By leveraging machine learning and data visualization, ResistAI helps in understanding trends and predicting future resistance, aiding in the fight against antibiotic-resistant bacteria.

## Features

- **Data Analysis:** Perform demographic, bacterial, and antibiotic analysis on AMR datasets.
- **Model Training:** Train machine learning models on selected datasets and analyze the model's performance metrics.
- **Forecasting:** Predict future trends of antibiotic resistance for different bacteria over selected periods.
- **Prediction:** Make predictions about the resistance of bacteria to specific antibiotics based on various parameters.
- **Interactive Visualizations:** Explore data through interactive charts and graphs.
- **Downloadable Models:** Download trained models for offline use or further analysis.

## Installation

To install ResistAI, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/SirImoleleAnthony/AMR-GENOMIC-ANALYSIS
    cd AMR-GENOMIC-ANALYSIS
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

Once the application is running, you can navigate through the various pages of the app using the top navigation bar. Each page serves a specific function, as outlined in the **Pages Overview** section below.

## Pages Overview

### Home

The Home page provides an overview of ResistAI, including an introduction to its features and some methodological overview. This page helps users understand what ResistAI can do and how to get started.

### Analysis

- **Select Dataset:** Choose between Antimicrobial Resistance in Europe Data, and Paratek-KEYSTONE data grouped into Gram-Negative Bacterial Surveillance Data, and Gram-Positive Bacterial Surveillance Data.
- **Choose Analysis Type:** Perform Demographic Analysis, Bacterial Analysis, or Antibiotic Analysis.
- **Interactive Visualizations:** Hover over charts and maps to view more details and insights.

### Train Model

- **Select Dataset:** Choose the dataset to train your machine learning model.
- **Choose Algorithm:** Select from 10 different machine learning algorithms.
- **View Outputs:** Analyze feature importance, model accuracy, precision, recall, F1-Score, and interactive confusion matrix.
- **Download Model:** Download the trained model as a pickle file for future use.

### Make a Forecast

- **Select Dataset:** Choose a dataset for forecasting.
- **Select Bacteria and Antibiotic:** Pick a bacterium and an antibiotic to forecast resistance trends.
- **Select Forecast Duration:** Choose the number of years for forecasting (1 to 10 years).
- **View Results:** Analyze the trend of resistance over the selected period.

### Make Prediction

- **Select Dataset:** Choose between Gram-Negative and Gram-Positive Bacterial Surveillance Data.
- **Set Parameters:** Select various parameters and conditions to tailor the prediction.
- **Make Prediction:** Click the button to get predictions on whether the bacteria would be resistant, intermediate, or susceptible to the chosen antibiotic.
- **Disclaimer:** Note that the predictions are for study purposes only and should be validated by a domain expert.

### About

Learn more about ResistAI, its purpose, and the team behind its development. This page may also include links to related resources and contact information.

## Disclaimer

The predictions and insights provided by ResistAI are for study and research purposes only. They should not be used as a sole basis for clinical decision-making without consulting a qualified healthcare professional.

## Contributing

This project is under the [2024 Vivli AMR Surveillance Data Challenge](https://amr.vivli.org/data-challenge/data-challenge-overview/). Contributing to it would depend on the outcome and descretion of the challenge organizers.

## License

This project is licensed under the MIT License as one of the licenses provided by GitHub.

## Contact

For questions or feedback, please contact us at:

- **Email:** anthonyimolele@gmail.com
