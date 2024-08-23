import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu # for setting up menu bar
import matplotlib.pyplot as plt # for data analysis and visualization
import seaborn as sns
import plotly.express as px
from plotly import graph_objs as go # for creating interactive visualizations
#import geopandas as gpd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import numpy as np
import pickle
import time
from io import BytesIO
import zipfile36 as zipfile
import utils


#-----------Web page setting-------------------#
page_title = "ResistAI"
page_icon = "🦠🧬💊"
picker_icon = "👇"
#layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = 'assets/resistAI_logo.png')

#--------------------Web App Design----------------------#

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Analysis', 'Train Model', 'Make a Forecast', 'Make Prediction', 'About'],
    icons = ["house-fill", "book-half", "gear", "activity", "robot", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)

@st.cache_data
#Antimicrobial Resistance in Europe Data
def load_euro_data():
    euro_link_id = "1aNulNFQQzvoDh75hbtZJ7_QcW4fRl9rs"
    euro_link = f'https://drive.google.com/uc?id={euro_link_id}'
    euro_df = pd.read_csv(euro_link)
    #euro_df = pd.read_csv("ecdc.csv")
    euro_df['Distribution'] = euro_df['Distribution'].str.split(',').str[1].str.split(' ').str[-1]
    euro_df = euro_df.drop(columns = ['Unit', 'RegionCode', 'Unnamed: 0'], axis = 1)
    return euro_df

euro_df = load_euro_data()

@st.cache_data
# Omadacycline Gram-Negative Data
def load_oma_ngram():
    negative_link_id = "1jHO-NFMsauUGVx9pfW5RPMvNGc6G0wAT"
    negative_link = f'https://drive.google.com/uc?id={negative_link_id}'
    gram_neg = pd.read_csv(negative_link)
    #gram_neg = pd.read_csv("omad_gram_neg_cleaned.csv")
    gram_neg = gram_neg.rename(columns={'Piperacillin-\ntazobactam': 'Piperacillin-tazobactam'})
    gram_neg = gram_neg.rename(columns={'Piperacillin-\ntazobactam_MIC': 'Piperacillin-tazobactam_MIC'})
    gram_neg = gram_neg[gram_neg['Age'] < 200]
    gram_neg['Gender'] = gram_neg['Gender'].replace({'M': 'Male', 'F': 'Female'})

    return gram_neg

gram_neg = load_oma_ngram()

@st.cache_data
# Omadacycline Gram-Negative Data
def load_oma_pgram():
    positive_link_id = "1NHR41hfyCN26EmQ7SrLCrhSBJnAVDQo0"
    positive_link = f'https://drive.google.com/uc?id={positive_link_id}'
    gram_pos = pd.read_csv(positive_link)
    #gram_pos = pd.read_csv("omad_gram_pos_cleaned.csv")
    gram_pos = gram_pos[gram_pos['Age'] < 200]
    gram_pos['Gender'] = gram_pos['Gender'].replace({'M': 'Male', 'F': 'Female'})
    return gram_pos

gram_pos = load_oma_pgram()

@st.cache_data
# Atlas Gram-Negative Data
def atlas_gram_neg():
    negative_link_id = "1SK9j-0gzKr7WFN-1F0pmcKHhF_oqlt9U"
    negative_link = f'https://drive.google.com/uc?id={negative_link_id}'
    atlas_gram_neg = pd.read_csv(negative_link)
    return atlas_gram_neg

atlas_gram_neg = atlas_gram_neg()

@st.cache_data
# Atlas Gram-Positive Data
def atlas_gram_pos():
    positive_link_id = "1HjME4Ef0Byz4XElh284KVZODmJjt9WKS"
    positive_link = f'https://drive.google.com/uc?id={positive_link_id}'
    atlas_gram_pos = pd.read_csv(positive_link)
    #atlas_gram_pos = atlas_gram_pos.drop('Phenotype', axis = 1)
    return atlas_gram_pos

atlas_gram_pos = atlas_gram_pos()


# List of items
model_list = ['Random Forest', 'Logistic Regression', 'Support Vector Machine (SVM)',
                    'Gradient Boosting Classifier', 'K-Nearest Neighbors (KNN)', 
                    'Decision Tree Classifier', 'Extreme Gradient Boosting (XGBoost)',
                    'Neural Network (MLPClassifier)', 'CatBoost Classifier', 'LightGBM Classifier']

ngram_antibiotic_list = ['Omadacycline', 'Doxycycline', 'Minocycline',
       'Tetracycline', 'Tigecycline', 'Ceftriaxone', 'Piperacillin-tazobactam',
       'Levofloxacin', 'Gentamicin', 'Amikacin', 'Aztreonam', 'Cefepime',
       'Ceftazidime', 'Imipenem', 'Trimethoprim-sulfamethoxazole']

pgram_antibiotic_list = ['Omadacycline', 'Doxycycline', 'Tetracycline',
       'Tigecycline', 'Oxacillin', 'Ceftaroline', 'Levofloxacin',
       'Erythromycin', 'Clindamycin', 'Linezolid', 'Daptomycin', 'Vancomycin',
       'Gentamicin', 'Trimethoprim-sulfamethoxazole']

atlas_pgram_anti_list = ['Clindamycin', 'Erythromycin', 'Levofloxacin', 'Linezolid', 'Minocycline',
                         'Penicillin', 'Tigecycline', 'Vancomycin','Ceftaroline', 'Daptomycin',
                         'Oxacillin','Teicoplanin']

atlas_ngram_anti_list = ['Amikacin', 'Amoxycillin clavulanate', 'Ampicillin', 'Cefepime', 'Ceftazidime', 
                         'Ceftriaxone', 'Imipenem', 'Levofloxacin', 'Meropenem', 'Minocycline', 
                         'Piperacillin tazobactam', 'Tigecycline', 'Aztreonam', 'Ceftaroline', 
                         'Ceftazidime avibactam', 'Colistin']

datasets = ["Antimicrobial Resistance in Europe Data", "Gram-Negative Bacterial Surveilance Data", 
                "Gram-Positive Bacterial Surveilance Data", "Atlas Gram-Negative Bacteria Data",
                "Atlas Gram-Positive Bacteria Data"]

analysis = ["Demographic Analysis", "Bacteria (Orgamism) Analysis", "Antibiotics Analysis"]

# Home page
if selected == "Home":
    st.image("assets/resistAI_banner.png", use_column_width=True)
    st.subheader("Welcome to AMR Web App")

    #
    st.subheader("Objectives")
    st.markdown(
        """
    The primary objective of ResistAI is to leverage advanced data analytics and machine learning techniques to analyze, predict, and forecast antimicrobial resistance (AMR) patterns. 
    The platform aims to empower healthcare professionals and researchers with actionable insights to enhance antibiotic stewardship, optimize treatment strategies, and contribute to global efforts in combating AMR.
        """
    )

    st.subheader("Methods")
    st.image("assets/pipeline.jpeg", use_column_width=True)
    st.markdown(
        """
    To achieve the overaching aim for the app and provide insightful and comprehensive usage, and through the use of domain knowledge, 
    the Pfizer's ATLAS and Paratek's KEYSTONE data provided were further grouped into Gram-Positive and Gram-Negative Bacterial data.
    Read more about this in the document [here](doc.here)
        
    ResistAI utilizes a comprehensive suite of tools, including demographic, bacterial, and antibiotic analysis, to provide a detailed understanding of AMR trends. 
    The platform employs state-of-the-art machine learning algorithms to train predictive models, and time series analysis for forecasting future resistance trends. 
    Data is visualized through interactive charts and maps, offering users an intuitive and in-depth exploration of AMR data.
        """,
        unsafe_allow_html=True
    )

    st.subheader("Results")
    st.image("assets/word_cloud.jpeg", use_column_width=True)
    st.markdown(
        """
    Through rigorous analysis and modeling, ResistAI delivers precise predictions on bacterial resistance patterns, highlighting the most influential factors driving AMR. 
    The platform’s forecasts offer clear insights into potential future scenarios, assisting in preemptive measures against rising resistance trends. 
    The results are presented in an easily interpretable format, allowing for quick and informed decision-making.
        """
    )

    st.subheader("Impact of the Work")
    st.markdown(
        """
    ResistAI's work significantly impacts public health by providing a robust tool for understanding and combating AMR. 
    By enabling data-driven decisions, the platform supports improved patient outcomes, reduces the misuse of antibiotics, and strengthens global efforts to curtail the spread of resistant bacterial strains. 
    The platform's adaptability and scalability make it a valuable resource in both high-resource and resource-limited settings, contributing to the global fight against AMR.
        """
    )




if selected == "Analysis":

    # Analys page Instructions Sidebar
    with st.sidebar:
        st.header("Welcome to the Analysis Page!")
        st.subheader("Instructions")
        
        with st.expander("**Step 1: Select Your Dataset**"):
            st.write("""
            - Use the first dropdown menu to choose the dataset you want to analyze.
                - Options include:
                    - **Antimicrobial Resistance in Europe Data**: Contains data on antimicrobial resistance trends in Europe.
                    - **Gram-Negative Bacterial Surveillance Data**: Focuses on surveillance data related to Gram-negative bacteria from the Paratek KEYSTONE data.
                    - **Gram-Positive Bacterial Surveillance Data**: Covers surveillance data on Gram-positive bacteria from Paratek KEYSTONE data.
                    - **Atlas Gram-Negative Bacteria Data**: Covers data on Gram-negative bacteria from the Pfizer ATLAS data.
                    - **Atlas Gram-Positive Bacteria Data**: Covers data on Gram-positive bacteria from the Pfizer ATLAS data.   
            """)

        with st.expander("**Step 2: Choose the Type of Analysis**"):
            st.write("""
            - Once you've selected your dataset, use the second dropdown menu to choose the type of analysis:
                - **Demographic Analysis**: Analyze data based on demographic factors.
                - **Bacterial Analysis**: Focus on analyzing bacterial strains and related metrics.
                - **Antibiotic Analysis**: Analyze the effectiveness of different antibiotics and resistance patterns.
            """)

        with st.expander("**Step 3: Interact with Visualizations**"):
            st.write("""
            - After selecting your dataset and analysis type, visualizations will be generated.
            - Interact with these visualizations by hovering over bars, choropleths, or other elements to see detailed information.
            - This interaction will provide you with deeper insights into the data.
            """)

        st.write("Note: Feel free to revisit this guide if you need assistance.")
        st.info("Happy analyzing!😊📈📊")


    st.subheader("Select preferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)


    if selected_dataset == "Antimicrobial Resistance in Europe Data":

        st.subheader("Select analysis")
        selected_analysis = st.selectbox("Pick analysis type " + picker_icon, analysis)

        if selected_analysis == "Demographic Analysis":

            data = euro_df
            # Age analysis
            age_subset = data[data['Distribution'] == "age"]
            age_count = age_subset['Category'].value_counts() 
            fig = px.bar(age_count, x=age_count.index, y='count', 
                    title='Age Group Distribution',
                    labels={'x': age_count.index, 'count': 'Frequency'})
            fig.update_layout(
                xaxis_title='Age Group',
                yaxis_title='Frequency',  
            )
            st.plotly_chart(fig)

            # Gender analysis
            gender_subset = data[data['Distribution'] == "gender"]
            gender_count = gender_subset['Category'].value_counts().sort_values(ascending=True)
            fig = px.bar(gender_count, x=gender_count.index, y='count', 
                    title='Gender Distribution',
                    labels={'x': gender_count.index, 'count': 'Frequency'})
            fig.update_layout(
                xaxis_title='Gender',
                yaxis_title='Frequency',  
            )
            st.plotly_chart(fig)

            # Country analysis
            country_count = data['RegionName'].value_counts().sort_values(ascending=False).head(10)
            fig = px.bar(country_count, x=country_count.index, y='count', 
                    title='Top 10 Countries of the Study',
                    labels={'x': country_count.index, 'count': 'Frequency'})
            fig.update_layout(
                xaxis_title='Countries',
                yaxis_title='Frequency',  
            )
            st.plotly_chart(fig)
        

        # Bacteria Analysis
        if selected_analysis == "Bacteria Analysis":

            st.write("**Bacteria Types Distribution**")
            bacteria_count = euro_df['Bacteria'].value_counts()
            st.bar_chart(bacteria_count)

        
        # Antibiotic Analysis
        if selected_analysis == "Antibiotics Analysis":
            st.write("**Antibiotic Distribution**")
            anti_count = euro_df['Antibiotic'].value_counts()
            st.bar_chart(anti_count)

            fig1 = px.box(
                euro_df,
                x='Antibiotic',
                y='Value',
                title='Antibiotic Efficacy Comparison',
                labels={'Antibiotic': 'Antibiotic', 'Value': 'Resistance Percentage'},
                category_orders={'Antibiotic': sorted(euro_df['Antibiotic'].unique())}
            )

            fig1.update_layout(
                xaxis_title='Year',
                yaxis_title='Resistance Percentage',
                height=800,  
                width=800   
            )

            st.plotly_chart(fig1)

            st.subheader("Bacteria-Antibiotic Interaction")
            st.write("Bacteria-Antibiotic Interaction Heatmap")
            interaction_data = euro_df.pivot_table(index='Bacteria', columns='Antibiotic', values='Value', aggfunc='mean')

            fig2 = px.imshow(
                interaction_data,
                color_continuous_scale=px.colors.sequential.Plasma_r,
                title='Bacteria-Antibiotic Interaction Heatmap',
                labels={'color': 'Resistance Percentage'},
                aspect='auto'
            )

            fig2.update_layout(
                title={'text': 'Bacteria-Antibiotic Interaction Heatmap', 'x': 0.5},
                xaxis_title='Antibiotic',
                yaxis_title='Bacteria',
                height=1200,  
                width=800   
            )
            st.plotly_chart(fig2)

            st.subheader("Time Series Analysis")
            aggregated_data = euro_df.groupby('Time').agg({'Value': 'mean'}).reset_index()
            fig1 = px.line(
                aggregated_data,
                x='Time', 
                y='Value',  
                title='Resistance Trend Over Time',
                labels={'Time': 'Year', 'Value': 'Average Resistance'},
                markers=True  
            )

            st.plotly_chart(fig1)
            st.info("Add summary analysis here!")

            st.subheader("Geographical Analysis")
            fig2 = px.choropleth(
                euro_df,
                locations='RegionName',
                locationmode='country names',
                color='Value',
                hover_name='RegionName',
                color_continuous_scale=px.colors.sequential.Plasma_r,
                title='Antimicrobial Resistance in Europe',
                scope='europe'
            )

            st.plotly_chart(fig2)
        

 
    
    # Gram-Negative Data Analysis
    if selected_dataset == "Gram-Negative Bacterial Surveilance Data":

        st.subheader("Select analysis")
        selected_analysis = st.selectbox("Pick analysis type " + picker_icon, analysis)

        # Demographic Analysis
        if selected_analysis == "Demographic Analysis":
            # Age distribution analysis
            utils.age_distribution(gram_neg)

            # Gender distribution analysis
            utils.gender_distribution(gram_neg)

            # Continent analys
            utils.continent_analysis(gram_neg)

            # Top 10 countries of study
            utils.top_10_countries(gram_neg)

        # Bactirial Analysis
        if selected_analysis == "Bacteria (Orgamism) Analysis":
            # Top 10 Organisms
            utils.top_10_organisms(gram_neg, "Organism")

            # Distribution of Organisms per Country
            utils.org_by_country(gram_neg, "Organism")

            #Organism per Gender
            utils.org_per_gender(gram_neg, "Organism")

            #Organism per Age Group
            utils.org_vs_age(gram_neg)

            # Yearly organism distribution
            utils.org_by_year(gram_neg)

            # Distribution of Organism per antibiotic
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, ngram_antibiotic_list)
            utils.org_vs_anti(gram_neg, anti, "Organism")

            # Organism Infection Source
            utils.org_infection_source(gram_neg)

            #Organism Infection Types
            utils.org_infection_type(gram_neg)

            #Organism Specimen Type
            utils.org_specimen_type(gram_neg)

        # Antibiotic Analysis
        if selected_analysis == "Antibiotics Analysis":
            # Antibiotic Resistance
            utils.anti_resistance(gram_neg)

            # Yearly Antibiotic Resistance
            st.subheader("Select an antibiotic for yearly resistance")
            anti_res = st.selectbox("Pick an Antibiotic " + picker_icon, ngram_antibiotic_list, key="yearly_antibiotic")
            utils.anti_resistance_yearly(gram_neg, anti_res)

            # Continent Resistance Distribution
            st.subheader("Select an antibiotic for continent resistance distribution")
            anti_continent = st.selectbox("Pick an Antibiotic " + picker_icon, ngram_antibiotic_list, key="continent_antibiotic")
            utils.anti_per_continent(gram_neg, anti_continent)

    # Gram-Positive Data Analysis
    if selected_dataset == "Gram-Positive Bacterial Surveilance Data":


        st.subheader("Select analysis")
        selected_analysis = st.selectbox("Pick analysis type " + picker_icon, analysis)

        # Demographic Analysis
        if selected_analysis == "Demographic Analysis":
            # Age distribution analysis
            utils.age_distribution(gram_pos)

            # Gender distribution analysis
            utils.gender_distribution(gram_pos)

            # Continent analys
            utils.continent_analysis(gram_pos)

            # Top 10 countries of study
            utils.top_10_countries(gram_pos)

        # Bactirial Analysis
        if selected_analysis == "Bacteria (Orgamism) Analysis":
            # Top 10 Organisms
            utils.top_10_organisms(gram_pos, "Organism")

            # Distribution of Organisms per Country
            utils.org_by_country(gram_pos, "Organism")

            #Organism per Gender
            utils.org_per_gender(gram_pos, "Organism")

            #Organism per Age Group
            utils.org_vs_age(gram_pos)

            # Yearly organism distribution
            utils.org_by_year(gram_pos)

            # Distribution of Organism per antibiotic
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, pgram_antibiotic_list)
            utils.org_vs_anti(gram_pos, anti, "Organism")

            # Organism Infection Source
            utils.org_infection_source(gram_pos)

            #Organism Infection Types
            utils.org_infection_type(gram_pos)

            #Organism Specimen Type
            utils.org_specimen_type(gram_pos)

        # Antibiotic Analysis
        if selected_analysis == "Antibiotics Analysis":
            # Antibiotic Resistance
            utils.anti_resistance(gram_pos)

            # Yearly Antibiotic Resistance
            st.subheader("Select an antibiotic for yearly resistance")
            anti_res = st.selectbox("Pick an Antibiotic " + picker_icon, pgram_antibiotic_list, key="yearly_antibiotic")
            utils.anti_resistance_yearly(gram_pos, anti_res)

            # Continent Resistance Distribution
            st.subheader("Select an antibiotic for continent resistance distribution")
            anti_continent = st.selectbox("Pick an Antibiotic " + picker_icon, pgram_antibiotic_list, key="continent_antibiotic")
            utils.anti_per_continent(gram_pos, anti_continent)

    # Atlas Gram-Negative Data Analysis
    if selected_dataset == "Atlas Gram-Negative Bacteria Data":
        st.subheader("Select analysis")
        selected_analysis = st.selectbox("Pick analysis type " + picker_icon, analysis)

        # Demographic Analysis
        if selected_analysis == "Demographic Analysis":

            # Age Group Analysis
            utils.atlas_age_group(atlas_gram_neg)

            # Country Analysis
            utils.top_10_countries(atlas_gram_neg)

            #Gender Distribution
            utils.gender_distribution(atlas_gram_neg)

            # Patient Type
            utils.atlas_patient_type(atlas_gram_neg)
        
        # Bactirial Analysis
        if selected_analysis == "Bacteria (Orgamism) Analysis":
            # Top 10 Organisms
            utils.top_10_organisms(atlas_gram_neg, "Species")

            # Age Group vs Organisms
            utils.atlas_age_vs_org(atlas_gram_neg, "Species")

            # Organism per Gender
            utils.org_per_gender(atlas_gram_neg, "Species")

            # Organism Distribution by Country
            utils.org_by_country(atlas_gram_neg, "Species")

            # Yearly organism distribution
            utils.atlas_org_by_year(atlas_gram_neg)

            # Organism by Family
            utils.atlas_org_by_family(atlas_gram_neg, "Species")

            # Organism Resistance per Antibiotic
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_ngram_anti_list)
            utils.org_vs_anti(atlas_gram_neg, anti, "Species")

        # Antibiotic Analysis
        if selected_analysis == "Antibiotics Analysis":
            # Antibiotic Resistance
            utils.anti_resistance(atlas_gram_neg)

            #Yearly Antibiotic Resistance
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_ngram_anti_list)
            utils.atlas_anti_resistance_yearly(atlas_gram_neg, anti)

            # Atlas Chloroplet
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_ngram_anti_list, key="continent_antibiotic")
            utils.atlas_anti_analysis(atlas_gram_neg, anti)
            
    
    # Atlas Gram-Positive Data Analysis
    if selected_dataset == "Atlas Gram-Positive Bacteria Data":
        st.subheader("Select analysis")
        selected_analysis = st.selectbox("Pick analysis type " + picker_icon, analysis)

        # Demographic Analysis
        if selected_analysis == "Demographic Analysis":

            # Age Group Analysis
            utils.atlas_age_group(atlas_gram_pos)

            # Country Analysis
            utils.top_10_countries(atlas_gram_pos)

            #Gender Distribution
            utils.gender_distribution(atlas_gram_pos)

            # Patient Type
            utils.atlas_patient_type(atlas_gram_pos)
        
        # Bactirial Analysis
        if selected_analysis == "Bacteria (Orgamism) Analysis":
            # Top 10 Organisms
            utils.top_10_organisms(atlas_gram_pos, "Species")

            # Age Group vs Organisms
            utils.atlas_age_vs_org(atlas_gram_pos, "Species")

            # Organism per Gender
            utils.org_per_gender(atlas_gram_pos, "Species")

            # Organism Distribution by Country
            utils.org_by_country(atlas_gram_pos, "Species")

            # Yearly organism distribution
            utils.atlas_org_by_year(atlas_gram_pos)

            # Organism by Family
            utils.atlas_org_by_family(atlas_gram_pos, "Species")

            # Organism Resistance per Antibiotic
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_pgram_anti_list)
            utils.org_vs_anti(atlas_gram_pos, anti, "Species")

        # Antibiotic Analysis
        if selected_analysis == "Antibiotics Analysis":
            # Antibiotic Resistance
            utils.anti_resistance(atlas_gram_pos)

            #Yearly Antibiotic Resistance
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_pgram_anti_list)
            utils.atlas_anti_resistance_yearly(atlas_gram_pos, anti)

            # Atlas Chloroplet
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_pgram_anti_list, key="continent_antibiotic")
            utils.atlas_anti_analysis(atlas_gram_pos, anti)


# Train Model Page
if selected == "Train Model":

    # Train Model page Instructions Sidebar
    with st.sidebar:
        st.header("Welcome to the Train Model Page!")
        st.subheader("Instructions")
        
        with st.expander("**Step 1: Select Dataset for Training**"):
            st.write("""
            - **What to Do:**
                - Use the first select box to choose the dataset you'd like to work with. The available datasets include:
                    - **Antimicrobial Resistance in Europe Data**
                    - **Gram-Negative Bacterial Surveillance Data**
                    - **Gram-Positive Bacterial Surveillance Data**
                    - **Atlas Gram-Negative Bacteria Data**
                    - **Atlas Gram-Positive Bacteria Data**
                     
            - **What to Expect:** The selected dataset will be loaded and prepared for analysis.
            """)

        with st.expander("**Step 2: Choose a Machine Learning Algorithm**"):
            st.write("""
            - **What to Do:** 
                - Use the second select box to select a machine learning algorithm from the list of 10 available options. 
                     
            - **What to Expect:** This algorithm will be used to train a model on the selected dataset.

            """)

        with st.expander("**Step 3: Choose an Antibiotic**"):
            st.write("""
            - **What to Do:** 
                - Use the third select box to select a particular antibiotic from the list of available antibiotics you wish to train the model for. 
                     
            - **What to Expect:** This antibiotic will be the target variable that will be used in the training of the model.

            """)

        with st.expander("**Step 4: Train Your Model**"):
            st.write("""
            - **What to Do:** 
                - Click the `Train your model` button to train the algorithm you have selected on the dataset chosen. 
                     
            - **What to Expect:** The training of the model would be done. This might take a few seconds to some minutes, depending on the algorithm selected.

            """)

        with st.expander("**Step 5: View and Interpret the Outputs**"):
            st.write("""
            - **Outputs:**
            - **Feature Importance (if applicable):**
                - **What to Do:** 
                    - Check the interactive bar chart that displays the most important features used by the model. 
                    - Hover over the bars to see details about each feature's importance.
                - **What to Expect:** This output will help you understand which features are most influential in the model's predictions.
            - **Metric Results:**
                - **What to Do:** 
                    - Review the following key performance metrics:
                        - **Accuracy Score:** Indicates the overall accuracy of the model.
                        - **Precision:** Measures the accuracy of positive predictions.
                        - **Recall:** Indicates how well the model identifies positive cases.
                        - **F1-Score:** The harmonic mean of precision and recall, balancing both metrics.
                - **What to Expect:** These metrics will help you evaluate the model's performance and suitability for your analysis.
            - **Confusion Matrix:**
                - **What to Do:** 
                    - Analyze the interactive confusion matrix plot to compare predicted labels versus actual labels.
                    - Hover over each cell to see the count of predictions.
                - **What to Expect:** This will give you insight into where the model is making correct or incorrect predictions.
            """)

        with st.expander("**Step 6: Download the Trained Model**"):
            st.write("""
            - **What to Do:**
                - If satisfied with the model’s performance, click the `Download Model` button to download the trained model as a pickle file.
            - **What to Expect:** This will allow you to save the model for future use or further analysis.
            """)

        st.write("Note: Feel free to revisit this guide if you need assistance.")
        st.info("Happy training!😊🛠️⚙️")

    st.subheader("Select preferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)


    if selected_dataset == "Antimicrobial Resistance in Europe Data":
        # Dummy feature encoding
        data_encoded = euro_df
        # Feature and target
        X = data_encoded.drop('Value', axis=1)
        y = (euro_df['Value'] > 50).astype(int)  # Binary classification: 0 for non-resistant, 1 for resistant

        # Apply label encoding to non-numerical columns in X
        le = LabelEncoder()
        non_numerical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in non_numerical_cols:
            X[col] = le.fit_transform(X[col])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Select model to train  
        st.subheader("Select Algorithm")
        model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
        
        def model_training_and_analysis(model_selected, X_train, y_train, X_test, y_test):
            # Initialize the model based on selection
            if model_selected == "Random Forest":
                model = RandomForestClassifier()
            elif model_selected == "Logistic Regression":
                model = LogisticRegression()
            elif model_selected == "Support Vector Machine (SVM)":
                model = SVC()
            elif model_selected == "Gradient Boosting Classifier":
                model = GradientBoostingClassifier()
            elif model_selected == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier()
            elif model_selected == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            elif model_selected == "Extreme Gradient Boosting (XGBoost)":
                model = XGBClassifier()
            elif model_selected == "Neural Network (MLPClassifier)":
                model = MLPClassifier()
            elif model_selected == "CatBoost Classifier":
                model = CatBoostClassifier(silent=True)
            elif model_selected == "LightGBM Classifier":
                model = LGBMClassifier()
            
            train_btn = st.button("Train your model")
            st.info("**Note**: Some algorithms might take long to train. Just give it some minutes!")
            
            # Get training 
            if train_btn:
                # Initialize the progress bar
                loading_text = st.text("Your model is training...")
                progress = st.progress(0)
                

                # Artificially increment progress
                for i in range(0, 101, 10):
                    time.sleep(0.1) 
                    progress.progress(i)
        
                # Train the model
                model.fit(X_train, y_train)

                # Remove the progress bar after completion
                progress.empty()
                loading_text.empty()

                # Feature importance (if applicable)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    # Prepare data for Plotly Express
                    importance_df = pd.DataFrame({
                        'Feature': [X_train.columns[i] for i in indices],
                        'Importance': importances[indices]
                    })


                    st.subheader("Feature Importance Analysis")
                    # Plot using Plotly Express
                    fig = px.bar(importance_df, x='Feature', y='Importance', title='Most Important Features',
                                labels={'Feature': 'Feature Name', 'Importance': 'Importance Value'})
                    fig.update_xaxes(tickangle=-90)
                    st.plotly_chart(fig)

                # Predictions
                st.subheader("Prediction Score")
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred) * 100
                precision = precision_score(y_test, y_pred, average='weighted') * 100
                recall = recall_score(y_test, y_pred, average='weighted') * 100
                f1 = f1_score(y_test, y_pred, average='weighted') * 100

                # Print metrics as percentages
                st.write(f"Accuracy: **{accuracy:.2f}%**")
                st.write(f"Precision: **{precision:.2f}%**")
                st.write(f"Recall: **{recall:.2f}%**")
                st.write(f"F1-score: **{f1:.2f}%**")

                # Confusion Matrix Analysis
                st.subheader("Confusion Matrix Analysis")
                cm = confusion_matrix(y_test, y_pred)

                # Convert confusion matrix to DataFrame for Plotly
                cm_df = pd.DataFrame(cm, index=['Resistant', 'Suceptible'], columns=['Resistant', 'Suceptible'])

                # Plot confusion matrix using Plotly Express
                fig1 = px.imshow(cm_df, text_auto=True, color_continuous_scale=px.colors.sequential.Plasma_r,
                                labels={'x': 'Predicted Label', 'y': 'Actual Label', 'color': 'Count'},
                                title='Confusion Matrix')
                fig1.update_xaxes(side="bottom") 
                st.plotly_chart(fig1)

            # Provide download button to download the trained model
            st.subheader("Download Trained Model")
            pickle_buffer = BytesIO()
            pickle.dump(model, pickle_buffer)
            pickle_buffer.seek(0)

            st.download_button(
                label="Download Model",
                data=pickle_buffer,
                file_name=f"{model_selected.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )


        model_training_and_analysis(model_selected, X_train, y_train, X_test, y_test)
    
    # Gram-Negative Data Training
    if selected_dataset == "Gram-Negative Bacterial Surveilance Data":        
        st.subheader("Select Algorithm")
        model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
        
        st.subheader("Select Antibiotic to train model for")
        anti = st.selectbox("Pick an Antibioticn " + picker_icon, ngram_antibiotic_list)

        utils.model_training(model_selected, gram_neg, anti)

    # Gram-Positive Data Training
    if selected_dataset == "Gram-Positive Bacterial Surveilance Data":        
        st.subheader("Select Algorithm")
        model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
        
        st.subheader("Select Antibiotic to train model for")
        anti = st.selectbox("Pick an Antibioticn " + picker_icon, pgram_antibiotic_list)

        utils.model_training(model_selected, gram_pos, anti)

    # Atlas Gram-Negative Data Training
    if selected_dataset == "Atlas Gram-Negative Bacteria Data":        
        st.subheader("Select Algorithm")
        model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
        
        st.subheader("Select Antibiotic to train model for")
        anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_ngram_anti_list)

        #Trained Data
        utils.atlas_model_training(model_selected, atlas_gram_neg, anti, pheno = None)

    # Atlas Gram-Positive Data Training
    if selected_dataset == "Atlas Gram-Positive Bacteria Data":        
        st.subheader("Select Algorithm")
        model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
        
        st.subheader("Select Antibiotic to train model for")
        anti = st.selectbox("Pick an Antibioticn " + picker_icon, atlas_pgram_anti_list)

        #Trained Data
        utils.atlas_model_training(model_selected, atlas_gram_pos, anti, pheno = "Phenotype")


if selected == "Make a Forecast":
    # Forecast page Instructions Sidebar
    with st.sidebar:
        st.header("Welcome to the Make a Forecast Page!")
        st.subheader("Instructions")
        
        with st.expander("**Step 1: Select Dataset for Analysis**"):
            st.write("""
            - **What to Do:** 
            - Use the first select box to choose the dataset you'd like to analyze. The available datasets include:
                - **Antimicrobial Resistance in Europe Data**
                - **Gram-Negative Bacterial Surveillance Data**
                - **Gram-Positive Bacterial Surveillance Data**
                - **Atlas Gram-Negative Bacteria Data**
                - **Atlas Gram-Positive Bacteria Data**
            - **What to Expect:** The selected dataset will be loaded and prepared for forecasting.

            """)

        with st.expander("**Step 2: Choose a Bacteria (Organism)**"):
            st.write("""
            - **What to Do:** 
                - Use the second select box to select a bacteria (organism) from the list of available options in the selected dataset.
            - **What to Expect:** This selection will allow you to focus the analysis on a specific bacteria.
            """)

        with st.expander("**Step 3: Choose an Antibiotic**"):
            st.write("""
            - **What to Do:** 
                - Use the third select box to select an antibiotic from the list of available options in the selected dataset.
            - **What to Expect:** This selection will set the context for forecasting resistance trends.

            """)

        with st.expander("**Step 4: Set the Forecast Period**"):
            st.write("""
            - **What to Do:** 
                - Use the slider to choose the number of years for the forecast, ranging from 1 to 10 years.
            - **What to Expect:** The forecast will be generated for the selected time period, providing a trend analysis of resistance.
            """)

        with st.expander("**Step 5: Make Your Forecast**"):
            st.write("""
            - **What to Do:** 
                - Click the `Make Your Forecast` button to forecast the resistance trend of your selected bacteria (organism) to the selected antibiotic over the given period of time. 
                     
            - **What to Expect:** A forecast would be made based on your selections. When done, a time series plot would be displayed. This might take a few seconds to some minutes so just give it a little time.
            """)

        with st.expander("**Step 6: Interpret the Time Series Trend Analysis**"):
            st.write("""
            - **Output:**
                - A time series chart will display the resistance values over the selected forecast period.
            - **What to Expect:** Based on the trend:
                - **Rising Trend:** Indicates increasing resistance of the selected bacteria to the chosen antibiotic over time.
                - **Falling Trend:** Indicates decreasing resistance (higher susceptibility) of the bacteria to the antibiotic over time.
                - **Plateauing Trend:** Indicates that the resistance of the bacteria to the antibiotic is stable and not changing significantly over time.
            """)

        st.write("Note: Feel free to revisit this guide if you need assistance.")
        st.info("Happy forecasting!😊📈")


    st.subheader("Select preferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)

    # Forecast using Antimicrobial Resistance in Europe Data
    if selected_dataset == "Antimicrobial Resistance in Europe Data":

        data = euro_df
        st.subheader("Select Bacteria (Organism) and a corresponding Antibiotic")
        bacteria_selected = st.selectbox("Pick a bacteria " + picker_icon, data['Bacteria'].unique())
        anti_selected = st.selectbox("Pick a antibiotic " + picker_icon, data['Antibiotic'].unique())

        filtered_data = data[(data['Bacteria'] == bacteria_selected) & (data['Antibiotic'] == anti_selected)]
        filtered_data = filtered_data.drop(columns=['Bacteria', 'Antibiotic'])

        if filtered_data.empty:
            st.info(f"**{bacteria_selected}** does not apply to **{anti_selected}**")
        else:

            period = st.slider("Pick number of years for forecast:", 1, 10, 1)
            
            # Prepare features and target
            filtered_data['ds'] = pd.to_datetime(filtered_data['Time'], format='%Y') #.dt.year
            filtered_data['y'] = filtered_data['Value']

            # Encode categorical variables
            filtered_data['RegionName_encoded'] = pd.factorize(filtered_data['RegionName'])[0]
            filtered_data['Category_encoded'] = pd.factorize(filtered_data['Category'])[0]
            filtered_data['Distribution_encoded'] = pd.factorize(filtered_data['Distribution'])[0]

            forecast_btn = st.button("Make Your Forecast")
            
            # Get training 
            if forecast_btn:
                # Initialize the progress bar
                loading_text = st.text("Your forecast results would display soon...")
                progress = st.progress(0)
                

                # Artificially increment progress
                for i in range(0, 101, 10):
                    time.sleep(0.1) 
                    progress.progress(i)

                # Initialize the Prophet model
                model = Prophet()

                # Add regressors
                model.add_regressor('RegionName_encoded')
                model.add_regressor('Category_encoded')
                model.add_regressor('Distribution_encoded')

                # Fit the model
                model.fit(filtered_data[['ds', 'y', 'RegionName_encoded', 'Category_encoded', 'Distribution_encoded']])

                # Make future dataframe 
                future = model.make_future_dataframe(periods=period, freq='YE')

                
                future['RegionName_encoded'] = filtered_data['RegionName_encoded'].iloc[-1]
                future['Category_encoded'] = filtered_data['Category_encoded'].iloc[-1]
                future['Distribution_encoded'] = filtered_data['Distribution_encoded'].iloc[-1]


                # Predict future values
                forecast = model.predict(future)

                # Plot the results
                fig = plt.figure(figsize=(10, 6)) 
                ax = fig.add_subplot(111)
                fig = model.plot(forecast, ax=ax)
                ax.set_xlabel("Year")
                ax.set_ylabel("Resistance Value")
                ax.set_title(f"Forecast Resistance of {bacteria_selected} to {anti_selected}")

                st.pyplot(fig)

                # Remove the progress bar after completion
                progress.empty()
                loading_text.empty()


    # Forecast using Gram-Negative Bactirial Surveilance Data
    if selected_dataset == "Gram-Negative Bacterial Surveilance Data":
        
        st.subheader("Select Bacteria (Organism) and a corresponding Antibiotic")
        bacteria_list = gram_neg['Organism'].unique()
        bacteria = st.selectbox("Pick an Bacteria (Organism) " + picker_icon, bacteria_list)
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, ngram_antibiotic_list)

        forecast_period = st.slider("Pick number of years for forecast:", 1, 10, 1)
        

        utils.forecast_data(gram_neg, anti, bacteria, forecast_period)

    # Forecast using Gram-Positive Bactirial Surveilance Data
    if selected_dataset == "Gram-Positive Bacterial Surveilance Data":
        
        st.subheader("Select Antibiotic to forecast its resistance for")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, pgram_antibiotic_list)

        bacteria_list = gram_pos['Organism'].unique()
        st.subheader("Select corresponding bacteria (organism)")
        bacteria = st.selectbox("Pick an Bacteria (Organism) " + picker_icon, bacteria_list)

        forecast_period = st.slider("Pick number of years for forecast:", 1, 10, 1)
        
        utils.forecast_data(gram_pos, anti, bacteria, forecast_period)

    # Forecast using Atlas Negative Bacteria Data
    if selected_dataset == "Atlas Gram-Negative Bacteria Data":
        
        st.subheader("Select Antibiotic to forecast its resistance for")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, atlas_ngram_anti_list)

        bacteria_list = atlas_gram_neg['Species'].unique()
        st.subheader("Select corresponding bacteria (organism)")
        bacteria = st.selectbox("Pick an Bacteria (Organism) " + picker_icon, bacteria_list)

        forecast_period = st.slider("Pick number of years for forecast:", 1, 10, 1)
        
        utils.forecast_atlas(atlas_gram_neg, anti, bacteria, forecast_period, pheno=None)

    # Forecast using "Atlas Gram-Positive Bacteria Data"
    if selected_dataset == "Atlas Gram-Positive Bacteria Data":
        
        st.subheader("Select Antibiotic to forecast its resistance for")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, atlas_pgram_anti_list)

        bacteria_list = atlas_gram_pos['Species'].unique()
        st.subheader("Select corresponding bacteria (organism)")
        bacteria = st.selectbox("Pick an Bacteria (Organism) " + picker_icon, bacteria_list)

        forecast_period = st.slider("Pick number of years for forecast:", 1, 10, 1)
        
        utils.forecast_atlas(atlas_gram_pos, anti, bacteria, forecast_period, pheno="Phenotype")


if selected == "Make Prediction":

    # Prediction page Instructions Sidebar
    with st.sidebar:
        st.header("Welcome to the **Make Prediction** Page!")
        st.subheader("Instructions")
        
        with st.expander("**Step 1: Select Dataset for Analysis**"):
            st.write("""
            - **What to Do:** 
            - Use the first select box to choose the dataset you'd like to analyze. The available datasets include:
                - **Gram-Negative Bacterial Surveillance Data**
                - **Gram-Positive Bacterial Surveillance Data**
                - **Atlas Gram-Negative Bacteria Data**
                - **Atlas Gram-Positive Bacteria Data**
            - **What to Expect:** The selected dataset will be loaded and prepared for your forecast.

            """)

        with st.expander("**Step 2: Set Parameters and Conditions**"):
            st.write("""
            - **What to Do:** 
                - Use the select boxes and sliders to choose various parameters and conditions.
            - **What to Expect:** These selections will allow you to make the prediction.
            """)

        with st.expander("**Step 3: Make Prediction**"):
            st.write("""
            - **What to Do:** 
                - Click on the `Make Prediction` button to generate the prediction based on your selections.
            - **What to Expect:** The prediction would be made based on your selections. This might take a few seconds to minutes so just give it a little time.

            """)

        with st.expander("**Step 4: View Results**"):
            st.write("""
            - **Output:**
                - There would be a display message indicating whether the selected organism (bacteria) is:
                    - **Resistant**
                    - **Intermediate**
                    - **Susceptible**
                - to the selected antibiotic under the chosen conditions.
            """)

        st.write("Note: Feel free to revisit this guide if you need assistance.")
        st.info("Happy making your predictions!🕵️‍♀️🔍🤔")
    

    st.subheader("Select preferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets[-4:])

    # Gram-Negative Prediction
    if selected_dataset == "Gram-Negative Bacterial Surveilance Data":
        st.subheader("Select the factors below to make the prediction")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, ngram_antibiotic_list)

        utils.make_prediction(gram_neg, anti)
    
    # Gram-Postivie Prediction
    if selected_dataset == "Gram-Positive Bacterial Surveilance Data":
        st.subheader("Select the factors below to make the prediction")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, pgram_antibiotic_list)

        utils.make_prediction(gram_pos, anti)

    # Atlas Gram-Negative Prediction
    if selected_dataset == "Atlas Gram-Negative Bacteria Data":
        
        st.subheader("Select the factors below to make the prediction")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, atlas_ngram_anti_list)

        # Make Prediction
        utils.atlas_make_prediction(atlas_gram_neg, anti)

    # Atlas Gram-Postive Prediction
    if selected_dataset == "Atlas Gram-Positive Bacteria Data":
        
        st.subheader("Select the factors below to make the prediction")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, atlas_pgram_anti_list)

        # Make Prediction
        utils.atlas_make_prediction(atlas_gram_pos, anti)


if selected == "About":
    st.subheader("About the Team Members")
    # Image list
    image1 = "assets/Anthony.jpg"
    image2 = "assets/Gamah.jpg"
    image3 = "assets/4B.jpg"
    image4 = "assets/Dr_Ken.jpg"

    # Create a 2x2 grid
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Place the images in the grid
    with col1:
        st.image(image1, use_column_width=True)

    with col2:
        st.image(image2, use_column_width=True)

    with col3:
        st.image(image3, use_column_width=True)

    with col4:
        st.image(image4, use_column_width=True)

    st.subheader("About the Competition")
    st.image("assets/amr_logo.png", use_column_width=True)
    st.markdown(
        """
    The 2024 Vivli AMR Surveillance Data Challenge, funded by GARDP, Paratek, Pfizer, and Vivli, is a groundbreaking initiative aimed at harnessing the power of the Vivli AMR Register to combat antimicrobial resistance (AMR). 
    
    This challenge seeks to drive critical research, foster collaboration and innovation, and push the boundaries of AMR research. 
    By leveraging the Vivli AMR Register's comprehensive datasets, participants can contribute meaningfully to reshaping our understanding and approach to AMR.

    Read more about the 2024 Vivli AMR Surveillance Data Challenge [here](https://amr.vivli.org/data-challenge/data-challenge-overview/).
        """,
        unsafe_allow_html=True
    )

    st.subheader("About the Datasets")
    st.markdown("""
    ResistAI is powered by three comprehensive datasets: Pfizer's ATLAS Program, Paratek's KEYSTONE Program, and the EARS Dataset from Kaggle. 
    The ATLAS Program, spanning 83 countries, tracks antibiotic susceptibility, resistance trends, and emerging mechanisms, with over 917,000 antibiotic isolates and 21,000 antifungal isolates.
    The KEYSTONE Program, covering 27 countries, focuses on omadacycline's effectiveness against challenging pathogens, with over 90,000 isolates. 
    The EARS Dataset, sourced from European institutions, provides a diverse view of antimicrobial resistance across the continent.

    These datasets are meticulously curated, updated regularly, and integrated with advanced AI to deliver unparalleled insights into AMR. 
    ATLAS offers genotypic data and forward-looking surveillance, while KEYSTONE provides real-world clinical scenario data. 
    EARS Dataset offers a structured and readable format, allowing users to explore resistance trends by country, gender, or age. 
    Together, these datasets empower ResistAI users to make data-driven decisions, predict AMR dynamics, and stay informed about the latest trends in antimicrobial resistance.
    
    """)
    st.markdown(
        """
        1. Read more about the Pfizer's ATLAS Program dataset [here](https://amr.vivli.org/members/research-programs/).
        2. Read more about the Paratek's KEYSTONE Program dataset [here](https://amr.vivli.org/members/research-programs/).
        3. Read more about the EARS Dataset from Kaggle [here](https://www.kaggle.com/datasets/samfenske/euro-resistance).
        """,
        unsafe_allow_html=True
    )

    st.subheader("About the Web App")
    st.image("assets/resistAI_about_page.png", use_column_width=True)
    st.markdown("""
    ResistAI is a robust web application designed to support researchers, healthcare professionals, and data scientists in tackling antimicrobial resistance (AMR). 
    The app provides comprehensive tools for analyzing AMR data, training predictive models, forecasting trends, and making informed predictions. 
    The app's user-friendly interface offers interactive visualizations that reveal deeper insights, making data exploration intuitive and informative.
    
    ResistAI's machine learning capabilities enable users to train models with various algorithms, assess performance through metrics like accuracy, precision, recall, and F1-score, and download trained models for further use. 
    The forecasting feature allows users to predict AMR trends over time, helping to anticipate and mitigate future resistance challenges. 
    While ResistAI provides powerful predictive insights, it is important to note that these predictions should be used for study purposes only and not for clinical decision-making without consulting a domain expert. 
    ResistAI is a valuable tool in the global effort to understand and combat AMR, providing data-driven insights that can inform research, policy, and practice.
""")
