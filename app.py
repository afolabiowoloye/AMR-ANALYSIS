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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import numpy as np
import pickle
from io import BytesIO
import utils


#-----------Web page setting-------------------#
page_title = "AMR Web App"
page_icon = "🦠🧬"
viz_icon = "📊"
stock_icon = "📋"
picker_icon = "👇"
#layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon)

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
    #euro_link_id = "1aNulNFQQzvoDh75hbtZJ7_QcW4fRl9rs"
    #euro_link = f'https://drive.google.com/uc?id={euro_link_id}'
    #euro_df = pd.read_csv(euro_link)
    euro_df = pd.read_csv("ecdc.csv")
    euro_df['Distribution'] = euro_df['Distribution'].str.split(',').str[1].str.split(' ').str[-1]
    euro_df = euro_df.drop(columns = ['Unit', 'RegionCode', 'Unnamed: 0'], axis = 1)
    return euro_df

euro_df = load_euro_data()

@st.cache_data
# Omadacycline Gram-Negative Data
def load_oma_ngram():
    #negative_link_id = "1jHO-NFMsauUGVx9pfW5RPMvNGc6G0wAT"
    #negative_link = f'https://drive.google.com/uc?id={negative_link_id}'
    #gram_neg = pd.read_csv(negative_link)
    gram_neg = pd.read_csv("omad_gram_neg_cleaned.csv")
    gram_neg = gram_neg.rename(columns={'Piperacillin-\ntazobactam': 'Piperacillin-tazobactam'})
    gram_neg = gram_neg.rename(columns={'Piperacillin-\ntazobactam_MIC': 'Piperacillin-tazobactam_MIC'})
    gram_neg = gram_neg[gram_neg['Age'] < 200]
    gram_neg['Gender'] = gram_neg['Gender'].replace({'M': 'Male', 'F': 'Female'})

    return gram_neg

gram_neg = load_oma_ngram()

@st.cache_data
# Omadacycline Gram-Negative Data
def load_oma_pgram():
    #positive_link_id = "1NHR41hfyCN26EmQ7SrLCrhSBJnAVDQo0"
    #positive_link = f'https://drive.google.com/uc?id={positive_link_id}'
    #gram_pos = pd.read_csv(positive_link)
    gram_pos = pd.read_csv("omad_gram_pos_cleaned.csv")
    gram_pos = gram_pos[gram_pos['Age'] < 200]
    gram_pos['Gender'] = gram_pos['Gender'].replace({'M': 'Male', 'F': 'Female'})
    return gram_pos

gram_pos = load_oma_pgram()


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

# Home page
if selected == "Home":
    st.subheader("Welcome to AMR Web App")
    st.write("Some dummy texts here")


if selected == "Analysis":
    datasets = ["Antimicrobial Resistance in Europe Data", "Gram-Negative Bactirial Surveilance Data", 
                "Gram-Positive Bactirial Surveilance Data"]
    st.subheader("Select oreferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)


    if selected_dataset == "Antimicrobial Resistance in Europe Data":

        analysis = ["Descriptive Statistics", "Resistance Trend Analysis", "Comparative Analysis",
                    "Demographic Analysis", "Bacteria Analysis", "Antibiotics Analysis"]
        st.subheader("Select analysis")
        selected_analysis = st.selectbox("Pick analysis type " + picker_icon, analysis)

        if selected_analysis == "Demographic Analysis":
            data = euro_df
            data['Distribution'] = data['Distribution'].str.split(',').str[1].str.split(' ').str[-1]
            age_subset = data[data['Distribution'] == "age"]
            age_count = age_subset['Category'].value_counts()


            fig = px.bar(age_subset, x='Category', title='Age Group Distribution',
                        labels={'Category': 'Age Group', 'count': 'Frequency'})
            st.plotly_chart(fig)
            st.info("Add summary analysis here!")
        
        if selected_analysis == "Descriptive Statistics":
            fig = px.histogram(
                euro_df, 
                x='Value', 
                nbins=30, 
                title='Histogram of Resistance Percentages',
                labels={'Value': 'Resistance Percentage'}, 
                color_discrete_sequence=['blue']
            )

            fig.update_layout(
                xaxis_title='Resistance Percentage',
                yaxis_title='Frequency',
                bargap=0.1,
                height=600
            )

            st.plotly_chart(fig)

            st.info("We can add some summary analysis here.")

            st.subheader("Bar Chart for Categorical Variables")
            st.write("Bar Chart of Bacteria Types")
            bacteria_count = euro_df['Bacteria'].value_counts()
            st.bar_chart(bacteria_count)
            st.info("Add summary analysis here!")

            st.write("Bar Chart of Antibiotic Types")
            antibiotic_count = euro_df['Antibiotic'].value_counts()
            st.bar_chart(antibiotic_count)
            st.info("Add summary analysis here!")

            st.subheader("Pie Chart of Distribution by Category")
            st.write("Pie Chart of Distribution by Category")
            gender_distribution = euro_df['Category'].value_counts()
            fig = px.pie(names=gender_distribution.index, values=gender_distribution.values)
            st.plotly_chart(fig)
            st.info("Add summary analysis here!")
        
        if selected_analysis == "Resistance Trend Analysis":
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
        
            st.info("Add summary analysis here!")

        if selected_analysis == "Comparative Analysis":
            st.subheader("Antibiotic Efficacy Comparison")
            fig1 = px.box(
                euro_df,
                x='Antibiotic',
                y='Value',
                title='Antibiotic Efficacy Comparison',
                labels={'Antibiotic': 'Antibiotic', 'Value': 'Resistance Percentage'},
                category_orders={'Antibiotic': sorted(euro_df['Antibiotic'].unique())}
            )

            fig1.update_layout(
                title={'text': 'Resistance Trends Over Time', 'x': 0.5},
                xaxis_title='Year',
                yaxis_title='Resistance Percentage',
                height=800,  
                width=800   
            )

            st.plotly_chart(fig1)
            st.info("Add summary analysis here!")

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
            st.info("Add summary analysis here!")
    
    # Gram-Negative Data Analysis
    if selected_dataset == "Gram-Negative Bactirial Surveilance Data":

        analysis = ["Demographic Analysis", "Bacteria (Orgamism) Analysis", "Antibiotics Analysis"]

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
            # Distribution of Organisms per Country
            utils.org_by_country(gram_neg)

            #Organism per Gender
            utils.org_per_gender(gram_neg)

            #Organism per Age Group
            utils.org_vs_age(gram_neg)

            # Yearly organism distribution
            utils.org_by_year(gram_neg)

            # Distribution of Organism per antibiotic
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, ngram_antibiotic_list)
            utils.org_vs_anti(gram_neg, anti)

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
    if selected_dataset == "Gram-Positive Bactirial Surveilance Data":

        analysis = ["Demographic Analysis", "Bacteria (Orgamism) Analysis", "Antibiotics Analysis"]

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
            # Distribution of Organisms per Country
            utils.org_by_country(gram_pos)

            #Organism per Gender
            utils.org_per_gender(gram_pos)

            #Organism per Age Group
            utils.org_vs_age(gram_pos)

            # Yearly organism distribution
            utils.org_by_year(gram_pos)

            # Distribution of Organism per antibiotic
            st.subheader("Select an antibiotic")
            anti = st.selectbox("Pick an Antibioticn " + picker_icon, pgram_antibiotic_list)
            utils.org_vs_anti(gram_pos, anti)

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


# Train Model Page
if selected == "Train Model":
    datasets = ["Antimicrobial Resistance in Europe Data", "Gram-Negative Bactirial Surveilance Data", 
                "Gram-Positive Bactirial Surveilance Data"]
    st.subheader("Select oreferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)


    if selected_dataset == "Antimicrobial Resistance in Europe Data":
        # Dummy feature encoding
        data_encoded = euro_df
        data_encoded = pd.get_dummies(euro_df, columns=['Distribution', 'RegionName', 'Bacteria', 'Antibiotic', 'Category'])
        # Feature and target
        X = data_encoded.drop('Value', axis=1)
        y = (euro_df['Value'] > 50).astype(int)  # Binary classification: 0 for non-resistant, 1 for resistant

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
            
            # Train the model
            model.fit(X_train, y_train)

            # Feature importance (if applicable)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Prepare data for Plotly Express
                importance_df = pd.DataFrame({
                    'Feature': [X_train.columns[i] for i in indices],
                    'Importance': importances[indices]
                })

                # Clean up feature names
                def clean_feature_name(name):
                    parts = name.split('_')
                    return parts[-1] if len(parts) > 1 else name
                
                importance_df['Cleaned_Feature'] = importance_df['Feature'].apply(clean_feature_name)

                st.subheader("Feature Importance Analysis")
                # Plot using Plotly Express
                fig = px.bar(importance_df.head(10), x='Cleaned_Feature', y='Importance', title='Top 10 Most Important Features',
                            labels={'Cleaned_Feature': 'Feature Name', 'Importance': 'Importance Value'})
                fig.update_xaxes(tickangle=-90)
                st.plotly_chart(fig)

            # Predictions
            st.subheader("Prediction Score")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_percentage = accuracy * 100

            st.markdown(f"Accuracy Score: **{accuracy_percentage:.2f}%**")

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
    if selected_dataset == "Gram-Negative Bactirial Surveilance Data":        
        st.subheader("Select Algorithm")
        model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
        
        st.subheader("Select Antibiotic to train model for")
        anti = st.selectbox("Pick an Antibioticn " + picker_icon, ngram_antibiotic_list)

        utils.model_training(model_selected, gram_neg, anti)

    # Gram-Positive Data Training
    if selected_dataset == "Gram-Positive Bactirial Surveilance Data":        
        st.subheader("Select Algorithm")
        model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
        
        st.subheader("Select Antibiotic to train model for")
        anti = st.selectbox("Pick an Antibioticn " + picker_icon, pgram_antibiotic_list)

        utils.model_training(model_selected, gram_pos, anti)


if selected == "Make a Forecast":

    datasets = ["Antimicrobial Resistance in Europe Data", "Gram-Negative Bactirial Surveilance Data", 
                "Gram-Positive Bactirial Surveilance Data"]
    st.subheader("Select oreferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)

    # Forecast using Antimicrobial Resistance in Europe Data
    if selected_dataset == "Antimicrobial Resistance in Europe Data":

        data = euro_df
        data = data.drop(['Unnamed: 0', 'Unit', 'RegionCode'], axis=1)
        data['Distribution'] = data['Distribution'].str.split(',').str[1].str.split(' ').str[-1]
        st.subheader("Select Bacteria and a corresponding Antibiotic")
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

    # Forecast using Gram-Negative Bactirial Surveilance Data
    if selected_dataset == "Gram-Negative Bactirial Surveilance Data":
        
        st.subheader("Select Antibiotic to forecast its resistance for")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, ngram_antibiotic_list)

        bacteria_list = gram_neg['Organism'].unique()
        st.subheader("Select corresponding bacteria (organism)")
        bacteria = st.selectbox("Pick an Bacteria (Organism) " + picker_icon, bacteria_list)

        forecast_period = st.slider("Pick number of years for forecast:", 1, 10, 1)
        

        utils.forecast_data(gram_neg, anti, bacteria, forecast_period)

    # Forecast using Gram-Positive Bactirial Surveilance Data
    if selected_dataset == "Gram-Positive Bactirial Surveilance Data":
        
        st.subheader("Select Antibiotic to forecast its resistance for")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, pgram_antibiotic_list)

        bacteria_list = gram_pos['Organism'].unique()
        st.subheader("Select corresponding bacteria (organism)")
        bacteria = st.selectbox("Pick an Bacteria (Organism) " + picker_icon, bacteria_list)

        forecast_period = st.slider("Pick number of years for forecast:", 1, 10, 1)
        
        utils.forecast_data(gram_pos, anti, bacteria, forecast_period)


if selected == "Make Prediction":
    
    datasets = ["Gram-Negative Bactirial Surveilance Data", "Gram-Positive Bactirial Surveilance Data"]
    st.subheader("Select oreferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)

    # Gram-Negative Prediction
    if selected_dataset == "Gram-Negative Bactirial Surveilance Data":
        st.subheader("Select Antibiotic to forecast its resistance for")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, ngram_antibiotic_list)

        utils.make_prediction(gram_neg, anti)
    
    # Gram-Postivie Prediction
    if selected_dataset == "Gram-Positive Bactirial Surveilance Data":
        st.subheader("Select Antibiotic to forecast its resistance for")
        anti = st.selectbox("Pick an Antibiotic " + picker_icon, pgram_antibiotic_list)

        utils.make_prediction(gram_pos, anti)


if selected == "About":
    st.markdown("About the Team Members")
    st.markdown("About the Competition")
    st.markdown("About the Web App")


