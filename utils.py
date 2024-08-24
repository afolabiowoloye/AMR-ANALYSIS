# Imports 
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from io import BytesIO

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Helper functions

# ATLAS DATA ANALYSIS
# Age Group Analysis
def atlas_age_group(df):
    top_10_countries = df['Age Group'].value_counts()

    fig = px.bar(top_10_countries, x=top_10_countries.index, y='count', 
                    title='Age Group Analysis',
                    labels={'x': top_10_countries.index, 'count': 'Frequency'})  

    return st.plotly_chart(fig)

# Atlas Patient Type
def atlas_patient_type(df):
    pat_type = df['Patient Type'].value_counts()

    fig = px.bar(pat_type, x=pat_type.index, y='count', 
                    title='Patient Type Analysis',
                    labels={'x': pat_type.index, 'count': 'Frequency'})  

    return st.plotly_chart(fig)

# Atlas Yearly Antibiotic Resistance
def atlas_anti_resistance_yearly(df, anti):

    anti_MIC = anti + "_MIC"
    resistance = df.groupby(['Year', anti_MIC])[anti_MIC].count().unstack().fillna(0)
    fig = px.bar(resistance, 
                    x=resistance.index, 
                    y=resistance.columns, 
                    title=f'Distribution of {anti} Resistance Over the Years',
                    labels={'x': 'Year', 'value': 'Frequency'},
                    color_discrete_map={'Resistant': 'red', 'Intermediate': 'orange', 'Susceptible': 'green'})
    return st.plotly_chart(fig)

# Atlas Yearly organism distribution
def atlas_org_by_year(df):
    organism_distribution = df.groupby(['Year', 'Species'])['Species'].count().unstack().fillna(0)

    fig = px.bar(organism_distribution, 
                    x=organism_distribution.index, 
                    y=organism_distribution.columns, 
                    title='Distribution of Organisms Over the Years',
                    labels={'x': 'Year', 'value': 'Number of Organisms'})
    return st.plotly_chart(fig)

# Atlas Age Group vs Organisms
def atlas_age_vs_org(df, col):
    fig = px.histogram(df, x="Age Group", color = col,
                        title="Distribution of Organisms by Age Group")
    fig.update_layout(yaxis_title="Number of Organisms")
    return st.plotly_chart(fig)

# Plot Organism by Family
def atlas_org_by_family(df, col):
    fig = px.histogram(df, x="Family", color=col, title="Distribution of Organisms by their Family")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    fig.update_xaxes(tickangle=-90)
    return st.plotly_chart(fig)

# Plot Antibiotic Resistance
def anti_resistance(df):
    mic_cols = [col for col in df.columns if col.endswith('MIC')]
    data = pd.DataFrame()

    for column in mic_cols:
        counts = df[column].value_counts().reset_index()
        counts.columns = ['Status', 'Count']
        counts['Antibiotic'] = column.replace('_MIC', '')
        data = pd.concat([data, counts], axis=0)

    fig = px.bar(data, x='Antibiotic', y='Count', color='Status', title='Antibiotic Resistance Distribution',
                    labels={'Count': 'Frequency', 'Antibiotic': 'Antibiotic'},
                    color_discrete_map={'Resistant': 'red', 'Intermediate': 'orange', 'Susceptible': 'green'})

    return st.plotly_chart(fig)

# Plot Organism Infection Source
def org_infection_source(df):
    fig = px.histogram(df, x="Infection Source", color="Organism", title="Distribution of Organisms by Source of Infection")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    fig.update_xaxes(tickangle=-90)
    return st.plotly_chart(fig)

# Plot Organism Specimen Type
def org_specimen_type(df):
    fig = px.histogram(df, x="Specimen Type", color="Organism", title="Distribution of Organisms by Type of Specimen")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    fig.update_xaxes(tickangle=-90)
    return st.plotly_chart(fig)
    
# Plot Organism Infection Types
def org_infection_type(df):
    fig = px.histogram(df, x="Infection Type", color="Organism", title="Distribution of Organisms by Type of Infection")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    fig.update_xaxes(tickangle=-90)
    return st.plotly_chart(fig)

# Plot Organism Distribution by Country
def org_by_country(df, col):
    fig = px.histogram(df, x="Country", color=col, title="Distribution of Organisms per Country of Study")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    fig.update_xaxes(tickangle=-90)
    return st.plotly_chart(fig)
    
# Plot Organism Resistance per Antibiotic
def org_vs_anti(df, anti, col):
    fig = px.histogram(df, x=anti + "_MIC", color= col, title=f"Distribution of Organisms per {anti}")
    fig.update_layout(yaxis_title="Number of Organisms",
                      xaxis_title=f"{anti} Resistance State")
    return st.plotly_chart(fig)

# Plot Organism per Gender
def org_per_gender(df, col):
    fig = px.histogram(df, x="Gender", color=col,
                   title="Distribution of Organisms by Gender",
                   labels={'Gender':'Gender'})
    fig.update_layout(yaxis_title="Number of Organisms")
    return st.plotly_chart(fig)

# Age Grouping 
def age_group(age):
    if age <= 5:
        return '0-5'
    elif 6 <= age <= 12:
        return '6-12'
    elif 13 <= age <= 19:
        return '13-19'
    elif 20 <= age <= 40:
        return '20-40'
    elif 41 <= age <= 60:
        return '41-60'
    else:
        return '60+'
    
# Plot Organism per Age Group
def org_vs_age(df):
    df['Age Group'] = df['Age'].apply(age_group)

    fig = px.histogram(df, x="Age Group", color="Organism",
                        title="Distribution of Organisms by Age Group")
    fig.update_layout(yaxis_title="Number of Organisms")
    return st.plotly_chart(fig)
    
# Continent Analysis
def continent_analysis(df):
    continents = df['Continent'].value_counts()

    fig = px.bar(continents, x=continents.index, y='count', 
                    title='Distribution of Continents of the Study',
                    labels={'x': continents.index, 'count': 'Frequency'})  

    return st.plotly_chart(fig)
    
# Country Analysis
def top_10_countries(df):
    top_10_countries = df['Country'].value_counts().head(10)

    fig = px.bar(top_10_countries, x=top_10_countries.index, y='count', 
                    title='Top 10 Countries of the Study',
                    labels={'x': top_10_countries.index, 'count': 'Frequency'})  

    return st.plotly_chart(fig)

# Top 10 Organisms
def top_10_organisms(df, col):
    top_10_organisms = df[col].value_counts().head(10)

    fig = px.bar(top_10_organisms, x=top_10_organisms.index, y='count', 
                    title='Top 10 Organisms in the Study',
                    labels={'x': top_10_organisms.index, 'count': 'Frequency'})  

    return st.plotly_chart(fig)
    
# Yearly organism distribution
def org_by_year(df):
    organism_distribution = df.groupby(['Study Year', 'Organism'])['Organism'].count().unstack().fillna(0)

    fig = px.bar(organism_distribution, 
                    x=organism_distribution.index, 
                    y=organism_distribution.columns, 
                    title='Distribution of Organisms Over the Years',
                    labels={'x': 'Study Year', 'value': 'Number of Organisms'})
    return st.plotly_chart(fig)

# Yearly Antibiotic Resistance
def anti_resistance_yearly(df, anti):

    anti_MIC = anti + "_MIC"
    resistance = df.groupby(['Study Year', anti_MIC])[anti_MIC].count().unstack().fillna(0)
    fig = px.bar(resistance, 
                    x=resistance.index, 
                    y=resistance.columns, 
                    title=f'Distribution of {anti} Resistance Over the Years',
                    labels={'x': 'Study Year', 'value': 'Frequency'},
                    color_discrete_map={'Resistant': 'red', 'Intermediate': 'orange', 'Susceptible': 'green'})
    return st.plotly_chart(fig)
    
# Plot age distribution
def age_distribution(df):
    fig = px.histogram(df, x="Age", title='Age Distribution')
    fig.update_layout(yaxis_title="Frequency")
    return st.plotly_chart(fig)

# Plot Gender Distribution
def gender_distribution(df):
    fig = px.histogram(df, x="Gender", 
                   title="Distribution of Gender",
                   labels={'Gender':'Patient Gender'})
    fig.update_layout(yaxis_title="Frequency")
    return st.plotly_chart(fig)

# Prepare data for forecasting
def forecast_data(df, anti, bacteria, period):
    df = df[df['Organism'] == bacteria]

    cols_to_use = ['Study Year', 'Continent', 'Country', 'Nosocomial',
               'Age', 'Gender', 'Infection Source', 'Infection Type', 'Specimen Type']
    
    cols_to_use.append(anti)
    df = df[cols_to_use]
   
    le = LabelEncoder()

    # Identify non-numerical columns in df
    non_numerical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Apply label encoding to non-numerical columns
    for col in non_numerical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Prepare features and target
    df['ds'] = pd.to_datetime(df['Study Year'], format='%Y') #.dt.year
    df['y'] = df[anti]

    forecast_btn = st.button("Make Your Forecast")
            
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
        model.add_regressor('Continent')
        model.add_regressor('Country')
        model.add_regressor('Nosocomial')
        model.add_regressor('Age')
        model.add_regressor('Gender')
        model.add_regressor('Infection Source')
        model.add_regressor('Infection Type')
        model.add_regressor('Specimen Type')


        # Fit the model
        model.fit(df[['ds', 'y', 'Continent', 'Country', 'Nosocomial', 'Age', 'Gender', 
                    'Infection Source', 'Infection Type', 'Specimen Type']])

        # Make future dataframe for the next 5 years
        future = model.make_future_dataframe(periods=period, freq='YE')

        
        future['Continent'] = df['Continent'].iloc[-1]
        future['Country'] = df['Country'].iloc[-1]
        future['Nosocomial'] = df['Nosocomial'].iloc[-1]
        future['Age'] = df['Age'].iloc[-1]
        future['Gender'] = df['Gender'].iloc[-1]
        future['Infection Source'] = df['Infection Source'].iloc[-1]
        future['Infection Type'] = df['Infection Type'].iloc[-1]
        future['Specimen Type'] = df['Specimen Type'].iloc[-1]


        # Predict future values
        forecast = model.predict(future)

        

        # Plot the results
        fig = plt.figure(figsize=(10, 6))  
        ax = fig.add_subplot(111)
        fig = model.plot(forecast, ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Resistance Value")
        ax.set_title(f"Forecast for {bacteria} with {anti}")

        st.pyplot(fig)

        # Remove the progress bar after completion
        progress.empty()
        loading_text.empty()


# User Trained Data
def model_training(model_selected, df, anti):
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

    cols_to_use = ['Study Year', 'Organism', 'Continent', 'Country', 'Nosocomial',
                   'Age', 'Gender', 'Infection Source', 'Infection Type', 'Specimen Type']

    anti_MIC = anti + "_MIC"
    cols_to_use.append(anti_MIC)
    df = df[cols_to_use]

    label_mapping = {'Resistant': 2, 'Intermediate': 1, 'Susceptible': 0}
    df[anti_MIC] = df[anti_MIC].map(label_mapping)

    X = df.drop(anti_MIC, axis=1)
    y = df[anti_MIC]

    # Apply label encoding to non-numerical columns in X
    le = LabelEncoder()
    non_numerical_cols = X.select_dtypes(include=['object', 'category']).columns

    for col in non_numerical_cols:
        X[col] = le.fit_transform(X[col])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
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

        # Make prediction
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100

        # Print metrics as percentages
        st.subheader("Prediction Score")
        st.write(f"Accuracy: **{accuracy:.2f}%**")
        st.write(f"Precision: **{precision:.2f}%**")
        st.write(f"Recall: **{recall:.2f}%**")
        st.write(f"F1-score: **{f1:.2f}%**")

        # Confusion Matrix Analysis
        cm = confusion_matrix(y_test, y_pred)

        # Mapping for labels
        label_mapping = {2: 'Resistant', 1: 'Intermediate', 0: 'Susceptible'}

        # Convert to DataFrame and map labels
        cm_df = pd.DataFrame(cm)
        cm_df.index = cm_df.index.map(label_mapping)
        cm_df.columns = cm_df.columns.map(label_mapping)

        # Plot confusion matrix using Plotly Express
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=px.colors.sequential.Plasma_r,
                        labels={'x': 'Predicted Label', 'y': 'Actual Label', 'color': 'Count'},
                        title='Confusion Matrix')
        # Update layout
        fig.update_xaxes(side="bottom")
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            title="Confusion Matrix for Classification Model"
        )

        st.plotly_chart(fig)

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

# Make Prediction Function
def make_prediction(df, anti):
    cols_to_use = ['Study Year', 'Organism', 'Continent', 'Country', 'Nosocomial',
                   'Age', 'Gender', 'Infection Source', 'Infection Type', 'Specimen Type']
    
    anti_MIC = anti + "_MIC"
    cols_to_use.append(anti_MIC)
    df = df[cols_to_use]

    label_mapping = {'Resistant': 2, 'Intermediate': 1, 'Susceptible': 0}
    df[anti_MIC] = df[anti_MIC].map(label_mapping)
    
    X = df.drop(anti_MIC, axis=1)
    y = df[anti_MIC]

    # Set encoders
    organism_encoder = LabelEncoder()
    continent_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    nosocomial_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    infection_source_encoder = LabelEncoder()
    infection_type_encoder = LabelEncoder()
    specimen_encoder = LabelEncoder()

    # Encode variables
    X['Organism'] = organism_encoder.fit_transform(X['Organism'])
    X['Continent'] = continent_encoder.fit_transform(X['Continent'])
    X['Country'] = country_encoder.fit_transform(X['Country'])
    X['Nosocomial'] = nosocomial_encoder.fit_transform(X['Nosocomial'])
    X['Gender'] = gender_encoder.fit_transform(X['Gender'])
    X['Infection Source'] = infection_source_encoder.fit_transform(X['Infection Source'])
    X['Infection Type'] = infection_type_encoder.fit_transform(X['Infection Type'])
    X['Specimen Type'] = specimen_encoder.fit_transform(X['Specimen Type'])

    gbc = GradientBoostingClassifier()
    gbc.fit(X, y)

    # Define some terms
    organisms = df['Organism'].unique()
    continents = df['Continent'].unique()
    countries = df['Country'].unique()
    nosocomial = df['Nosocomial'].unique()
    gender = df['Gender'].unique()
    infection_source = df['Infection Source'].unique()
    infection_type = df['Infection Type'].unique()
    specimen = df['Specimen Type'].unique()

    # Selections
    year = st.slider("Pick year of study:", min_value=2010, max_value=2030, value=2010, step=1)
    organisms_selected = st.selectbox("Select bacteria (organism) under study: ", organisms)
    continent_selected = st.selectbox("Select continent of study: ", continents)
    country_selected = st.selectbox("Select country of study: ", countries)
    nosocomial_selected = st.selectbox("Select Nosocomial type: ", nosocomial)
    age = st.slider("Pick age:", 0, 150, 1)
    gender_selected = st.selectbox("Select gender: ", gender)
    infection_source_selected = st.selectbox("Select the source of infection: ", infection_source)
    infection_type_selected = st.selectbox("Select the type of infection: ", infection_type)
    specimen_selected = st.selectbox("Select the type of Specimen: ", specimen)
    pred_btn = st.button("Make Prediction")

    # Get prediction
    if pred_btn:
        # Initialize the progress bar
        loading_text = st.text("Your prediction is loading...")
        progress = st.progress(0)
        

        # Artificially increment progress
        for i in range(0, 101, 10):
            time.sleep(0.1) 
            progress.progress(i)

        X_test = np.array([[
            year,
            organisms_selected,
            continent_selected,
            country_selected,
            nosocomial_selected,
            age,
            gender_selected,
            infection_source_selected,
            infection_type_selected,
            specimen_selected,
        ]])

        # Apply encoding
        X_test[:,1] = organism_encoder.transform(X_test[:,1])
        X_test[:,2] = continent_encoder.transform(X_test[:,2])
        X_test[:,3] = country_encoder.transform(X_test[:,3])
        X_test[:,4] = nosocomial_encoder.transform(X_test[:,4])
        X_test[:,6] = gender_encoder.transform(X_test[:,6])
        X_test[:,7] = infection_source_encoder.transform(X_test[:,7])
        X_test[:,8] = infection_type_encoder.transform(X_test[:,8])
        X_test[:,9] = specimen_encoder.transform(X_test[:,9])

        X_test = X_test.astype(float)

        pred = gbc.predict(X_test)

        # Remove the progress bar after completion
        progress.empty()
        loading_text.empty()

        
        if pred[0] == 0:
            st.write(f"The bacteria (organism) **{organisms_selected}** would be **Susceptible** to the antibiotic **{anti}** on the conditions selected")
            st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please consult a domain expert before making any decisions based on these predictions.")
        elif pred[0] == 1:
            st.write(f"The bacteria (organism) **{organisms_selected}** would have **Intermediate Resistance** to the antibiotic **{anti}** on the conditions selected")
            st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please consult a domain expert before making any decisions based on these predictions.")
        elif pred[0] == 2:
            st.write(f"The bacteria (organism) **{organisms_selected}** would be **Resistant** to the antibiotic **{anti}** on the conditions selected")
            st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please consult a domain expert before making any decisions based on these predictions.")



# Forecast Atlas
def forecast_atlas(df, anti, bacteria, period, pheno=None):
    # Filter the DataFrame by the selected bacteria species
    df = df[df['Species'] == bacteria]
    
    # Define the common columns to use
    base_cols = ['Study', 'Family', 'Country', 'Gender', 'Age Group', 'Speciality', 'Source', 'Patient Type', 'Year']
    
    # If pheno is provided, add the 'Phenotype' column
    if pheno:
        base_cols.append('Phenotype')
    
    # Add the antibiotic column to the list
    cols_to_use = base_cols + [anti]
    
    # Filter the DataFrame to use only the selected columns
    df = df[cols_to_use]
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Identify and encode non-numerical columns
    non_numerical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[non_numerical_cols] = df[non_numerical_cols].apply(le.fit_transform)
    
    # Prepare features and target for Prophet
    df['ds'] = pd.to_datetime(df['Year'], format='%Y')
    df['y'] = df[anti]
    
    # Check if the DataFrame has enough data points
    if len(df) < 10:
        st.info(f"There is no much study on {bacteria} with {anti}")
    else:
        # Display the forecast button
        forecast_btn = st.button("Make Your Forecast")
        
        if forecast_btn:
            # Initialize the progress bar
            loading_text = st.text("Your forecast results would display soon...")
            progress = st.progress(0)
            
            # Artificially increment progress
            for i in range(0, 101, 10):
                time.sleep(0.1)
                progress.progress(i)
            
            # Initialize and configure the Prophet model
            model = Prophet()
            for col in base_cols:
                if col != 'Year':  # Skip 'Year' as it's used for 'ds'
                    model.add_regressor(col)
            
            # Fit the model
            model.fit(df[['ds', 'y'] + base_cols])
            
            # Make future dataframe for the next 'period' years
            future = model.make_future_dataframe(periods=period, freq='YE')
            
            # Assign the last available value of each regressor to the future dataframe
            for col in base_cols:
                future[col] = df[col].iloc[-1]
            
            # Predict future values
            forecast = model.predict(future)
            
            # Plot the forecast results
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            fig = model.plot(forecast, ax=ax)
            ax.set_xlabel("Year")
            ax.set_ylabel("Resistance Value")
            ax.set_title(f"Forecast for {bacteria} with {anti}")
            st.pyplot(fig)
            
            # Remove the progress bar after completion
            progress.empty()
            loading_text.empty()



# Atlas User Trained Data
def atlas_model_training(model_selected, df, anti, pheno = None):
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

    # Define the common columns to use
    base_cols = ['Study', 'Species', 'Family', 'Country', 'Gender', 'Age Group', 'Speciality', 'Source', 'Patient Type', 'Year']
    
    # If pheno is provided, add the 'Phenotype' column
    if pheno:
        base_cols.append('Phenotype')
    
    # Add the antibiotic column to the list
    anti_MIC = anti + "_MIC"
    #base_cols.append(anti_MIC)
    cols_to_use = base_cols + [anti_MIC]
    
    # Filter the DataFrame to use only the selected columns
    df = df[cols_to_use]

    # Map Antibiotic MIC column
    label_mapping = {'Resistant': 2, 'Intermediate': 1, 'Susceptible': 0}
    df[anti_MIC] = df[anti_MIC].map(label_mapping)

    X = df.drop(anti_MIC, axis=1)
    y = df[anti_MIC]

    # Apply label encoding to non-numerical columns in X
    le = LabelEncoder()
    non_numerical_cols = X.select_dtypes(include=['object', 'category']).columns

    for col in non_numerical_cols:
        X[col] = le.fit_transform(X[col])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
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

        # Make prediction
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100

        # Print metrics as percentages
        st.subheader("Prediction Score")
        st.write(f"Accuracy: **{accuracy:.2f}%**")
        st.write(f"Precision: **{precision:.2f}%**")
        st.write(f"Recall: **{recall:.2f}%**")
        st.write(f"F1-score: **{f1:.2f}%**")

        # Confusion Matrix Analysis
        cm = confusion_matrix(y_test, y_pred)

        # Mapping for labels
        label_mapping = {2: 'Resistant', 1: 'Intermediate', 0: 'Susceptible'}

        # Convert to DataFrame and map labels
        cm_df = pd.DataFrame(cm)
        cm_df.index = cm_df.index.map(label_mapping)
        cm_df.columns = cm_df.columns.map(label_mapping)

        # Plot confusion matrix using Plotly Express
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=px.colors.sequential.Plasma_r,
                        labels={'x': 'Predicted Label', 'y': 'Actual Label', 'color': 'Count'},
                        title='Confusion Matrix')
        # Update layout
        fig.update_xaxes(side="bottom")
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            title="Confusion Matrix for Classification Model"
        )

        st.plotly_chart(fig)

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


# Atlas - Make Prediction Function
def atlas_make_prediction(df, anti):
    # Define the common columns to use
    base_cols = ['Study', 'Species', 'Family', 'Country', 'Gender', 'Age Group', 'Speciality', 'Source', 'Patient Type', 'Year']
        
    # Add the antibiotic column to the list
    anti_MIC = anti + "_MIC"
    cols_to_use = base_cols + [anti_MIC]
    
    # Filter the DataFrame to use only the selected columns
    df = df[cols_to_use]

    label_mapping = {'Resistant': 2, 'Intermediate': 1, 'Susceptible': 0}
    df[anti_MIC] = df[anti_MIC].map(label_mapping)
    
    X = df.drop(anti_MIC, axis=1)
    y = df[anti_MIC]

    # Set encoders
    study_encoder = LabelEncoder()
    species_encoder = LabelEncoder()
    family_encoder = LabelEncoder()
    country_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    age_group_encoder = LabelEncoder()
    speciality_encoder = LabelEncoder()
    source_encoder = LabelEncoder()
    patient_type_encoder = LabelEncoder()

    # Encode variables
    X['Study'] = study_encoder.fit_transform(X['Study'])
    X['Species'] = species_encoder.fit_transform(X['Species'])
    X['Family'] = family_encoder.fit_transform(X['Family'])
    X['Country'] = country_encoder.fit_transform(X['Country'])
    X['Gender'] = gender_encoder.fit_transform(X['Gender'])
    X['Age Group'] = age_group_encoder.fit_transform(X['Age Group'])
    X['Speciality'] = speciality_encoder.fit_transform(X['Speciality'])
    X['Source'] = source_encoder.fit_transform(X['Source'])
    X['Patient Type'] = patient_type_encoder.fit_transform(X['Patient Type'])

    gbc = GradientBoostingClassifier()
    gbc.fit(X, y)


    # Selections
    year = st.slider("Pick year of study:", min_value=2000, max_value=2030, value=2010, step=1)
    study_selected =  st.selectbox("Select the type of Study done: ", df['Study'].unique())
    species_selected = st.selectbox("Select bacteria (organism) under study: ", df['Species'].unique())
    family_selected = st.selectbox("Select the family that bacteria (organism) under study belong to: ", df['Family'].unique())
    country_selected = st.selectbox("Select country of study: ", df['Country'].unique())
    gender_selected = st.selectbox("Select gender: ", df['Gender'].unique())
    age_group_selected = st.selectbox("Select the age group: ", df['Age Group'].unique())
    speciality_selected = st.selectbox("Select the kind of Specialty: ", df['Speciality'].unique())
    source_selected = st.selectbox("Select Source: ", df['Source'].unique())
    patient_type_selected = st.selectbox("Select the patient type: ", df['Patient Type'].unique())
    pred_btn = st.button("Make Prediction")

    # Get prediction
    if pred_btn:
        # Initialize the progress bar
        loading_text = st.text("Your prediction is loading...")
        progress = st.progress(0)
        

        # Artificially increment progress
        for i in range(0, 101, 10):
            time.sleep(0.1) 
            progress.progress(i)



        X_test = np.array([[
            study_selected,
            species_selected,
            family_selected,
            country_selected,
            gender_selected,
            age_group_selected,
            speciality_selected,
            source_selected,
            patient_type_selected,
            year
        ]])

        # Apply encoding
        X_test[:,0] = study_encoder.transform(X_test[:,0])
        X_test[:,1] = species_encoder.transform(X_test[:,1])
        X_test[:,2] = family_encoder.transform(X_test[:,2])
        X_test[:,3] = country_encoder.transform(X_test[:,3])
        X_test[:,4] = gender_encoder.transform(X_test[:,4])
        X_test[:,5] = age_group_encoder.transform(X_test[:,5])
        X_test[:,6] = speciality_encoder.transform(X_test[:,6])
        X_test[:,7] = source_encoder.transform(X_test[:,7])
        X_test[:,8] = patient_type_encoder.transform(X_test[:,8])

        X_test = X_test.astype(float)

        pred = gbc.predict(X_test)

        # Remove the progress bar after completion
        progress.empty()
        loading_text.empty()

        
        if pred[0] == 0:
            st.write(f"The bacteria (organism) **{species_selected}** would be **Susceptible** to the antibiotic **{anti}** on the conditions selected")
            st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please consult a domain expert before making any decisions based on these predictions.")
        elif pred[0] == 1:
            st.write(f"The bacteria (organism) **{species_selected}** would have **Intermediate Resistance** to the antibiotic **{anti}** on the conditions selected")
            st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please consult a domain expert before making any decisions based on these predictions.")
        elif pred[0] == 2:
            st.write(f"The bacteria (organism) **{species_selected}** would be **Resistant** to the antibiotic **{anti}** on the conditions selected")
            st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please consult a domain expert before making any decisions based on these predictions.")
