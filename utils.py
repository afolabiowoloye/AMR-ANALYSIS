# Imports 
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import pickle
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Helper functions
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
                    labels={'Count': 'Value Count', 'Antibiotic': 'Antibiotic'},
                    color_discrete_map={'Resistant': 'red', 'Intermediate': 'orange', 'Susceptible': 'green'})

    return st.plotly_chart(fig)

# Plot Organism Infection Source
def org_infection_source(df):
    fig = px.histogram(df, x="Infection Source", color="Organism", title="Distribution of Organisms by Type of Infection")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    return st.plotly_chart(fig)

# Plot Organism Specimen Type
def org_specimen_type(df):
    fig = px.histogram(df, x="Specimen Type", color="Organism", title="Distribution of Organisms by Type of Specimen")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    return st.plotly_chart(fig)
    
# Plot Organism Infection Types
def org_infection_type(df):
    fig = px.histogram(df, x="Infection Type", color="Organism", title="Distribution of Organisms by Type of Infection")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    return st.plotly_chart(fig)

# Plot Organism Distribution by Country
def org_by_country(df):
    fig = px.histogram(df, x="Country", color="Organism", title="Distribution of Organisms per Country of Study")
    fig.update_layout(yaxis_title="Number of Organisms",
                      #height=1200,  
                    width=800)
    return st.plotly_chart(fig)
    
# Plot Organism Resistance per Antibiotic
def org_vs_anti(df, anti):
    fig = px.histogram(df, x=anti + "_MIC", color="Organism", title=f"Distribution of Organisms per {anti}")
    fig.update_layout(yaxis_title="Number of Organisms")
    return st.plotly_chart(fig)

# Plot Organism per Gender
def org_per_gender(df):
    fig = px.histogram(df, x="Gender", color="Organism",
                   title="Distribution of Organisms by Gender",
                   labels={'Gender':'Patient Gender'})
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
    
# Plot Chloroplet
def anti_per_continent(df, anti):
    # Europe
    fig_europe = px.choropleth(df, 
                           locations='Country', 
                           locationmode='country names', 
                           color=anti, 
                           hover_name='Country', 
                           color_continuous_scale=px.colors.sequential.Plasma_r, 
                           title=f'{anti} Resistance in Europe', 
                           scope='europe')
    # North America
    fig_na = px.choropleth(df, 
                           locations='Country', 
                           locationmode='country names', 
                           color=anti, 
                           hover_name='Country', 
                           color_continuous_scale=px.colors.sequential.Plasma_r, 
                           title=f'{anti} Resistance in North America', 
                           scope='north america')
    return st.plotly_chart(fig_europe), st.plotly_chart(fig_na)

# Yearly organism distribution
def org_by_year(df):
    organism_distribution = df.groupby(['Study Year', 'Organism'])['Organism'].count().unstack().fillna(0)

    fig = px.bar(organism_distribution, 
                    x=organism_distribution.index, 
                    y=organism_distribution.columns, 
                    title='Distribution of Organisms Over the Years',
                    labels={'x': 'Study Year', 'value': 'Count'})
    return st.plotly_chart(fig)

# Yearly Antibiotic Resistance
def anti_resistance_yearly(df, anti):

    anti_MIC = anti + "_MIC"
    resistance = df.groupby(['Study Year', anti_MIC])[anti_MIC].count().unstack().fillna(0)
    fig = px.bar(resistance, 
                    x=resistance.index, 
                    y=resistance.columns, 
                    title=f'Distribution of {anti} Resistance Over the Years',
                    labels={'x': 'Study Year', 'value': 'Count'},
                    color_discrete_map={'Resistant': 'red', 'Intermediate': 'orange', 'Susceptible': 'green'})
    return st.plotly_chart(fig)
    
# Plot age distribution
def age_distribution(df):
    fig = px.histogram(df, x="Age", title='Age Distribution')
    fig.update_layout(yaxis_title="Count")
    return st.plotly_chart(fig)

# Plot Gender Distribution
def gender_distribution(df):
    fig = px.histogram(df, x="Gender", 
                   title="Distribution of Gender",
                   labels={'Gender':'Patient Gender'})
    fig.update_layout(yaxis_title="Count")
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


        #fig.show()
    return st.pyplot(fig)


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

    # Train the model
    model.fit(X_train, y_train)

    # Make prediction
    y_pred = model.predict(X_test)

    # Calculate metrics
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
    
    X = df.drop(anti_MIC, axis = 1)
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

    
    dec = DecisionTreeClassifier()
    dec.fit(X, y)

    # Define some terms
    organisms = df['Organism'].unique()
    continents = df['Continent'].unique()
    countries = df['Country'].unique()
    nosocomial = df['Nosocomial'].unique()
    gender = df['Gender'].unique()
    infection_source = df['Infection Source'].unique()
    infection_type = df['Infection Type'].unique()
    specimen = df['Specimen Type'].unique()


    #Selections
    year = st.slider("Pick year of study:", min_value=2010, max_value=2030, value=2010, step=1)
    organisms_selected = st.selectbox("Select bacteria (organism) under study: ", organisms)
    continent_selected = st.selectbox("Select continent of study: ", continents)
    country_selected = st.selectbox("Select country of study: ", countries)
    nosocomial_selected = st.selectbox("Select Nocosomial type: ", nosocomial)
    age = st.slider("Pick age:", 0, 150, 1)
    gender_selected = st.selectbox("Select gender: ", gender)
    infection_source_selected = st.selectbox("Select the source of infection: ", infection_source)
    infection_type_selected = st.selectbox("Select the type of infection: ", infection_type)
    specimen_selected = st.selectbox("Select the type of Specimen: ", specimen)
    pred_btn = st.button("Make Prediction")


    # Get salary prediction
    if pred_btn:
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

        X_test[:,1] = organism_encoder.transform(X_test[:,1])
        X_test[:,2] = continent_encoder.transform(X_test[:,2])
        X_test[:,3] = country_encoder.transform(X_test[:,3])
        X_test[:,4] = nosocomial_encoder.transform(X_test[:,4])
        X_test[:,6] = gender_encoder.transform(X_test[:,6])
        X_test[:,7] = infection_source_encoder.transform(X_test[:,7])
        X_test[:,8] = infection_type_encoder.transform(X_test[:,8])
        X_test[:,9] = specimen_encoder.fit_transform(X_test[:,9])


        X_test = X_test.astype(float)


        pred = dec.predict(X_test)
        if pred[0] == 0:
            st.write(f"The bacteria (organism) **{organisms_selected}** would be **Susceptible** to the antibiotic **{anti}** on the parameters selected")
        elif pred[0] == 1:
            st.write(f"The bacteria (organism) **{organisms_selected}** would have **Intermediate Resistance** to the antibiotic **{anti}** on the parameters selected")
        elif pred[0] == 2:
            (f"The bacteria (organism) **{organisms_selected}** would be **Resistant** to the antibiotic **{anti}** on the parameters selected")

