import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and encoders
@st.cache_data
def load_model_and_encoders():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, encoders, feature_columns

# Load model and encoders
model, label_encoders, feature_columns = load_model_and_encoders()

# Streamlit app
st.title('Financial Inclusion in Africa Predictor')
st.write('Predict whether an individual is likely to have a bank account based on demographic and socioeconomic factors.')

# Create input fields
st.header('Enter Individual Information:')

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox('Country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
    location_type = st.selectbox('Location Type', ['Rural', 'Urban'])
    cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 16, 100, 35)
    household_size = st.slider('Household Size', 1, 20, 4)

with col2:
    relationship_with_head = st.selectbox('Relationship with Head of Household', 
                                        ['Head of Household', 'Spouse', 'Child', 'Parent', 
                                         'Other relative', 'Other non-relatives', 'Dont know'])
    
    marital_status = st.selectbox('Marital Status', 
                                ['Married/Living together', 'Divorced/Seperated', 
                                 'Widowed', 'Single/Never Married', "Don't know"])
    
    education_level = st.selectbox('Education Level', 
                                 ['No formal education', 'Primary education', 
                                  'Secondary education', 'Vocational/Specialised training', 
                                  'Tertiary education', 'Other/Dont know/RTA'])
    
    job_type = st.selectbox('Job Type', 
                          ['Farming and Fishing', 'Self employed', 
                           'Formally employed Government', 'Formally employed Private', 
                           'Informally employed', 'Remittance Dependent', 
                           'Government Dependent', 'Other Income', 'No Income', 
                           'Dont Know/Refuse to answer'])
    
    year = st.selectbox('Year', [2016, 2017, 2018])

# Create prediction button
if st.button('Predict Bank Account Ownership'):
    # Create input dataframe
    input_data = {
        'country': country,
        'year': year,
        'location_type': location_type,
        'cellphone_access': cellphone_access,
        'household_size': household_size,
        'age_of_respondent': age,
        'gender_of_respondent': gender,
        'relationship_with_head': relationship_with_head,
        'marital_status': marital_status,
        'education_level': education_level,
        'job_type': job_type
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for column in input_df.columns:
        if column in label_encoders:
            try:
                input_df[column] = label_encoders[column].transform([input_data[column]])[0]
            except ValueError:
                # Handle unseen categories
                input_df[column] = 0
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Display results
    st.header('Prediction Results:')
    
    if prediction == 1:
        st.success('✅ This individual is likely to HAVE a bank account')
        st.write(f'Confidence: {prediction_proba[1]:.2%}')
    else:
        st.error('❌ This individual is likely to NOT have a bank account')
        st.write(f'Confidence: {prediction_proba[0]:.2%}')
    
    # Show probability breakdown
    st.subheader('Probability Breakdown:')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("No Bank Account", f"{prediction_proba[0]:.2%}")
    with col2:
        st.metric("Has Bank Account", f"{prediction_proba[1]:.2%}")

# Add some information about the model
st.sidebar.header('About')
st.sidebar.info('''
This application predicts financial inclusion in Africa based on demographic 
and socioeconomic factors. The model was trained on data from Kenya, Rwanda, 
Tanzania, and Uganda.

Features used:
- Country
- Location type (Rural/Urban)
- Cellphone access
- Age and gender
- Household size
- Education level
- Job type
- Marital status
- Relationship with head of household
''')