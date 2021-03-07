import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Title and subtitle

st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python.
""")

image = Image.open('glitter-heart.jpg')
st.image(image, caption="Machine Learning", use_column_width=True)

# Data: Outcome - 1 - has diabetes, 0 - doesn't have

df = pd.read_csv('diabetes.csv')

st.subheader('Data Information: ')
# Display data as table
st.dataframe(df)
# Display statistics on the data
st.write(df.describe())
# Display chart
chart = st.bar_chart(df)

# Split data into variables - X and outcome - Y
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# Split data into 75% training and 25% testing. Random state keeps the states of the data.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# User input


def get_user_input():

    pregnancies = st.sidebar.slider('Pregnancies', 0, 20, 2)
    glucose = st.sidebar.slider('Glucose', 0, 199, 100)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 32.2)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'bmi': bmi,
                 'diabetes_pedigree_function': diabetes_pedigree_function,
                 'age': age
                 }

    # data to data frame. Data frame = list of equal length vectors
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store and display user input
user_input = get_user_input()
st.subheader('User Input: ')
st.write(user_input)

# Create and train model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Display metrics
st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

# Store the model prediction
prediction = RandomForestClassifier.predict(user_input)

# Display classification
st.subheader('Classification: ')
st.write(prediction)
