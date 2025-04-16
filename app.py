import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

breast_cancer = load_breast_cancer()

st.write("""
# Breast cancer dataset
         """)
st.subheader("""
    - **radius** (mean of distances from center to points on the perimeter)
    - **texture** (standard deviation of gray-scale values)
    - **perimeter**
    - **area**
    - **smoothness** (local variation in radius lengths)
    - **compactness** (perimeter^2 / area - 1.0)
    - **concavity** (severity of concave portions of the contour)
    - **concave points** (number of concave portions of the contour)
    - **symmetry**
    - **fractal dimension** ("coastline approximation" - 1)

    The mean, standard error, and "worst" or largest (mean of the three
    worst/largest values) of these features were computed for each image,
    resulting in 30 features.  For instance, field 0 is Mean Radius, field
    10 is Radius SE, field 20 is Worst Radius.
             """)


st.sidebar.header("""
User Input Parameters
                  """)


def get_user_parameters():
    radius_mean = st.sidebar.slider("Radius (Mean)", 6.981,  28.11, 28.11/2)
    texture_mean = st.sidebar.slider("Texture (Mean)", 9.71, 39.28, 39.28/2)
    perimeter_mean = st.sidebar.slider(
        "Perimeter (Mean)", 43.79, 188.5, 188.5/2)
    area_mean = st.sidebar.slider("Area (Mean)", 143.5, 2501.0, 2501.0/2)
    smoothness_mean = st.sidebar.slider(
        "Smoothness (Mean)", 0.053, 0.163, 0.163/2)
    compactness_mean = st.sidebar.slider(
        "Compactness (Mean)", 0.019, 0.345, 0.345/2)
    concavity_mean = st.sidebar.slider("Concavity (Mean)", 0.0, 0.427, 0.427/2)
    concave_points_mean = st.sidebar.slider(
        "Concave Points (Mean)", 0.0, 0.201, 0.201/2)
    symmetry_mean = st.sidebar.slider("Symmetry (Mean)", 0.106, 0.304, 0.304/2)
    fractal_dimension_mean = st.sidebar.slider(
        "Fractal Dimension (Mean)", 0.05,  0.097, 0.097/2)
    radius_se = st.sidebar.slider(
        "Radius (SE)", 0.112, 2.873,  2.873/2)
    texture_se = st.sidebar.slider(
        "Texture (SE)", 0.36, 4.885, 4.885/2)
    perimeter_se = st.sidebar.slider(
        "Perimeter (SE)", 0.757, 21.98, 21.98/2)
    area_se = st.sidebar.slider(
        "Area (SE)", 6.802, 542.2,  542.2/2)
    smoothness_se = st.sidebar.slider(
        "Smoothness (SE)", 0.002, 0.031, 0.031/2)
    compactness_se = st.sidebar.slider(
        "Compactness (SE)", 0.002, 0.135, 0.135/2)
    concavity_se = st.sidebar.slider("Concavity (SE)", 0.0, 0.396, 0.396/2)
    concave_points_se = st.sidebar.slider(
        "Concave Points (SE)", 0.0,  0.053, 0.053/2)
    symmetry_se = st.sidebar.slider("Symmetry (SE)",  0.008, 0.079, 0.079/2)
    fractal_dimension_se = st.sidebar.slider(
        "Fracal Dimension (SE)", 0.001, 0.03, 0.03/2)
    radius_worst = st.sidebar.slider("Radius (Worst)", 7.93, 36.04, 36.04/2)
    texture_worst = st.sidebar.slider("Texture (Worst)", 12.02, 49.54, 49.54/2)
    perimeter_worst = st.sidebar.slider(
        "Perimeter (Worst)", 50.41, 251.2, 251.2/2)
    area_worst = st.sidebar.slider("Area (Worst)", 185.2, 4254.0, 4254.0/2)
    smoothness_worst = st.sidebar.slider(
        "Smoothness (Worst)", 0.071, 0.223, 0.223/2)
    compactness_worst = st.sidebar.slider(
        "Compactness (Worst)", 0.027, 1.058, 1.058/2)
    concavity_worst = st.sidebar.slider(
        "Concavity (Worst)", 0.0, 1.252, 1.252/2)
    concave_points_worst = st.sidebar.slider(
        "Concave Points (Worst)", 0.0, 0.291, 0.291/2)
    symmetry_worst = st.sidebar.slider(
        "Symmetry (Worst)", 0.156, 0.664, 0.664/2)
    fractal_dimension_worst = st.sidebar.slider(
        "Fractal Dimension (Worst)", 0.055, 0.208, 0.208/2)

    data = {
        "Radius (Mean)": radius_mean,
        "Texture (Mean)": texture_mean,
        "Permiter (Mean)": perimeter_mean,
        "Area (Mean)": area_mean,
        "Smoothness (Mean)": smoothness_mean,
        "Compactness (Mean)": compactness_mean,
        "Concavity (Mean)": concavity_mean,
        "Concave Points (Mean)": concave_points_mean,
        "Symmetry (Mean)": symmetry_mean,
        "Fractal Dimension (Mean)": fractal_dimension_mean,
        "Radius (SE)": radius_se,
        "Texture (SE)": texture_se,
        "Perimeter (SE)": perimeter_se,
        "Area (SE)": area_se,
        "Smoothness (SE)": smoothness_se,
        "Compactness (SE)": compactness_se,
        "Concavity (SE)": concavity_se,
        "Concave Points (SE)": concave_points_se,
        "Symmetry (SE)": symmetry_se,
        "Fracal Dimension (SE)": fractal_dimension_se,
        "Radius (Worst)": radius_worst,
        "Texture (Worst)": texture_worst,
        "Perimeter (Worst)": perimeter_worst,
        "Area (Worst)": area_worst,
        "Smoothness (Worst)": smoothness_worst,
        "Compactness (Worst)": compactness_worst,
        "Concavity (Worst)": concavity_worst,
        "Concave Points (Worst)": concave_points_worst,
        "Symmetry (Worst)": symmetry_worst,
        "Fractal Dimension (Worst)": fractal_dimension_worst


    }

    feature_names = pd.DataFrame(data, index=[0])
    return feature_names


df = get_user_parameters()

st.write(df)

clf = RandomForestClassifier()

X = breast_cancer.data
y = breast_cancer.target

clf.fit(X, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
