import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer


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
