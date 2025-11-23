import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title("🚗 Auto MPG Prediction App (Linear Regression Model)")

@st.cache_data
def load_data():
    df = pd.read_csv("D:/JUPYTER/Machine learning PROJECTS/auto-mpg.csv")
    df = df.replace("?", None)
    df = df.dropna()
    df["horsepower"] = df["horsepower"].astype(float)
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df[["mpg","horsepower","weight","acceleration"]].corr(), annot=True, ax=ax)
st.pyplot(fig)

st.header("🔧 MPG Prediction")

X = df[["horsepower", "weight", "acceleration"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

hp = st.number_input("Horsepower", 40.0, 250.0, 100.0)
wt = st.number_input("Weight", 1000.0, 6000.0, 2500.0)
acc = st.number_input("Acceleration", 5.0, 25.0, 15.0)

if st.button("Predict MPG"):
    pred = model.predict([[hp, wt, acc]])[0]
    st.success(f"Estimated MPG: {pred:.2f}")

st.subheader("📊 Model Performance")
pred_test = model.predict(X_test)
st.write("R² Score:", r2_score(y_test, pred_test))
st.write("MSE:", mean_squared_error(y_test, pred_test))
