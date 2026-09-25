
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction System")
st.write("Enter the details of a house to predict its estimated price.")

# Training dataset
data = {
    "Area": [600,700,800,900,1000,1100,1200,1300,1400,1500,
             1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,
             2600,2700,2800,2900,3000],

    "Bedrooms": [1,1,2,2,2,2,3,3,3,3,
                 3,3,4,4,4,4,4,4,5,5,
                 5,5,5,5,6],

    "Bathrooms": [1,1,1,2,2,2,2,2,2,3,
                  3,3,3,3,3,4,4,4,4,4,
                  4,5,5,5,5],

    "Age": [20,18,15,14,12,10,9,8,7,6,
            5,5,4,4,3,3,2,2,2,1,
            1,1,1,0,0],

    "Price": [25,29,34,38,42,47,53,57,62,67,
              72,77,82,88,94,100,106,112,119,126,
              133,140,148,156,165]
}

df = pd.DataFrame(data)

# Train Linear Regression model
X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# User input
st.header("🏡 Enter House Details")

area = st.number_input(
    "House Area (sqft)",
    min_value=300,
    max_value=10000,
    value=1500,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

age = st.number_input(
    "House Age (Years)",
    min_value=0,
    max_value=100,
    value=5
)

# Prediction
if st.button("🔮 Predict House Price"):

    user_house = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Age": [age]
    })

    prediction = model.predict(user_house)[0]

    st.subheader("📋 House Details")

    st.write(f"**Area:** {area} sqft")
    st.write(f"**Bedrooms:** {bedrooms}")
    st.write(f"**Bathrooms:** {bathrooms}")
    st.write(f"**House Age:** {age} years")

    st.subheader("💰 Predicted House Price")

    st.success(f"₹ {prediction:.2f} Lakhs")

    st.info(f"Approximately ₹ {prediction * 100000:,.0f}")

# Show training dataset
with st.expander("📊 View Training Dataset"):
    st.dataframe(df, use_container_width=True)
