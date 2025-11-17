
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import load_data, train_models, predict_new

st.title("ReFillHub Marketing & ML Dashboard")

df, X, y = load_data("ReFillHub_SyntheticSurvey.csv")

st.header("Top 5 Marketing Insights")
fig, ax = plt.subplots()
sns.countplot(x=df["Likely_to_Use_ReFillHub"], ax=ax)
st.pyplot(fig)

st.header("Run Algorithms")
results = train_models(X, y)

st.write(pd.DataFrame({
    m: {
        "Accuracy": r["acc"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "AUC": r["auc"]
    } for m, r in results.items()
}).T)

st.header("Predict New Customer")
inputs = {}
for col in df.columns:
    inputs[col] = st.text_input(col, "")

if st.button("Predict"):
    model = results["GBRT"]["model"]
    updated_df = predict_new(model, df, inputs)
    st.write(updated_df.tail())
    updated_df.to_csv("Updated_Predictions.csv", index=False)
    st.download_button("Download Updated Dataset", data=open("Updated_Predictions.csv","rb"), file_name="Updated_Predictions.csv")
