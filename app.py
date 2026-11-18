
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# Professional styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef5ff, #f8fbff);
}
[data-testid="stSidebar"] {
    background: #102d55;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
h1, h2, h3 {
    color: #102d55;
}
div.stButton > button {
    background: #1264e8;
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    background: #084dbb;
    color: white;
}
.metric-card {
    background: white;
    border: 1px solid #d8e6fb;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 10px;
}
.result-card {
    background: #e1f8ed;
    border: 1px solid #a6e8c5;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎓 Student AI")
st.sidebar.caption("MACHINE LEARNING MINI PROJECT")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Home", "Prediction", "Dataset", "About"]
)
st.sidebar.divider()
st.sidebar.info("Study Smarter, Perform Better")

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [20, 25, 35, 40, 50, 60, 65, 75, 85, 95]
}
df = pd.DataFrame(data)

# Train ML model
X = df[["Study_Hours"]]
y = df["Marks"]
model = LinearRegression()
model.fit(X, y)

# Header
st.title("Student Performance Prediction")
st.write(
    "Predict student marks using Machine Learning "
    "and explore the relationship between study hours and marks."
)
st.divider()

# Top information cards
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        '<div class="metric-card">📘 <b>ML Algorithm</b><br>'
        'Linear Regression</div>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        '<div class="metric-card">🗂️ <b>Dataset Size</b><br>'
        '10 sample records</div>',
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        '<div class="metric-card">🎯 <b>Prediction Target</b><br>'
        'Marks out of 100</div>',
        unsafe_allow_html=True
    )

if page in ["Home", "Prediction"]:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("⚙️ Enter Study Details")
        hours = st.slider(
            "Daily study hours",
            min_value=1,
            max_value=10,
            value=5
        )

        predict = st.button("📊 Predict Marks")

        if predict:
            new_data = pd.DataFrame({"Study_Hours": [hours]})
            prediction = float(model.predict(new_data)[0])
            prediction = max(0, min(100, prediction))

            st.session_state["prediction"] = prediction
            st.session_state["hours"] = hours

        if "prediction" in st.session_state:
            result = st.session_state["prediction"]
            used_hours = st.session_state["hours"]

            st.markdown(
                f'<div class="result-card">'
                f'<h3>Predicted Marks</h3>'
                f'<h1 style="color:#087443">{result:.2f} / 100</h1>'
                f'<p>Daily study hours: {used_hours}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "This is an estimate from a small sample dataset, "
                "not a guarantee of actual exam performance."
            )
        else:
            st.info("Choose study hours and click Predict Marks.")

        st.subheader("💡 Key Takeaways")
        st.markdown(
            "- The model learns a relationship between study hours and marks.\n"
            "- Linear Regression is used for prediction.\n"
            "- Real student data can improve the project."
        )

    with right:
        st.subheader("📈 Study Hours vs Marks")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(
            df["Study_Hours"], df["Marks"],
            label="Sample data"
        )
        ax.plot(
            df["Study_Hours"], model.predict(X),
            label="Regression line"
        )

        if "prediction" in st.session_state:
            ax.scatter(
                [st.session_state["hours"]],
                [st.session_state["prediction"]],
                label="Your prediction",
                marker="*",
                s=220
            )

        ax.set_xlabel("Study Hours per Day")
        ax.set_ylabel("Marks out of 100")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif page == "Dataset":
    st.subheader("🗂️ Sample Training Dataset")
    st.write("These sample records are used to train the model.")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False),
        file_name="student_performance.csv",
        mime="text/csv"
    )

elif page == "About":
    st.subheader("About This Project")
    st.write(
        "Student Performance Prediction is a Machine Learning "
        "mini project built with Python, Streamlit, Pandas, "
        "Matplotlib and Scikit-learn."
    )
    st.markdown("""
    **Algorithm:** Linear Regression

    **Input:** Daily study hours

    **Output:** Estimated marks

    **Current limitation:** The model uses 10 illustrative sample records.
    It should not be treated as a validated model of real student performance.
    """)
