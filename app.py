import streamlit as st
import pickle
import numpy as np


with open('pcos_model.pkl', 'rb') as f:
    model = pickle.load(f)

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

tab1, tab2 = st.tabs(["Welcome", "PCOS Risk Form"])


with tab1:
    st.title("PCOS Risk Predictor App")
    st.subheader("Welcome! Please enter your name to begin:")
    
    name_input = st.text_input("Enter your name:")
    if name_input:
        st.session_state.user_name = name_input.strip()

    st.markdown("### Make sure you have the following info ready:")
    st.markdown("""
    - Age  
                
    - BMI  
                
    - Menstrual Irregularity 
                 
    - Testosterone Level (ng/dL)  
                
    - Antral Follicle Count  
    """)


with tab2:
    if not st.session_state.user_name:
        st.warning("🚫 Please enter your name in the 'Welcome' tab to fill the form.")
    else:
        st.title(f"Hello, {st.session_state.user_name}! ✨")
        st.markdown("### Fill in the details below:")

        with st.form("pcos_form"):
            age = st.slider("Age", 18, 45, 25)
            bmi = st.slider("BMI", 10.0, 50.0, 22.0)
            menstrual_irregularity = st.selectbox("Menstrual Irregularity", ["Yes", "No"])
            testosterone = st.slider("Testosterone Level (ng/dL)", 10.0, 200.0, 50.0)
            afc = st.slider("Antral Follicle Count", 0, 50, 10)

            submit = st.form_submit_button("Predict")

        if submit:
            input_data = np.array([[age, bmi, 1 if menstrual_irregularity == 'Yes' else 0,
                                    testosterone, afc]])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("⚠️ You might be at risk of PCOS. Please consult a healthcare provider.")
            else:
                st.success("✅ You are likely not at risk of PCOS.")
