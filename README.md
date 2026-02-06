☀️ Solar Plant DC Power Predictor

A machine learning–based web application that predicts the total DC power output of a solar power plant using environmental conditions such as solar irradiation, module temperature, and ambient temperature.

The app is built with Python, Scikit-Learn, and Streamlit, and is deployed on Streamlit Community Cloud for real-time predictions.

🚀 Live Demo:-

https://solar-plant-dc-power-predictor-083.streamlit.app/

📌 What This Project Does
Predicts plant-level DC power output
Uses Linear Regression for prediction
Takes real-time user input through a web interface
Displays results in kW (converted from Watts)
Designed for academic projects, internships, and ML deployment practice


🧠 Machine Learning Overview:-

Model: Linear Regression

Input Features:
Solar Irradiation (W/m²)
Module Temperature (°C)
Ambient Temperature (°C)

Target: Total DC Power Output (Plant Level)

Performance:

R² Score ≈ 0.99


Preprocessing:
Feature scaling using StandardScaler (inputs only)
🖥️ Web App Features
Simple & user-friendly UI
Real-time predictions
Handles night conditions (zero irradiation)
Clear unit conversion and explanation


⚙️ Run Locally
pip install -r requirements.txt
streamlit run app.py

⚠️ Important Note

This model predicts total DC power of the entire solar plant, not individual panels or inverters. Output values are high because they represent aggregated plant-level power.

🧑‍💻 Technologies Used

Python • Pandas • NumPy • Scikit-Learn • Streamlit

🙌 Author

Bhavsar Kush Sunilbhai
CSE Student | Machine Learning Enthusiast