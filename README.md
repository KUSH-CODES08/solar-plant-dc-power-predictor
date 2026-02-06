☀️ Solar Plant DC Power Predictor

A machine learning–based web application that predicts the total DC power output of a solar power plant using environmental and weather conditions such as solar irradiation, module temperature, and ambient temperature.

The application is built using Python, Scikit-Learn, and Streamlit, and is deployed on Streamlit Community Cloud for real-time user interaction.

🚀 Live Demo

👉 Live App URL: https://solar-plant-dc-power-predictor-083.streamlit.app/

📌 Project Overview
Predicts plant-level DC power output (not individual panels or inverters)
Uses Linear Regression for prediction
Trained on real solar power generation and weather sensor data
Provides an interactive web interface using Streamlit
Designed for academic projects, internships, and learning ML deployment


🧠 Machine Learning Details
Model: Linear Regression
Input Features:
Solar Irradiation (W/m²)
Module Temperature (°C)
Ambient Temperature (°C)

Target Variable:
DC Power (Total plant-level DC output)
Evaluation Metrics:
R² Score ≈ 0.99
Mean Squared Error (MSE) used for performance evaluation


Preprocessing:
Feature scaling using StandardScaler
Scaling applied only to input features

Prediction Output:
DC Power displayed in kW (converted from Watts)
🖥️ Web Application Features
User-friendly interface
Real-time DC power prediction
Automatic handling of night conditions (zero irradiation)
Clear unit conversion and explanation
Suitable for both desktop and mobile devices

📂 Project Structure

solar-plant-dc-power-predictor/
│
├── app.py                          # Streamlit web application
├── solar_Power_generation_model.pkl # Trained ML model
├── scaler.pkl                      # StandardScaler for input features
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

⚙️ How to Run Locally

Clone the repository:-

git clone https://github.com/your-username/solar-plant-dc-power-predictor.git
cd solar-plant-dc-power-predictor


Install dependencies:-
pip install -r requirements.txt

Run the Streamlit app
streamlit run app.py

The app will open automatically in your browser.

🌐 Deployment
This application is deployed using Streamlit Community Cloud, which is well-suited for Python-based ML applications and interactive dashboards.

⚠️ Important Notes
The model predicts total DC power output of the entire solar plant
Output values are high because they represent aggregated plant-level power
This is not a per-panel or per-inverter prediction

The app is intended for educational and demonstration purposes

🧪 Example Input
Parameter	Value
Irradiation	800 W/m²
Module Temperature	44 °C
Ambient Temperature	30 °C

Predicted Output:
⚡ ~220,000 kW (plant-level DC power)

🧑‍💻 Technologies Used

Python
Pandas
NumPy
Scikit-Learn
Joblib
Streamlit


🎯 Use Cases

Academic projects & internships
Learning ML model deployment
Demonstrating end-to-end ML pipelines
Solar energy data analysis


📜 License
This project is for educational and learning purposes.

🙌 Author
Bhavsar Kush Sunilbhai
Computer Science Engineering Student
Machine Learning & Data Science Enthusiast
