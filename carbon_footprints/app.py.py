import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf
import google.generativeai as genai
from datetime import datetime, timedelta
import base64

# --- Constants ---
REDUCTION_TIPS = {
    "transport": "Consider biking, walking, or using public transport more often. If you need a car, think about an electric or hybrid model.",
    "energy": "Switch to renewable energy sources, use energy-efficient appliances, and insulate your home to reduce energy consumption.",
    "food": "Reduce meat consumption, buy local and seasonal produce, and minimize food waste to lower your food carbon footprint.",
    "shopping": "Buy less stuff, choose products with minimal packaging, and support companies committed to sustainability.",
    "waste": "Recycle as much as possible, compost food scraps, and reduce your overall waste generation."
}

# --- Google Gemini API Setup ---
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

def chat_with_gemini(prompt, model_name="gemini-pro"):
    """Generate a response using Google Gemini AI."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

# --- Model Loading ---
# Register MSE for model loading
tf.keras.losses.mse = tf.keras.losses.MeanSquaredError()

# File paths for individual model
individual_model_path = "carbon_footprint_model.h5"
individual_encoder_path = "encoder.pkl"
individual_scaler_path = "scaler.pkl"

# File paths for business model
business_model_path = "carbon_emission_model1.h5"
business_encoder_path = "encoder1.pkl"
business_scaler_path = "scaler1.pkl"

# Load individual Model
individual_model = None
individual_encoder = None
individual_scaler = None

if os.path.exists(individual_model_path):
    try:
        individual_model = load_model(individual_model_path, compile=False)
        individual_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    except Exception as e:
        st.error(f"Error loading individual model: {e}")
        individual_model = None

if os.path.exists(individual_encoder_path):
    try:
        individual_encoder = joblib.load(individual_encoder_path)
    except Exception as e:
        st.error(f"Error loading individual encoder: {e}")
        individual_encoder = None

if os.path.exists(individual_scaler_path):
    try:
        individual_scaler = joblib.load(individual_scaler_path)
    except Exception as e:
        st.error(f"Error loading individual scaler: {e}")
        individual_scaler = None


# Load business Model
business_model = None
business_encoder = None
business_scaler = None

if os.path.exists(business_model_path):
    try:
        business_model = load_model(business_model_path, compile=False)
        business_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    except Exception as e:
        st.error(f"Error loading business model: {e}")
        business_model = None

if os.path.exists(business_encoder_path):
    try:
        business_encoder = joblib.load(business_encoder_path)
    except Exception as e:
        st.error(f"Error loading business encoder: {e}")
        business_encoder = None

if os.path.exists(business_scaler_path):
    try:
        business_scaler = joblib.load(business_scaler_path)
    except Exception as e:
        st.error(f"Error loading business scaler: {e}")
        business_scaler = None

# --- Helper Functions ---
def update_carbon_history(emission, history_key):
    """Updates the carbon footprint history in the session state."""
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    st.session_state[history_key].append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emission))
    # Limit history to, say, 10 entries
    st.session_state[history_key] = st.session_state[history_key][-10:]

def calculate_progress(history_key, target_reduction):
    """Calculates progress towards the carbon reduction goal."""
    if history_key in st.session_state and len(st.session_state[history_key]) >= 2:
        initial_emission = st.session_state[history_key][0][1]
        latest_emission = st.session_state[history_key][-1][1]
        reduction = initial_emission - latest_emission
        target = initial_emission * (target_reduction / 100)

        if target > 0:  # Avoid division by zero
            progress = (reduction / target) * 100
        else:
            progress = 0

        progress = max(0, progress)  # Ensure progress is not negative
        progress = min(100, progress) # Ensure progress does not exceed 100

        return float(progress)  # Cap at 100% and convert to float
    else:
        return 0

def generate_report_html(title, sections):
    """Generates an HTML report from the given sections."""
    html = f"""
    <html>
    <head>
        <title>{title}</title>
    </head>
    <body>
        <h1>{title}</h1>
    """
    for section_title, content in sections.items():
        html += f"""
        <h2>{section_title}</h2>
        <p>{content}</p>
        """
    html += """
    </body>
    </html>
    """
    return html

def download_html_report(html, filename, button_text):
    """Generates a download link for the given HTML."""
    b64 = base64.b64encode(html.encode()).decode()  # Encode to base64
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def generate_smart_alert(current_emission, previous_emission, category, industry_sector=None, diet=None):
    """Generates a smart alert based on emission change."""
    if previous_emission is None:
        return "This is your first calculation. Keep tracking to see how your footprint changes over time."

    change = current_emission - previous_emission
    percent_change = (change / previous_emission) * 100 if previous_emission != 0 else 0

    if change < 0:
        alert_type = "decrease"
        prompt_details = f"Your {category} carbon emission has decreased by {abs(percent_change):.2f}% since your last calculation. "
        prompt_details += "Provide a short, encouraging message and suggest one way to maintain or further improve this trend."
    else:
        alert_type = "increase"
        prompt_details = f"Your {category} carbon emission has increased by {percent_change:.2f}% since your last calculation. "

        if category == "individual":
             prompt_details += f"Knowing their diet is {diet}, suggest one specific area they could focus on to reduce it."
        elif category == "business":
             prompt_details += f"Knowing their industry is {industry_sector}, suggest one specific area they could focus on to reduce it."
        prompt_details += "Keep the message concise and actionable."
    prompt = prompt_details
    return chat_with_gemini(prompt)

# --- Sidebar and App Selection ---
app_type = st.sidebar.selectbox("Select Carbon Footprint Calculator", ["Individual", "Business", "Chatbot"])

# --- Chatbot App ---
if app_type == "Chatbot":
    st.title("💬 AI Chatbot for Carbon Footprint Reduction")
    st.write("Ask me anything about reducing your carbon footprint!")

    user_input = st.text_input("You:")
    if st.button("Ask AI"):
        if user_input:
            response = chat_with_gemini(user_input)
            st.text_area("AI Response:", response, height=200)
        else:
            st.warning("Please enter a question or message.")

# --- Individual Carbon Footprint App ---
elif app_type == "Individual":
    st.title("🌍 Individual Carbon Footprint Tracker")
    st.write("Enter your details to estimate your carbon footprint and get AI-powered reduction strategies.")

    # Initialize session state for persistent data storage
    if 'individual_data' not in st.session_state:
        st.session_state['individual_data'] = None
    if 'previous_emission' not in st.session_state:
        st.session_state['previous_emission'] = None
    if 'alert_message' not in st.session_state:
        st.session_state['alert_message'] = None
    if 'last_calculated_date' not in st.session_state:
        st.session_state['last_calculated_date'] = None
    if 'report' not in st.session_state:
        st.session_state['report'] = None
    if 'individual_carbon_history' not in st.session_state:
        st.session_state['individual_carbon_history'] = []
    if 'individual_target_reduction' not in st.session_state:
        st.session_state['individual_target_reduction'] = 0

    # User input fields
    diet = st.selectbox("Diet", ["omnivore", "vegetarian", "vegan", "pescatarian"])
    shower = st.selectbox("How Often Do You Shower?", ["daily", "more frequently", "less frequently"])
    energy_source = st.selectbox("Heating Energy Source", ["coal", "natural gas", "wood", "renewable"])
    transport = st.selectbox("Transport Mode", ["public", "private", "walk/bicycle"])
    vehicle_type = st.selectbox("Vehicle Type", ["petrol", "diesel", "electric", "none"])
    social_activity = st.selectbox("Social Activity", ["often", "rarely", "never"])
    monthly_grocery_bill = st.number_input("Monthly Grocery Bill ($)", min_value=0)
    air_travel = st.selectbox("Frequency of Traveling by Air", ["never", "rarely", "frequently"])
    vehicle_distance = st.number_input("Vehicle Monthly Distance (km)", min_value=0)
    waste_size = st.selectbox("Waste Bag Size", ["small", "large", "extra large"])
    waste_count = st.number_input("Waste Bag Weekly Count", min_value=0)
    tv_pc_hours = st.number_input("How Long TV/PC Daily (Hours)", min_value=0)
    new_clothes = st.number_input("New Clothes Purchased Monthly", min_value=0)
    internet_hours = st.number_input("How Long Internet Daily (Hours)", min_value=0)
    energy_efficiency = st.selectbox("Energy Efficiency Measures?", ["Yes", "No", "Sometimes"])
    recycling = st.multiselect("Recycling Items", ["Plastic", "Metal", "Glass", "Paper"])
    cooking_methods = st.multiselect("Cooking With", ["Stove", "Oven", "Microwave"])
    body_type = st.selectbox("Body Type", ["slim", "average", "overweight"])
    sex = st.selectbox("Sex", ["male", "female", "other"])

    # Goal Setting
    st.subheader("🎯 Set Your Carbon Reduction Goal")
    individual_target_reduction = st.number_input("Target Reduction (%)", min_value=0, max_value=100, value=st.session_state['individual_target_reduction'], key="individual_reduction_input")
    st.session_state['individual_target_reduction'] = individual_target_reduction

    if st.button("Calculate Carbon Footprint"):
        # Convert multiselect values to strings
        recycling_str = ", ".join(recycling) if recycling else "None"
        cooking_methods_str = ", ".join(cooking_methods) if cooking_methods else "None"

        # Create DataFrame with user input
        user_data = pd.DataFrame([[diet, shower, energy_source, transport, vehicle_type, social_activity,
                                   monthly_grocery_bill, air_travel, vehicle_distance, waste_size, waste_count,
                                   tv_pc_hours, new_clothes, internet_hours, energy_efficiency,
                                   recycling_str, cooking_methods_str,body_type,sex]],
                                 columns=["Diet", "How Often Shower", "Heating Energy Source", "Transport",
                                          "Vehicle Type", "Social Activity", "Monthly Grocery Bill",
                                          "Frequency of Traveling by Air", "Vehicle Monthly Distance Km",
                                          "Waste Bag Size", "Waste Bag Weekly Count", "How Long TV PC Daily Hour",
                                          "How Many New Clothes Monthly", "How Long Internet Daily Hour",
                                          "Energy efficiency", "Recycling", "Cooking_With","Body Type","Sex"])

        # Ensure all categorical features from the encoder are present
        expected_categorical_features = individual_encoder.feature_names_in_ if individual_encoder is not None else []
        for col in expected_categorical_features:
            if col not in user_data.columns:
                user_data[col] = "Unknown"

        # Encode categorical features
        try:
            user_data_encoded = pd.DataFrame(individual_encoder.transform(user_data[expected_categorical_features]),
                                             columns=individual_encoder.get_feature_names_out())
        except Exception as e:
            st.error(f"Error encoding data: {e}")
            user_data_encoded = None

        # Ensure all numerical features from the scaler are present
        expected_numerical_features = individual_scaler.feature_names_in_ if individual_scaler is not None else []
        for col in expected_numerical_features:
            if col not in user_data.columns:
                user_data[col] = 0

        # Scale numerical features
        try:
            user_data_scaled = pd.DataFrame(individual_scaler.transform(user_data[expected_numerical_features]),
                                            columns=expected_numerical_features)
        except Exception as e:
            st.error(f"Error scaling data: {e}")
            user_data_scaled = None

        if user_data_encoded is not None and user_data_scaled is not None:
            # Combine processed data
            user_final = pd.concat([user_data_encoded, user_data_scaled], axis=1)

            # Predict carbon footprint
            carbon_emission = None  # Define carbon_emission here with a default value
            try:
                carbon_emission = individual_model.predict(user_final)[0][0] if individual_model is not None else None
            except Exception as e:
                st.error(f"Error predicting carbon footprint: {e}")

            if carbon_emission is not None:
                # Store current calculation time
                st.session_state['last_calculated_date'] = datetime.now()

                # Update carbon footprint history
                update_carbon_history(carbon_emission, 'individual_carbon_history')

                # * Smart Alert based on emission change *
                if st.session_state['previous_emission'] is not None:
                    smart_alert = generate_smart_alert(carbon_emission, st.session_state['previous_emission'], "individual", diet=diet)
                    st.info(smart_alert)

                # Display results
                st.subheader(f"🌱 Estimated Carbon Footprint: {carbon_emission:.2f} kg CO₂/month")

                # Breakdown visualization
                categories = ["Transport", "Energy", "Food", "Shopping", "Waste"]
                emissions = np.random.uniform(0.1, 0.3, size=5) * carbon_emission  # Simulated breakdown

                fig, ax = plt.subplots()
                sns.barplot(x=categories, y=emissions, palette=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"], ax=ax)
                ax.set_ylabel("Emissions (kg CO₂)")
                ax.set_title("Carbon Emission Breakdown by Category")
                st.pyplot(fig)

                # Identify the category with the highest emissions
                max_index = np.argmax(emissions)
                max_category = categories[max_index]

                # * Dynamic AI-Powered Reduction Strategy using Gemini *
                prompt = f"""Given that the user's highest carbon emission category is '{max_category}' and their
                overall monthly carbon footprint is {carbon_emission:.2f} kg CO2, provide 3 specific and actionable recommendations
                for reducing their emissions in that category.  Tailor the recommendations to be suitable for someone with a '{diet}' diet.
                Format the recommendations as a numbered list."""

                ai_recommendations = chat_with_gemini(prompt)
                st.subheader(f"✅ AI-Powered Reduction Strategies for {max_category}")
                st.write(ai_recommendations)
                st.write("Take small steps today to create a sustainable future! 🌎")

                # Store current data for the next calculation
                st.session_state['previous_emission'] = carbon_emission
                st.session_state['individual_data'] = user_data.to_dict() # Store the user data

                 # Progress towards goal
                progress = calculate_progress('individual_carbon_history', st.session_state['individual_target_reduction'])
                st.subheader("📈 Progress Towards Your Goal")
                st.progress(progress / 100)
                st.write(f"You have made {progress:.2f}% progress towards your {st.session_state['individual_target_reduction']}% reduction goal.")

                # Carbon Footprint History
                st.subheader("History")
                if 'individual_carbon_history' in st.session_state and st.session_state['individual_carbon_history']:
                    history_df = pd.DataFrame(st.session_state['individual_carbon_history'], columns=['Date', 'Carbon Footprint (kg CO₂)'])
                    st.dataframe(history_df)
                    # Plotting the history
                    fig, ax = plt.subplots()
                    ax.plot(history_df['Date'], history_df['Carbon Footprint (kg CO₂)'], marker='o')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Carbon Footprint (kg CO₂)")
                    ax.set_title("Carbon Footprint History")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
                    st.pyplot(fig)

                # * Automated Report Generation *
                report_prompt = f"""Generate a concise report summarizing the individual's carbon footprint. Include the following:
                - The estimated monthly carbon footprint: {carbon_emission:.2f} kg CO2/month.
                - Key factors influencing the carbon footprint based on the user's data (e.g., diet, transport, energy source).
                - A brief overview of the AI-powered reduction strategies provided, focusing on the most impactful category: {max_category}.
                - An encouraging closing statement to motivate the user to reduce their carbon footprint.
                Keep the report under 150 words."""
                report = chat_with_gemini(report_prompt)
                st.subheader("📝 Carbon Footprint Report")
                st.write(report)

                # * Download Report *
                report_sections = {
                    "Estimated Carbon Footprint": f"{carbon_emission:.2f} kg CO₂/month",
                    "Key Influencing Factors": "Based on user data (e.g., diet, transport, energy source)",
                    "Reduction Strategies": f"AI-powered strategies for {max_category}",
                    "Report": report
                }
                report_html = generate_report_html("Individual Carbon Footprint Report", report_sections)
                download_html_report(report_html, "individual_carbon_footprint_report.html", "Download Report")

            else:
                st.warning("Could not estimate carbon footprint. Please check your inputs and model loading.")

# --- Business Carbon Emission App ---
elif app_type == "Business":
    st.title("🏢 Business Carbon Emission Calculator")
    st.write("Enter your business details to estimate carbon emissions.")

    # Initialize session state for persistent data storage
    if 'business_data' not in st.session_state:
        st.session_state['business_data'] = None
    if 'business_previous_emission' not in st.session_state:
        st.session_state['business_previous_emission'] = None
    if 'alert_message' not in st.session_state:
        st.session_state['business_alert_message'] = None
    if 'business_last_calculated_date' not in st.session_state:
        st.session_state['business_last_calculated_date'] = None
    if 'business_report' not in st.session_state:
        st.session_state['business_report'] = None
    if 'business_carbon_history' not in st.session_state:
        st.session_state['business_carbon_history'] = []
    if 'business_target_reduction' not in st.session_state:
        st.session_state['business_target_reduction'] = 0

    # User Input
    industry_sector = st.selectbox("Industry Sector", ["Manufacturing", "Tech", "Retail", "Agriculture"])
    location = st.selectbox("Location", ["USA", "Europe", "Asia", "Other"])
    raw_material = st.selectbox("Raw Material", ["Steel", "Plastic", "Wood", "Other"])
    employees = st.number_input("Number of Employees", min_value=1)
    electricity = st.number_input("Annual Electricity Usage (kWh)", min_value=0)
    fuel = st.number_input("Annual Fuel Consumption (liters)", min_value=0)
    renewable_energy = st.number_input("Percentage of Renewable Energy Used (%)", min_value=0, max_value=100)
    raw_material_qty = st.number_input("Raw Material Quantity (tons)", min_value=0)
    waste_generated = st.number_input("Total Annual Waste Generated (tons)", min_value=0)
    waste_recycled = st.number_input("Percentage of Waste Recycled (%)", min_value=0, max_value=100)

    # Goal Setting
    st.subheader("🎯 Set Your Carbon Reduction Goal")
    business_target_reduction = st.number_input("Target Reduction (%)", min_value=0, max_value=100, value=st.session_state['business_target_reduction'], key="business_reduction_input")
    st.session_state['business_target_reduction'] = business_target_reduction

    if st.button("Calculate Carbon Emission"):
        # Create DataFrame with user input
        user_data = pd.DataFrame([[industry_sector, location, raw_material, employees, electricity, fuel,
                                   renewable_energy, raw_material_qty, waste_generated, waste_recycled]],
                                 columns=["Industry Sector", "Location", "Raw Material", "Number of Employees",
                                          "Annual Electricity Usage kWh", "Annual Fuel Consumption liters",
                                          "Percentage of Renewable Energy Used", "Raw Material Quantity tons",
                                          "Total Annual Waste Generated tons", "Percentage of Waste Recycled"])

        # Print the user data right after creation
        print("Business User Data:")
        print(user_data)

        # Ensure all categorical features from the encoder are present
        expected_categorical_features = business_encoder.feature_names_in_ if business_encoder is not None else []
        print(f"Business Expected categorical features: {expected_categorical_features}")  # Print expected features
        for col in expected_categorical_features:
            if col not in user_data.columns:
                user_data[col] = "Unknown"

        # Encode categorical features
        try:
            user_data_encoded = pd.DataFrame(business_encoder.transform(user_data[expected_categorical_features]),
                                             columns=business_encoder.get_feature_names_out())
            print("Business Encoded User Data:")
            print(user_data_encoded)
        except Exception as e:
            st.error(f"Error encoding data: {e}")
            user_data_encoded = None

        # Ensure all numerical features from the scaler are present
        expected_numerical_features = business_scaler.feature_names_in_ if business_scaler is not None else []
        print(f"Business Expected numerical features: {expected_numerical_features}") #print expected
        for col in expected_numerical_features:
            if col not in user_data.columns:
                user_data[col] = 0

        # Scale numerical features
        try:
            user_data_scaled = pd.DataFrame(business_scaler.transform(user_data[expected_numerical_features]),
                                            columns=expected_numerical_features)
            print("Business Scaled User Data:")
            print(user_data_scaled)
        except Exception as e:
            st.error(f"Error scaling data: {e}")
            user_data_scaled = None

        if user_data_encoded is not None and user_data_scaled is not None:
            # Combine processed data
            user_final = pd.concat([user_data_encoded, user_data_scaled], axis=1)

            # Check for missing or non-finite values before prediction
            if user_final.isnull().values.any():
                st.error("Error: Missing values in input data.")
                st.stop()
            if not np.isfinite(user_final.values).all():
                st.error("Error: Non-finite values in input data.")
                st.stop()

            # Predict carbon emission
            carbon_emission = None  # Define carbon_emission here with a default value
            try:
                carbon_emission = business_model.predict(user_final)[0][0] if business_model is not None else None
            except Exception as e:
                st.error(f"Error predicting carbon emission: {e}")

            if carbon_emission is not None:
                # Store current calculation time
                st.session_state['business_last_calculated_date'] = datetime.now()

                # Update carbon emission history
                update_carbon_history(carbon_emission, 'business_carbon_history')

                # * Smart Alert based on emission change *
                if st.session_state['business_previous_emission'] is not None:
                    smart_alert = generate_smart_alert(carbon_emission, st.session_state['business_previous_emission'], "business", industry_sector=industry_sector)
                    st.info(smart_alert)

                # Display results
                st.subheader(f"🔥 Estimated Carbon Emission: {carbon_emission:.2f} tons CO₂/year")

                 # Breakdown visualization
                categories = ["Electricity", "Fuel", "Raw Material", "Waste", "Other"]
                emissions = np.random.uniform(0.1, 0.3, size=5) * carbon_emission  # Simulated breakdown

                fig, ax = plt.subplots()
                sns.barplot(x=categories, y=emissions, palette=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"], ax=ax)
                ax.set_ylabel("Emissions (tons CO₂)")
                ax.set_title("Carbon Emission Breakdown by Category")
                st.pyplot(fig)

                # Identify the category with the highest emissions
                max_index = np.argmax(emissions)
                max_category = categories[max_index]

                # * Dynamic AI-Powered Reduction Strategy using Gemini *
                prompt = f"""Given that the user's highest carbon emission category is '{max_category}' and their
                overall annual carbon emission is {carbon_emission:.2f} tons CO2, provide 3 specific and actionable recommendations
                for reducing their emissions in that category.  Tailor the recommendations to be suitable for a business in the '{industry_sector}' sector.
                Format the recommendations as a numbered list."""

                ai_recommendations = chat_with_gemini(prompt)
                st.subheader(f"✅ AI-Powered Reduction Strategies for {max_category}")
                st.write(ai_recommendations)
                st.write("Take small steps today to create a sustainable future! 🌎")

                # Store current data for the next calculation
                st.session_state['business_previous_emission'] = carbon_emission
                st.session_state['business_data'] = user_data.to_dict()  # Store the user data

                 # Progress towards goal
                progress = calculate_progress('business_carbon_history', st.session_state['business_target_reduction'])
                st.subheader("📈 Progress Towards Your Goal")
                st.progress(progress / 100)
                st.write(f"You have made {progress:.2f}% progress towards your {st.session_state['business_target_reduction']}% reduction goal.")

                # Carbon Emission History
                st.subheader("History")
                if 'business_carbon_history' in st.session_state and st.session_state['business_carbon_history']:
                    history_df = pd.DataFrame(st.session_state['business_carbon_history'], columns=['Date', 'Carbon Emission (tons CO₂)'])
                    st.dataframe(history_df)
                    
                # * Automated Report Generation *
                report_prompt = f"""Generate a concise report summarizing the business's carbon emission. Include the following:
                - The estimated annual carbon emission: {carbon_emission:.2f} tons CO2/year.
                - Key factors influencing the carbon emission based on the business's data (e.g., industry sector, location, raw materials, energy usage).
                - A brief overview of the AI-powered reduction strategies provided, focusing on the most impactful category: {max_category}.
                - An encouraging closing statement to motivate the business to reduce its carbon footprint.
                Keep the report under 150 words."""
                report = chat_with_gemini(report_prompt)
                st.subheader("📝 Carbon Emission Report")
                st.write(report)

                 # * Download Report *
                report_sections = {
                    "Estimated Carbon Emission": f"{carbon_emission:.2f} tons CO₂/year",
                    "Key Influencing Factors": "Based on business data (e.g., industry sector, location, raw materials)",
                    "Reduction Strategies": f"AI-powered strategies for {max_category}",
                    "Report": report
                }
                report_html = generate_report_html("Business Carbon Emission Report", report_sections)
                download_html_report(report_html, "business_carbon_emission_report.html", "Download Report")

            else:
                st.warning("Could not estimate carbon emission. Please check your inputs and model loading.")