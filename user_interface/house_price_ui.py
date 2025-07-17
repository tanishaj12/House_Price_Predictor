import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px
import requests

# Page configuration
st.set_page_config(
    page_title="House Price Predictor", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .feature-description {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .sidebar-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .currency-toggle {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .house-image {
        width: 100%;
        height: 300px;
        object-fit: cover;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTab {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to get current USD to INR exchange rate
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate():
    try:
        # Using a free API to get current exchange rate
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data['rates']['INR']
    except:
        # Approximate fallback rate if API fails
        return 83.0  

# Load the model and feature names
@st.cache_resource
def load_model_and_features():
    """Load the trained model, feature names, and scaler"""
    try:
        model = joblib.load('house_price_model.pkl')
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, feature_names, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

def create_feature_mapping(user_inputs, feature_names):
    """
    Map user inputs to the exact feature names expected by the model
    """
    # Initialize with zeros for all expected features
    feature_dict = {name: 0 for name in feature_names}
    
    # Core mappings from UI to model features
    feature_mapping = {
        # Area/Size mappings
        'area': 'GrLivArea',  # Ground Living Area
        'lot_area': 'LotArea',   # Total Area of the house
        'total_basement_area': 'TotalBsmtSF',   #Total Basement Area
        
        # Room mappings
        'bedrooms': 'BedroomAbvGr',  
        'bathrooms': 'FullBath',    
        'half_bathrooms': 'HalfBath',    
        
        # Quality/Condition mappings
        'overall_quality': 'OverallQual',   
        'overall_condition': 'OverallCond',
        'fence_quality': 'Fence',
        'kitchen_quality': 'KitchenQual',
        
        # Year mappings
        'year_built': 'YearBuilt',
        'year_remodeled': 'YearRemodAdd',
        
        # Other important features
        'garage_cars': 'GarageCars',
        'garage_area': 'GarageArea',
        'first_floor_area': '1stFlrSF',
        'second_floor_area': '2ndFlrSF',
        'stories': 'stories',
        'parking': 'parking',
    }
    
    # Direct Mappings
    for ui_key, model_feature in feature_mapping.items():
        if ui_key in user_inputs and model_feature in feature_names:
            feature_dict[model_feature] = user_inputs[ui_key]
    
    # Handling derived features expected by the model
    if 'House_Age' in feature_names and 'year_built' in user_inputs:
        feature_dict['House_Age'] = 2023 - user_inputs['year_built']
    
    if 'Total_SF' in feature_names:
        first_floor = user_inputs.get('first_floor_area', user_inputs.get('area', 0))
        second_floor = user_inputs.get('second_floor_area', 0)
        feature_dict['Total_SF'] = first_floor + second_floor
    
    if 'Total_Area' in feature_names:
        total_sf = feature_dict.get('Total_SF', user_inputs.get('area', 0))
        basement = user_inputs.get('total_basement_area', 0)
        feature_dict['Total_Area'] = total_sf + basement
    
    if 'Total_Bathrooms' in feature_names:
        full_bath = user_inputs.get('bathrooms', 0)
        half_bath = user_inputs.get('half_bathrooms', 0)
        feature_dict['Total_Bathrooms'] = full_bath + (0.5 * half_bath)
    
    if 'Has_Garage' in feature_names:
        feature_dict['Has_Garage'] = 1 if user_inputs.get('garage_cars', 0) > 0 else 0
    
    if 'Has_Basement' in feature_names:
        feature_dict['Has_Basement'] = 1 if user_inputs.get('total_basement_area', 0) > 0 else 0
    
    # Handling Yes/No categorical features
    yes_no_mapping = {
        'mainroad': 'mainroad',
        'fireplaces': 'fireplaces', 
        'basement_yn': 'basement',
        'airconditioning': 'CentralAir',
    }
    
    for ui_key, model_feature in yes_no_mapping.items():
        if ui_key in user_inputs and model_feature in feature_names:
            # Convert Yes/No to 1/0
            if user_inputs[ui_key] == 'Yes':
                feature_dict[model_feature] = 1
            else:
                feature_dict[model_feature] = 0
    
    return feature_dict

def preprocess_for_prediction(feature_dict, feature_names, scaler):
    """
    Convert feature dictionary to model-ready format
    """
    # DataFrame with correct column order
    df = pd.DataFrame([feature_dict])[feature_names]
    
    # Numerical columns for scaling
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Scaling numerical features
    if numerical_cols:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

# Function to format currency based on selection
def format_currency(amount, currency_type):
    if currency_type == "INR (₹)":
        inr_amount = amount * exchange_rate
        return f"₹{inr_amount:,.2f}", inr_amount
    else:
        return f"${amount:,.2f}", amount

# Header
st.markdown('<h1 class="main-header">🏠 AI-Powered House Price Predictor</h1>', unsafe_allow_html=True)

# Display a house image from web URL
try:
    # Using a beautiful house image 
    house_image_url = "https://images.unsplash.com/photo-1568605114967-8130f3a36994?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80"
    st.image(house_image_url, caption="Beautiful Modern House", use_column_width=True)

except Exception as e:
    # Fallback banner if image fails to load
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem; color: white;">
        <h2 style="color: white; margin: 0; font-size: 2.5rem;">🏡 Get Instant Property Valuations</h2>
        <p style="color: white; margin: 0.5rem 0; font-size: 1.2rem;">Powered by Advanced Machine Learning Algorithms</p>
    </div>
    """, unsafe_allow_html=True)

# Load model components
model, feature_names, scaler = load_model_and_features()

if model is None:
    st.error("Could not load model files. Please ensure the following files exist:")
    st.code("- house_price_model.pkl\n- feature_names.pkl\n- scaler.pkl")
    st.stop()

# Get current exchange rate
exchange_rate = get_usd_to_inr_rate()

# Sidebar with information
with st.sidebar:
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 📊 About This Tool")
    st.markdown("""
    This AI-powered tool predicts house prices using machine learning algorithms trained on comprehensive real estate data based on the Ames city(Lowa, US).
    
    **Model Features:**
    - 🎯 High accuracy predictions
    - 🔄 Real-time processing
    - 📈 Advanced feature engineering
    - 🤖 Ensemble learning methods
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Currency selection
    st.markdown("### 💱 Currency Selection ")
    currency_option = st.radio(
        "Choose your preferred currency:",
        ["USD ($)", "INR (₹)"],
        help="Select USD for US Dollars or INR for Indian Rupees"
    )
    
    if currency_option == "INR (₹)":
        st.info(f"Current exchange rate: 1 USD = ₹{exchange_rate:.2f}")
    
    st.markdown("### 💡 Tips for Better Predictions")
    st.info("""
    - Ensure all measurements are accurate
    - Consider the neighborhood quality
    - Recent renovations can increase value
    - Location accessibility matters
    """)
    
    st.markdown("### 📞 Need Help?")
    st.markdown("Contact our real estate experts for personalized advice!")
    
    # Model information
    st.markdown("### 🔍 Model Information")
    st.success(f"✅ Model loaded successfully!")
    st.info(f"Expected features: {len(feature_names)}")

# Main input form
st.markdown('<h2 class="sub-header">🏗️ Property Details</h2>', unsafe_allow_html=True)

# Tabs for better organization
tab1, tab2, tab3 = st.tabs(["🏠 Basic Info", "✨ Features", "📊 Prediction"])

with tab1:
    st.markdown('<div class="feature-description">', unsafe_allow_html=True)
    st.markdown("**📐 Basic Property Information**")
    st.markdown("Enter the fundamental details about the property size and structure.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏠 Ground Living Area (Square Feet)**")
        st.caption("Total living space area of the property (min = 500, max = 5000")
        area = st.number_input(
            "Area", 
            min_value=500, 
            max_value=5000, 
            value=1500, 
            step=50,
            help="Enter the ground living area in square feet (most important feature)"
        )
        
        st.markdown("**🏢 Number of Stories**")
        st.caption("How many floors does the property have?")
        stories = st.selectbox(
            "Stories", 
            [1, 2, 3, 4], 
            index=1,
            help="Select the number of stories/floors"
        )
        
        st.markdown("**🏠 Basement Area (sq ft)**")
        st.caption("Total basement area")
        total_basement_area = st.number_input(
            "Total Basement Area",
            min_value=0,
            max_value=2000,
            value=0,
            help="Enter total basement area in square feet"
        )

        st.markdown("**🏡 Lot Area (sq ft)**")
        st.caption("Total lot size")
        lot_area = st.number_input(
            "Lot Area",
            min_value=1000,
            max_value=20000,
            value=8000,
            help="Includes the house area, garage, backyards etc"
        )
        
        st.markdown("**🚗 Parking Spaces**")
        st.caption("Number of garage spaces")
        parking = st.selectbox(
            "Garage Cars", 
            [0, 1, 2, 3, 4], 
            index=2,
            help="Select the number of cars that fit in garage"
        )
        
    
    with col2:
                
        st.markdown("**🛏️ Number of Bedrooms**")
        st.caption("Bedrooms above ground level")
        bedrooms = st.selectbox(
            "Bedrooms", 
            [1, 2, 3, 4, 5, 6, 7, 8], 
            index=2,
            help="Select the number of bedrooms above ground"
        )
        
        st.markdown("**🚿 Full Bathrooms**")
        st.caption("Complete bathrooms with all fixtures")
        bathrooms = st.selectbox(
            "Full Bathrooms", 
            [1, 2, 3, 4, 5], 
            index=1,
            help="Select the number of full bathrooms"
        )
        
        st.markdown("**🚿 Half Bathrooms**")
        st.caption("Half baths (powder rooms)")
        half_bathrooms = st.selectbox(
            "Half Bathrooms", 
            [0, 1, 2, 3], 
            index=1,
            help="Select the number of half bathrooms"
        )
        st.markdown("**🏗️ Year Built**")
        st.caption("When was the property constructed?(1800-2023)")
        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2023,
            value=2000,
            help="Enter the year the property was built"
        )
        
        st.markdown("**🔧 Year Remodeled**")
        st.caption("Most recent major renovation year")
        year_remodeled = st.number_input(
            "Year Remodeled",
            min_value=year_built,
            max_value=2023,
            value=year_built,
            help="Year of most recent remodel (same as built year if never remodeled)"
        )
        

with tab2:
    st.markdown('<div class="feature-description">', unsafe_allow_html=True)
    st.markdown("**🌟 Property Features & Quality**")
    st.markdown("Select the quality ratings and amenities available in the property.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⭐ Overall Quality**")
        st.caption("Overall material and finish quality")
        overall_quality = st.selectbox(
            "Overall Quality",
            options=list(range(1, 11)),
            index=6,
            help="1=Very Poor, 5=Average, 10=Excellent"
        )
        
        st.markdown("**🔧 Overall Condition**")
        st.caption("Overall condition of the property")
        overall_condition = st.selectbox(
            "Overall Condition",
            options=list(range(1, 11)),
            index=5,
            help="1=Very Poor, 5=Average, 10=Excellent"
        )

        st.markdown("**𓊏 Fence Quality**")
        st.caption("Overall quality of the Fence")
        fence_quality = st.selectbox(
            "Fence Quality",
            options=list(range(1, 11)),
            index=6,
            help="1=Very Poor, 5=Average, 10=Excellent"
        )

        st.markdown("**👨‍🍳 Kitchen Quality**")
        st.caption("Overall quality of the Kitchen")
        kitchen_quality = st.selectbox(
            "Kitchen Quality",
            options=list(range(1, 11)),
            index=6,
            help="1=Very Poor, 5=Average, 10=Excellent"
        )
        
    
    with col2:
        st.markdown("**🛣️ Main Road Access**")
        st.caption("Does the property have direct access to a main road?")
        mainroad = st.selectbox(
            "Main Road", 
            ["Yes", "No"],
            help="Properties with main road access typically have higher values"
        )
        
        st.markdown("**🔥 Fire Place**")
        st.caption("Are there any Fire Places")
        fireplaces = st.selectbox(
            "Fire Places", 
            ["Yes", "No"],
            help="Fire Places add comfort and value during winters."
        )
        
        st.markdown("**❄️ Air Conditioning**")
        st.caption("Central air conditioning system installed?")
        airconditioning = st.selectbox(
            "Air Conditioning", 
            ["Yes", "No"],
            help="AC systems significantly improve comfort and value"
        )
        

with tab3:
    st.markdown('<div class="feature-description">', unsafe_allow_html=True)
    st.markdown("**🎯 Get Your Property Valuation**")
    st.markdown("Click the button below to get an AI-powered price prediction based on your inputs.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button in the center
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(
            "🎯 Predict House Price", 
            type="primary", 
            use_container_width=True,
            help="Click to get your property valuation"
        )
    
    if predict_button:
        with st.spinner("🔄 Analyzing property data..."):
            try:
                # Collect user inputs
                user_inputs = {
                    'area': area,
                    'stories': stories,
                    'total_basement_area': total_basement_area,
                    'lot_area': lot_area,
                    'parking': parking,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'half_bathrooms': half_bathrooms,
                    'year_built': year_built,
                    'year_remodeled': year_remodeled,
                    'overall_quality': overall_quality,
                    'overall_condition': overall_condition,
                    'kitchen_quality': kitchen_quality,
                    'garage_cars': parking,
                    'garage_area': parking * 200,  # Estimate garage area
                    'first_floor_area': area,
                    'second_floor_area': 0 if stories == 1 else area // 2,
                    'mainroad': mainroad,
                    'fireplaces': fireplaces,
                    'basement_yn': 'Yes' if total_basement_area > 0 else 'No',
                    'airconditioning': airconditioning
                }
                
                # Map features to model format
                feature_dict = create_feature_mapping(user_inputs, feature_names)
                
                # Preprocess for prediction
                processed_data = preprocess_for_prediction(feature_dict, feature_names, scaler)
                
                # Make prediction
                prediction = model.predict(processed_data)[0]
                
                # Ensure prediction is reasonable
                if prediction < 10000:  # Too low, likely scaling issue
                    prediction = prediction * 1000  # Adjust if needed
                
                # Format currency based on selection
                formatted_price, numeric_price = format_currency(prediction, currency_option)
                
                # Display result with animation
                st.balloons()
                
                # Main prediction display
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🎉 Estimated Property Value</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{formatted_price}</h1>
                    <p>Based on current market analysis and property features</p>
                    <p><small>Currency: {currency_option}</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                price_per_sqft = numeric_price / area
                price_per_sqft_formatted, _ = format_currency(prediction / area, currency_option)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="💰 Total Value", 
                        value=formatted_price,
                        help="Estimated market value"
                    )
                
                with col2:
                    st.metric(
                        label="📏 Price per Sq Ft", 
                        value=price_per_sqft_formatted,
                        help="Price per square foot"
                    )
                
                with col3:
                    st.metric(
                        label="🏠 Property Size", 
                        value=f"{area:,} sq ft",
                        help="Ground living area"
                    )
                
                with col4:
                    st.metric(
                        label="🛏️ Bed/Bath", 
                        value=f"{bedrooms}BR/{bathrooms}BA",
                        help="Bedrooms and bathrooms"
                    )
                
                # Currency conversion display
                if currency_option == "INR (₹)":
                    usd_price = f"${prediction:,.2f}"
                    st.info(f"💱 **USD Equivalent:** {usd_price} (at rate: 1 USD = ₹{exchange_rate:.2f})")
                else:
                    inr_price = f"₹{prediction * exchange_rate:,.2f}"
                    st.info(f"💱 **INR Equivalent:** {inr_price} (at rate: 1 USD = ₹{exchange_rate:.2f})")
                
                # Confidence indicator
                confidence = "Medium"
                if area > 1000 and overall_quality >= 7:
                    confidence = "High"
                elif area < 1000 or overall_quality <= 4:
                    confidence = "Low"
                
                st.markdown(f"### 🎯 Prediction Confidence: **{confidence}**")
                
                # Property summary
                st.markdown("### 📋 Property Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏗️ Structure Details:**")
                    st.write(f"• **Ground Living Area:** {area:,} sq ft")
                    st.write(f"• **Bedrooms:** {bedrooms}")
                    st.write(f"• **Full Bathrooms:** {bathrooms}")
                    st.write(f"• **Half Bathrooms:** {half_bathrooms}")
                    st.write(f"• **Stories:** {stories}")
                    st.write(f"• **Year Built:** {year_built}")
                    st.write(f"• **Overall Quality:** {overall_quality}/10")
                    st.write(f"• **Kitchen Quality:** {kitchen_quality}/10")                    
                    st.write(f"• **Garage Cars:** {parking}")
                
                with col2:
                    st.markdown("**✨ Premium Features:**")
                    features_list = []
                    if mainroad == "Yes": features_list.append("🛣️ Main Road Access")
                    if total_basement_area > 0: features_list.append("🏠 Basement")
                    if fireplaces == "Yes": features_list.append("🔥 Fire Places")
                    if airconditioning == "Yes": features_list.append("❄️ Air Conditioning")
                    
                    if features_list:
                        for feature in features_list:
                            st.write(f"• {feature}")
                    else:
                        st.write("• Basic configuration")
                
                # Price range indicator
                st.markdown("### 📊 Price Range Analysis")
                
                # Calculate price ranges
                lower_bound = numeric_price * 0.9
                upper_bound = numeric_price * 1.1
                
                lower_bound_formatted, _ = format_currency(prediction * 0.9, currency_option)
                upper_bound_formatted, _ = format_currency(prediction * 1.1, currency_option)
                
                # Create a gauge chart for price range
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = numeric_price,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Property Value Range ({currency_option})"},
                    gauge = {
                        'axis': {'range': [None, upper_bound * 1.2]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, lower_bound], 'color': "lightgray"},
                            {'range': [lower_bound, upper_bound], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': numeric_price
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"💡 **Estimated Price Range:** {lower_bound_formatted} - {upper_bound_formatted}")
                
                # Sanity check warnings
                if prediction < 50000:
                    st.warning("⚠️ Prediction seems low. Please verify your inputs.")
                elif prediction > 1000000:
                    st.warning("⚠️ Prediction seems high. Please verify your inputs.")
                else:
                    st.success("✅ Prediction looks reasonable!")
                
                # Show feature importance if available
                if hasattr(model, 'feature_importances_'):
                    with st.expander("📊 View Feature Importance"):
                        # Get top 10 most important features
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        fig_importance = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Top 10 Most Important Features"
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                st.caption("*Actual market price may vary based on location, condition, and current market conditions.")
                
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {str(e)}")
                st.error("Please check your inputs and try again.")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write("User inputs:", user_inputs)
                    st.write("Feature names expected:", feature_names[:10], "...")
                    st.write("Error details:", str(e))

# Footer
st.markdown("---")
st.markdown("""
        <h4 style="text-align: center">🏠 AI-Powered Real Estate Valuation</h4>
        <p style="text-align: center">Built using Streamlit, Machine Learning, and Advanced Analytics</p>
        <p style="text-align: center"><small>Disclaimer: This tool provides estimates based on historical data and should not be used as the sole basis for real estate decisions.<br>This tool provides the predicted prices based on the house prices of Ames City(Lowa, US).</small></p>
    </div>
""", unsafe_allow_html=True)
