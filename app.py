import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="UrbanFlow Green Mobility Intelligence",
    page_icon="🚗🍃",
    layout="wide"
)

# --- 2. GLOBAL STYLING & PLOTLY DARK CONFIGURATION ---
pio.templates.default = "plotly_dark"

def style_plotly_fig(fig):
    """Applies consistent dark theme transparency and clean margins to Plotly figures."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#fafafa", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        transition_duration=500
    )
    return fig

st.markdown("""
    <style>
    /* Global Background & Font Styling */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Glassmorphism Container Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 22px;
        border-radius: 14px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Glowing Status Pills */
    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .status-smooth { background-color: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid #2ecc71; }
    .status-moderate { background-color: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid #f1c40f; }
    .status-heavy { background-color: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid #e74c3c; }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD ARTIFACTS (Cached for performance) ---
@st.cache_resource
def load_ml_artifacts():
    try:
        cat_mappings = joblib.load('categorical_mappings.joblib')
        num_meta = joblib.load('numerical_metadata.joblib')
        
        model_wait = joblib.load('model_xgb_wait_time.joblib')
        model_emit = joblib.load('model_xgb_emissions.joblib')
        model_class = joblib.load('model_xgb_congestion_classifier.joblib')
        label_encoder = joblib.load('congestion_label_encoder.joblib')
        feature_importances_df = joblib.load('feature_importances.joblib')
        
        return cat_mappings, num_meta, model_wait, model_emit, model_class, label_encoder, feature_importances_df
    except FileNotFoundError as e:
        st.error(f"Missing artifact file: {e}. Please ensure all .joblib files are in the app directory.")
        st.stop()

cat_mappings, num_meta, model_wait, model_emit, model_class, label_encoder, feature_importances_df = load_ml_artifacts()

expected_features = list(cat_mappings.keys()) + num_meta['binary_features'] + list(num_meta['continuous_features'].keys())

# --- 4. SESSION STATE INITIALIZATION ---
if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "baseline_predictions" not in st.session_state:
    st.session_state.baseline_predictions = None

if "baseline_df" not in st.session_state:
    st.session_state.baseline_df = None

# Initialize widget state
for col_name, details in cat_mappings.items():
    if f"ui_{col_name}" not in st.session_state:
        st.session_state[f"ui_{col_name}"] = details['options'][0]

for feat in num_meta['binary_features']:
    if f"ui_{feat}" not in st.session_state:
        st.session_state[f"ui_{feat}"] = 'No'

for feat, meta in num_meta['continuous_features'].items():
    if f"ui_{feat}" not in st.session_state:
        default_v = float(meta['default'])
        st.session_state[f"ui_{feat}"] = max(float(meta['min']), min(float(meta['max']), default_v))

def randomize_inputs():
    """Randomizes all widget keys in session state."""
    for col_name, details in cat_mappings.items():
        st.session_state[f"ui_{col_name}"] = random.choice(details['options'])
        
    for feat in num_meta['binary_features']:
        st.session_state[f"ui_{feat}"] = random.choice(['No', 'Yes'])
        
    for feat, meta in num_meta['continuous_features'].items():
        rand_val = round(random.uniform(float(meta['min']), float(meta['max'])), 2)
        st.session_state[f"ui_{feat}"] = rand_val
        
    st.session_state.predictions = None
    st.session_state.input_df = None
    st.toast("Scenario randomized! Screen inputs updated.", icon="✨")

def clear_predictions():
    """Clears the output prediction results and baseline."""
    st.session_state.predictions = None
    st.session_state.input_df = None
    st.session_state.baseline_predictions = None
    st.session_state.baseline_df = None
    st.toast("Prediction results and baseline cleared.", icon="🧹")

def generate_html_report(input_df, predictions):
    """Generates a professional executive HTML report that can be printed/saved as PDF."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    inputs_html = ""
    if input_df is not None and not input_df.empty:
        for col in input_df.columns:
            val = input_df.iloc[0][col]
            display_val = val
            if col in cat_mappings:
                id_to_label = cat_mappings[col]['id_to_label']
                if int(val) in id_to_label:
                    display_val = id_to_label[int(val)]
            inputs_html += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><b>{col.replace('_', ' ').title()}</b></td><td style='padding: 8px; border: 1px solid #ddd;'>{display_val}</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Intelligence Executive Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            .header {{ background: #1E3A8A; color: white; padding: 20px; border-radius: 8px; }}
            .metric-box {{ display: inline-block; background: #F3F4F6; padding: 15px 25px; margin: 10px 5px; border-radius: 6px; border-left: 5px solid #2563EB; }}
            .section {{ margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background: #F9FAFB; padding: 10px; border: 1px solid #ddd; text-align: left; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚗🍃 UrbanFlow Green Mobility Intelligence</h1>
            <p>Executive Node Evaluation Report | Generated: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>📊 Key Predictive Metrics</h2>
            <div class="metric-box">
                <h3>Congestion Level</h3>
                <p style="font-size: 20px; font-weight: bold; color: #1E3A8A;">{predictions['class']}</p>
            </div>
            <div class="metric-box">
                <h3>Estimated Wait Time</h3>
                <p style="font-size: 20px; font-weight: bold; color: #2563EB;">{predictions['wait_time']:.2f} mins</p>
            </div>
            <div class="metric-box">
                <h3>Carbon Emissions</h3>
                <p style="font-size: 20px; font-weight: bold; color: #059669;">{predictions['emission']:.2f} kg</p>
            </div>
        </div>

        <div class="section">
            <h2>🛠️ Evaluated Intersection Parameters</h2>
            <table>
                <tr><th>Parameter Name</th><th>Configured Value</th></tr>
                {inputs_html}
            </table>
        </div>
        
        <div class="section" style="margin-top: 40px; font-size: 12px; color: #666; border-top: 1px solid #ddd; padding-top: 10px;">
            <p>UrbanFlow Green Mobility Intelligence Platform — Automated ML Pipeline Report. Confidential & Proprietary.</p>
        </div>
    </body>
    </html>
    """
    return html_content

# --- 5. SIDEBAR CONTROLS & MODE SELECTOR ---
st.sidebar.title("🎮 Control Panel")
app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    options=["Interactive Single Node", "Batch CSV Upload"],
    index=0
)

st.sidebar.divider()

if app_mode == "Interactive Single Node":
    if st.sidebar.button("🎲 Randomize Scenario", use_container_width=True):
        randomize_inputs()
        st.rerun()

    if st.sidebar.button("🗑️ Clear Results & Baseline", use_container_width=True):
        clear_predictions()
        st.rerun()

st.sidebar.info("💡 Tip: Toggle between interactive mode and batch CSV evaluation depending on your analysis scope.")

# --- 6. MAIN HEADER ---
st.title("🚗🍃 UrbanFlow Green Mobility Intelligence")
st.markdown("Evaluate real-time intersection node performance, congestion levels, wait times, and carbon emission metrics.")

# ==============================================================================
# MODE 1: BATCH CSV UPLOAD MODE
# ==============================================================================
if app_mode == "Batch CSV Upload":
    st.subheader("📂 Bulk CSV Batch Evaluation")
    st.markdown("Upload a CSV file containing intersection parameters to run predictions across all rows simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Raw Input Features CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data", batch_df.head())
        
        if st.button("Process Batch Predictions", use_container_width=True):
            with st.spinner("Analyzing bulk IoT telemetry & computing XGBoost batch inferences..."):
                try:
                    processed_batch = batch_df.copy()
                    
                    for col_name, details in cat_mappings.items():
                        if col_name in processed_batch.columns:
                            if processed_batch[col_name].dtype == object:
                                processed_batch[col_name] = processed_batch[col_name].map(details['label_to_id']).fillna(0)

                    aligned_batch = processed_batch.reindex(columns=expected_features, fill_value=0)
                    
                    log_wait_preds = model_wait.predict(aligned_batch)
                    batch_df['Predicted_Wait_Time_mins'] = np.expm1(log_wait_preds)
                    batch_df['Predicted_Wait_Time_mins'] = batch_df['Predicted_Wait_Time_mins'].clip(lower=0.0).round(2)
                    
                    log_emit_preds = model_emit.predict(aligned_batch)
                    batch_df['Predicted_Carbon_Emission_kg'] = np.expm1(log_emit_preds)
                    batch_df['Predicted_Carbon_Emission_kg'] = batch_df['Predicted_Carbon_Emission_kg'].clip(lower=0.0).round(2)
                    
                    encoded_class_preds = model_class.predict(aligned_batch)
                    batch_df['Predicted_Congestion_Level'] = label_encoder.inverse_transform(encoded_class_preds)
                    
                    st.success("✅ Batch processing completed successfully!")
                    st.toast("Batch CSV predictions generated successfully!", icon="🫡")
                    st.write("### Enriched Dataset with Predictions", batch_df.head())
                    
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Batch Predictions CSV",
                        data=csv_data,
                        file_name="urban_flow_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error executing batch inference: {e}")

# ==============================================================================
# MODE 2: INTERACTIVE SINGLE NODE MODE
# ==============================================================================
else:
    user_inputs = {}
    
    with st.form("prediction_form"):
        st.markdown("### 🛠️ Configure Intersection Parameters")
        
        with st.expander("🔍 Location & Environmental Categories", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            cat_emojis = {
                'city_zone': "🏙️", 'road_type': "🛣️", 'peak_period': "⏰",
                'weather_condition': "🌦️", 'road_condition': "🚧", 'iot_sensor_health': "📡"
            }
            cat_help_text = {
                'city_zone': "Metropolitan sector or urban zone where the node is situated.",
                'road_type': "Classification of the roadway structure.",
                'peak_period': "Daily operational time band.",
                'weather_condition': "Atmospheric weather state.",
                'road_condition': "Physical surface state of the asphalt.",
                'iot_sensor_health': "Operational status of surrounding IoT sensors."
            }
            
            cat_keys = list(cat_mappings.keys())
            for i, col_name in enumerate(cat_keys):
                target_col = [col1, col2, col3][i % 3]
                with target_col:
                    details = cat_mappings[col_name]
                    emoji = cat_emojis.get(col_name, "📌")
                    readable_label = f"{emoji} {col_name.replace('_', ' ').title()}"
                    tooltip = cat_help_text.get(col_name, "Categorical parameter.")
                    
                    selected_label = st.selectbox(
                        readable_label, options=details['options'], 
                        help=tooltip, key=f"ui_{col_name}"
                    )
                    user_inputs[col_name] = details['label_to_id'][selected_label]

        with st.expander("🚩 Operational Flags & Conditions", expanded=False):
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            binary_feats = num_meta['binary_features']
            
            binary_emojis = {
                'is_weekend': "📅", 'is_holiday': "🎉", 'rush_hour': "🚕",
                'nearby_school': "🏫", 'nearby_hospital': "🏥", 'nearby_market': "🛒",
                'construction_activity': "🏗️", 'public_event': "🏟️",
                'accident_reported': "⚠️", 'emergency_vehicle_detected': "🚑"
            }
            binary_help_text = {
                'is_weekend': "Saturday or Sunday flag.", 'is_holiday': "Official public holiday flag.",
                'rush_hour': "Peak commuter window flag.", 'nearby_school': "School zone proximity.",
                'nearby_hospital': "Medical emergency route proximity.", 'nearby_market': "Commercial activity zone.",
                'construction_activity': "Roadwork or lane restrictions.", 'public_event': "Concerts/sports events.",
                'accident_reported': "Active collision reported.", 'emergency_vehicle_detected': "Siren/transponder alert."
            }
            
            for i, feat in enumerate(binary_feats):
                target_col = [b_col1, b_col2, b_col3, b_col4][i % 4]
                with target_col:
                    emoji = binary_emojis.get(feat, "🔹")
                    readable_label = f"{emoji} {feat.replace('_', ' ').title()}"
                    tooltip = binary_help_text.get(feat, "Boolean operational condition flag.")
                    
                    choice = st.radio(
                        readable_label, options=['No', 'Yes'], 
                        horizontal=True, key=f"ui_{feat}", help=tooltip
                    )
                    user_inputs[feat] = 1 if choice == 'Yes' else 0

        with st.expander("📈 Continuous Metrics & Sensors", expanded=False):
            c_col1, c_col2, c_col3 = st.columns(3)
            continuous_feats = num_meta['continuous_features']
            
            continuous_emojis = {
                'latitude': "🌐", 'longitude': "🗺️", 'lanes': "🛣️", 'speed_limit': "🛑",
                'vehicle_count': "🚙", 'average_speed': "⏩", 'heavy_vehicle_count': "🚚",
                'motorcycle_count': "🏍️", 'public_transport_count': "🚌", 'traffic_density': "👥",
                'queue_length': "📏", 'average_wait_time': "⏳", 'hour': "🕒", 'day_of_week': "📆",
                'temperature': "🌡️", 'rainfall_mm': "🌧️", 'visibility_km': "🌫️",
                'air_quality_index': "🍃", 'parking_occupancy': "🅿️", 'traffic_camera_count': "📷",
                'signal_cycle_seconds': "🚥", 'green_light_duration': "🟢", 'traffic_flow_rate': "🔀",
                'log_queue_length': "🔄"
            }
            
            unit_labels = {
                'latitude': ' (°)', 'longitude': ' (°)', 'speed_limit': ' (km/h)',
                'average_speed': ' (km/h)', 'temperature': ' (°C)', 'rainfall_mm': ' (mm)',
                'visibility_km': ' (km)', 'signal_cycle_seconds': ' (s)', 'green_light_duration': ' (s)',
                'average_wait_time': ' (mins)', 'queue_length': ' (m)', 'traffic_flow_rate': ' (veh/h)',
                'vehicle_count': ' (veh)', 'heavy_vehicle_count': ' (veh)', 'motorcycle_count': ' (veh)',
                'public_transport_count': ' (veh)'
            }
            
            continuous_help_text = {
                'latitude': "Geographic coordinate latitude position across the urban network.",
                'longitude': "Geographic coordinate longitude position across the urban network.",
                'lanes': "Total number of active lanes available at the intersection node.",
                'speed_limit': "Legal maximum speed restriction configured for the roadway.",
                'vehicle_count': "Total count of passenger vehicles detected passing through the node.",
                'average_speed': "Mean observed speed of traffic flow across the sector.",
                'heavy_vehicle_count': "Count of freight trucks and heavy commercial vehicles.",
                'motorcycle_count': "Count of two-wheeled motorized vehicles.",
                'public_transport_count': "Count of active buses and mass transit units.",
                'traffic_density': "Calculated concentration of vehicles per square unit.",
                'queue_length': "Measured physical backup length of vehicles waiting at the light.",
                'average_wait_time': "Observed average delay experienced by commuters.",
                'hour': "Current hour of the daily operational timeline (0–23).",
                'day_of_week': "Numerical day index of the weekly cycle.",
                'temperature': "Ambient atmospheric temperature recorded by IoT weather sensors.",
                'rainfall_mm': "Precipitation volume measured in millimeters.",
                'visibility_km': "Atmospheric clarity distance measured in kilometers.",
                'air_quality_index': "Environmental pollution index score at the location.",
                'parking_occupancy': "Percentage of utilized street parking capacity.",
                'traffic_camera_count': "Number of active closed-circuit monitoring cameras.",
                'signal_cycle_seconds': "Total duration length of a complete traffic light phase cycle.",
                'green_light_duration': "Active green phase window duration.",
                'traffic_flow_rate': "Rate of vehicle passage volume over time.",
                'log_queue_length': "Log-transformed statistical index measuring traffic backup severity."
            }
            
            i = 0
            for feat, meta in continuous_feats.items():
                target_col = [c_col1, c_col2, c_col3][i % 3]
                with target_col:
                    emoji = continuous_emojis.get(feat, "🔢")
                    unit = unit_labels.get(feat, '')
                    
                    display_name = "Traffic Backup Severity Index" if feat == 'log_queue_length' else feat.replace('_', ' ').title()
                    # Clean label without jarring min/max numbers inline
                    readable_label = f"{emoji} {display_name}{unit}"
                    
                    # Concise summary explanation + explicit min/max bounds inside the help tooltip
                    base_desc = continuous_help_text.get(feat, "Continuous sensor telemetry metric.")
                    tooltip = f"{base_desc} (Allowed Range — Min: {meta['min']}, Max: {meta['max']})"
                    
                    step_val = float(meta.get('step', 0.1 if 'latitude' in feat or 'longitude' in feat else 1.0 if meta['min'].is_integer() and meta['max'].is_integer() else 0.1))
                    
                    user_inputs[feat] = st.slider(
                        label=readable_label,
                        min_value=float(meta['min']),
                        max_value=float(meta['max']),
                        step=step_val,
                        help=tooltip,
                        key=f"ui_{feat}"
                    )
                i += 1

        st.divider()
        submit_button = st.form_submit_button(label="Run Traffic Prediction Analysis", use_container_width=True)

    # --- 7. INFERENCE & SESSION STORAGE PIPELINE ---
    if submit_button:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        st.session_state.input_df = input_df

        with st.spinner("Analyzing node telemetry & computing multi-target XGBoost inferences..."):
            try:
                log_wait_pred = model_wait.predict(input_df)[0]
                wait_time_actual = max(0.0, float(np.expm1(log_wait_pred)))
                
                log_emit_pred = model_emit.predict(input_df)[0]
                emission_actual = max(0.0, float(np.expm1(log_emit_pred)))
                
                encoded_class_pred = model_class.predict(input_df)
                class_pred_text = str(label_encoder.inverse_transform(encoded_class_pred)[0])

                st.session_state.predictions = {
                    'class': class_pred_text,
                    'wait_time': wait_time_actual,
                    'emission': emission_actual
                }
                st.toast("Traffic intelligence analysis completed successfully!", icon="🫡")
            except Exception as e:
                st.error(f"An error occurred during model inference: {e}")

    # --- 8. DISPLAY RESULTS, GIS MAP, GAUGES & EXPLAINABILITY ---
    if st.session_state.predictions is not None:
        st.markdown("---")
        
        res_col_hdr1, res_col_hdr2 = st.columns([3, 1])
        with res_col_hdr1:
            st.subheader("📋 Latest Traffic Intelligence Results")
        with res_col_hdr2:
            html_report = generate_html_report(st.session_state.input_df, st.session_state.predictions)
            st.download_button(
                label="📥 Download Executive Report",
                data=html_report,
                file_name=f"UrbanFlow_Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        # --- SCENARIO COMPARISON PIN BUTTON ---
        col_pin1, col_pin2 = st.columns([1, 3])
        with col_pin1:
            if st.button("📌 Pin as Baseline Scenario", use_container_width=True):
                st.session_state.baseline_predictions = st.session_state.predictions.copy()
                st.session_state.baseline_df = st.session_state.input_df.copy()
                st.toast("Current configuration pinned as baseline!", icon="📌")
        with col_pin2:
            if st.session_state.baseline_predictions is not None:
                b_wait = st.session_state.baseline_predictions['wait_time']
                b_emit = st.session_state.baseline_predictions['emission']
                curr_wait = st.session_state.predictions['wait_time']
                curr_emit = st.session_state.predictions['emission']
                
                wait_diff = curr_wait - b_wait
                emit_diff = curr_emit - b_emit
                
                wait_color = "red" if wait_diff > 0 else "green"
                emit_color = "red" if emit_diff > 0 else "green"
                
                st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 8px 15px; border-radius: 8px; font-size: 0.85rem;">
                        <b>Delta vs. Baseline:</b> 
                        Wait Time: <span style="color: {wait_color}; font-weight: bold;">{wait_diff:+.2f}m</span> | 
                        Emissions: <span style="color: {emit_color}; font-weight: bold;">{emit_diff:+.2f}kg CO₂</span>
                    </div>
                """, unsafe_allow_html=True)

        g_col1, g_col2, g_col3 = st.columns([1, 1, 1])
        
        with g_col1:
            congestion_cls = st.session_state.predictions['class']
            pill_class = "status-smooth"
            if "Moderate" in str(congestion_cls):
                pill_class = "status-moderate"
            elif "Heavy" in str(congestion_cls) or "Severe" in str(congestion_cls):
                pill_class = "status-heavy"
                
            st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <h4 style="margin: 0; color: #9ca3af;">Predicted Congestion Level</h4>
                    <div style="margin: 15px 0;">
                        <span class="status-pill {pill_class}">{congestion_cls}</span>
                    </div>
                    <p style="font-size: 0.85rem; color: #9ca3af; margin: 0;">Node classified based on active sensors.</p>
                </div>
            """, unsafe_allow_html=True)

        with g_col2:
            fig_wait = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = st.session_state.predictions['wait_time'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Estimated Wait Time (Mins)"},
                gauge = {
                    'axis': {'range': [0, 60]},
                    'bar': {'color': "#2563EB"},
                    'steps': [
                        {'range': [0, 10], 'color': "rgba(46, 204, 113, 0.2)"},
                        {'range': [10, 25], 'color': "rgba(241, 196, 15, 0.2)"},
                        {'range': [25, 60], 'color': "rgba(231, 76, 60, 0.2)"}
                    ]
                }
            ))
            fig_wait = style_plotly_fig(fig_wait)
            fig_wait.update_layout(height=220)
            st.plotly_chart(fig_wait, use_container_width=True)

        with g_col3:
            fig_emit = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = st.session_state.predictions['emission'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Carbon Emission (CO₂ Kg)"},
                gauge = {
                    'axis': {'range': [0, 300]},
                    'bar': {'color': "#059669"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(46, 204, 113, 0.2)"},
                        {'range': [50, 150], 'color': "rgba(241, 196, 15, 0.2)"},
                        {'range': [150, 300], 'color': "rgba(231, 76, 60, 0.2)"}
                    ]
                }
            ))
            fig_emit = style_plotly_fig(fig_emit)
            fig_emit.update_layout(height=220)
            st.plotly_chart(fig_emit, use_container_width=True)

        # --- GIS MAP NODE VISUALIZER & LOCAL EXPLAINABILITY ---
        map_col, explain_col = st.columns([1, 1])

        with map_col:
            st.markdown("#### 📍 Evaluated Intersection Node GIS Pin")
            if "latitude" in st.session_state.input_df.columns and "longitude" in st.session_state.input_df.columns:
                lat_val = float(st.session_state.input_df['latitude'].iloc[0])
                lon_val = float(st.session_state.input_df['longitude'].iloc[0])
                node_map_df = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                st.map(node_map_df, zoom=13, use_container_width=True)
            else:
                st.info("Latitude and longitude coordinates not available.")

        with explain_col:
            st.markdown("#### 🧩 Local Feature Contribution Drivers")
            st.markdown("Top active telemetry triggers influencing this specific node prediction:")
            
            try:
                sample_row = st.session_state.input_df.iloc[0]
                local_drivers = []
                
                # Check key continuous traffic drivers
                if 'vehicle_count' in sample_row and sample_row['vehicle_count'] > 50:
                    local_drivers.append(f"High vehicle volume ({int(sample_row['vehicle_count'])} vehicles)")
                if 'average_speed' in sample_row and sample_row['average_speed'] < 30:
                    local_drivers.append(f"Restricted average speed ({sample_row['average_speed']} km/h)")
                if 'queue_length' in sample_row and sample_row['queue_length'] > 100:
                    local_drivers.append(f"Extended queue backup ({sample_row['queue_length']}m)")
                if 'heavy_vehicle_count' in sample_row and sample_row['heavy_vehicle_count'] > 15:
                    local_drivers.append(f"Elevated heavy freight/truck presence ({int(sample_row['heavy_vehicle_count'])} units)")
                
                # Check binary operational flags
                for b_feat in num_meta['binary_features']:
                    if b_feat in sample_row and sample_row[b_feat] == 1:
                        clean_name = b_feat.replace('_', ' ').title()
                        local_drivers.append(f"Active flag: {clean_name}")
                
                if local_drivers:
                    for driver in local_drivers[:4]:  # Show top 4 active drivers
                        st.markdown(f"- **{driver}**: Accelerates queue wait times and CO₂ emissions.")
                else:
                    st.markdown("- **Normal Operating Parameters**: Traffic flow rate and sensor telemetry remain within nominal baseline thresholds.")
            except Exception as e:
                st.caption(f"Local explainability breakdown unavailable: {e}")

        # --- GLOBAL FEATURE IMPORTANCE MODULE (DYNAMICALLY HIGHLIGHTED) ---
        st.markdown("### ⚙️ Global Model Intelligence")
        st.markdown("Global XGBoost feature weights across the urban network. *Green bars indicate features active in your current scenario.*")
        
        try:
            fi_subset = feature_importances_df.sort_values(by='Importance', ascending=False).head(8).copy()
            display_features = [str(f).replace('_', ' ').title() for f in fi_subset['Feature']]
            
            sample_row = st.session_state.input_df.iloc[0]
            bar_colors = []
            
            for _, row in fi_subset.iterrows():
                f_raw = str(row['Feature'])
                # Clean up display feature names like "Cat Peak Period Evening Peak" or "Num Traffic Density"
                clean_feat = f_raw.replace('Cat ', '').replace('Num ', '').strip().lower().replace(' ', '_')
                
                is_active = False
                for col_key, val in sample_row.items():
                    if clean_feat in str(col_key).lower() or str(col_key).lower() in clean_feat:
                        if isinstance(val, (int, float, np.number)) and val > 0:
                            is_active = True
                            break
                        elif isinstance(val, str) and val != '' and val != 'No':
                            is_active = True
                            break
                
                # Highlight active scenario drivers in emerald green, others in professional blue
                bar_colors.append('#059669' if is_active else '#3B82F6')

            fig_imp = go.Figure(go.Bar(
                x=fi_subset['Importance'],
                y=display_features,
                orientation='h',
                marker=dict(color=bar_colors)
            ))
            fig_imp.update_layout(
                title="Top 8 Global Factors (Green = Active in Current Node)",
                xaxis_title="Relative Feature Importance Score",
                yaxis=dict(autorange="reversed"),
                height=300
            )
            fig_imp = style_plotly_fig(fig_imp)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.caption(f"Feature explainability chart unavailable: {e}")
    else:
        st.info("💡 Adjust the parameters above or use the sidebar controls, then click **'Run Traffic Prediction Analysis'** to evaluate traffic conditions.")