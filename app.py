import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Urban Traffic & Congestion Intelligence",
    page_icon="🚦",
    layout="wide"
)

# --- 2. LOAD ARTIFACTS (Cached for performance) ---
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

# --- 3. SESSION STATE INITIALIZATION ---
if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "input_df" not in st.session_state:
    st.session_state.input_df = None

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
    """Clears the output prediction results."""
    st.session_state.predictions = None
    st.session_state.input_df = None
    st.toast("Prediction results cleared.", icon="🧹")

def generate_html_report(input_df, predictions):
    """Generates a professional executive HTML report that can be printed/saved as PDF."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format input parameters for display table
    inputs_html = ""
    if input_df is not None and not input_df.empty:
        for col in input_df.columns:
            val = input_df.iloc[0][col]
            # Reverse map categorical IDs if needed
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
            <h1>🚦 Urban Traffic & Congestion Intelligence</h1>
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
            <p>CityFlow MLOps Intelligence Platform — Automated ML Pipeline Report. Confidential & Proprietary.</p>
        </div>
    </body>
    </html>
    """
    return html_content

# --- 4. SIDEBAR CONTROLS & MODE SELECTOR ---
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

    if st.sidebar.button("🗑️ Clear Results", use_container_width=True):
        clear_predictions()
        st.rerun()

st.sidebar.info("💡 Tip: Toggle between interactive mode and batch CSV evaluation depending on your analysis scope.")

# --- 5. MAIN HEADER ---
st.title("🚦 Urban Traffic & Congestion Prediction Intelligence")
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
                st.write("### Enriched Dataset with Predictions", batch_df.head())
                
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Predictions CSV",
                    data=csv_data,
                    file_name="traffic_batch_predictions.csv",
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
        
        with st.expander("📍 Location & Environmental Categories", expanded=True):
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
                'is_weekend': "📅", 'is_holiday': "🎉", 'rush_hour': "🚗",
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
                'latitude': "🌐", 'longitude': "🗺️", 'lanes': "🛣️", 'speed_limit': "⏱️",
                'vehicle_count': "🚙", 'average_speed': "🚗", 'heavy_vehicle_count': "🚚",
                'motorcycle_count': "🏍️", 'public_transport_count': "🚌", 'traffic_density': "📊",
                'queue_length': "📏", 'average_wait_time': "⏳", 'hour': "🕒", 'day_of_week': "📆",
                'temperature': "🌡️", 'rainfall_mm': "🌧️", 'visibility_km': "🌫️",
                'air_quality_index': "🍃", 'parking_occupancy': "🅿️", 'traffic_camera_count': "📷",
                'signal_cycle_seconds': "🚦", 'green_light_duration': "🟢", 'traffic_flow_rate': "🌊",
                'log_queue_length': "📈"
            }
            
            unit_labels = {
                'latitude': ' (°)', 'longitude': ' (°)', 'speed_limit': ' (km/h)',
                'average_speed': ' (km/h)', 'temperature': ' (°C)', 'rainfall_mm': ' (mm)',
                'visibility_km': ' (km)', 'signal_cycle_seconds': ' (s)', 'green_light_duration': ' (s)',
                'average_wait_time': ' (mins)', 'queue_length': ' (m)', 'traffic_flow_rate': ' (veh/h)',
                'vehicle_count': ' (veh)', 'heavy_vehicle_count': ' (veh)', 'motorcycle_count': ' (veh)',
                'public_transport_count': ' (veh)'
            }
            
            i = 0
            for feat, meta in continuous_feats.items():
                target_col = [c_col1, c_col2, c_col3][i % 3]
                with target_col:
                    emoji = continuous_emojis.get(feat, "🔢")
                    unit = unit_labels.get(feat, '')
                    
                    display_name = "Traffic Backup Severity Index" if feat == 'log_queue_length' else feat.replace('_', ' ').title()
                    readable_label = f"{emoji} {display_name}{unit} [Min: {meta['min']}, Max: {meta['max']}]"
                    step_val = float(meta.get('step', 0.1 if 'latitude' in feat or 'longitude' in feat else 1.0 if meta['min'].is_integer() and meta['max'].is_integer() else 0.1))
                    
                    user_inputs[feat] = st.slider(
                        label=readable_label,
                        min_value=float(meta['min']),
                        max_value=float(meta['max']),
                        step=step_val,
                        key=f"ui_{feat}"
                    )
                i += 1

        st.divider()
        submit_button = st.form_submit_button(label="Run Traffic Prediction Analysis", use_container_width=True)

    # --- 6. INFERENCE & SESSION STORAGE PIPELINE ---
    if submit_button:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        st.session_state.input_df = input_df

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
            st.toast("Analysis re-run successfully with updated parameters!", icon="🔄")
        except Exception as e:
            st.error(f"An error occurred during model inference: {e}")

    # --- 7. DISPLAY RESULTS, GAUGES & REPORT DOWNLOAD ---
    if st.session_state.predictions is not None:
        st.markdown("---")
        
        # Results Header & Download Report Button
        res_col_hdr1, res_col_hdr2 = st.columns([3, 1])
        with res_col_hdr1:
            st.subheader("📊 Latest Traffic Intelligence Results")
        with res_col_hdr2:
            html_report = generate_html_report(st.session_state.input_df, st.session_state.predictions)
            st.download_button(
                label="📥 Download Executive Report",
                data=html_report,
                file_name=f"Traffic_Intelligence_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        g_col1, g_col2, g_col3 = st.columns([1, 1, 1])
        
        with g_col1:
            st.metric(label="🚦 Predicted Congestion Level", value=st.session_state.predictions['class'])
            st.info(f"**Node Status:** Classified as **{st.session_state.predictions['class']}** congestion based on current volume and environmental sensors.")

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
                        {'range': [0, 10], 'color': "#DCFCE7"},
                        {'range': [10, 25], 'color': "#FEF08A"},
                        {'range': [25, 60], 'color': "#FCA5A5"}
                    ]
                }
            ))
            fig_wait.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
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
                        {'range': [0, 50], 'color': "#D1FAE5"},
                        {'range': [50, 150], 'color': "#FEF08A"},
                        {'range': [150, 300], 'color': "#FCA5A5"}
                    ]
                }
            ))
            fig_emit.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_emit, use_container_width=True)

        # --- FEATURE IMPORTANCE / EXPLAINABILITY MODULE ---
        st.markdown("### 🔍 Model Explainability & Key Drivers")
        st.markdown("Top feature contributions driving the classification and regression predictions:")
        
        try:
            fi_subset = feature_importances_df.sort_values(by='Importance', ascending=False).head(8)
            display_features = [str(f).replace('_', ' ').title() for f in fi_subset['Feature']]
            
            fig_imp = go.Figure(go.Bar(
                x=fi_subset['Importance'],
                y=display_features,
                orientation='h',
                marker=dict(color='#3B82F6')
            ))
            fig_imp.update_layout(
                title="Top 8 Global Factors Influencing Traffic Predictions",
                xaxis_title="Relative Feature Importance Score",
                yaxis=dict(autorange="reversed"),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.caption(f"Feature explainability chart unavailable: {e}")
    else:
        st.info("💡 Adjust the parameters above or use the sidebar controls, then click **'Run Traffic Prediction Analysis'** to evaluate traffic conditions.")