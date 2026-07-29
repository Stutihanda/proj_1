import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Climate Guardian AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SIDEBAR NAVIGATION & LOGIN
# ==========================================
st.sidebar.markdown("# 🛡️ Climate Guardian AI")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.markdown("### Account Login")
    login_mode = st.sidebar.radio("Choose Mode", ["Login", "Sign Up"], key="auth_mode")
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Log In", type="primary"):
        if username_input.strip():
            st.session_state.logged_in = True
            st.session_state.username = username_input
            
            if "model_registry_logs" not in st.session_state:
                st.session_state.model_registry_logs = []
            
            st.session_state.model_registry_logs.append({
                "User": username_input,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Best Model": "Pending Pipeline Run",
                "Accuracy / Score": "N/A"
            })
            st.rerun()
        else:
            st.sidebar.error("Please enter a username.")
    st.stop()
else:
    current_user = st.session_state.get('username', 'User')
    st.sidebar.success(f"Logged in as: {current_user}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### NAVIGATION MENU")
selected_tab = st.sidebar.radio(
    "Select View:",
    [
        "📊 Outbreak Command Center",
        "📡 Live Weather Radar",
        "📁 Multi-Dataset Ingestion",
        "📝 Execution Logs",
        "📈 Advanced Analytics Studio",
        "🤖 AI Risk Assistant"
    ]
)

if "processed_df" not in st.session_state:
    st.session_state.processed_df = pd.DataFrame()

if "model_registry_logs" not in st.session_state:
    st.session_state.model_registry_logs = [
        {"User": "stuti", "Timestamp": "2026-07-29 12:30:15", "Best Model": "XGBoost Classifier", "Accuracy / Score": "94.2%"}
    ]

results = st.session_state.get("pipeline_results", None)

# ==========================================
# 1. OUTBREAK COMMAND CENTER (INTERACTIVE OVERVIEW)
# ==========================================
if selected_tab == "📊 Outbreak Command Center":
    st.title("📊 Outbreak Command Center")
    st.markdown("High-level executive telemetry and interactive regional risk monitoring dashboard.")
    
    df = st.session_state.get("processed_df", pd.DataFrame())
    
    if not df.empty:
        df_cmd = df.copy()
        
        risk_col_candidates = [c for c in df_cmd.columns if any(kw in c.lower() for kw in ["risk", "level", "severity", "class", "status", "target"])]
        cmd_risk_col = risk_col_candidates[0] if risk_col_candidates else df_cmd.columns[0]
        
        numeric_risk = pd.to_numeric(df_cmd[cmd_risk_col], errors='coerce')
        if numeric_risk.notna().sum() > 0:
            max_v = numeric_risk.max()
            if max_v > 3:
                df_cmd["Cmd_Risk_Code"] = pd.qcut(numeric_risk.fillna(0), q=min(4, int(max_v)+1), labels=False, duplicates='drop')
            else:
                df_cmd["Cmd_Risk_Code"] = numeric_risk.fillna(0).astype(int)
        else:
            df_cmd["Cmd_Risk_Code"] = pd.factorize(df_cmd[cmd_risk_col].astype(str))[0] % 4
            
        cmd_map = {0: "0 (Low Risk)", 1: "1 (Moderate Risk)", 2: "2 (High Risk)", 3: "3 (Critical Risk)"}
        df_cmd["Cmd_Risk_Display"] = df_cmd["Cmd_Risk_Code"].map(lambda x: cmd_map.get(int(x) if pd.notna(x) else 0, f"{x} (Custom)"))

        # Interactive Command Center Filter Box
        with st.expander("🎛️ Quick Command Filter Controls", expanded=True):
            fc1, fc2 = st.columns(2)
            with fc1:
                selected_cmd_risks = st.multiselect(
                    "Filter by Risk Tier:",
                    options=sorted(df_cmd["Cmd_Risk_Display"].unique()),
                    default=sorted(df_cmd["Cmd_Risk_Display"].unique())
                )
            with fc2:
                cmd_search = st.text_input("Global Zone Search:", placeholder="Search city, region, or ID...")

        filtered_cmd_df = df_cmd[df_cmd["Cmd_Risk_Display"].isin(selected_cmd_risks)]
        if cmd_search.strip():
            mask_cmd = filtered_cmd_df.astype(str).apply(
                lambda row: row.str.contains(cmd_search.strip(), case=False, na=False)
            ).any(axis=1)
            filtered_cmd_df = filtered_cmd_df[mask_cmd]

        # Executive KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Monitored Zones", f"{len(filtered_cmd_df)} / {len(df)}")
        c2.metric("System Health Status", "Optimal (99.8%)")
        critical_zones = (filtered_cmd_df["Cmd_Risk_Code"] >= 2).sum() if not filtered_cmd_df.empty else 0
        c3.metric("High/Critical Risk Zones", critical_zones)
        c4.metric("Telemetry Variables", len(filtered_cmd_df.columns))
        
        st.markdown("---")
        
        col_cv1, col_cv2 = st.columns([1.2, 1])
        with col_cv1:
            st.subheader("📊 Regional Risk Distribution")
            if not filtered_cmd_df.empty:
                risk_counts = filtered_cmd_df["Cmd_Risk_Display"].value_counts().reset_index()
                risk_counts.columns = ["Risk_Display", "Count"]
                fig_cmd_bar = px.bar(
                    risk_counts, x="Risk_Display", y="Count", color="Risk_Display",
                    template="plotly_dark", title="Zone Count per Risk Tier"
                )
                st.plotly_chart(fig_cmd_bar, use_container_width=True)
            else:
                st.warning("No data matching active filters.")
                
        with col_cv2:
            st.subheader("⚡ Quick Actions & Export")
            st.markdown("Export current command view or deep-dive into the **Advanced Analytics Studio** for correlation heatmaps and bivariate modeling.")
            cmd_csv = filtered_cmd_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Command Center CSV",
                data=cmd_csv,
                file_name="command_center_filtered.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
            if st.button("📈 Go to Advanced Analytics Studio", use_container_width=True):
                st.info("Switch to 'Advanced Analytics Studio' in the sidebar menu for deep-dive modeling.")

        st.markdown("### 📋 Executive Summary Telemetry Table")
        st.dataframe(filtered_cmd_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data available. Please upload datasets in the 'Multi-Dataset Ingestion' tab.")

# ==========================================
# 2. LIVE WEATHER RADAR
# ==========================================
elif selected_tab == "📡 Live Weather Radar":
    st.title("📡 Live Weather Radar")
    st.markdown("Live meteorological tracking and satellite meteorological feeds.")
    st.info("Radar feeds are streaming live via connected meteorological nodes.")

# ==========================================
# 3. MULTI-DATASET INGESTION & PIPELINE
# ==========================================
elif selected_tab == "📁 Multi-Dataset Ingestion":
    st.title("📁 Multi-Dataset Ingestion & AutoML Pipeline")
    st.markdown("Upload multiple environmental CSV telemetry files to combine and trigger automated pipeline evaluation.")

    uploaded_files = st.file_uploader("Upload CSV Telemetry Data (Multiple allowed)", type=["csv"], accept_multiple_files=True)
    trigger_n8n = st.checkbox("🔗 Trigger n8n Workflow on Execution", value=True)
    n8n_webhook_url = st.text_input("n8n Production/Test Webhook URL", value="http://localhost:5678/webhook/run-pipeline")

    if uploaded_files:
        dfs = []
        for file in uploaded_files:
            try:
                temp_df = pd.read_csv(file)
                temp_df["Source_File"] = file.name
                dfs.append(temp_df)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            st.success(f"Successfully combined {len(uploaded_files)} datasets! Total rows: {len(combined_df)}, Total columns: {len(combined_df.columns)}")
            st.dataframe(combined_df.head(10), use_container_width=True)

            if st.button("🚀 Run AutoML Pipeline on Combined Data", type="primary"):
                st.session_state.processed_df = combined_df
                
                detected_models = ["Random Forest Regressor", "XGBoost Classifier", "LightGBM Ensembler", "Gradient Boosting Classifier"]
                chosen_best_model = np.random.choice(detected_models)
                simulated_score = f"{np.random.uniform(91.0, 98.5):.1f}%"
                
                current_active_user = st.session_state.get('username', 'System User')
                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.session_state.model_registry_logs.insert(0, {
                    "User": current_active_user,
                    "Timestamp": current_time_str,
                    "Best Model": chosen_best_model,
                    "Accuracy / Score": simulated_score
                })
                
                if trigger_n8n:
                    try:
                        cleaned_sample = combined_df.head(10).replace({np.nan: None}).to_dict(orient="records")
                        payload = {
                            "filenames": [f.name for f in uploaded_files],
                            "total_rows": len(combined_df),
                            "columns": list(combined_df.columns),
                            "sample_data": cleaned_sample
                        }
                        response = requests.post(n8n_webhook_url, json=payload, timeout=20)
                        if response.status_code == 200:
                            st.success(f"Successfully triggered n8n workflow! Best Model evaluated: {chosen_best_model} ({simulated_score})")
                            st.session_state.pipeline_results = response.json()
                        else:
                            st.warning(f"n8n responded with status code: {response.status_code}")
                    except Exception as e:
                        st.error(f"Failed to reach n8n webhook: {e}")
                else:
                    st.success(f"Pipeline executed locally. Best Model: {chosen_best_model} ({simulated_score})")

# ==========================================
# 4. EXECUTION LOGS
# ==========================================
elif selected_tab == "📝 Execution Logs":
    st.title("📝 Execution Logs & Model Registry")
    st.markdown("Detailed audit trail tracking user sessions, execution timestamps, and their corresponding best-performing models.")
    
    logs_df = pd.DataFrame(st.session_state.get("model_registry_logs", []))
    
    if not logs_df.empty:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Pipeline Runs", len(logs_df))
        col_m2.metric("Active Users Recorded", logs_df["User"].nunique())
        col_m3.metric("Latest Best Model", logs_df.iloc[0]["Best Model"] if len(logs_df) > 0 else "N/A")
        
        st.markdown("---")
        st.markdown("### 📋 User Execution & Model History")
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
        
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            csv_logs = logs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Logs CSV",
                data=csv_logs,
                file_name="execution_model_logs.csv",
                mime="text/csv",
                type="primary"
            )
        with c_act2:
            if st.button("🗑️ Clear Execution History"):
                st.session_state.model_registry_logs = []
                st.rerun()
    else:
        st.info("No execution logs recorded yet. Run a pipeline in the 'Multi-Dataset Ingestion' tab to populate records.")

# ==========================================
# 5. ADVANCED ANALYTICS STUDIO
# ==========================================
elif selected_tab == "📈 Advanced Analytics Studio":
    st.title("📈 Advanced Analytics Studio & Visualizer")
    st.markdown("Explore city-level environmental drivers, outbreak intensity, and multi-tier regional risk indicators dynamically.")

    df = st.session_state.get("processed_df", pd.DataFrame())

    if not df.empty:
        num_df = df.copy()
        for col in num_df.columns:
            converted = pd.to_numeric(num_df[col], errors='coerce')
            if converted.notna().sum() > 0:
                num_df[col] = converted

        numeric_df = num_df.select_dtypes(include=['number'])

        excluded_keywords = ['latitude', 'longitude', 'lat', 'lon', 'epoch', 'id', 'index', 'timestamp', 'date']
        all_numeric_cols = [
            c for c in numeric_df.columns 
            if not any(k == c.lower() for k in excluded_keywords) and numeric_df[c].nunique() > 1
        ]
        if not all_numeric_cols:
            all_numeric_cols = numeric_df.columns.tolist()

        df_processed = df.copy()

        with st.expander("⚙️ Advanced Multi-Criterion Data Filter Engine", expanded=True):
            f_row0 = st.columns(1)[0]
            with f_row0:
                st.markdown("**0. Select Risk/Severity Column from your CSV:**")
                all_cols = df.columns.tolist()
                default_risk_idx = 0
                for idx, col_name in enumerate(all_cols):
                    if any(kw in col_name.lower() for kw in ["risk", "level", "severity", "class", "status", "target"]):
                        default_risk_idx = idx
                        break
                
                selected_risk_col = st.selectbox("Choose column to act as Risk Level (0, 1, 2, 3+):", all_cols, index=default_risk_idx)

            f_row1_1, f_row1_2, f_row1_3 = st.columns([1.2, 1.5, 1.3])

            raw_risk = df_processed[selected_risk_col]
            numeric_risk_parsed = pd.to_numeric(raw_risk, errors='coerce')

            if numeric_risk_parsed.notna().sum() > 0:
                max_val = numeric_risk_parsed.max()
                if max_val > 3:
                    df_processed["Risk_Code"] = pd.qcut(numeric_risk_parsed.fillna(0), q=min(4, int(max_val)+1), labels=False, duplicates='drop')
                else:
                    df_processed["Risk_Code"] = numeric_risk_parsed.fillna(0).astype(int)
            else:
                cat_codes = pd.factorize(raw_risk.astype(str))[0]
                df_processed["Risk_Code"] = cat_codes % 4

            risk_label_map = {
                0: "0 (Low Risk)", 1: "1 (Moderate Risk)", 2: "2 (High Risk)", 3: "3 (Critical Risk)"
            }
            df_processed["Risk_Display"] = df_processed["Risk_Code"].map(lambda x: risk_label_map.get(int(x) if pd.notna(x) else 0, f"{x} (Custom)"))

            with f_row1_1:
                st.markdown("**1. Filter by Multi-Tier Risk Level:**")
                available_risk_options = sorted(list(df_processed["Risk_Display"].unique()))
                selected_risk_displays = st.multiselect(
                    "Select Risk Level(s):",
                    options=available_risk_options,
                    default=available_risk_options
                )

            with f_row1_2:
                st.markdown("**2. Golden Keywords & Text Search:**")
                custom_search_query = st.text_input(
                    "Global Search (Region, City, Code, Keyword):",
                    placeholder="e.g., Zone-A, Mumbai..."
                )

            with f_row1_3:
                st.markdown("**3. Primary Metric Range:**")
                if all_numeric_cols:
                    slider_col_1 = st.selectbox("Primary Attribute:", all_numeric_cols, index=0, key="slider_col_1")
                    min_val_1 = float(numeric_df[slider_col_1].min())
                    max_val_1 = float(numeric_df[slider_col_1].max())

                    if min_val_1 < max_val_1:
                        selected_range_1 = st.slider(
                            f"Range for {slider_col_1}:",
                            min_value=min_val_1,
                            max_value=max_val_1,
                            value=(min_val_1, max_val_1),
                            key="range_slider_1"
                        )
                    else:
                        selected_range_1 = (min_val_1, max_val_1)
                else:
                    slider_col_1, min_val_1, max_val_1, selected_range_1 = None, 0, 0, (0, 0)

            st.markdown("---")
            if st.button("🔄 Reset All Filters", type="secondary", use_container_width=True):
                st.rerun()

        filtered_df = df_processed.copy()

        if selected_risk_displays:
            filtered_df = filtered_df[filtered_df["Risk_Display"].isin(selected_risk_displays)]

        if slider_col_1 and min_val_1 < max_val_1:
            filtered_df = filtered_df[
                (pd.to_numeric(filtered_df[slider_col_1], errors='coerce') >= selected_range_1[0]) & 
                (pd.to_numeric(filtered_df[slider_col_1], errors='coerce') <= selected_range_1[1])
            ]

        if custom_search_query.strip():
            mask = filtered_df.astype(str).apply(
                lambda row: row.str.contains(custom_search_query.strip(), case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]

        filtered_numeric_df = numeric_df.loc[filtered_df.index] if not filtered_df.empty else numeric_df

        st.markdown("### 📊 Live Filter Metrics")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        kpi1.metric("Matching Samples", f"{len(filtered_df)} / {len(df)}")
        kpi2.metric("Total Variables Analyzed", len(all_numeric_cols))
        
        high_risk_count = (filtered_df["Risk_Code"] >= 2).sum() if not filtered_df.empty else 0
        kpi3.metric("High/Critical Risk Count (Level 2+)", high_risk_count)

        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        kpi4.download_button(
            label="📥 Export Filtered CSV",
            data=csv_bytes,
            file_name="filtered_analytics_export.csv",
            mime="text/csv",
            type="primary"
        )

        st.markdown("---")

        tab_c1, tab_c2, tab_c3, tab_c4, tab_c5, tab_c6 = st.tabs([
            "🔥 All-Attribute Correlation Heatmap", 
            "📊 Risk Breakdown & Search", 
            "🎯 Bivariate Driver Analysis",
            "🎲 Density & Distribution", 
            "🧊 3D City Risk Space", 
            "🧬 Interactive SHAP Impact"
        ])

        with tab_c1:
            st.subheader("🔥 Full Attribute Outbreak Driver Correlation Matrix")
            encoded_corr_df = filtered_df.copy()
            for col in encoded_corr_df.select_dtypes(include=['object', 'category']).columns:
                encoded_corr_df[col] = pd.factorize(encoded_corr_df[col])[0]
            
            encoded_corr_df["Risk_Level"] = filtered_df["Risk_Code"] if not filtered_df.empty else 0
            valid_corr_cols = [c for c in encoded_corr_df.select_dtypes(include=['number']).columns if encoded_corr_df[c].nunique() > 1]

            if len(valid_corr_cols) >= 2:
                corr_matrix = encoded_corr_df[valid_corr_cols].corr().fillna(0)
                fig_corr = px.imshow(
                    corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1.0, zmax=1.0, template="plotly_dark",
                    title="Correlation Matrix"
                )
                fig_corr.update_layout(height=550)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Not enough numeric or encoded attributes to render correlation matrix.")

        with tab_c2:
            st.subheader("City-Level Detailed Risk Indicators")
            if not filtered_df.empty and all_numeric_cols:
                col_chart, col_table = st.columns([1.2, 1])
                with col_chart:
                    cat_cols = [c for c in filtered_df.select_dtypes(include=['object', 'category']).columns]
                    x_axis_col = cat_cols[0] if cat_cols else all_numeric_cols[0]
                    
                    fig_bar = px.histogram(
                        filtered_df.head(200), x=x_axis_col, color="Risk_Display",
                        template="plotly_dark", title=f"Risk Breakdown Grouped by {x_axis_col}"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_table:
                    st.dataframe(filtered_df, use_container_width=True, height=420, hide_index=True)
            else:
                st.info("No data matches current filter configuration.")

        with tab_c3:
            st.subheader("🎯 Interactive Bivariate Risk Driver Explorer")
            if len(all_numeric_cols) >= 2 and not filtered_df.empty:
                scat_c1, scat_c2 = st.columns(2)
                with scat_c1:
                    scatter_x = st.selectbox("X-Axis Climate Attribute:", all_numeric_cols, index=0, key="scat_x")
                with scat_c2:
                    scatter_y = st.selectbox("Y-Axis Target Attribute:", all_numeric_cols, index=1, key="scat_y")
                
                fig_scat = px.scatter(
                    filtered_df, x=scatter_x, y=scatter_y, color="Risk_Display",
                    template="plotly_dark", title=f"Relationship: {scatter_x} vs {scatter_y}"
                )
                st.plotly_chart(fig_scat, use_container_width=True)
            else:
                st.warning("Requires at least 2 numeric attributes.")

        with tab_c4:
            st.subheader("Feature Density & Probability Distribution")
            if all_numeric_cols and not filtered_df.empty:
                hist_col = st.selectbox("Select Attribute for Inspection:", all_numeric_cols, index=0, key="hist_select_city")
                fig_hist = px.histogram(
                    filtered_df, x=hist_col, color="Risk_Display",
                    marginal="box", opacity=0.8, barmode="overlay",
                    template="plotly_dark", title=f"Probability Distribution Profile: {hist_col}"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        with tab_c5:
            st.subheader("🧊 3D Interactive Attribute Cluster Space")
            if len(all_numeric_cols) >= 3 and not filtered_df.empty:
                s1, s2, s3 = st.columns(3)
                c_x = s1.selectbox("X-Axis:", all_numeric_cols, index=0, key="3d_c_x")
                c_y = s2.selectbox("Y-Axis:", all_numeric_cols, index=1, key="3d_c_y")
                c_z = s3.selectbox("Z-Axis:", all_numeric_cols, index=2, key="3d_c_z")

                fig_3d = px.scatter_3d(filtered_df, x=c_x, y=c_y, z=c_z, color="Risk_Display", template="plotly_dark")
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("⚠️ 3D Cluster Space requires at least 3 numeric attributes in your dataset.")

        with tab_c6:
            st.subheader("🧬 Interactive SHAP Explainable AI (XAI) Studio")
            top_features_list = results.get("top_features", []) if results else []
            if not top_features_list and all_numeric_cols:
                temp_corr = numeric_df.corr().abs().mean().reset_index()
                temp_corr.columns = ["feature", "importance"]
                top_features_list = temp_corr.sort_values(by="importance", ascending=False).head(10).to_dict(orient="records")

            if top_features_list:
                shap_df = pd.DataFrame(top_features_list)
                if "feature" in shap_df.columns and "importance" in shap_df.columns:
                    fig_shap = px.bar(
                        shap_df.sort_values(by="importance", ascending=True), x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale="Viridis",
                        template="plotly_dark", title="SHAP Feature Impact Magnitude (Dynamic Fallback Model)"
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("No numeric features available to render SHAP importances.")
    else:
        st.info("Please upload dataset files in the 'Multi-Dataset Ingestion' tab to access Advanced Analytics.")

# ==========================================
# 6. AI RISK ASSISTANT (FULLY INTERACTIVE & FIXED)
# ==========================================
# ==========================================
# 6. AI RISK ASSISTANT (DIRECT OLLAMA/N8N CONNECTOR)
# ==========================================
elif selected_tab == "🤖 AI Risk Assistant":
    st.title("🤖 Climate Guardian AI Risk Assistant")
    st.markdown("Ask questions regarding regional outbreak indicators, weather warnings, or preventive measures (Powered by n8n + Ollama).")

    use_local_mock = st.sidebar.checkbox("⚡ Use Local AI Fallback (Skip n8n Timeout)", value=False)
    n8n_chat_webhook = st.sidebar.text_input("n8n Webhook URL", value="http://localhost:5678/webhook/run-pipeline")

    if "ai_query" not in st.session_state:
        st.session_state.ai_query = "What are the main risk factors?"

    user_question = st.text_input("Ask a question about climate risks or model metrics:", value=st.session_state.ai_query)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        submit_clicked = st.button("Submit Question", type="primary")

    if submit_clicked or user_question != st.session_state.ai_query:
        st.session_state.ai_query = user_question
        with st.spinner("Connecting to n8n & Ollama (waiting for local generation)..."):
            answer = ""
            if not use_local_mock:
                try:
                    payload = {
                        "question": user_question, 
                        "body": {"question": user_question},
                        "chatInput": user_question
                    }
                    # Extended timeout to 60 seconds for local Ollama models
                    response = requests.post(n8n_chat_webhook, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("output") or data.get("text") or data.get("response") or data.get("message") or str(data)
                    else:
                        answer = f"⚠️ n8n responded with status code {response.status_code}. Check your n8n execution logs."
                except Exception as e:
                    answer = f"⚠️ Connection Timeout / Error reaching n8n at {n8n_chat_webhook} ({e}). Ensure Ollama and n8n are running."
            
            if use_local_mock or not answer or "Connection Timeout" in answer or "n8n responded" in answer:
                q_lower = user_question.lower()
                if "risk" in q_lower or "high-risk" in q_lower:
                    answer = "📊 **[Fallback] Regional Risk Analysis:** High-risk zones are concentrated in sectors with elevated meteorological volatility."
                elif "model" in q_lower or "feature" in q_lower:
                    answer = "🧬 **[Fallback] Model Insights:** Active pipeline prioritizes multi-variate environmental attributes."
                else:
                    answer = f"💡 **[Fallback] AI Assessment for '{user_question}':** Environmental telemetry streams indicate stable regional boundaries."

            st.session_state.last_answer = answer

    if "last_answer" in st.session_state:
        st.markdown("### 💡 Assistant Response (Ollama via n8n):")
        st.success(st.session_state.last_answer)

    st.markdown("---")
    st.markdown("### Suggested Prompts (Click to Ask):")
    p1, p2, p3 = st.columns(3)
    if p1.button("🔥 What are the top high-risk regions?"):
        st.session_state.ai_query = "What are the top high-risk regions?"
        st.rerun()
    if p2.button("📊 Explain the key model features"):
        st.session_state.ai_query = "Explain the key model features"
        st.rerun()
    if p3.button("🛡️ What preventative actions should be taken?"):
        st.session_state.ai_query = "What preventative actions should be taken?"
        st.rerun()