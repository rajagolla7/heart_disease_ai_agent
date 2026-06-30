import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

MODEL_FILENAME = "heart_model .pkl"

try:
	model = joblib.load(MODEL_FILENAME)
except Exception as e:
	model = None
	st.error(f"Failed to load model '{MODEL_FILENAME}': {e}")

st.markdown("""
<style>
.big-title { font-size:32px; font-weight:700; }
.card { background: linear-gradient(90deg,#ffffff,#f7fbff); padding:16px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">❤️ Heart Disease AI — Premium Dashboard</div>', unsafe_allow_html=True)

# Additional premium styles
st.markdown("""
<style>
body { background: linear-gradient(180deg, #f3f7ff 0%, #ffffff 100%); }
.stApp { color-scheme: light; }
.kpi { display:flex; gap:12px; }
.kpi .card { flex:1; padding:12px; border-radius:10px; box-shadow: 0 2px 10px rgba(16,24,40,0.06); }
.logo { float:right; width:48px; height:48px; border-radius:8px; }
.small { font-size:12px; color:#6b7280 }
</style>
""", unsafe_allow_html=True)

# header row with small logo
colh1, colh2 = st.columns([6,1])
with colh1:
		st.markdown("""
		<div style='display:flex; align-items:center; gap:12px;'>
			<div style='font-size:20px; font-weight:700;'>Heart Disease AI — Premium</div>
			<div class='small'>Interactive predictions · Explainability · Export</div>
		</div>
		""", unsafe_allow_html=True)
with colh2:
		st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=48)

if model is None:
	st.warning("Model is not loaded. Fix the model file and refresh.")
else:
	# Sidebar removed per user request
	show_advanced = False

	# helper: prediction
	def predict_df(df: pd.DataFrame):
		try:
			out = df.copy()
			if hasattr(model, "predict_proba"):
				probs = model.predict_proba(df)
				if probs.ndim == 2 and probs.shape[1] == 2:
					out["probability_of_disease"] = probs[:, 1]
				else:
					out["prediction_probs"] = probs.tolist()
			out["prediction"] = model.predict(df)
			return out
		except Exception as e:
			st.error(f"Prediction failed: {e}")
			return None

	# tabs for a premium feel
	tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Batch", "Explain", "About"])

	# Predict tab
	with tab1:
		col1, col2 = st.columns([2, 1])
		with col1:
			st.subheader("Patient inputs")
			with st.form("single_patient_form"):
				age = st.number_input("Age", min_value=1, max_value=120, value=55)
				sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
				cp = st.selectbox("Chest pain type (cp)", [0, 1, 2, 3])
				trestbps = st.number_input("Resting BP", value=130)
				chol = st.number_input("Cholesterol", value=250)
				fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])
				restecg = st.selectbox("Resting ECG", [0, 1, 2])
				thalach = st.number_input("Max heart rate (thalach)", value=150)
				exang = st.selectbox("Exercise induced angina (exang)", [0, 1])
				oldpeak = st.number_input("Oldpeak", value=1.0, format="%.2f")
				slope = st.selectbox("ST slope", [0, 1, 2])
				ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3, 4])
				thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
				submit = st.form_submit_button("Run prediction")

		with col2:
			st.subheader("Prediction")
			st.write("Results will appear here after prediction.")
			result_card = st.empty()
			# KPI placeholders
			k1, k2, k3 = st.columns(3)
			k1.metric("Model loaded", "Yes" if model is not None else "No")
			k2.metric("Example prob", "—")
			k3.metric("Batch ready", "Yes")

		# example patient loading removed

		if submit:
			row = {"age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps], "chol": [chol],
				   "fbs": [fbs], "restecg": [restecg], "thalach": [thalach], "exang": [exang],
				   "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal]}
			df_row = pd.DataFrame(row)
			res = predict_df(df_row)
			if res is not None:
				prob = res.loc[0, "probability_of_disease"] if "probability_of_disease" in res.columns else None
				# Human-readable label and risk
				pred_label = "Heart Disease Detected" if int(res.loc[0, "prediction"]) == 1 else "No Heart Disease Detected"
				if prob is not None:
					if prob >= 0.75:
						risk = "High"
					elif prob >= 0.4:
						risk = "Medium"
					else:
						risk = "Low"
				else:
					# fallback when probabilities aren't available
					risk = "High" if int(res.loc[0, "prediction"]) == 1 else "Low"

				# Recommendations by risk
				recommendations = []
				if risk == "High":
					recommendations = [
						"Consult a cardiologist.",
						"Exercise regularly under medical advice.",
						"Reduce salt and oily foods.",
						"Monitor your blood pressure.",
						"Schedule regular health check-ups."
					]
				elif risk == "Medium":
					recommendations = [
						"Follow up with your primary care doctor.",
						"Adopt heart-healthy diet and exercise.",
						"Monitor blood pressure and cholesterol regularly."
					]
				else:
					recommendations = [
						"Maintain a healthy diet.",
						"Exercise regularly.",
						"Continue regular health check-ups."
					]

				# Render a styled result card similar to the example
				card_html = f"""
				<div style='background:#0b1220;color:#ffffff;padding:18px;border-radius:12px;border:1px solid rgba(255,255,255,0.06);'>
				  <h3 style='margin:0 0 8px 0;'>Prediction: {pred_label}</h3>
				  <div style='margin-bottom:8px;'><strong>Risk Level:</strong> {risk}</div>
				  <div style='margin-top:8px;'><strong>Health Recommendations</strong></div>
				  <ul style='margin-top:6px;'>
				    {''.join(f'<li style="margin-bottom:6px;">{r}</li>' for r in recommendations)}
				  </ul>
				</div>
				"""
				with result_card:
					st.markdown(card_html, unsafe_allow_html=True)

				# Also show a small summary table in an expander and keep download button
				with st.expander("View raw prediction data"):
					st.dataframe(res)
				csv = res.to_csv(index=False)
				st.download_button("Download result CSV", data=csv, file_name="prediction.csv")

	# Batch tab
	with tab2:
		st.subheader("Batch predictions")
		uploaded = st.file_uploader("Upload CSV", type=["csv"])        
		if uploaded:
			try:
				df = pd.read_csv(uploaded)
			except Exception as e:
				st.error(f"Failed to read CSV: {e}")
			else:
				res = predict_df(df)
				if res is not None:
					st.success("Batch predictions complete")
					st.dataframe(res)
					st.download_button("Download predictions", res.to_csv(index=False), file_name="batch_predictions.csv")

	# Explain tab
	with tab3:
		st.subheader("Model explainability & feature importance")
		fi = None
		try:
			if hasattr(model, "feature_importances_"):
				names = list(getattr(model, "feature_names_in_", None) or [])
				importances = model.feature_importances_
				if len(names) != len(importances):
					# fallback: index names
					names = [f"f{i}" for i in range(len(importances))]
				fi = pd.Series(importances, index=names).sort_values(ascending=False)
			elif hasattr(model, "coef_"):
				coef = np.ravel(model.coef_)
				names = list(getattr(model, "feature_names_in_", None) or [])
				if len(names) != len(coef):
					names = [f"f{i}" for i in range(len(coef))]
				fi = pd.Series(np.abs(coef), index=names).sort_values(ascending=False)
		except Exception as e:
			st.write("Could not compute feature importance:", e)

		if fi is None or fi.empty:
			st.info("Feature importances not available for this model.")
		else:
			# use plotly for nicer bar chart
			fig = px.bar(fi.reset_index(), x=0, y='index', orientation='h', labels={0:'importance','index':'feature'}, height=400)
			fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white')
			st.plotly_chart(fig, width='stretch')
			st.dataframe(fi.reset_index().rename(columns={"index": "feature", 0: "importance"}))

	# About tab
	with tab4:
		st.subheader("About this premium dashboard")
		st.markdown("- Interactive single and batch predictions\n- Probability visualization and download\n- Feature importance chart")
		if show_advanced:
			st.markdown("Advanced: raw model type and pipeline")
			st.write(type(model))
			try:
				if hasattr(model, "named_steps"):
					st.write({k: type(v).__name__ for k, v in model.named_steps.items()})
			except Exception:
				pass