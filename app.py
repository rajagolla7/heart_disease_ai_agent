import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
	page_title="Heart Disease AI",
	page_icon="❤️",
	layout="wide",
	initial_sidebar_state="collapsed",
)

MODEL_FILENAME = "heart_model .pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

try:
	model = joblib.load(MODEL_PATH)
except Exception as e:
	model = None
	st.error(f"Failed to load model '{MODEL_FILENAME}': {e}")

st.markdown(
	"""
<style>
html, body, [data-testid="stAppViewContainer"] {
	background:
		radial-gradient(circle at top left, rgba(255, 255, 255, 0.95) 0%, rgba(255,255,255,0.78) 24%, rgba(240,244,255,0.95) 54%, rgba(227,236,255,1) 100%),
		linear-gradient(135deg, #eaf2ff 0%, #ffffff 42%, #f5f9ff 100%);
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
	background: transparent;
}

.stApp {
	color: #0f172a;
}

.hero-shell {
	border: 1px solid rgba(15, 23, 42, 0.08);
	background: linear-gradient(135deg, rgba(6, 95, 70, 0.96) 0%, rgba(30, 64, 175, 0.92) 55%, rgba(15, 23, 42, 0.98) 100%);
	color: white;
	padding: 28px 28px 24px;
	border-radius: 28px;
	box-shadow: 0 22px 50px rgba(15, 23, 42, 0.18);
	position: relative;
	overflow: hidden;
}

.hero-shell:before {
	content: "";
	position: absolute;
	right: -80px;
	top: -80px;
	width: 220px;
	height: 220px;
	border-radius: 50%;
	background: rgba(255,255,255,0.12);
}

.hero-kicker {
	display: inline-flex;
	align-items: center;
	gap: 8px;
	padding: 6px 12px;
	border-radius: 999px;
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	background: rgba(255,255,255,0.14);
	border: 1px solid rgba(255,255,255,0.16);
}

.hero-title {
	font-size: 40px;
	line-height: 1.05;
	font-weight: 800;
	margin: 14px 0 10px;
}

.hero-copy {
	font-size: 16px;
	max-width: 840px;
	opacity: 0.92;
	margin-bottom: 18px;
}

.hero-pills {
	display: flex;
	flex-wrap: wrap;
	gap: 10px;
	margin-top: 16px;
}

.pill {
	padding: 8px 12px;
	border-radius: 999px;
	background: rgba(255,255,255,0.12);
	border: 1px solid rgba(255,255,255,0.15);
	font-size: 13px;
	font-weight: 600;
}

.glass-card {
	background: rgba(255,255,255,0.72);
	backdrop-filter: blur(16px);
	border: 1px solid rgba(255,255,255,0.6);
	border-radius: 24px;
	box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
	padding: 20px 20px 18px;
}

.section-title {
	font-size: 16px;
	font-weight: 800;
	color: #0f172a;
	margin: 0 0 10px;
}

.section-subtitle {
	color: #475569;
	font-size: 13px;
	margin: 0 0 16px;
}

.metric-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 12px;
}

.metric-card {
	padding: 14px 16px;
	border-radius: 18px;
	background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(241,245,249,0.9));
	border: 1px solid rgba(148,163,184,0.18);
	box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
}

.metric-label {
	font-size: 12px;
	text-transform: uppercase;
	letter-spacing: 0.08em;
	color: #64748b;
	margin-bottom: 6px;
}

.metric-value {
	font-size: 20px;
	font-weight: 800;
	color: #0f172a;
}

.metric-note {
	font-size: 12px;
	color: #64748b;
	margin-top: 4px;
}

.result-panel {
	border-radius: 24px;
	padding: 20px;
	color: white;
	background: linear-gradient(145deg, #0f172a 0%, #1d4ed8 48%, #0ea5e9 100%);
	box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
	border: 1px solid rgba(255,255,255,0.1);
}

.result-panel.high {
	background: linear-gradient(145deg, #7f1d1d 0%, #dc2626 50%, #f97316 100%);
}

.result-panel.low {
	background: linear-gradient(145deg, #064e3b 0%, #059669 54%, #22c55e 100%);
}

.result-badge {
	display: inline-flex;
	padding: 6px 12px;
	border-radius: 999px;
	background: rgba(255,255,255,0.18);
	border: 1px solid rgba(255,255,255,0.18);
	font-size: 12px;
	font-weight: 700;
	margin-bottom: 12px;
}

.result-title {
	font-size: 28px;
	font-weight: 800;
	margin: 0 0 8px;
}

.result-subtitle {
	font-size: 14px;
	opacity: 0.9;
	margin-bottom: 14px;
}

.result-probability {
	font-size: 15px;
	font-weight: 700;
	margin-bottom: 12px;
}

.recommendation-list {
	margin: 10px 0 0;
	padding-left: 18px;
}

.recommendation-list li {
	margin-bottom: 7px;
	line-height: 1.35;
}

div[data-testid="stTabs"] button {
	font-weight: 700;
}

.small-helper {
	font-size: 13px;
	color: #64748b;
}

</style>
""",
	unsafe_allow_html=True,
)

st.markdown(
	"""
<div class="hero-shell">
	<div class="hero-kicker">Heart Disease Intelligence</div>
	<div class="hero-title">A sharper, more polished health risk dashboard</div>
	<div class="hero-copy">Enter patient values, get a clear high-risk or low-risk result, and review model insights in a cleaner layout designed to feel closer to a professional clinical tool.</div>
	<div class="hero-pills">
		<div class="pill">Live prediction</div>
		<div class="pill">Risk-focused result</div>
		<div class="pill">Batch CSV support</div>
		<div class="pill">Feature importance view</div>
	</div>
</div>
""",
	unsafe_allow_html=True,
)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

st.markdown(
	"""
<div class="metric-grid">
	<div class="metric-card">
		<div class="metric-label">Model status</div>
		<div class="metric-value">Loaded</div>
		<div class="metric-note">Ready for interactive predictions</div>
	</div>
	<div class="metric-card">
		<div class="metric-label">Output style</div>
		<div class="metric-value">High / Low Risk</div>
		<div class="metric-note">Clear result instead of raw codes</div>
	</div>
	<div class="metric-card">
		<div class="metric-label">Views</div>
		<div class="metric-value">4 Sections</div>
		<div class="metric-note">Predict, Batch, Explain, About</div>
	</div>
</div>
""",
	unsafe_allow_html=True,
)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

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
					# In this dataset/model, class 0 is disease (High Risk), class 1 is normal (Low Risk).
					# So probability of disease is probs[:, 0].
					out["probability_of_disease"] = probs[:, 0]
				else:
					out["prediction_probs"] = probs.tolist()
			# Predict returns 0 for disease and 1 for normal.
			# We invert it so prediction is 1 for disease (High Risk) and 0 for normal (Low Risk).
			raw_preds = model.predict(df)
			out["prediction"] = 1 - raw_preds
			return out
		except Exception as e:
			st.error(f"Prediction failed: {e}")
			return None

	# tabs for a premium feel
	tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Batch", "Explain", "About"])

	# Predict tab
	with tab1:
		col1, col2 = st.columns([1.25, 0.95], gap="large")
		with col1:
			st.markdown("<div class='section-title'>Patient inputs</div><div class='section-subtitle'>Fill in the clinical fields below, then run the model to see the risk summary.</div>", unsafe_allow_html=True)
			with st.form("single_patient_form"):
				input_top = st.columns(2)
				with input_top[0]:
					age = st.number_input("Age", min_value=1, max_value=120, value=55)
					sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)
					cp = st.selectbox("Chest pain type (cp)", [0, 1, 2, 3])
					fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])
					thalach = st.number_input("Max heart rate (thalach)", value=150)
					oldpeak = st.number_input("Oldpeak", value=1.0, format="%.2f")
				with input_top[1]:
					trestbps = st.number_input("Resting BP", value=130)
					chol = st.number_input("Cholesterol", value=250)
					restecg = st.selectbox("Resting ECG", [0, 1, 2])
					exang = st.selectbox("Exercise induced angina (exang)", [0, 1])
					slope = st.selectbox("ST slope", [0, 1, 2])
					ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3, 4])
					thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
				st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
				submit = st.form_submit_button("Run prediction")

		with col2:
			st.markdown("<div class='section-title'>Prediction summary</div><div class='section-subtitle'>Your result appears here in a high-contrast card.</div>", unsafe_allow_html=True)
			result_card = st.empty()
			st.markdown(
				"""
				<div class="metric-grid" style="grid-template-columns: 1fr;">
					<div class="metric-card">
						<div class="metric-label">Model loaded</div>
						<div class="metric-value">Yes</div>
						<div class="metric-note">The trained model is available</div>
					</div>
					<div class="metric-card">
						<div class="metric-label">Prediction type</div>
						<div class="metric-value">Clinical risk</div>
						<div class="metric-note">High risk or low risk verdict</div>
					</div>
					<div class="metric-card">
						<div class="metric-label">Batch mode</div>
						<div class="metric-value">Ready</div>
						<div class="metric-note">Upload CSV files in the Batch tab</div>
					</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

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
				predicted_class = int(res.loc[0, "prediction"])
				pred_label = "Heart Disease Detected" if predicted_class == 1 else "No Heart Disease Detected"
				if prob is not None:
					risk = "High Risk" if prob >= 0.5 else "Low Risk"
				else:
					# fallback when probabilities aren't available
					risk = "High Risk" if predicted_class == 1 else "Low Risk"

				# Recommendations by risk
				recommendations = []
				if risk == "High Risk":
					recommendations = [
						"Consult a cardiologist.",
						"Exercise regularly under medical advice.",
						"Reduce salt and oily foods.",
						"Monitor your blood pressure.",
						"Schedule regular health check-ups."
					]
				else:
					recommendations = [
						"Maintain a healthy diet.",
						"Exercise regularly.",
						"Continue regular health check-ups."
					]

				# Render a styled result card similar to the example
				card_class = "high" if risk == "High Risk" else "low"
				card_label = "High Risk" if risk == "High Risk" else "Low Risk"
				card_html = f"""
				<div class='result-panel {card_class}'>
				  <div class='result-badge'>Prediction result</div>
				  <div class='result-title'>{card_label}</div>
				  <div class='result-subtitle'>{pred_label}</div>
				  <div class='result-probability'>Probability: {f'{prob:.2f}' if prob is not None else 'Not available'}</div>
				  <div style='font-size:14px;font-weight:700;margin-bottom:8px;'>Health recommendations</div>
				  <ul class='recommendation-list'>
				    {''.join(f'<li style="margin-bottom:6px;">{r}</li>' for r in recommendations)}
				  </ul>
				</div>
				"""
				with result_card:
					st.markdown(card_html, unsafe_allow_html=True)
					if risk == "High Risk":
						st.error("High Risk")
					else:
						st.success("Low Risk")

				# Also show a small summary table in an expander and keep download button
				with st.expander("View raw prediction data"):
					st.dataframe(res)
				csv = res.to_csv(index=False)
				st.download_button("Download detailed result CSV", data=csv, file_name="prediction.csv")

	# Batch tab
	with tab2:
		st.markdown("<div class='section-title'>Batch predictions</div><div class='section-subtitle'>Upload a CSV to get predictions for multiple patients at once.</div>", unsafe_allow_html=True)
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
		st.markdown("<div class='section-title'>Model explainability & feature importance</div><div class='section-subtitle'>Inspect which features are pushing predictions up or down.</div>", unsafe_allow_html=True)
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
			fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white', margin=dict(l=10, r=10, t=20, b=10))
			st.plotly_chart(fig, width='stretch')
			st.dataframe(fi.reset_index().rename(columns={"index": "feature", 0: "importance"}))

	# About tab
	with tab4:
		st.markdown("<div class='section-title'>About this dashboard</div><div class='section-subtitle'>Designed to make the result easy to read at a glance.</div>", unsafe_allow_html=True)
		st.markdown("- Interactive single and batch predictions\n- High-risk or low-risk verdict\n- Feature importance chart\n- CSV export for detailed review")
		if show_advanced:
			st.markdown("Advanced: raw model type and pipeline")
			st.write(type(model))
			try:
				if hasattr(model, "named_steps"):
					st.write({k: type(v).__name__ for k, v in model.named_steps.items()})
			except Exception:
				pass
