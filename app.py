from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="!", layout="wide")
DATA_FILE = Path(__file__).with_name("creditcard.csv")
TARGET_COLUMN = "Class"


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
	return pd.read_csv(file_path)


@st.cache_resource
def train_models(data: pd.DataFrame):
	features = [column for column in data.columns if column != TARGET_COLUMN]
	scaler = StandardScaler().fit(data[features])
	scaled = scaler.transform(data[features])
	fraud_rate = float(data[TARGET_COLUMN].mean())
	isolation_forest = IsolationForest(
		n_estimators=150, contamination=max(0.001, min(fraud_rate, 0.5)), random_state=42, n_jobs=-1
	).fit(scaled)
	random_forest = RandomForestClassifier(
		n_estimators=100, class_weight="balanced_subsample", max_depth=18, random_state=42, n_jobs=-1
	).fit(scaled, data[TARGET_COLUMN])
	return scaler, isolation_forest, random_forest, features


def score_transactions(data, scaler, model, features, model_name):
	result = data.copy()
	scaled = scaler.transform(result[features])
	if model_name == "Random Forest":
		result["FraudScore"] = model.predict_proba(scaled)[:, 1]
		result["PredictedFraud"] = model.predict(scaled).astype(int)
	else:
		result["FraudScore"] = -model.decision_function(scaled)
		result["PredictedFraud"] = (model.predict(scaled) == -1).astype(int)
	result["RiskLevel"] = pd.cut(
		result["FraudScore"], [-np.inf, 0, 0.1, np.inf], labels=["Low", "Medium", "High"]
	).astype(str)
	return result


def explain_transaction(transaction, data, features):
	z_scores = ((transaction[features].iloc[0] - data[features].mean()) /
		data[features].std().replace(0, 1)).abs()
	return z_scores.sort_values(ascending=False).head(5).rename("Deviation").to_frame()


st.title("Credit Card Fraud Detection")
st.caption("Anomaly detection powered by the transactions in creditcard.csv")
try:
	data = load_data(str(DATA_FILE))
	scaler, isolation_forest, random_forest, features = train_models(data)
except Exception as error:
	st.error(f"Could not load the dataset: {error}")
	st.stop()

model_name = st.sidebar.selectbox("Detection model", ["Isolation Forest", "Random Forest"])
active_model = random_forest if model_name == "Random Forest" else isolation_forest
scored_data = score_transactions(data, scaler, active_model, features, model_name)
actual_fraud = int(data[TARGET_COLUMN].sum())
predicted_fraud = int(scored_data["PredictedFraud"].sum())

metrics = st.columns(4)
metrics[0].metric("Transactions", f"{len(data):,}")
metrics[1].metric("Detected anomalies", f"{predicted_fraud:,}")
metrics[2].metric("Actual fraud", f"{actual_fraud:,}")
metrics[3].metric("Detection rate", f"{predicted_fraud / len(data) * 100:.2f}%")

test_features, _, test_target, _ = train_test_split(
	data[features], data[TARGET_COLUMN], test_size=0.2, random_state=42, stratify=data[TARGET_COLUMN]
)
test_predictions = score_transactions(test_features, scaler, active_model, features, model_name)["PredictedFraud"]
st.subheader(f"{model_name} performance")
performance = st.columns(4)
performance[0].metric("Precision", f"{precision_score(test_target, test_predictions, zero_division=0):.2%}")
performance[1].metric("Recall", f"{recall_score(test_target, test_predictions, zero_division=0):.2%}")
performance[2].metric("F1 score", f"{f1_score(test_target, test_predictions, zero_division=0):.2%}")
performance[3].metric("Accuracy", f"{accuracy_score(test_target, test_predictions):.2%}")

overview_tab, batch_tab, predict_tab = st.tabs(["Overview", "Analyze CSV", "Predict transaction"])
with overview_tab:
	left, right = st.columns(2)
	with left:
		st.subheader("Risk distribution")
		st.bar_chart(scored_data["RiskLevel"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0))
	with right:
		st.subheader("Transaction amount distribution")
		st.line_chart(data["Amount"].describe()[["min", "25%", "50%", "75%", "max"]])
	st.subheader("Highest-risk transactions")
	st.dataframe(scored_data.sort_values("FraudScore", ascending=False).head(20), use_container_width=True, hide_index=True)

with batch_tab:
	uploaded_file = st.file_uploader("Upload a transaction CSV", type="csv")
	batch_data = data.drop(columns=[TARGET_COLUMN]) if uploaded_file is None else pd.read_csv(uploaded_file)
	missing = [column for column in features if column not in batch_data.columns]
	if missing:
		st.error("Missing required columns: " + ", ".join(missing))
	else:
		batch_scored = score_transactions(batch_data[features], scaler, active_model, features, model_name)
		st.write(f"Analyzed {len(batch_scored):,} transactions")
		st.dataframe(batch_scored.sort_values("FraudScore", ascending=False).head(100), use_container_width=True, hide_index=True)
		st.download_button("Download scored transactions", batch_scored.to_csv(index=False).encode(), "fraud_predictions.csv", "text/csv")
		st.download_button("Download high-risk alerts", batch_scored[batch_scored["PredictedFraud"] == 1].to_csv(index=False).encode(), "fraud_alerts.csv", "text/csv")

with predict_tab:
	prediction_file = st.file_uploader("Upload real transaction data", type="csv", key="prediction_file")
	prediction_data = data.drop(columns=[TARGET_COLUMN]) if prediction_file is None else pd.read_csv(prediction_file).drop(columns=[TARGET_COLUMN], errors="ignore")
	missing = [column for column in features if column not in prediction_data.columns]
	if missing:
		st.error("Missing required columns: " + ", ".join(missing))
	else:
		row_number = st.number_input("Transaction row", 0, len(prediction_data) - 1, 0, key="prediction_row")
		selected = prediction_data.iloc[[int(row_number)]][features]
		edited = st.data_editor(selected, use_container_width=True, hide_index=True)
		prediction = score_transactions(edited, scaler, active_model, features, model_name).iloc[0]
		if prediction["PredictedFraud"]:
			st.error(f"High-risk transaction detected (score: {prediction['FraudScore']:.4f})")
		else:
			st.success(f"Transaction appears normal (score: {prediction['FraudScore']:.4f})")
		st.subheader("Why this result?")
		st.dataframe(explain_transaction(edited, data, features), use_container_width=True)