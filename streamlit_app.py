"""Streamlit интерактивное демо для sentiment analysis.

Возможности:
- Загрузка и обучение модели
- Real-time инференс
- Визуализация метрик
- Мониторинг дрифта
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = "http://localhost:8000"
MODEL_PATH = Path("artefacts/best_model.joblib")
METRICS_PATH = Path("artefacts/model_artefacts")


st.set_page_config(
    page_title="Sentiment Analysis Demo",
    page_icon="📊",
    layout="wide",
)


def load_model_metadata():
    """Загружает метаданные модели."""
    meta_file = METRICS_PATH / "best_model_meta.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_drift_report():
    """Загружает drift report."""
    drift_file = Path("artefacts/drift_artefacts/drift_report.csv")
    if drift_file.exists():
        return pd.read_csv(drift_file)
    return None


def predict_sentiment(text: str):
    """Отправляет запрос к API для предсказания."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "reviewText": text,
                "text_len": float(len(text)),
                "word_count": float(len(text.split())),
            },
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


st.title("📊 Sentiment Analysis MLOps Demo")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Inference", "📈 Metrics", "🔍 Drift Monitor", "ℹ️ Model Info"]
)

with tab1:
    st.header("Real-time Inference")

    input_text = st.text_area(
        "Введите текст отзыва:",
        placeholder="This book was absolutely amazing! Highly recommend.",
        height=150,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("Предсказать", type="primary", use_container_width=True)

    if predict_btn and input_text:
        with st.spinner("Анализирую..."):
            result = predict_sentiment(input_text)

        if "error" in result:
            st.error(f"Ошибка API: {result['error']}")
        else:
            sentiment = result.get("predicted_sentiment", "unknown")
            confidence = result.get("confidence", 0.0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", sentiment.upper())
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            with col3:
                sentiment_emoji = "😊" if sentiment == "positive" else "😔"
                st.metric("", sentiment_emoji)

            st.progress(confidence)

            if confidence < 0.6:
                st.warning(
                    "⚠️ Низкая уверенность модели. Результат может быть неточным."
                )

with tab2:
    st.header("Model Metrics")

    meta = load_model_metadata()
    if meta:
        col1, col2, col3, col4 = st.columns(4)
        metrics = meta.get("test_metrics", {})

        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")

        st.markdown("---")

        feature_imp_file = METRICS_PATH / "feature_importances.json"
        if feature_imp_file.exists():
            with open(feature_imp_file, encoding="utf-8") as f:
                feature_importances = json.load(f)

            if feature_importances:
                df_imp = pd.DataFrame(
                    list(feature_importances.items()), columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)

                fig = px.bar(
                    df_imp.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 15 Feature Importances",
                )
                st.plotly_chart(fig, use_container_width=True)

        optuna_file = METRICS_PATH / "optuna_top_trials.csv"
        if optuna_file.exists():
            st.subheader("Optuna HPO Trials")
            df_optuna = pd.read_csv(optuna_file)
            st.dataframe(df_optuna.head(10), use_container_width=True)

    else:
        st.warning("Метаданные модели не найдены. Запустите обучение через Airflow.")

with tab3:
    st.header("Data Drift Monitoring")

    drift_df = load_drift_report()
    if drift_df is not None and not drift_df.empty:
        drift_df = drift_df.sort_values("psi_score", ascending=False)

        st.subheader("PSI Scores по признакам")
        fig = px.bar(
            drift_df,
            x="feature",
            y="psi_score",
            color="drift_detected",
            title="Population Stability Index (PSI)",
            color_discrete_map={True: "red", False: "green"},
        )
        fig.add_hline(
            y=0.1, line_dash="dash", line_color="orange", annotation_text="Threshold"
        )
        st.plotly_chart(fig, use_container_width=True)

        drifted = drift_df[drift_df["drift_detected"] == True]
        if not drifted.empty:
            st.error(f"⚠️ Обнаружен дрифт в {len(drifted)} признаках!")
            st.dataframe(drifted, use_container_width=True)
        else:
            st.success("✅ Дрифт не обнаружен.")

        st.subheader("История дрифта")
        st.dataframe(drift_df, use_container_width=True)
    else:
        st.info(
            "Данные мониторинга дрифта отсутствуют. Запустите DAG с параметром run_drift_monitor=true."
        )

with tab4:
    st.header("Model Information")

    meta = load_model_metadata()
    if meta:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("General Info")
            st.json(
                {
                    "Model Type": meta.get("model_kind", "unknown"),
                    "MLflow Run ID": meta.get("run_id", "N/A"),
                    "Training Date": meta.get("created_at", "N/A"),
                }
            )

        with col2:
            st.subheader("Hyperparameters")
            best_params = meta.get("best_params", {})
            if best_params:
                st.json(best_params)

        schema_file = METRICS_PATH / "model_schema.json"
        if schema_file.exists():
            st.subheader("Model Schema")
            with open(schema_file, encoding="utf-8") as f:
                schema = json.load(f)
            st.json(schema)
    else:
        st.warning("Информация о модели недоступна.")

    st.markdown("---")
    st.markdown("**API Health Check**")
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        if response.ok:
            st.success(f"✅ API доступен: {API_URL}")
        else:
            st.error(f"❌ API недоступен (HTTP {response.status_code})")
    except Exception as e:
        st.error(f"❌ Не удалось подключиться к API: {e}")
