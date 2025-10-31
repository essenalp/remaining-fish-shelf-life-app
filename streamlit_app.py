import streamlit as st
import joblib
import pandas as pd
import os
import re

st.title("🐟 Balık Raf Ömrü Tahmin Uygulaması")
st.write("Balığın depolama koşullarına göre tahmini kalan raf ömrünü hesaplayabilirsiniz.")

# ---- Kullanıcı girişleri ----
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
storage_hours = st.number_input("Depolama süresi (saat)", min_value=0, value=24, step=1)
storage_temp = st.number_input("Depolama sıcaklığı (°C)", value=0)
violation_hours = st.number_input("İhlal süresi (saat)", min_value=0, value=0)
violation_temp = st.number_input("İhlal sıcaklığı (°C)", value=0)
model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

# ---- Model dosya yolunu çöz ----
def resolve_model_path(choice: str) -> str:
    fname = "rf_model_app.joblib" if choice == "Random Forest" else "xgb_model_app.joblib"
    paths = [os.path.join("models", fname), os.path.join("Models", fname)]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

# ---- Tahmin butonu ----
if st.button("Tahmin Et"):
    try:
        model_path = resolve_model_path(model_choice)
        if not os.path.exists(model_path):
            st.error(f"Model dosyası bulunamadı: {model_path}")
        else:
            model = joblib.load(model_path)

            # Modelin beklediği özellik adlarını tespit et
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                candidate = ["storage_hours", "storage_temp", "violation_hours", "violation_temp", 
                             "species_Somon", "species_Levrek"]
                expected_features = candidate

            # Kullanıcı girdisini modelin beklediği kolon setine çevir
            row = {col: 0 for col in expected_features}

            # Saat/sıcaklık kolonları
            if "storage_hours" in expected_features:
                row["storage_hours"] = storage_hours
            if "storage_temp" in expected_features:
                row["storage_temp"] = storage_temp
            if "violation_hours" in expected_features:
                row["violation_hours"] = violation_hours
            if "violation_temp" in expected_features:
                row["violation_temp"] = violation_temp

            # Tür kolonları
            sp_col = f"species_{species}"
            if sp_col in expected_features:
                row[sp_col] = 1

            # DataFrame oluştur
            X_input = pd.DataFrame([[row[c] for c in expected_features]], columns=expected_features)

            pred = model.predict(X_input)[0]
            st.success(f"Tahmini kalan raf ömrü: {pred:.1f} saat")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
