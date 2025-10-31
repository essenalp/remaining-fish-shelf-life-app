import streamlit as st
import pandas as pd
import joblib
import os

st.title("🐟 Balık Raf Ömrü Tahmin Uygulaması")
st.write("Balığın depolama koşullarına göre tahmini kalan raf ömrünü hesaplayabilirsiniz.")

# ---- Kullanıcı girişleri ----
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
storage_hours = st.number_input("Depolama süresi (saat)", min_value=0, max_value=1440, value=24, step=1)
storage_temp = st.number_input("Depolama sıcaklığı (°C)", min_value=-5.0, max_value=25.0, value=4.0, step=0.1)
violation_hours = st.number_input("İhlal süresi (saat)", min_value=0, max_value=1440, value=0, step=1)
violation_temp = st.number_input("İhlal sıcaklığı (°C)", min_value=-5.0, max_value=25.0, value=0.0, step=0.1)

model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

# ---- Model yolunu çöz ----
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

            # Modelin beklediği feature'ları tespit et
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                # Örnek feature seti
                expected_features = ["storage_hours","storage_temp",
                                     "violation_hours","violation_temp",
                                     "species_Levrek","species_Somon"]

            # Kullanıcı girdilerini model formatına çevir
            row = {col: 0 for col in expected_features}

            for col, val in [("storage_hours", storage_hours),
                             ("storage_temp", storage_temp),
                             ("violation_hours", violation_hours),
                             ("violation_temp", violation_temp)]:
                if col in row:
                    row[col] = val

            sp_col = f"species_{species}"
            if sp_col in row:
                row[sp_col] = 1

            X_input = pd.DataFrame([row], columns=expected_features)

            pred = model.predict(X_input)[0]
            st.success(f"Tahmini kalan raf ömrü: {pred:.1f} saat")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
