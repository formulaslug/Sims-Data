from explainerdashboard import ExplainerDashboard
import joblib

# Change this to load any saved scenario:
#   "explainer.joblib"                — original
#   "explainer_all.joblib"
#   "explainer_high_current.joblib"
#   "explainer_very_high_current.joblib"
#   "explainer_contactor_stable.joblib"
#   "explainer_contactor_flicker.joblib"
#   "explainer_hot_controller.joblib"
#   "explainer_large_v_drop.joblib"
LOAD_FILE = "explainer_early_in_session.joblib"

print("SCRIPT START")

explainer = joblib.load(LOAD_FILE)
print("Loaded explainer:", type(explainer))

if __name__ == "__main__":
    print("Starting dashboard...")
    db = ExplainerDashboard(explainer)
    db.run(host="127.0.0.1", port=8050, use_waitress=True)
print("Has MotorFlags?", "SME_TRQSPD_MotorFlags" in explainer.X.columns)