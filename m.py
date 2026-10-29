# ==============================================================================
# PIPELINE: BIOMODEL SIMULATION, DUAL-NN TRAINING & ARDUINO C COMPILATION
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

# 1. SPAC MECHANISTIC BIOMODEL GENERATOR
class AgroBiomodel:
    def __init__(self, days=730):
        self.days = days
        self.theta_sat = 0.48   # Saturation point (m3/m3)
        self.theta_fc  = 0.32   # Field capacity
        self.theta_pwp = 0.14   # Permanent wilting point
        self.root_depth = 600.0 # Effective root zone (mm)
        self.kc = 1.05          # Crop factor
        self.t_base = 10.0      # Base temp for GDD

    def simulate(self):
        records = []
        theta = self.theta_fc
        ndf = 35.0  # Initial Neutral Detergent Fiber %
        cp = 22.0   # Initial Crude Protein %

        for day in range(self.days):
            season = np.sin(2 * np.pi * day / 365.0)
            t_air = 25.0 + 12.0 * season + np.random.normal(0, 2.0)
            rh = np.clip(55.0 - 25.0 * season + np.random.normal(0, 6.0), 15.0, 95.0)
            solar_rad = np.clip(550.0 + 350.0 * season + np.random.normal(0, 40.0), 120.0, 950.0)

            # Hydrological events (Intermittent drought cycles)
            rain, irrigation = 0.0, 0.0
            if day not in range(130, 175) and day not in range(495, 540):
                if np.random.rand() < 0.15: rain = np.random.uniform(10.0, 40.0)
                if np.random.rand() < 0.20: irrigation = 25.0
            else:
                if np.random.rand() < 0.02: rain = 3.0 # Drought period

            # Reference Evapotranspiration (Hargreaves formulation)
            et0 = 0.0023 * (solar_rad / 28.5) * (t_air + 17.8) * np.sqrt(max(1.0, 35.0 - rh / 3.0))

            # Transpiration Stress Factor (Ks)
            if theta >= self.theta_fc:
                ks = 1.0
            elif theta <= self.theta_pwp:
                ks = 0.0
            else:
                ks = (theta - self.theta_pwp) / (self.theta_fc - self.theta_pwp)

            eta = ks * self.kc * et0
            water_in = rain + irrigation
            drainage = max(0.0, ((theta * self.root_depth) + water_in - eta) - (self.theta_fc * self.root_depth))

            # Euler integration for soil water balance
            d_theta = (water_in - eta - drainage) / self.root_depth
            theta = np.clip(theta + d_theta, 0.08, self.theta_sat)

            # Canopy temperature differential (cooling failure)
            delta_t = 3.5 * (1.0 - ks) + 2.0 * (solar_rad / 1000.0) - 1.5 * ks
            canopy_temp = t_air + delta_t

            # Phenological changes
            gdd = max(0.0, t_air - self.t_base)
            ndf = np.clip(ndf + 0.04 * gdd * (1.0 + 0.5 * (1.0 - ks)), 32.0, 68.0)
            cp = np.clip(cp - 0.025 * gdd, 7.0, 24.0)

            # Assign categorical ground truth
            if theta < (self.theta_pwp + 0.03):
                label = 1  # Drought Stress
            elif theta >= (self.theta_sat - 0.02):
                label = 3  # Overwatered
            elif canopy_temp > 38.0 and delta_t > 2.0:
                label = 2  # Heat Stress
            else:
                label = 0  # Normal

            records.append({
                "soil_moisture": (theta / self.theta_sat) * 100.0,
                "air_temp": t_air,
                "humidity": rh,
                "light_idx": (solar_rad / 1000.0) * 100.0,
                "ndf": ndf,
                "cp": cp,
                "stress": label
            })
        return pd.DataFrame(records)

df = AgroBiomodel(days=730).simulate()

# 2. MODEL 1: CROP STRESS CLASSIFIER (4 -> 8 -> 4 MLP)
Xc = df[["soil_moisture", "air_temp", "humidity", "light_idx"]].values
yc = df["stress"].values
scaler_c = StandardScaler()
Xc_scaled = scaler_c.fit_transform(Xc)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc_scaled, yc, test_size=0.2, random_state=42)
mlp_crop = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=500, random_state=42)
mlp_crop.fit(Xc_train, yc_train)

print("Crop Stress Classifier Metrics:")
print(classification_report(yc_test, mlp_crop.predict(Xc_test), target_names=["Normal", "Drought", "Heat", "Overwatered"]))

# 3. MODEL 2: FODDER DMI REGRESSOR (4 -> 6 -> 1 MLP)
cow_bw = np.random.uniform(400, 650, len(df))
milk_yield = np.random.uniform(10, 32, len(df))
dmi = (0.012 * cow_bw) / (df["ndf"].values / 100.0) + (0.30 * milk_yield) + np.random.normal(0, 0.2, len(df))

Xf = np.column_stack([cow_bw, milk_yield, df["ndf"].values, df["cp"].values])
yf = dmi
scaler_f = StandardScaler()
Xf_scaled = scaler_f.fit_transform(Xf)

Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf_scaled, yf, test_size=0.2, random_state=42)
mlp_fodder = MLPRegressor(hidden_layer_sizes=(6,), activation='relu', max_iter=500, random_state=42)
mlp_fodder.fit(Xf_train, yf_train)

print(f"Fodder Optimizer DMI RMSE: {np.sqrt(mean_squared_error(yf_test, mlp_fodder.predict(Xf_test))):.3f} kg/day")

# 4. EXPORT WEIGHT MATRICES TO C-HEADER (ZERO-SRAM PROGMEM)
def export_to_c_header(filename="agri_models.h"):
    with open(filename, "w") as f:
        f.write("#ifndef AGRI_MODELS_H\n#define AGRI_MODELS_H\n\n#include <avr/pgmspace.h>\n#include <math.h>\n\n")

        # Crop Scaler & Weights
        f.write(f"const float C_MEAN[4] PROGMEM = {{{', '.join(map(str, scaler_c.mean_))}}};\n")
        f.write(f"const float C_SCALE[4] PROGMEM = {{{', '.join(map(str, scaler_c.scale_))}}};\n")
        f.write(f"const float W_CROP_L1[32] PROGMEM = {{{', '.join(map(str, mlp_crop.coefs_[0].flatten()))}}};\n")
        f.write(f"const float B_CROP_L1[8] PROGMEM = {{{', '.join(map(str, mlp_crop.intercepts_[0]))}}};\n")
        f.write(f"const float W_CROP_L2[32] PROGMEM = {{{', '.join(map(str, mlp_crop.coefs_[1].flatten()))}}};\n")
        f.write(f"const float B_CROP_L2[4] PROGMEM = {{{', '.join(map(str, mlp_crop.intercepts_[1]))}}};\n\n")

        # Fodder Scaler & Weights
        f.write(f"const float F_MEAN[4] PROGMEM = {{{', '.join(map(str, scaler_f.mean_))}}};\n")
        f.write(f"const float F_SCALE[4] PROGMEM = {{{', '.join(map(str, scaler_f.scale_))}}};\n")
        f.write(f"const float W_FOD_L1[24] PROGMEM = {{{', '.join(map(str, mlp_fodder.coefs_[0].flatten()))}}};\n")
        f.write(f"const float B_FOD_L1[6] PROGMEM = {{{', '.join(map(str, mlp_fodder.intercepts_[0]))}}};\n")
        f.write(f"const float W_FOD_L2[6] PROGMEM = {{{', '.join(map(str, mlp_fodder.coefs_[1].flatten()))}}};\n")
        f.write(f"const float B_FOD_L2[1] PROGMEM = {{{', '.join(map(str, mlp_fodder.intercepts_[1]))}}};\n\n")

        # Static Float Inference Pipelines
        f.write("""
static inline int predict_crop_stress(float* x) {
    float h[8];
    for(int j = 0; j < 8; j++) {
        float sum = pgm_read_float(&B_CROP_L1[j]);
        for(int i = 0; i < 4; i++) {
            float x_scaled = (x[i] - pgm_read_float(&C_MEAN[i])) / pgm_read_float(&C_SCALE[i]);
            sum += x_scaled * pgm_read_float(&W_CROP_L1[i * 8 + j]);
        }
        h[j] = (sum > 0.0f) ? sum : 0.0f;
    }
    int best_class = 0;
    float max_val = -10000.0f;
    for(int k = 0; k < 4; k++) {
        float out = pgm_read_float(&B_CROP_L2[k]);
        for(int j = 0; j < 8; j++) {
            out += h[j] * pgm_read_float(&W_CROP_L2[j * 4 + k]);
        }
        if(out > max_val) {
            max_val = out;
            best_class = k;
        }
    }
    return best_class;
}

static inline float predict_fodder_dmi(float* x) {
    float h[6];
    for(int j = 0; j < 6; j++) {
        float sum = pgm_read_float(&B_FOD_L1[j]);
        for(int i = 0; i < 4; i++) {
            float x_scaled = (x[i] - pgm_read_float(&F_MEAN[i])) / pgm_read_float(&F_SCALE[i]);
            sum += x_scaled * pgm_read_float(&W_FOD_L1[i * 6 + j]);
        }
        h[j] = (sum > 0.0f) ? sum : 0.0f;
    }
    float dmi = pgm_read_float(&B_FOD_L2[0]);
    for(int j = 0; j < 6; j++) {
        dmi += h[j] * pgm_read_float(&W_FOD_L2[j]);
    }
    return dmi;
}
#endif
""")

export_to_c_header()
print("Successfully generated: agri_models.h")