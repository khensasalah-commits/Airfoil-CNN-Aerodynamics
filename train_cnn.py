import numpy as np
import pandas as pd
import matplotlib.path as mpath
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# 1. NACA 4-Digit Coordinate Generator
def generate_naca4_coords(m, p, t, num_points=100):
    x = np.linspace(0, 1, num_points // 2)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    if p > 0 and p < 1:
        yc = np.where(
            x < p, 
            (m / (p**2 + 1e-6)) * (2 * p * x - x**2),
            (m / ((1 - p)**2 + 1e-6)) * ((1 - 2 * p) + 2 * p * x - x**2)
        )
    else:
        yc = np.zeros_like(x)
    
    x_upper, y_upper = x, yc + yt
    x_lower, y_lower = x[::-1], (yc - yt)[::-1]
    
    return np.concatenate([x_upper, x_lower]), np.concatenate([y_upper, y_lower])

# 2. Synthetic Dataset Generation
np.random.seed(42)
num_samples = 1500
data = []

for _ in range(num_samples):
    m = np.random.uniform(0.0, 0.06)
    p = np.random.uniform(0.2, 0.5)
    t = np.random.uniform(0.06, 0.18)
    alpha = np.random.uniform(-5, 15)
    reynolds = np.random.uniform(1e5, 2e6)
    
    x, y = generate_naca4_coords(m, p, t)
    
    alpha_rad = np.radians(alpha)
    cl_ideal = 2 * np.pi * (alpha_rad + 2 * m)
    cl = cl_ideal + np.random.normal(0, 0.05)
    
    cd_0 = 0.006 + 0.01 * t + (100000 / reynolds)**0.2 * 0.002
    cd = max(cd_0 + 0.04 * (cl**2) + np.random.normal(0, 0.002), 0.001)
    
    # STORE GEOMETRY PARAMETERS DIRECTLY IN DATA DICT
    data.append({
        'm': m, 'p': p, 't': t,
        'x': x, 'y': y, 
        'angle_of_attack': alpha, 
        'reynolds_number': reynolds,
        'cl': cl, 'cd': cd
    })

df = pd.DataFrame(data)

# 3. Rasterization preserving aspect ratio
def coords_to_image(x, y, img_size=64):
    x_norm = x  # x is already in [0, 1]
    y_norm = y + 0.5  # shift y to center it in [0, 1] without scaling height independently
    y_norm = np.clip(y_norm, 0, 1)
    
    xv, yv = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    
    path = mpath.Path(np.column_stack([x_norm, y_norm]))
    inside = path.contains_points(points)
    return inside.reshape(img_size, img_size).astype(np.float32)

img_list = [coords_to_image(row['x'], row['y']) for _, row in df.iterrows()]
X_img = np.array(img_list).reshape(-1, 64, 64, 1)

# Normalization
aoa_norm = (df['angle_of_attack'].values - (-5)) / (15 - (-5))
re_norm = (np.log10(df['reynolds_number'].values) - 5) / (np.log10(2e6) - 5)

X_cond = np.column_stack([aoa_norm, re_norm])
y = df[['cl', 'cd']].values

# Train/Test Split
X_img_train, X_img_test, X_cond_train, X_cond_test, y_train, y_test = train_test_split(
    X_img, X_cond, y, test_size=0.2, random_state=42
)

# 4. Multi-Input CNN Model Construction
input_img = Input(shape=(64, 64, 1), name='geometry')
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

input_cond = Input(shape=(2,), name='conditions')

combined = Concatenate()([x, input_cond])
z = Dense(64, activation='relu')(combined)
z = Dense(32, activation='relu')(z)
output = Dense(2, name='coefficients')(z)

model = Model(inputs=[input_img, input_cond], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    [X_img_train, X_cond_train], y_train,
    validation_data=([X_img_test, X_cond_test], y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# 5. Evaluate CNN
y_pred = model.predict([X_img_test, X_cond_test])

cnn_r2_cl = r2_score(y_test[:, 0], y_pred[:, 0])
cnn_r2_cd = r2_score(y_test[:, 1], y_pred[:, 1])

print(f"CNN R² Score Cl: {cnn_r2_cl:.4f}")
print(f"CNN R² Score Cd: {cnn_r2_cd:.4f}")
print(f"CNN MAE Cl:      {mean_absolute_error(y_test[:, 0], y_pred[:, 0]):.4f}")
print(f"CNN MAE Cd:      {mean_absolute_error(y_test[:, 1], y_pred[:, 1]):.4f}")

# 6. Fixed Baseline Linear Regression
X_handcrafted = np.column_stack([
    df['t'].values,
    df['m'].values,
    df['p'].values,
    df['angle_of_attack'].values,
    np.log10(df['reynolds_number'].values)
])

X_train_hc, X_test_hc, y_train_hc, y_test_hc = train_test_split(
    X_handcrafted, y, test_size=0.2, random_state=42
)

lr_cl = LinearRegression().fit(X_train_hc, y_train_hc[:, 0])
lr_cd = LinearRegression().fit(X_train_hc, y_train_hc[:, 1])

cl_pred_lr = lr_cl.predict(X_test_hc)
cd_pred_lr = lr_cd.predict(X_test_hc)

lr_r2_cl = r2_score(y_test_hc[:, 0], cl_pred_lr)
lr_r2_cd = r2_score(y_test_hc[:, 1], cd_pred_lr)

print("\nLinear Regression Baseline:")
print(f"  R² Cl:  {lr_r2_cl:.4f}")
print(f"  R² Cd:  {lr_r2_cd:.4f}")
print(f"  MAE Cl: {mean_absolute_error(y_test_hc[:, 0], cl_pred_lr):.4f}")
print(f"  MAE Cd: {mean_absolute_error(y_test_hc[:, 1], cd_pred_lr):.4f}")

print(f"\nCNN Improvement in R² (Cl): {(cnn_r2_cl - lr_r2_cl) * 100:.2f} pp")
print(f"CNN Improvement in R² (Cd): {(cnn_r2_cd - lr_r2_cd) * 100:.2f} pp")