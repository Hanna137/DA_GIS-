import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 


# HÀM TÍNH METRIC 
def nse(obs, sim):
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

# THÊM: Hàm tính đầy đủ metrics
def calc_metrics_full(obs, sim):
    r2 = r2_score(obs, sim)
    rmse = mean_squared_error(obs, sim) ** 0.5
    mae = mean_absolute_error(obs, sim)
    nse_val = nse(obs, sim)
    return r2, rmse, mae, nse_val


# HÀM VẼ BIỂU ĐỒ ĐÁNH GIÁ (Từng Trạm) 

def plot_evaluation_charts(station, obs, pred, time, outdir, r2, rmse, nse_val):
    os.makedirs(outdir, exist_ok=True)

    # 1. ===== Biểu đồ Chuỗi Thời Gian (Time series) =====
    plt.figure(figsize=(12, 4))
    plt.plot(time, obs, label="Thực đo (Test)", linewidth=2.5, color="blue")
    plt.plot(time, pred, label="Mô phỏng (Test)", linewidth=1.2, color="orange")
    plt.title(f"Time series - {station} (Test Set 2021-2022)")
    plt.xlabel("Thời gian")
    plt.ylabel(station)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"timeseries_test_{station}.png"), dpi=200)
    plt.close()

    # 2. Biểu đồ Phân Tán (Scatter)
    plt.figure(figsize=(5, 5))
    plt.scatter(obs, pred, s=15, alpha=0.6)

    mn = min(np.nanmin(obs), np.nanmin(pred))
    mx = max(np.nanmax(obs), np.nanmax(pred))
    pad = (mx - mn) * 0.05

    plt.plot([mn-pad, mx+pad], [mn-pad, mx+pad], "k--", linewidth=1.5)
    plt.xlabel("Thực đo (Test)")
    plt.ylabel("Mô phỏng (Test)")
    plt.title(f"Scatter - {station} (Test Set 2021-2022)")

    txt = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nNSE = {nse_val:.3f}"
    plt.text(
        0.02, 0.98, txt, transform=plt.gca().transAxes,
        va="top", bbox=dict(facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"scatter_test_{station}.png"), dpi=200)
    plt.close()


# HÀM VẼ BIỂU ĐỒ TỔNG HỢP (Tất cả Trạm) - MỚI

def plot_all_stations_summary(all_obs, all_pred, all_time_list, station_names, outdir):
    
    # Gom tất cả các mảng lại thành một mảng lớn để tính metrics tổng hợp
    all_obs_flat = np.concatenate(all_obs)
    all_pred_flat = np.concatenate(all_pred)
    r2_all, rmse_all, mae_all, nse_all = calc_metrics_full(all_obs_flat, all_pred_flat)

    # 1. Biểu đồ Phân tán Tổng hợp
    plt.figure(figsize=(6, 6))
    for obs_arr, pred_arr, station in zip(all_obs, all_pred, station_names):
        plt.scatter(obs_arr, pred_arr, s=15, alpha=0.6, label=station)
    
    mn = min(np.nanmin(all_obs_flat), np.nanmin(all_pred_flat))
    mx = max(np.nanmax(all_obs_flat), np.nanmax(all_pred_flat))
    pad = (mx - mn) * 0.05
    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth=1.5)
    
    txt = f"Tổng hợp Test Set:\nR² = {r2_all:.3f}\nRMSE = {rmse_all:.3f}\nNSE = {nse_all:.3f}"
    plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va="top", bbox=dict(facecolor="white", alpha=0.8))
    
    plt.xlabel("Thực đo (Test Set)")
    plt.ylabel("Mô phỏng (Test Set)")
    plt.title("Scatter - Tổng hợp Đánh giá Mô hình (2021-2022)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_test_AllStations.png"), dpi=200)
    plt.close()
    print("✔ Đã lưu Biểu đồ Phân tán Tổng hợp (Test Set)")

    # 2. Biểu đồ Chuỗi Thời gian Nhóm (Grouped Time Series)
    plt.figure(figsize=(15, 8))
    
    num_stations = len(station_names)
    rows = int(np.ceil(num_stations / 3))
    
    for i, (obs_arr, pred_arr, time_arr, station) in enumerate(zip(all_obs, all_pred, all_time_list, station_names)):
        plt.subplot(rows, 3, i + 1)
        
        st_r2, _, _, _ = calc_metrics_full(obs_arr, pred_arr)

        plt.plot(time_arr, obs_arr, label="Thực đo", linewidth=2.0, color="blue")
        plt.plot(time_arr, pred_arr, label="Mô phỏng", linewidth=1.0, color="orange")
        
        plt.title(f"{station} (R²: {st_r2:.2f})", fontsize=10)
        plt.xlabel("Thời gian", fontsize=8)
        plt.ylabel("Độ Mặn", fontsize=8)
        plt.xticks(fontsize=7, rotation=45)
        plt.yticks(fontsize=7)
    
    plt.figlegend(labels=["Thực đo", "Mô phỏng"], loc='upper right', fontsize=10)
    plt.suptitle("Chuỗi Thời Gian Độ Mặn - Đánh giá Mô hình (2021-2022)", fontsize=16, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(outdir, "timeseries_test_GroupedStations.png"), dpi=200)
    plt.close()
    print("✔ Đã lưu Biểu đồ Chuỗi Thời gian Nhóm (Test Set)")


# KHAI BÁO BIẾN GLOBAL VÀ THƯ MỤC

OUTDIR_EVAL = r"D:\DA GIS\RF_Evaluation_Charts"
os.makedirs(OUTDIR_EVAL, exist_ok=True)
os.makedirs(os.path.join(OUTDIR_EVAL, "Individual_Stations"), exist_ok=True)

all_test_obs = []     # Lưu trữ toàn bộ dữ liệu thực đo Test Set
all_test_pred = []    # Lưu trữ toàn bộ dữ liệu mô phỏng Test Set
all_test_time = []    # Lưu trữ toàn bộ thời gian Test Set
station_names = []    # Lưu trữ tên trạm


print("=== LOAD DATA ===")
df_raw = pd.read_excel(r"D:\DA GIS\DuLieuChung_1Hang1Ngay_SapXepTheoNamff.xlsx")
print("Số dòng dữ liệu gốc:", len(df_raw))

# ---- Chuẩn hóa Date ----
df_raw["Date"] = pd.to_datetime(df_raw["Date"])

# ---- Tạo full range ngày 2007–2022 ----
full_days = pd.DataFrame({"Date": pd.date_range("2007-01-01", "2022-12-31", freq="D")})
df_merged = full_days.merge(df_raw, on="Date", how="left")
print("Kích thước sau merge:", df_merged.shape)

# ---- Danh sách cột ----
salinity_cols = [c for c in df_merged.columns if "ĐộMặn" in c or "DoMan" in c]
feature_cols_raw = [c for c in df_merged.columns if c not in salinity_cols + ["Date"]]

print("Số cột độ mặn:", len(salinity_cols))
print("Số cột feature:", len(feature_cols_raw))

# ====== SET DATETIME INDEX ======
df_merged = df_merged.set_index("Date")


#  BƯỚC 1: INTERPOLATE CÁC FEATURE KHÍ TƯỢNG / THỦY VĂN
print("\n=== BƯỚC 1: INTERPOLATE FEATURE KHÍ TƯỢNG / THỦY VĂN ===")
df_merged[feature_cols_raw] = df_merged[feature_cols_raw].interpolate(method="time")
print("✔ Hoàn thành interpolate feature.")

#  BƯỚC 2: TẠO LAG CHO TỪNG TRẠM ĐỘ MẶN
print("\n=== BƯỚC 2: TẠO LAG CHO TỪNG TRẠM ĐỘ MẶN ===")
lag_cols = []
for col in salinity_cols:
    df_merged[f"{col}_lag1"] = df_merged[col].shift(3)
    df_merged[f"{col}_lag2"] = df_merged[col].shift(5)
    df_merged[f"{col}_lag3"] = df_merged[col].shift(7)
    lag_cols += [f"{col}_lag1", f"{col}_lag2", f"{col}_lag3"]

df_merged[lag_cols] = df_merged[lag_cols].interpolate(method="time")
print("✔ Hoàn thành tạo lag.")

#  BƯỚC 3: TRAIN RF + TEST SPLIT
print("\n=== BƯỚC 3: TRAIN RF + TEST SPLIT (2007–2020 | 2021–2022) ===")

model_features = feature_cols_raw + lag_cols
metrics_summary = [] # Lưu R2, RMSE, NSE

for col in salinity_cols:
    print(f"\n---- Đang xử lý trạm: {col} ----")

    df_full = df_merged[df_merged[col].notna()].copy()
    df_full = df_full.loc["2007-06-05":"2022-12-31"]

    # Train-test split theo thời gian
    df_train = df_full.loc["2007-06-05":"2020-12-31"]
    df_test = df_full.loc["2021-01-01":"2022-12-31"]

    # Nếu test không có dữ liệu thật → bỏ qua
    if len(df_test) == 0:
        print("⚠ Không có dữ liệu test (2021–2022)")
        continue

    X_train = df_train[model_features]
    y_train = df_train[col]

    X_test = df_test[model_features]
    y_test = df_test[col]

    # Train model trên TRAIN SET
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=20
    )
    model.fit(X_train, y_train)

    # ---- ĐÁNH GIÁ TRÊN TEST SET ----
    y_test_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
    nse_val = nse(y_test.values, y_test_pred)

    print(f"→ R² (test): {r2:.4f}")
    print(f"→ RMSE (test): {rmse:.4f}")
    print(f"→ NSE (test): {nse_val:.4f}")

    metrics_summary.append([col, r2, rmse, nse_val])
    
    # ---- THU THẬP DỮ LIỆU ĐỂ VẼ BIỂU ĐỒ TỔNG HỢP 
    all_test_obs.append(y_test.values)
    all_test_pred.append(y_test_pred)
    all_test_time.append(y_test.index)
    station_names.append(col)

    # ---- VẼ BIỂU ĐỒ ĐÁNH GIÁ CHO TỪNG TRẠM 
    plot_evaluation_charts(
        col, y_test.values, y_test_pred, y_test.index, 
        os.path.join(OUTDIR_EVAL, "Individual_Stations"), 
        r2, rmse, nse_val
    )
    print(f"✔ Đã lưu biểu đồ đánh giá cho trạm {col}")

    # ---- Dùng FULL MODEL (2007–2020 only) để fill NA ----
    df_pred = df_merged[df_merged[col].isna()]
    df_pred = df_pred.loc["2007-06-05":"2022-12-31"]
    if len(df_pred) > 0:
        y_pred_fill = model.predict(df_pred[model_features])
        df_merged.loc[df_pred.index, col] = y_pred_fill
        print(f"✔ Fill NA xong cho trạm {col}")

#  BƯỚC 4: VẼ BIỂU ĐỒ TỔNG HỢP (ALL STATIONS) 
print("\n=== BƯỚC 4: VẼ BIỂU ĐỒ TỔNG HỢP TEST SET (2021-2022) ===")
if len(all_test_obs) > 0:
    plot_all_stations_summary(all_test_obs, all_test_pred, all_test_time, station_names, OUTDIR_EVAL)
else:
    print("⚠ Không có dữ liệu hợp lệ để vẽ biểu đồ tổng hợp.")
    
#  BƯỚC CUỐI: XUẤT FILE
output = df_merged.reset_index()
output.to_excel(r"D:\DA GIS\DoMan_Fill_2007_2022_Final1.xlsx", index=False)

df_metrics = pd.DataFrame(metrics_summary, 
                          columns=["Station", "R2_test", "RMSE_test", "NSE_test"])
df_metrics.to_excel(r"D:\DA GIS\DanhGia_R2_RMSE_NSEf.xlsx", index=False)