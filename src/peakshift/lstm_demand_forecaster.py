import math, random, numpy as np, pandas as pd, torch
from torch import nn
from datetime import datetime, timedelta
from pathlib import Path

class LSTMDemandDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = features
        self.y = targets
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMDemandModel(nn.Module):
    def __init__(self, input_dim: int, hidden=128, layers=2, horizon=48):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        out,_ = self.lstm(x)
        last = out[:,-1,:]
        pred = self.head(last)
        return pred.unsqueeze(-1)

class LSTMDemandForecaster:
    def __init__(self, seq_len=96, horizon=48, device=None, scaler_eps=1e-6):
        self.seq_len = seq_len
        self.horizon = horizon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.demand_min = None
        self.demand_max = None
        self.scaler_eps = scaler_eps

    def _build_time_features(self, df: pd.DataFrame):
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60
        df["dow"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
        df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
        df["doy"] = df["timestamp"].dt.dayofyear
        df["doy_sin"] = np.sin(2*np.pi*df["doy"]/365)
        df["doy_cos"] = np.cos(2*np.pi*df["doy"]/365)
        df["lag_48"] = df["demand_kW"].shift(48)
        df["ma_48"] = df["demand_kW"].rolling(48, min_periods=1).mean()
        df["ma_336"] = df["demand_kW"].rolling(336, min_periods=1).mean()
        df["std_336"] = df["demand_kW"].rolling(336, min_periods=1).std().fillna(0)
        return df

    def _scale(self, series: pd.Series, fit=False):
        if fit:
            self.demand_min = float(series.min())
            self.demand_max = float(series.max())
        if self.demand_min is None or self.demand_max is None:
            raise RuntimeError("Scaler has not been fitted. Call _scale with fit=True first.")
        return (series - self.demand_min)/(self.demand_max - self.demand_min + self.scaler_eps)

    def _inverse_scale(self, arr: np.ndarray):
        if self.demand_min is None or self.demand_max is None:
            raise RuntimeError("Scaler has not been fitted.")
        return arr*(self.demand_max - self.demand_min + self.scaler_eps) + self.demand_min

    def _make_windows(self, feat_df: pd.DataFrame, demand_scaled: pd.Series):
        numeric_cols = feat_df.select_dtypes(include=np.number).columns.tolist()
        cols_to_use = [c for c in numeric_cols if c not in ["timestamp","demand_kW"]]
        
        feat_mat = feat_df[cols_to_use].to_numpy(dtype=np.float32)
        y = demand_scaled.to_numpy(dtype=np.float32)
        X_list, Y_list = [], []
        total = len(feat_df)
        for start in range(0, total - self.seq_len - self.horizon + 1):
            end_in = start + self.seq_len
            end_out = end_in + self.horizon
            X_list.append(feat_mat[start:end_in])
            Y_list.append(y[end_in:end_out])
        return np.stack(X_list), np.stack(Y_list)

    def fit_until(self, df_all: pd.DataFrame, cutoff_day: pd.Timestamp, epochs=8, lr=1e-3, batch=64):
        df = df_all[df_all["timestamp"] < cutoff_day].copy()
        if df.empty:
            raise ValueError("No history before cutoff_day")
        df["demand_kW"] = df["demand_mw"]*1000
        df.sort_values("timestamp", inplace=True)
        df["demand_kW"] = df["demand_kW"].interpolate().bfill().ffill()
        df = self._build_time_features(df)
        df["lag_48"].fillna(df["demand_kW"], inplace=True)
        demand_scaled = self._scale(df["demand_kW"], fit=True)
        X, Y = self._make_windows(df, demand_scaled)
        if len(X)==0:
            raise ValueError("Not enough data for windows")
        dataset = LSTMDemandDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=False)
        input_dim = X.shape[2]
        self.model = LSTMDemandModel(input_dim, horizon=self.horizon).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for ep in range(epochs):
            total_loss=0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)[:,:,0]
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()*len(xb)
        return {"train_samples": len(dataset)}

    def predict_day(self, df_all: pd.DataFrame, target_day: pd.Timestamp):
        start_idx_mask = (df_all["timestamp"] < target_day)
        df_hist = df_all[start_idx_mask].copy()
        df_future = df_all[(df_all["timestamp"] >= target_day) &
                           (df_all["timestamp"] < target_day + pd.Timedelta(hours=self.horizon/2))].copy()
        if len(df_future) < self.horizon:
            raise ValueError("Future horizon rows insufficient")
        df_hist["demand_kW"] = df_hist["demand_mw"]*1000
        df_hist = self._build_time_features(df_hist)
        df_sub = df_hist.iloc[-self.seq_len:].copy()
        if len(df_sub) < self.seq_len:
            raise ValueError("Not enough history for prediction")
        df_sub["lag_48"].fillna(df_sub["demand_kW"], inplace=True)
        if self.demand_min is None or self.demand_max is None:
            raise RuntimeError("Scaler has not been fitted. Call fit_until first.")
        demand_scaled = (df_sub["demand_kW"] - self.demand_min)/(self.demand_max - self.demand_min + self.scaler_eps)
        
        numeric_cols = df_sub.select_dtypes(include=np.number).columns.tolist()
        cols_to_use = [c for c in numeric_cols if c not in ["timestamp", "demand_kW"]]
        
        X = df_sub[cols_to_use].to_numpy(dtype=np.float32)[None,...]
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit_until first.")
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(X).to(self.device))[:,:,0].cpu().numpy()[0]
        pred = self._inverse_scale(pred_scaled)
        df_future = df_future.copy()
        df_future["demand_predict_kW"] = pred[:len(df_future)]
        df_future["demand_predict_mw"] = df_future["demand_predict_kW"]/1000
        return df_future[["timestamp","demand_predict_mw","demand_predict_kW"]]
