"""
Screener saham IDX harga kecil — mode Scalping / Day Trade / Swing.
Jalankan sore/malam setelah market tutup -> kandidat untuk dipantau berikutnya.
Data: Yahoo Finance via yfinance (end-of-day, bisa delay).

Pilih mode trading dulu -> tiap mode punya kriteria, target profit, dan
stop loss yang berbeda sesuai horizon holding-nya -> klik "Mulai Screening"
-> keluar list saham berpotensi lengkap dengan rekomendasi Buy / Sell (TP) /
Stop Loss.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Screener Saham IDX", layout="wide")

FEE_TOTAL_PCT = 0.15 + 0.25               # estimasi fee beli+jual broker (%)
CHUNK_SIZE = 60                           # jumlah ticker per batch request ke Yahoo Finance
TOP_N_DISPLAY = 50                        # batas baris yang ditampilkan di tabel utama

# ---------------- Mode trading: tiap mode punya kriteria & sizing SL/TP sendiri ----------------
MODES = {
    "scalping": {
        "label": "🔥 Scalping",
        "subtitle": "Beli Pagi Jual Sore — intraday, jual hari yang sama/besok pagi",
        "horizon": "Hari ini s/d besok pagi",
        "period": "3mo",
        "min_rows": 21,
        "price_range": (50, 1000),
        "min_avg_value": 1.0,      # Rp miliar, rata2 nilai transaksi 20 hari
        "min_vol_ratio": 1.2,      # lonjakan volume vs rata2 20 hari
        "chg_range": (0.5, 20.0),  # % kenaikan hari ini
        "min_close_pos": 0.5,      # posisi close di range harian
        "trend_check": "ma5",
        "sl_range": (1.5, 5.0),
        "rr_ratio": 2.0,
        "tp_cap": 10.0,
        "vol_mult": 0.6,
    },
    "daytrade": {
        "label": "⚡ Day Trade",
        "subtitle": "Hold 2-5 hari bursa, ikuti momentum jangka pendek",
        "horizon": "2-5 hari bursa",
        "period": "3mo",
        "min_rows": 25,
        "price_range": (50, 2000),
        "min_avg_value": 2.0,
        "min_vol_ratio": 1.0,
        "chg_range": (-2.0, 15.0),
        "min_close_pos": 0.4,
        "trend_check": "ma5_ma10",
        "sl_range": (3.0, 8.0),
        "rr_ratio": 2.5,
        "tp_cap": 20.0,
        "vol_mult": 0.8,
    },
    "swing": {
        "label": "📈 Swing",
        "subtitle": "Hold 1-4 minggu, ikuti tren menengah",
        "horizon": "1-4 minggu",
        "period": "6mo",
        "min_rows": 55,
        "price_range": (50, 5000),
        "min_avg_value": 3.0,
        "min_vol_ratio": 0.8,
        "chg_range": (-5.0, 20.0),
        "min_close_pos": 0.3,
        "trend_check": "ma20_ma50",
        "sl_range": (5.0, 15.0),
        "rr_ratio": 2.5,
        "tp_cap": 30.0,
        "vol_mult": 1.0,
    },
}


def load_universe() -> list[str]:
    f = Path(__file__).parent / "idx_tickers.csv"
    df = pd.read_csv(f)
    return df["code"].tolist()


def tick_size(price: float) -> int:
    # Fraksi harga IDX
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25


def round_tick(price: float, mode: str = "nearest") -> int:
    t = tick_size(price)
    if mode == "down":
        return int(price // t) * t
    if mode == "up":
        return -int(-price // t) * t
    return int(round(price / t)) * t


@st.cache_data(ttl=1800, show_spinner=False)
def download_chunk(symbols: tuple, period: str) -> pd.DataFrame:
    return yf.download(
        list(symbols), period=period, interval="1d", group_by="ticker",
        auto_adjust=False, threads=True, progress=False,
    )


def download_all(tickers: list, period: str) -> pd.DataFrame:
    symbols = [t + ".JK" for t in tickers]
    chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
    frames = []
    progress = st.progress(0.0, text=f"Mengambil data 0/{len(chunks)} batch...")
    for i, chunk in enumerate(chunks):
        try:
            frames.append(download_chunk(tuple(chunk), period))
        except Exception:
            pass  # lewati batch yang gagal (mis. rate-limit), lanjut ke batch berikutnya
        progress.progress((i + 1) / len(chunks), text=f"Mengambil data {i + 1}/{len(chunks)} batch...")
    progress.empty()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def get_one(df: pd.DataFrame, sym: str, min_rows: int) -> pd.DataFrame | None:
    try:
        d = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
    except KeyError:
        return None
    d = d.dropna(subset=["Close", "Volume"])
    d = d[d["Volume"] > 0]
    return d if len(d) >= min_rows else None


def check_trend(d: pd.DataFrame, close: float, trend_check: str) -> bool:
    if trend_check == "ma5":
        return close > float(d["Close"].iloc[-5:].mean())
    if trend_check == "ma5_ma10":
        ma5 = float(d["Close"].iloc[-5:].mean())
        ma10 = float(d["Close"].iloc[-10:].mean())
        return close > ma5 and ma5 > ma10
    if trend_check == "ma20_ma50":
        ma20 = float(d["Close"].iloc[-20:].mean())
        ma50 = float(d["Close"].iloc[-50:].mean())
        return close > ma20 and ma20 > ma50
    return True


def compute(df: pd.DataFrame, tickers: list, mode: dict) -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = get_one(df, t + ".JK", mode["min_rows"])
        if d is None:
            continue
        last, prev = d.iloc[-1], d.iloc[-2]
        hist = d.iloc[-21:-1]  # 20 hari sebelum hari terakhir, dasar volatilitas & likuiditas

        close = float(last["Close"])
        high, low = float(last["High"]), float(last["Low"])
        vol = float(last["Volume"])
        avg_vol = float(hist["Volume"].mean())
        avg_value = float((hist["Close"] * hist["Volume"]).mean())
        avg_range_pct = float(((hist["High"] - hist["Low"]) / hist["Close"]).mean() * 100)
        close_pos = (close - low) / (high - low) if high > low else 0.5
        chg_pct = (close / float(prev["Close"]) - 1) * 100
        vol_ratio = vol / avg_vol if avg_vol else 0

        # ukuran SL/TP mengikuti volatilitas harian saham itu sendiri, diskalakan per mode
        sl_lo, sl_hi = mode["sl_range"]
        sl_pct = min(max(avg_range_pct * mode["vol_mult"], sl_lo), sl_hi)
        tp_pct = min(sl_pct * mode["rr_ratio"], mode["tp_cap"])

        buy = round_tick(close, "nearest")
        sl = round_tick(buy * (1 - sl_pct / 100), "down")
        tp = round_tick(buy * (1 + tp_pct / 100), "down")

        pmin, pmax = mode["price_range"]
        cmin, cmax = mode["chg_range"]
        lolos = (
            pmin <= close <= pmax
            and avg_value >= mode["min_avg_value"]
            and vol_ratio >= mode["min_vol_ratio"]
            and cmin <= chg_pct <= cmax
            and close_pos >= mode["min_close_pos"]
            and check_trend(d, close, mode["trend_check"])
        )

        skor = vol_ratio * close_pos * (tp_pct / sl_pct)

        rows.append({
            "Kode": t,
            "Tanggal": d.index[-1].date(),
            "Close": close,
            "Chg %": chg_pct,
            "Vol/Avg20": vol_ratio,
            "Avg Value20 (Rp M)": avg_value,
            "Buy": buy,
            "Sell/TP": tp,
            "Stop Loss": sl,
            "Potensi Profit %": tp_pct,
            "Risiko %": sl_pct,
            "R:R": tp_pct / sl_pct,
            "Skor": skor,
            "Lolos": lolos,
        })
    return pd.DataFrame(rows)


# ---------------- UI ----------------
st.title("Screener Saham IDX")
st.caption(
    "Pilih mode trading, lalu klik Mulai Screening. Bukan rekomendasi/nasihat investasi, "
    "gunakan sebagai referensi pantauan saja."
)

mode_key = st.segmented_control(
    "Mode Trading",
    options=list(MODES.keys()),
    format_func=lambda k: MODES[k]["label"],
    default="scalping",
)
if mode_key is None:
    mode_key = "scalping"
mode = MODES[mode_key]

st.caption(f"**{mode['label']}** — {mode['subtitle']} · Target hold: **{mode['horizon']}**")

tickers = load_universe()

c1, c2 = st.columns([1, 2])
with c1:
    run = st.button("🔍 Mulai Screening", type="primary", use_container_width=True)
with c2:
    modal = st.number_input(
        "Modal per saham (Rp, opsional — untuk hitung jumlah lot)",
        min_value=0, step=500_000, value=0,
    )

st.caption(
    f"Auto-scan **{len(tickers)}** saham di BEI (papan Utama/Pengembangan/Akselerasi/"
    "Ekonomi Baru — papan Pemantauan Khusus dikecualikan karena mekanisme lelangnya "
    "tidak cocok untuk trading harian). Proses bisa makan waktu beberapa menit."
)

if "hasil" not in st.session_state:
    st.session_state.hasil = None
    st.session_state.hasil_mode = None

if run:
    raw = download_all(tickers, mode["period"])
    st.session_state.raw = raw
    st.session_state.hasil = compute(raw, tickers, mode)
    st.session_state.hasil_mode = mode_key

if st.session_state.hasil is None:
    st.info(f"Klik **Mulai Screening** untuk mencari kandidat mode **{mode['label']}**.")
    st.stop()

if st.session_state.hasil_mode != mode_key:
    st.warning(
        f"Mode diganti ke **{mode['label']}** — hasil di bawah masih dari mode sebelumnya. "
        "Klik **Mulai Screening** lagi untuk memindai ulang dengan kriteria mode ini."
    )
    st.stop()

res = st.session_state.hasil
if res.empty:
    st.error("Data tidak didapat dari Yahoo Finance. Coba klik Mulai Screening lagi (kemungkinan rate-limit sementara).")
    st.stop()

n_lolos = int(res["Lolos"].sum())

hasil_full = res.sort_values(["Lolos", "Skor"], ascending=[False, False]).copy()
hasil_full["Status"] = hasil_full["Lolos"].map({True: "✅ Lolos", False: "⏳ Pantau"})
hasil_full = hasil_full.drop(columns=["Lolos"])
hasil = hasil_full.head(TOP_N_DISPLAY).copy()

if modal > 0:
    hasil["Lot (100 lbr)"] = (modal // (hasil["Buy"] * 100)).astype(int)

st.subheader(f"Hasil screening ({mode['label']}): {n_lolos} lolos kriteria, {len(res)} saham berhasil dipindai")
st.caption(
    f"Target hold: **{mode['horizon']}**. Menampilkan top {len(hasil)} saham diurutkan dari yang "
    "paling berpotensi (Skor tertinggi). Tandai **⏳ Pantau** = belum memenuhi semua kriteria "
    "tapi masih layak dipantau."
)

fmt = {
    "Close": "{:,.0f}", "Chg %": "{:+.2f}", "Vol/Avg20": "{:.2f}x",
    "Avg Value20 (Rp M)": "{:.2f}", "Buy": "{:,.0f}", "Sell/TP": "{:,.0f}",
    "Stop Loss": "{:,.0f}", "Potensi Profit %": "{:+.2f}", "Risiko %": "{:.2f}",
    "R:R": "1:{:.1f}", "Skor": "{:.2f}",
}
st.dataframe(hasil.style.format(fmt), use_container_width=True, hide_index=True)
st.caption(
    f"Buy = harga acuan beli (dekat close terakhir). Sell/TP = target jual. "
    f"Stop Loss = batas cut loss. Estimasi fee broker beli+jual ~{FEE_TOTAL_PCT:.2f}% sudah "
    "perlu diperhitungkan saat hitung untung bersih."
)

pilih = st.selectbox("Lihat grafik", hasil["Kode"])
d = get_one(st.session_state.raw, pilih + ".JK", mode["min_rows"])
if d is not None:
    g1, g2 = st.columns(2)
    g1.line_chart(d["Close"], height=260)
    g2.bar_chart(d["Volume"], height=260)
