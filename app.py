"""
Screener BPJS (Beli Pagi Jual Sore) untuk saham IDX harga kecil.
Jalankan sore/malam setelah market tutup -> kandidat untuk dibeli besok pagi.
Data: Yahoo Finance via yfinance (end-of-day, bisa delay).

Tinggal pencet tombol "Mulai Screening" -> otomatis scan seluruh saham IDX
(papan Utama/Pengembangan/Akselerasi/Ekonomi Baru) -> keluar list saham
berpotensi scalping lengkap dengan rekomendasi harga Buy, Sell (TP), dan
Stop Loss.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Screener BPJS", layout="wide")

# ---------------- Kriteria & parameter scalping (fixed, tidak perlu diatur) ----------------
PRICE_MIN, PRICE_MAX = 50, 1000          # saham harga kecil
MIN_AVG_VALUE = 1.0                       # Rp miliar, rata2 nilai transaksi 20 hari (likuiditas)
MIN_VOL_RATIO = 1.2                       # lonjakan volume vs rata2 20 hari
CHG_MIN, CHG_MAX = 0.5, 20.0              # % kenaikan hari ini (hindari yang sudah mepet ARA)
MIN_CLOSE_POS = 0.5                       # posisi close di range harian (buyer dominan)
FEE_TOTAL_PCT = 0.15 + 0.25               # estimasi fee beli+jual broker (%)

SL_MIN_PCT, SL_MAX_PCT = 1.5, 5.0         # batas stop loss (%)
RR_RATIO = 2.0                            # target profit = SL x rasio ini (risk:reward 1:2)
TP_MAX_PCT = 10.0                         # batas atas target profit (%)

CHUNK_SIZE = 60                           # jumlah ticker per batch request ke Yahoo Finance
TOP_N_DISPLAY = 50                        # batas baris yang ditampilkan di tabel utama


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
def download_chunk(symbols: tuple, period: str = "3mo") -> pd.DataFrame:
    return yf.download(
        list(symbols), period=period, interval="1d", group_by="ticker",
        auto_adjust=False, threads=True, progress=False,
    )


def download_all(tickers: list, period: str = "3mo") -> pd.DataFrame:
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


def get_one(df: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    try:
        d = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
    except KeyError:
        return None
    d = d.dropna(subset=["Close", "Volume"])
    d = d[d["Volume"] > 0]
    return d if len(d) >= 21 else None


def compute(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = get_one(df, t + ".JK")
        if d is None:
            continue
        last, prev = d.iloc[-1], d.iloc[-2]
        hist = d.iloc[-21:-1]  # 20 hari sebelum hari terakhir

        close = float(last["Close"])
        high, low = float(last["High"]), float(last["Low"])
        vol = float(last["Volume"])
        avg_vol = float(hist["Volume"].mean())
        avg_value = float((hist["Close"] * hist["Volume"]).mean())
        avg_range_pct = float(((hist["High"] - hist["Low"]) / hist["Close"]).mean() * 100)
        close_pos = (close - low) / (high - low) if high > low else 0.5
        chg_pct = (close / float(prev["Close"]) - 1) * 100
        vol_ratio = vol / avg_vol if avg_vol else 0

        # ukuran SL/TP mengikuti volatilitas harian saham itu sendiri
        sl_pct = min(max(avg_range_pct * 0.6, SL_MIN_PCT), SL_MAX_PCT)
        tp_pct = min(sl_pct * RR_RATIO, TP_MAX_PCT)

        buy = round_tick(close, "nearest")
        sl = round_tick(buy * (1 - sl_pct / 100), "down")
        tp = round_tick(buy * (1 + tp_pct / 100), "down")

        lolos = (
            PRICE_MIN <= close <= PRICE_MAX
            and avg_value >= MIN_AVG_VALUE
            and vol_ratio >= MIN_VOL_RATIO
            and CHG_MIN <= chg_pct <= CHG_MAX
            and close_pos >= MIN_CLOSE_POS
            and close > float(d["Close"].iloc[-5:].mean())
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
st.title("Screener BPJS saham harga kecil")
st.caption(
    "Beli Pagi Jual Sore — klik tombol di bawah untuk mencari saham berpotensi scalping "
    "lengkap dengan rekomendasi Buy / Sell (TP) / Stop Loss. Bukan rekomendasi/nasihat "
    "investasi, gunakan sebagai referensi pantauan saja."
)

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
    "tidak cocok untuk scalping). Proses bisa makan waktu beberapa menit."
)

if "hasil" not in st.session_state:
    st.session_state.hasil = None

if run:
    raw = download_all(tickers)
    st.session_state.raw = raw
    st.session_state.hasil = compute(raw, tickers)

if st.session_state.hasil is None:
    st.info("Klik **Mulai Screening** untuk mulai mencari kandidat.")
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

st.subheader(f"Hasil screening: {n_lolos} lolos kriteria, {len(res)} saham berhasil dipindai")
st.caption(
    f"Menampilkan top {len(hasil)} saham diurutkan dari yang paling berpotensi (Skor tertinggi). "
    "Tandai **⏳ Pantau** = belum memenuhi semua kriteria tapi masih layak dipantau."
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
d = get_one(st.session_state.raw, pilih + ".JK")
if d is not None:
    g1, g2 = st.columns(2)
    g1.line_chart(d["Close"], height=260)
    g2.bar_chart(d["Volume"], height=260)
