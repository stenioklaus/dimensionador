import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Dimensionador Fotovoltaico", page_icon="☀️", layout="wide")

st.title("☀️ Dimensionador Fotovoltaico ☀️")
st.caption("Cálculo de geração solar com dados da NASA POWER, elaborado por Estevão Krause - Der Krause")

N_DIAS_REFERENCIA = [17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]

def calcular_fator_rb_preciso(mes, lat, inc, azi=0):
    delta = np.radians(23.45 * np.sin(np.radians(360 * (284 + N_DIAS_REFERENCIA[mes - 1]) / 365)))
    phi = np.radians(lat)
    beta = np.radians(inc)
    gamma = np.radians(azi - 180)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    w_range = np.linspace(-ws, ws, 500)
    cos_theta = (
        np.sin(delta) * np.sin(phi) * np.cos(beta)
        - np.sin(delta) * np.cos(phi) * np.sin(beta) * np.cos(gamma)
        + np.cos(delta) * np.cos(phi) * np.cos(beta) * np.cos(w_range)
        + np.cos(delta) * np.sin(phi) * np.sin(beta) * np.cos(gamma) * np.cos(w_range)
        + np.cos(delta) * np.sin(beta) * np.sin(gamma) * np.sin(w_range)
    )
    prod_incl = np.trapezoid(np.maximum(cos_theta, 0), w_range)
    cos_theta_z = np.cos(phi) * np.cos(delta) * np.cos(w_range) + np.sin(phi) * np.sin(delta)
    prod_horiz = np.trapezoid(np.maximum(cos_theta_z, 0), w_range)
    return max(prod_incl / prod_horiz, 0.1) if prod_horiz > 0 else 0.1

@st.cache_data(ttl=86400)
def buscar_coordenadas(cidade):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={cidade}&format=json&limit=1"
        headers = {"User-Agent": "SolarSimulatorStreamlit/1.0"}
        response = requests.get(url, headers=headers, timeout=5).json()
        if response:
            return float(response[0]["lat"]), float(response[0]["lon"])
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=3600)
def buscar_dados_nasa(lat, lon):
    api_url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,T2M,T2M_MIN,T2M_MAX"
        f"&community=RE&latitude={lat}&longitude={lon}"
        f"&start=20220101&end=20221231&format=JSON"
    )
    return requests.get(api_url, timeout=30).json()

@st.cache_data(ttl=600)
def processar_dados(res, lat, inc1, azi1, inc2, azi2, ef_sys, temp_coef, ref_temp):
    solar_data = res["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    df = pd.DataFrame.from_dict(solar_data, orient="index", columns=["HSP"])
    df.index = pd.to_datetime(df.index)
    df_m = df.resample("MS").sum()
    t_max_dict = res["properties"]["parameter"]["T2M_MAX"]
    t_min_dict = res["properties"]["parameter"]["T2M_MIN"]
    t_max_m = (pd.DataFrame.from_dict(t_max_dict, orient="index", columns=["T"])
               .set_index(pd.to_datetime(list(t_max_dict.keys()))).resample("MS").mean())
    t_min_m = (pd.DataFrame.from_dict(t_min_dict, orient="index", columns=["T"])
               .set_index(pd.to_datetime(list(t_min_dict.keys()))).resample("MS").mean())
    f1 = [calcular_fator_rb_preciso(m, lat, inc1, azi1) for m in range(1, 13)]
    f2 = [calcular_fator_rb_preciso(m, lat, inc2, azi2) for m in range(1, 13)]
    effs = []
    for i in range(12):
        t_ref_amb = t_max_m["T"].iloc[i] if i in [10, 11, 0, 1, 2, 3] else t_min_m["T"].iloc[i]
        effs.append(max(0.01, ef_sys * (1 + temp_coef * (t_ref_amb - ref_temp))))
    return df_m, f1, f2, effs

def calcular_modulos(df_m, f1, f2, effs, consumo, pot_mod1_wp, pot_mod2_wp):
    hsp1 = sum([df_m["HSP"].iloc[i] * f1[i] * effs[i] for i in range(12)])
    hsp2 = sum([df_m["HSP"].iloc[i] * f2[i] * effs[i] for i in range(12)])
    n1 = int(np.ceil((consumo * 6) / hsp1 / (pot_mod1_wp / 1000))) if hsp1 > 0 else 0
    n2 = int(np.ceil((consumo * 6) / hsp2 / (pot_mod2_wp / 1000))) if hsp2 > 0 else 0
    return n1, n2

def enviar_telegram(mensagem, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": mensagem, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.warning(f"Erro ao enviar Telegram: {e}")
        return False

@st.cache_data(ttl=600)
def gerar_tabela_e_grafico(df_m, f1, f2, effs, n1, n2, pot_mod1_wp, pot_mod2_wp, meta):
    prod1 = [n1 * (pot_mod1_wp / 1000) * df_m["HSP"].iloc[i] * f1[i] * effs[i] for i in range(12)]
    prod2 = [n2 * (pot_mod2_wp / 1000) * df_m["HSP"].iloc[i] * f2[i] * effs[i] for i in range(12)]
    total = [prod1[i] + prod2[i] for i in range(12)]
    meses_abrev = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    df_res = pd.DataFrame({"Mês": meses_abrev, "Arr 1 (kWh)": prod1, "Arr 2 (kWh)": prod2, "Total (kWh)": total})
    df_res.set_index("Mês", inplace=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(meses_abrev, prod1, color="#1a3a6b", alpha=0.85, label="Arranjo 1")
    ax.bar(meses_abrev, prod2, bottom=prod1, color="#4da6e0", alpha=0.85, label="Arranjo 2")
    ax.axhline(y=meta, color="red", linestyle="--", linewidth=1.5, label=f"Meta ({meta} kWh)")
    ax.set_title("Geração Estimada por Arranjo", fontsize=13, pad=12)
    ax.set_ylabel("kWh")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return df_res, fig

# Inicializa session_state
for k, v in [("inc1", 20), ("azi1", 0), ("inc2", 20), ("azi2", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configurações")

    st.subheader("📍 Localização")
    cidade = st.text_input("Cidade", value="Igrejinha", placeholder="Digite e pressione Enter", key="cidade")

    if cidade != st.session_state.get("cidade_anterior", ""):
        st.session_state["cidade_anterior"] = cidade
        if len(cidade) > 3:
            lat_auto, lon_auto = buscar_coordenadas(cidade)
            if lat_auto:
                st.session_state["lat"] = lat_auto
                st.session_state["lon"] = lon_auto
                st.success(f"📍 {cidade}: {lat_auto:.4f}, {lon_auto:.4f}")
            else:
                st.warning("Cidade não encontrada. Ajuste as coordenadas manualmente.")

    lat = st.number_input("Latitude", value=st.session_state.get("lat", -29.57), format="%.4f", key="lat_input")
    lon = st.number_input("Longitude", value=st.session_state.get("lon", -50.79), format="%.4f", key="lon_input")
    meta = st.number_input("Meta de consumo (kWh/mês)", value=1000, min_value=200, step=100, key="meta_input")

    # Arranjo 1
    st.subheader("🔆 Arranjo 1")
    st.caption("Inclinação")
    a1, b1, c1 = st.columns([1, 2, 1])
    if a1.button("−", key="inc1_menos", use_container_width=True):
        st.session_state["inc1"] = max(0, st.session_state["inc1"] - 1)
    b1.markdown(f"<div style='text-align:center;padding:6px 0;font-size:18px;font-weight:600'>{st.session_state['inc1']}°</div>", unsafe_allow_html=True)
    if c1.button("+", key="inc1_mais", use_container_width=True):
        st.session_state["inc1"] = min(90, st.session_state["inc1"] + 1)

    st.caption("Orientação")
    a2, b2, c2 = st.columns([1, 2, 1])
    if a2.button("−", key="azi1_menos", use_container_width=True):
        st.session_state["azi1"] = max(-180, st.session_state["azi1"] - 1)
    b2.markdown(f"<div style='text-align:center;padding:6px 0;font-size:18px;font-weight:600'>{st.session_state['azi1']}°</div>", unsafe_allow_html=True)
    if c2.button("+", key="azi1_mais", use_container_width=True):
        st.session_state["azi1"] = min(180, st.session_state["azi1"] + 1)

    pot_mod1 = st.number_input("Potência módulo (Wp)", value=610, min_value=250, step=5, key="pot_mod1")

    # Arranjo 2
    st.subheader("🔆 Arranjo 2")
    st.caption("Inclinação")
    a3, b3, c3 = st.columns([1, 2, 1])
    if a3.button("−", key="inc2_menos", use_container_width=True):
        st.session_state["inc2"] = max(0, st.session_state["inc2"] - 1)
    b3.markdown(f"<div style='text-align:center;padding:6px 0;font-size:18px;font-weight:600'>{st.session_state['inc2']}°</div>", unsafe_allow_html=True)
    if c3.button("+", key="inc2_mais", use_container_width=True):
        st.session_state["inc2"] = min(90, st.session_state["inc2"] + 1)

    st.caption("Orientação")
    a4, b4, c4 = st.columns([1, 2, 1])
    if a4.button("−", key="azi2_menos", use_container_width=True):
        st.session_state["azi2"] = max(-180, st.session_state["azi2"] - 1)
    b4.markdown(f"<div style='text-align:center;padding:6px 0;font-size:18px;font-weight:600'>{st.session_state['azi2']}°</div>", unsafe_allow_html=True)
    if c4.button("+", key="azi2_mais", use_container_width=True):
        st.session_state["azi2"] = min(180, st.session_state["azi2"] + 1)

    pot_mod2 = st.number_input("Potência módulo (Wp)", value=610, min_value=250, step=5, key="pot_mod2")

    inc1 = st.session_state["inc1"]
    azi1 = st.session_state["azi1"]
    inc2 = st.session_state["inc2"]
    azi2 = st.session_state["azi2"]

    st.subheader("⚙️ Parâmetros do Sistema")
    ef_sys = st.slider("PR Base (eficiência do sistema)", 0.50, 1.00, 0.75, step=0.05, key="ef_sys")
    temp_coef = st.number_input("Coeficiente de temperatura", value=-0.004, format="%.4f", key="temp_coef")
    ref_temp = st.number_input("Temperatura de referência (°C)", value=25, step=1, key="ref_temp")

    st.divider()
    st.subheader("📲 Telegram (opcional)")
    tg_token = st.text_input("Token do bot", type="password", key="tg_token_input")
    tg_chat = st.text_input("Chat ID", key="tg_chat_input")
    if not tg_token:
        tg_token = os.environ.get("TELEGRAM_TOKEN", "")
    if not tg_chat:
        tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")

    calcular = st.button("🚀 Calcular com dados da NASA", type="primary", use_container_width=True)

# ── Área principal ────────────────────────────────────────────────────────────
if calcular:
    with st.spinner(f"Buscando dados NASA para {cidade}..."):
        try:
            res = buscar_dados_nasa(lat, lon)
            df_m, f1, f2, effs = processar_dados(res, lat, inc1, azi1, inc2, azi2, ef_sys, temp_coef, ref_temp)
            n1, n2 = calcular_modulos(df_m, f1, f2, effs, meta, pot_mod1, pot_mod2)
            st.session_state["dados"] = {"df_m": df_m, "f1": f1, "f2": f2, "effs": effs}
            st.session_state["n1"] = n1
            st.session_state["n2"] = n2
            st.success("Dados carregados com sucesso!")
        except requests.exceptions.Timeout:
            st.error("A API da NASA demorou demais. Tente novamente.")
        except requests.exceptions.ConnectionError:
            st.error("Sem conexão com a internet.")
        except KeyError:
            st.error("Resposta inesperada da NASA. Verifique as coordenadas.")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")

if "dados" in st.session_state:
    dados = st.session_state["dados"]

    st.subheader("Ajuste manual de módulos")
    col1, col2 = st.columns(2)
    n1_manual = col1.number_input("Módulos Arranjo 1", value=int(st.session_state.get("n1", 0)), min_value=0, key="n1_manual")
    n2_manual = col2.number_input("Módulos Arranjo 2", value=int(st.session_state.get("n2", 0)), min_value=0, key="n2_manual")

    recalc = st.button("🔄 Recalcular com ajuste manual")

    df_res, fig = gerar_tabela_e_grafico(
        dados["df_m"], dados["f1"], dados["f2"], dados["effs"],
        n1_manual, n2_manual, pot_mod1, pot_mod2, meta
    )
    total_anual = df_res.sum()
    media_mensal = int(total_anual["Total (kWh)"] / 12)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Módulos Arr. 1", n1_manual)
    m2.metric("Módulos Arr. 2", n2_manual)
    m3.metric("Geração média/mês", f"{media_mensal} kWh")
    m4.metric("Meta", f"{meta} kWh", delta=f"{media_mensal - meta:+.0f} kWh")

    st.pyplot(fig)

    with st.expander("📋 Tabela mensal detalhada"):
        st.dataframe(df_res.style.format("{:.0f}"), use_container_width=True)

    if (recalc or calcular) and tg_token and tg_chat:
        msg = (
            f"<b>Resultado Dimensionador Solar</b>\n"
            f"📍 {cidade}\n"
            f"Meta: {meta} kWh/mês\n"
            f"🛠 Arr1: {n1_manual}x {pot_mod1}Wp | {inc1}° inclinação | {azi1}° orientação\n"
            f"🛠 Arr2: {n2_manual}x {pot_mod2}Wp | {inc2}° inclinação | {azi2}° orientação\n"
            f"⚡ Média gerada: {media_mensal} kWh/mês"
        )
        ok = enviar_telegram(msg, tg_token, tg_chat)
        if ok:
            st.toast("Resultado enviado ao Telegram!", icon="📲")

else:
    st.info("Configure os parâmetros na barra lateral e clique em **Calcular** para iniciar.")
    st.markdown("""
    **Como usar:**
    1. Digite a cidade — as coordenadas são buscadas automaticamente
    2. Configure os dois arranjos (inclinação, orientação e potência dos módulos)
    3. Ajuste a eficiência e coeficientes do sistema
    4. Clique em *Calcular com dados da NASA*
    5. Se quiser, ajuste o número de módulos manualmente e recalcule
    """)
