import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="해양 미세플라스틱 포집 시뮬레이터")

# ---------------------------------------------------------
# 1. 해류 데이터베이스 및 지구 해류 유속 추정 모델
# ---------------------------------------------------------
MAJOR_CURRENTS = [
    {"name": "쿠로시오 해류 (Kuroshio)", "lat": 29.5, "lon": 131.0, "u": 1.70, "v": 0.55},
    {"name": "북태평양 쓰레기 지대 (GPGP)", "lat": 32.0, "lon": -145.0, "u": 0.12, "v": -0.05},
    {"name": "캘리포니아 해류 (California)", "lat": 36.0, "lon": -122.0, "u": -0.15, "v": -0.45},
    {"name": "북적도 해류 (North Equatorial)", "lat": 15.0, "lon": 140.0, "u": -0.65, "v": 0.08},
    {"name": "멕시코 만류 (Gulf Stream)", "lat": 35.0, "lon": -75.0, "u": 1.50, "v": 0.70},
    {"name": "한국 남해 연안류 (South Sea Korea)", "lat": 34.2, "lon": 128.5, "u": 0.45, "v": 0.15},
]

def estimate_ocean_current(lat, lon):
    if lon > 180: lon -= 360
    # 1차: 근묵 주요 해류 확인
    for c in MAJOR_CURRENTS:
        if np.hypot(lat - c["lat"], lon - c["lon"]) < 6.0:
            return c["name"], c["u"], c["v"]
            
    # 2차: 임의 해역 위도 대기 순환 기반 유속 추정 모델
    rad = np.radians(lat)
    u_est = round(0.45 * np.cos(3 * rad) + 0.10 * np.sin(rad), 2)
    v_est = round(0.15 * np.sin(2 * rad), 2)
    return f"해역 관측지점 (위도 {lat:.2f}°, 경도 {lon:.2f}°)", u_est, v_est

# ---------------------------------------------------------
# 2. 물리 포집 효율 및 최적 각도 계산 엔진
# ---------------------------------------------------------
def calculate_efficiency_curve(net_speed, plastic_density):
    RHO_WATER = 1025.0
    reserve_buoyancy = max(0.01, (RHO_WATER - plastic_density) / RHO_WATER)
    submergence_factor = np.clip(reserve_buoyancy / 0.1366, 0.2, 1.0)
    
    angles = np.linspace(10, 90, 81)
    eff_list = []
    
    for ang in angles:
        half_rad = np.radians(ang / 2.0)
        span = np.sin(half_rad)                           # 유효 포집 폭
        slide = np.cos(half_rad) ** 2.2                   # 슬라이딩 유도 효율
        drag_penalty = (net_speed ** 1.2) * (np.sin(half_rad) ** 2) * 15.0
        v_normal = net_speed * np.sin(half_rad)
        washout = ((v_normal / submergence_factor) ** 1.3) * 22.0
        
        raw_eff = (100.0 * span * slide * submergence_factor) - drag_penalty - washout
        eff = max(5.0, min(98.0, raw_eff))
        eff_list.append(eff)
        
    best_idx = np.argmax(eff_list)
    return angles, np.array(eff_list), angles[best_idx], eff_list[best_idx]

# ---------------------------------------------------------
# 3. Session State 초기화 및 지도 클릭 동기화
# ---------------------------------------------------------
if "u_vel" not in st.session_state: st.session_state["u_vel"] = 0.50
if "v_vel" not in st.session_state: st.session_state["v_vel"] = 0.10
if "location_name" not in st.session_state: st.session_state["location_name"] = "기본 선택 해역"

st.title("📍 해 지도 상호작용 및 3D 물리 시뮬레이션")

col_map, col_controls = st.columns([1.2, 0.8])

with col_map:
    m = folium.Map(location=[25, 120], zoom_start=2, tiles="OpenStreetMap")
    for c in MAJOR_CURRENTS:
        folium.Marker(
            [c["lat"], c["lon"]],
            popup=f"<b>{c['name']}</b><br>유속: {np.hypot(c['u'], c['v']):.2f} m/s",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
    map_data = st_folium(m, width=650, height=450, key="ocean_map")

    # 지도 클릭 시 Session State 자동 업데이트
    clicked_lat, clicked_lon = None, None
    if map_data.get("last_object_clicked"):
        clicked_lat = map_data["last_object_clicked"]["lat"]
        clicked_lon = map_data["last_object_clicked"]["lng"]
    elif map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

    if clicked_lat is not None and clicked_lon is not None:
        loc_name, auto_u, auto_v = estimate_ocean_current(clicked_lat, clicked_lon)
        # 중요: 세션 값 강제 업데이트로 입력창 자동 수정 버그 해결
        st.session_state["u_vel"] = auto_u
        st.session_state["v_vel"] = auto_v
        st.session_state["location_name"] = loc_name
        st.rerun()

with col_controls:
    st.success(f"🎯 **선택된 위치**: {st.session_state['location_name']}")
    st.info("💡 지도 위의 해류 핀이나 아무 바다나 클릭하면 유속 값이 자동으로 설정됩니다.")
    
    u_val = st.number_input("동서 방향 유속 u (m/s)", value=float(st.session_state["u_vel"]), step=0.05, key="input_u")
    v_val = st.number_input("남북 방향 유속 v (m/s)", value=float(st.session_state["v_vel"]), step=0.05, key="input_v")
    
    plastic_type = st.selectbox("포집 대상 플라스틱 재질", ["PP (폴리프로필렌 - 900 kg/m³)", "PE (폴리에틸렌 - 920 kg/m³)", "PS (폴리스티렌 - 1040 kg/m³)"])
    density_map = {"PP (폴리프로필렌 - 900 kg/m³)": 900.0, "PE (폴리에틸렌 - 920 kg/m³)": 920.0, "PS (폴리스티렌 - 1040 kg/m³)": 1040.0}
    
    net_speed = np.hypot(u_val, v_val)
    angles, effs, opt_angle, opt_eff = calculate_efficiency_curve(net_speed, density_map[plastic_type])
    
    c1, c2 = st.columns(2)
    c1.metric("권장 최적 V-각도", f"{opt_angle:.0f}°")
    c2.metric("예상 포집 효율", f"{opt_eff:.1f} %")

# ---------------------------------------------------------
# 4. 3D Surface 입체 포집 그래프 렌더링
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🌐 3D 유속-각도-포집효율 종합 곡면 분석")

# 3D Meshgrid 생성
ang_grid = np.linspace(10, 90, 40)
spd_grid = np.linspace(0.1, 2.5, 40)
A, S = np.meshgrid(ang_grid, spd_grid)

Z = []
for s_row, a_row in zip(S, A):
    z_row = []
    for s_elem, a_elem in zip(s_row, a_row):
        _, _, _, eff_val = calculate_efficiency_curve(s_elem, density_map[plastic_type])
        z_row.append(eff_val)
    Z.append(z_row)

fig_3d = go.Figure(data=[go.Surface(x=A, y=S, z=np.array(Z), colorscale="Viridis", opacity=0.85)])

# 현재 선택된 해역 상태 점(Marker) 표시
fig_3d.add_trace(go.Scatter3d(
    x=[opt_angle], y=[net_speed], z=[opt_eff],
    mode='markers+text',
    marker=dict(size=8, color='red', symbol='diamond'),
    name='현재 해역 최적점',
    text=[f"최적: {opt_angle:.0f}° ({opt_eff:.1f}%)"],
    textposition="top center"
))

fig_3d.update_layout(
    scene=dict(
        xaxis_title="V-차단막 각도 (°)",
        yaxis_title="해류 총 유속 (m/s)",
        zaxis_title="포집 효율 (%)"
    ),
    margin=dict(l=10, r=10, b=10, t=30),
    height=550
)

st.plotly_chart(fig_3d, use_container_width=True)
