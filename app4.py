import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# =========================================================
# 1. 페이지 기본 설정 및 Custom CSS
# =========================================================
st.set_page_config(
    page_title="해양 미세플라스틱 V-차단막 포집 최적화 시뮬레이터",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 800; color: #0F172A; margin-bottom: 4px; }
    .sub-header { font-size: 1.0rem; color: #475569; margin-bottom: 20px; }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 1.7rem; font-weight: 800; color: #0284C7; }
    .metric-label { font-size: 0.85rem; color: #64748B; margin-top: 4px; font-weight: 600; }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        background-color: #E0F2FE;
        color: #0369A1;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. Session State 상태 관리
# =========================================================
if "u_val" not in st.session_state:
    st.session_state["u_val"] = 1.70
if "v_val" not in st.session_state:
    st.session_state["v_val"] = 0.55
if "selected_location" not in st.session_state:
    st.session_state["selected_location"] = "쿠로시오 해류 (Kuroshio)"
if "last_click_coord" not in st.session_state:
    st.session_state["last_click_coord"] = None

# 주요 해류 데이터베이스
MAJOR_CURRENTS = [
    {"name": "쿠로시오 해류 (Kuroshio)", "lat": 29.5, "lon": 131.0, "u": 1.70, "v": 0.55},
    {"name": "북태평양 쓰레기 지대 (GPGP)", "lat": 32.0, "lon": -145.0, "u": 0.12, "v": -0.05},
    {"name": "캘리포니아 해류 (California)", "lat": 36.0, "lon": -122.0, "u": -0.15, "v": -0.45},
    {"name": "북적도 해류 (North Equatorial)", "lat": 15.0, "lon": 140.0, "u": -0.65, "v": 0.08},
    {"name": "멕시코 만류 (Gulf Stream)", "lat": 35.0, "lon": -75.0, "u": 1.50, "v": 0.70},
    {"name": "남극 순환류 (ACC)", "lat": -55.0, "lon": 20.0, "u": 0.85, "v": -0.10},
    {"name": "한국 남해 연안류 (South Sea Korea)", "lat": 34.2, "lon": 128.5, "u": 0.45, "v": 0.15},
]

def get_ocean_current_info(lat, lon):
    if lon > 180:
        lon -= 360
    for c in MAJOR_CURRENTS:
        if np.hypot(lat - c["lat"], lon - c["lon"]) < 7.0:
            return c["name"], c["u"], c["v"]
            
    rad = np.radians(lat)
    u_est = round(0.50 * np.cos(3 * rad) + 0.10 * np.sin(rad), 2)
    v_est = round(0.10 * np.sin(2 * rad), 2)
    loc_str = f"임의 선택 해역 (위도 {lat:.2f}°, 경도 {lon:.2f}°)"
    return loc_str, u_est, v_est

def calculate_physics_efficiency(net_speed, plastic_density):
    RHO_WATER = 1025.0
    buoyancy_factor = max(0.2, (RHO_WATER - plastic_density) / 125.0)
    
    angles = np.linspace(10, 90, 81)
    eff_list = []
    
    for ang in angles:
        half_rad = np.radians(ang / 2.0)
        span = np.sin(half_rad)
        hydro_factor = np.exp(-((ang - 35.0) ** 2) / 500.0)
        
        eff = 92.0 * (0.30 * span + 0.70 * hydro_factor) * (1.0 - 0.06 * max(0, net_speed - 0.5)) * buoyancy_factor
        eff = max(10.0, min(96.5, eff))
        eff_list.append(eff)
        
    best_idx = np.argmax(eff_list)
    return angles, np.array(eff_list), angles[best_idx], eff_list[best_idx]

# =========================================================
# 3. 사이드바 (파라미터 및 해류 설정)
# =========================================================
with st.sidebar:
    st.header("⚙️ 환경 및 환경 변수 설정")
    st.markdown("---")
    
    st.subheader("📍 선택된 해역 및 유속")
    st.info(f"**현재 위치:** {st.session_state['selected_location']}")
    
    col_u, col_v = st.columns(2)
    with col_u:
        u_in = st.number_input("동서 유속 u (m/s)", step=0.05, key="u_val")
    with col_v:
        v_in = st.number_input("남북 유속 v (m/s)", step=0.05, key="v_val")
        
    net_speed = np.hypot(u_in, v_in)
    
    st.markdown("---")
    st.subheader("🧪 미세플라스틱 및 차단막 규격")
    
    plastic_type = st.selectbox(
        "대상 플라스틱 종류",
        ["PE (Polyethylene - 0.92 g/cm³)", 
         "PP (Polypropylene - 0.90 g/cm³)", 
         "PS (Polystyrene - 1.04 g/cm³)",
         "PET (Polyethylene Terephthalate - 1.38 g/cm³)"]
    )
    
    density_map = {
        "PE (Polyethylene - 0.92 g/cm³)": 920.0,
        "PP (Polypropylene - 0.90 g/cm³)": 900.0,
        "PS (Polystyrene - 1.04 g/cm³)": 1040.0,
        "PET (Polyethylene Terephthalate - 1.38 g/cm³)": 1380.0
    }
    plastic_density = density_map[plastic_type]
    
    boom_length = st.slider("차단막 한쪽 날개 길이 L (m)", min_value=5.0, max_value=30.0, value=10.0, step=1.0)

# =========================================================
# 4. 메인 화면 레이아웃 (헤더 및 KPI 카드)
# =========================================================
st.markdown('<div class="main-header">🌊 해양 미세플라스틱 V-차단막 포집 최적화 시뮬레이터</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">해류 유속 및 차단막 각도별 포집 성능 동적 분석 시스템</div>', unsafe_allow_html=True)

angles, efficiencies, opt_angle, opt_eff = calculate_physics_efficiency(net_speed, plastic_density)
eff_span = boom_length * np.sin(np.radians(opt_angle / 2.0)) * 2.0

map_col, kpi_col = st.columns([1.1, 0.9])

with map_col:
    st.markdown("<span class='status-badge'>📍 해양 해류 관측 지도 (클릭하여 유속 선택)</span>", unsafe_allow_html=True)
    
    m = folium.Map(location=[20, 140], zoom_start=2, tiles="CartoDB positron")
    
    for cur in MAJOR_CURRENTS:
        folium.CircleMarker(
            location=[cur["lat"], cur["lon"]],
            radius=7,
            popup=f"<b>{cur['name']}</b><br>u={cur['u']} m/s, v={cur['v']} m/s",
            color="#0284C7",
            fill=True,
            fill_color="#38BDF8",
            fill_opacity=0.8
        ).add_to(m)
        
    map_data = st_folium(m, height=270, width=None, key="ocean_map")
    
    if map_data and (map_data.get("last_object_clicked") or map_data.get("last_clicked")):
        clicked = map_data.get("last_object_clicked") or map_data.get("last_clicked")
        clat, clon = clicked["lat"], clicked["lng"]
        current_coord = (round(clat, 3), round(clon, 3))
        
        if current_coord != st.session_state["last_click_coord"]:
            st.session_state["last_click_coord"] = current_coord
            loc_name, auto_u, auto_v = get_ocean_current_info(clat, clon)
            st.session_state["u_val"] = auto_u
            st.session_state["v_val"] = auto_v
            st.session_state["selected_location"] = loc_name
            st.rerun()

with kpi_col:
    st.markdown("<span class='status-badge'>📊 실시간 포집 최적화 핵심 연산 결과</span>", unsafe_allow_html=True)
    
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{net_speed:.2f} m/s</div><div class="metric-label">합성 유속 ($V_{{net}}$)</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#059669;">{opt_angle:.1f}°</div><div class="metric-label">최적 V-차단막 각도 ($\theta_{{opt}}$)</div></div>', unsafe_allow_html=True)
        
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    k3, k4 = st.columns(2)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2563EB;">{opt_eff:.1f} %</div><div class="metric-label">최대 예상 포집 효율</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#D97706;">{eff_span:.1f} m</div><div class="metric-label">차단막 유효 포집폭</div></div>', unsafe_allow_html=True)

# =========================================================
# 5. 하단 시각화 탭 (2D / 3D / 동적 입자 추적 애니메이션)
# =========================================================
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📈 2D 단면 최적화 곡선", 
    "🌐 3D 유속-각도-포집효율 입체 곡면",
    "🎬 실시간 미세플라스틱 포집 동적 추적 (Particle Flow)"
])

# ----- TAB 1: 2D 최적화 곡선 -----
with tab1:
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=angles, y=efficiencies, mode="lines", name="포집 효율 (%)", line=dict(color="#0284C7", width=3)))
    fig2d.add_trace(go.Scatter(x=[opt_angle], y=[opt_eff], mode="markers+text", name="최적 정점", marker=dict(color="#DC2626", size=12, symbol="star"), text=[f"  최적 각도: {opt_angle:.1f}° ({opt_eff:.1f}%)"], textposition="top right"))
    fig2d.update_layout(title=f"V-차단막 각도에 따른 포집 효율 (합성 유속: {net_speed:.2f} m/s)", xaxis_title="V-차단막 포함각도 θ (도)", yaxis_title="예상 포집 효율 (%)", height=420, template="plotly_white")
    st.plotly_chart(fig2d, use_container_width=True)

# ----- TAB 2: 3D 입체 곡면 -----
with tab2:
    speeds_3d = np.linspace(0.1, 3.0, 30)
    angles_3d = np.linspace(10, 90, 30)
    S, A = np.meshgrid(speeds_3d, angles_3d)
    
    Z = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ang, spd = A[i, j], S[i, j]
            half_r = np.radians(ang / 2.0)
            span = np.sin(half_r)
            hydro = np.exp(-((ang - 35.0) ** 2) / 500.0)
            eff_val = 92.0 * (0.30 * span + 0.70 * hydro) * (1.0 - 0.06 * max(0, spd - 0.5)) * max(0.2, (1025.0 - plastic_density)/125.0)
            Z[i, j] = max(10.0, min(96.5, eff_val))

    fig3d = go.Figure(data=[go.Surface(z=Z, x=S, y=A, colorscale="Viridis")])
    fig3d.update_layout(title="유속(Speed) - 각도(Angle) - 포집효율(Efficiency) 3D 응답 곡면", scene=dict(xaxis_title="유속 (m/s)", yaxis_title="차단막 각도 (°)", zaxis_title="포집 효율 (%)"), height=500)
    st.plotly_chart(fig3d, use_container_width=True)

# ----- TAB 3: 동적 입자(Particle) 포집 시뮬레이션 -----
with tab3:
    st.markdown("##### 🌊 선택한 V-각도 및 유속 조건에서의 입자 행동 시뮬레이션")
    st.caption("▶️ 아래 그래프 왼쪽 상단의 **[Play]** 버튼을 누르면 미세플라스틱 입자가 해류를 따라 이동하는 과정을 관찰할 수 있습니다.")

    half_rad = np.radians(opt_angle / 2.0)
    rx, ry = boom_length * np.sin(half_rad), boom_length * np.cos(half_rad)
    lx, ly = -rx, ry

    boom_x = [lx, 0, rx]
    boom_y = [ly, 0, ry]

    n_particles = 32
    n_frames = 16
    np.random.seed(42)
    init_x = np.linspace(-12, 12, n_particles) + np.random.uniform(-0.3, 0.3, n_particles)
    init_y = np.full(n_particles, -11.0)

    frames_data = []
    t_steps = np.linspace(0, 1, n_frames)

    for t in t_steps:
        fx, fy, colors = [], [], []
        for i in range(n_particles):
            x0, y0 = init_x[i], init_y[i]
            dist = max(0.4, net_speed) * t * 15.0
            
            y_curr = y0 + dist
            x_curr = x0

            barrier_y = np.abs(x0) / np.tan(half_rad) if np.sin(half_rad) > 0 else 0

            if np.abs(x0) <= rx:
                if y_curr >= barrier_y:
                    if opt_angle <= 48:
                        x_curr = x0 * max(0.0, 1.0 - t * 1.8)
                        y_curr = min(barrier_y, y0 + dist * 0.4)
                        colors.append("#10B981")
                    else:
                        x_curr = x0 + np.random.uniform(-0.7, 0.7)
                        y_curr = barrier_y + 1.2
                        colors.append("#EF4444")
                else:
                    colors.append("#0284C7")
            else:
                colors.append("#9CA3AF")

            fx.append(x_curr)
            fy.append(y_curr)
            
        frames_data.append((fx, fy, colors))

    fig_anim = go.Figure(
        data=[
            go.Scatter(x=boom_x, y=boom_y, mode="lines", name="V-차단막 (Boom)", line=dict(color="#0F172A", width=5)),
            go.Scatter(x=[0], y=[0], mode="markers", name="포집망 (Apex)", marker=dict(size=15, color="#F59E0B", symbol="star")),
            go.Scatter(x=frames_data[0][0], y=frames_data[0][1], mode="markers", name="미세플라스틱 입자", marker=dict(size=9, color=frames_data[0][2], opacity=0.85))
        ],
        layout=go.Layout(
            xaxis=dict(range=[-16, 16], title="수평 거리 (m)", zeroline=False),
            yaxis=dict(range=[-13, 13], title="해류 진행 방향 (m)", zeroline=False),
            height=480,
            template="plotly_white",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.0, y=1.15,
                buttons=[dict(
                    label="▶️ 포집 시뮬레이션 재생 (Play)",
                    method="animate",
                    args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                )]
            )]
        ),
        frames=[
            go.Frame(data=[
                go.Scatter(x=boom_x, y=boom_y),
                go.Scatter(x=[0], y=[0]),
                go.Scatter(x=fd[0], y=fd[1], mode="markers", marker=dict(size=9, color=fd[2], opacity=0.85))
            ]) for fd in frames_data
        ]
    )

    st.plotly_chart(fig_anim, use_container_width=True)
    st.info("💡 **시뮬레이션 해석:** 파란색 입자는 접근 중인 미세플라스틱이며, 차단막에 도달한 뒤 **초록색**으로 변해 중앙 포집망(Star)으로 모이면 정상 포집된 것입니다. (각도가 너무 클 경우 **빨간색**으로 변하며 와류에 유실됩니다.)")
