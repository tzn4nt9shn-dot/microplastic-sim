import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------
# 1. 페이지 기본 설정 및 고성능 UI CSS 스타일링
# ---------------------------------------------------------
st.set_page_config(
    page_title="해양 미세플라스틱 포집 시뮬레이터",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 세련되고 정돈된 대시보드 CSS 디자인
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif; }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 22px 28px;
        border-radius: 14px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .header-title {
        font-size: 25px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-subtitle {
        font-size: 14px;
        color: #94A3B8;
        line-height: 1.5;
    }

    .metric-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 10px;
    }
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }
    .stat-card:hover {
        border-color: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }

    .info-badge {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        border-radius: 4px 8px 8px 4px;
        font-size: 13.5px;
        color: #1E40AF;
        margin-bottom: 15px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Session State (상태 변수) 초기화
# ---------------------------------------------------------
if "u_val" not in st.session_state:
    st.session_state["u_val"] = 1.70  # 쿠로시오 기본 유속 u
if "v_val" not in st.session_state:
    st.session_state["v_val"] = 0.55  # 쿠로시오 기본 유속 v
if "selected_location" not in st.session_state:
    st.session_state["selected_location"] = "쿠로시오 해류 (Kuroshio)"

# ---------------------------------------------------------
# 3. 주요 해류 데이터베이스 및 추정 모델
# ---------------------------------------------------------
MAJOR_CURRENTS = [
    {"name": "쿠로시오 해류 (Kuroshio)", "lat": 29.5, "lon": 131.0, "u": 1.70, "v": 0.55},
    {"name": "북태평양 환류 / 쓰레기 지대 (GPGP)", "lat": 32.0, "lon": -145.0, "u": 0.12, "v": -0.05},
    {"name": "캘리포니아 해류 (California)", "lat": 36.0, "lon": -122.0, "u": -0.15, "v": -0.45},
    {"name": "북적도 해류 (North Equatorial)", "lat": 15.0, "lon": 140.0, "u": -0.65, "v": 0.08},
    {"name": "멕시코 만류 (Gulf Stream)", "lat": 35.0, "lon": -75.0, "u": 1.50, "v": 0.70},
    {"name": "남극 순환류 (ACC)", "lat": -55.0, "lon": 20.0, "u": 0.85, "v": -0.10},
    {"name": "한국 남해 연안류 (South Sea Korea)", "lat": 34.2, "lon": 128.5, "u": 0.45, "v": 0.15},
]

def get_ocean_current_info(lat, lon):
    if lon > 180: lon -= 360
    for c in MAJOR_CURRENTS:
        if np.hypot(lat - c["lat"], lon - c["lon"]) < 7.0:
            return c["name"], c["u"], c["v"]
            
    rad = np.radians(lat)
    u_est = round(0.50 * np.cos(3 * rad) + 0.10 * np.sin(rad), 2)
    v_est = round(0.10 * np.sin(2 * rad), 2)
    loc_str = f"임의 선택 해역 (위도 {lat:.2f}°, 경도 {lon:.2f}°)"
    return loc_str, u_est, v_est

# ---------------------------------------------------------
# 4. 유체역학 포집 효율 계산 공식 (물리 모델)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 5. 메인 대시보드 타이틀
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">🌊 해양 미세플라스틱 차단막 포집 수치 시뮬레이터</div>
    <div class="header-subtitle">해류 유속 및 차단막 기하학 구조(V-Angle)에 따른 입자 포집 유체역학 시뮬레이션 및 현장 설치 가이드</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. 상단 탭 구성
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📍 대화형 Folium 해지도 연동 & 시뮬레이션",
    "🎯 맞춤형 설치 조건 도출기 (정부/기업용)",
    "⚠️ 현장 설치 및 운영 주의사항"
])

# =========================================================
# TAB 1: 지도 연동 및 포집 시뮬레이션
# =========================================================
with tab1:
    col_map, col_input = st.columns([1.2, 0.8], gap="medium")

    # --- 왼쪽: Folium 지도 ---
    with col_map:
        st.markdown("##### 🗺️ 해역 선택 (마커 클릭 또는 위치 임의 지정)")
        m = folium.Map(location=[25, 125], zoom_start=2, tiles="OpenStreetMap")
        for c in MAJOR_CURRENTS:
            spd = np.hypot(c['u'], c['v'])
            popup_html = f"""
            <div style="font-family:'Pretendard',sans-serif; width:180px; font-size:12.5px; line-height:1.5;">
                <b style="font-size:13.5px; color:#0F172A;">{c['name']}</b><br>
                <span style="color:#2563EB; font-weight:600;">평균 유속: {spd:.2f} m/s</span>
            </div>
            """
            folium.Marker(
                [c["lat"], c["lon"]],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=c["name"],
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

        map_data = st_folium(m, height=440, use_container_width=True, key="ocean_folium_map")

        if map_data:
            clicked_lat, clicked_lon = None, None
            if map_data.get("last_object_clicked"):
                clicked_lat = map_data["last_object_clicked"]["lat"]
                clicked_lon = map_data["last_object_clicked"]["lng"]
            elif map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lon = map_data["last_clicked"]["lng"]

            if clicked_lat is not None and clicked_lon is not None:
                loc_name, auto_u, auto_v = get_ocean_current_info(clicked_lat, clicked_lon)
                if (abs(st.session_state["u_val"] - auto_u) > 1e-3 or 
                    abs(st.session_state["v_val"] - auto_v) > 1e-3 or 
                    st.session_state["selected_location"] != loc_name):
                    
                    st.session_state["u_val"] = auto_u
                    st.session_state["v_val"] = auto_v
                    st.session_state["selected_location"] = loc_name
                    st.rerun()

    # --- 오른쪽: 입력 제어 및 실시간 분석 카드 ---
    with col_input:
        st.markdown("##### ⚙️ 해역 파라미터 및 대상 플라스틱 설정")
        st.markdown(f'''
        <div class="info-badge">
            📍 <b>현재 선택된 위치</b>: {st.session_state['selected_location']}
        </div>
        ''', unsafe_allow_html=True)

        i_col1, i_col2 = st.columns(2)
        with i_col1:
            u_in = st.number_input("동서 유속 u (m/s)", value=float(st.session_state["u_val"]), step=0.05, format="%.2f")
        with i_col2:
            v_in = st.number_input("남북 유속 v (m/s)", value=float(st.session_state["v_val"]), step=0.05, format="%.2f")

        st.session_state["u_val"] = u_in
        st.session_state["v_val"] = v_in

        plastic_options = {
            "PP (폴리프로필렌 - 900 kg/m³)": 900.0,
            "PE (폴리에틸렌 - 920 kg/m³)": 920.0,
            "PS (폴리스티렌 - 1040 kg/m³)": 1040.0,
            "PET (페트 - 1380 kg/m³)": 1380.0
        }
        
        plastic_selected = st.selectbox(
            "포집 대상 미세플라스틱 재질", 
            list(plastic_options.keys()), 
            index=0
        )

        net_speed = np.hypot(u_in, v_in)
        st.session_state["net_speed"] = net_speed
        
        plastic_density = plastic_options[plastic_selected]
        angles, eff_curve, opt_angle, opt_eff = calculate_physics_efficiency(net_speed, plastic_density)

        st.markdown(f'''
        <div class="metric-container">
            <div class="stat-card">
                <div class="metric-label">권장 최적 V-각도</div>
                <div class="metric-value">{opt_angle:.0f}°</div>
            </div>
            <div class="stat-card">
                <div class="metric-label">최대 포집 효율</div>
                <div class="metric-value">{opt_eff:.1f} %</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.button("🚀 해석 파라미터 갱신", use_container_width=True, type="primary")

    # --- 하단 시각화 및 입자 추적 시뮬레이션 ---
    st.markdown("---")
    st.subheader("📊 유체역학 시각화 및 입자 동적 추적")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "📈 2D 단면 최적화 곡선", 
        "🌐 3D 유속-각도-효율 곡면",
        "🎬 실시간 포집 동적 추적 (Particle Flow)"
    ])

    with chart_tab1:
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(
            x=angles, y=eff_curve,
            mode='lines', name='포집 효율 (%)',
            line=dict(color='#2563EB', width=3)
        ))
        fig_2d.add_trace(go.Scatter(
            x=[opt_angle], y=[opt_eff],
            mode='markers+text', name='최적 작동점',
            marker=dict(size=12, color='#EF4444', symbol='diamond'),
            text=[f"최적: {opt_angle:.0f}° ({opt_eff:.1f}%)"],
            textposition="top center"
        ))
        fig_2d.update_layout(
            xaxis_title="V-차단막 각도 (°)",
            yaxis_title="포집 효율 (%)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    with chart_tab2:
        ang_grid = np.linspace(10, 90, 35)
        spd_grid = np.linspace(0.1, 2.5, 35)
        A, S = np.meshgrid(ang_grid, spd_grid)

        Z = np.zeros_like(A)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                _, _, _, eff_val = calculate_physics_efficiency(S[i, j], plastic_density)
                Z[i, j] = eff_val

        fig_3d = go.Figure(data=[go.Surface(x=A, y=S, z=Z, colorscale="Viridis", opacity=0.9)])
        fig_3d.add_trace(go.Scatter3d(
            x=[opt_angle], y=[net_speed], z=[opt_eff],
            mode='markers+text',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='현재 선택 지점',
            text=[f"현재: {opt_angle:.0f}°"],
            textposition="top center"
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="V-각도 (°)",
                yaxis_title="유속 (m/s)",
                zaxis_title="포집 효율 (%)"
            ),
            height=450,
            margin=dict(l=10, r=10, b=10, t=10)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with chart_tab3:
        st.markdown("##### 🌊 해류 유입 방향(아래 → 위) 및 V-차단막 입자 행동 시뮬레이션")
        st.caption("▶️ Play 버튼을 클릭하면 아래에서 위(+Y방향)로 흘러드는 해류 속 미세플라스틱 입자가 V자 차단막으로 유입/슬라이딩/유실되는 움직임을 관찰할 수 있습니다.")

        # 올바른 차단막 기하학 (해류 진행: 아래 -> 위)
        half_rad = np.radians(opt_angle / 2.0)
        boom_len = 10.0
        rx = boom_len * np.sin(half_rad)      
        apex_y = boom_len * np.cos(half_rad)  

        boom_x = [-rx, 0, rx]
        boom_y = [0, apex_y, 0]

        n_particles = 40
        n_frames = 20
        np.random.seed(42)
        
        init_x = np.linspace(-12, 12, n_particles) + np.random.uniform(-0.2, 0.2, n_particles)
        init_y = np.full(n_particles, -12.0)

        frames_data = []
        t_steps = np.linspace(0, 1, n_frames)

        for t in t_steps:
            fx, fy, colors, sizes = [], [], [], []
            
            for i in range(n_particles):
                x0 = init_x[i]
                y0 = init_y[i]
                
                dist = net_speed * t * 18.0
                y_curr = y0 + dist
                x_curr = x0

                if abs(x0) <= rx and rx > 0:
                    barrier_y = apex_y * (1.0 - abs(x0) / rx)
                    
                    if y_curr >= barrier_y:
                        if opt_angle <= 48:
                            slide = max(0.0, 1.0 - (y_curr - barrier_y) * 0.2)
                            x_curr = x0 * slide
                            y_curr = min(apex_y, barrier_y + (apex_y - barrier_y) * (1.0 - slide))
                            colors.append("#10B981") # 초록색: 포집 성공
                            sizes.append(9)
                        else:
                            x_curr = x0 + np.random.uniform(-0.6, 0.6)
                            y_curr = barrier_y + 1.2
                            colors.append("#EF4444") # 빨간색: 유실
                            sizes.append(7)
                    else:
                        colors.append("#38BDF8") # 시안색: 유입 중
                        sizes.append(7)
                else:
                    colors.append("#64748B") # 회색: 범위 밖
                    sizes.append(5)

                fx.append(x_curr)
                fy.append(y_curr)
                
            frames_data.append((fx, fy, colors, sizes))

        # 안전한 Plotly Trace 기반 애니메이션 구현 (traces=[0] 갱신 방식)
        fig_anim = go.Figure(
            data=[
                go.Scatter(x=frames_data[0][0], y=frames_data[0][1], mode="markers",
                           marker=dict(size=frames_data[0][3], color=frames_data[0][2]), name="미세플라스틱 입자"),
                go.Scatter(x=boom_x, y=boom_y, mode="lines+markers", name="V-차단막 (Boom)",
                           line=dict(color="#F59E0B", width=5), marker=dict(size=6, color="#FBBF24")),
                go.Scatter(x=[0], y=[apex_y], mode="markers+text", name="포집 정점 (Apex Net)",
                           marker=dict(size=15, color="#10B981", symbol="star"),
                           text=["🎯 Apex Net"], textposition="top center", textfont=dict(color="#10B981", size=12))
            ],
            layout=go.Layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#020617",
                xaxis=dict(range=[-15, 15], title="수평 거리 (m)", zeroline=False, gridcolor="#1E293B", font=dict(color="#94A3B8")),
                yaxis=dict(range=[-13, 13], title="해류 진행 방향 ↑ (m)", zeroline=False, gridcolor="#1E293B", font=dict(color="#94A3B8")),
                height=500,
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(font=dict(color="#E2E8F0"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    bgcolor="#1E293B",
                    font=dict(color="#F8FAFC"),
                    x=0.01, y=0.98,
                    buttons=[dict(label="▶️ 포집 시뮬레이션 재생", method="animate",
                                 args=[None, {"frame": {"duration": 90, "redraw": False}, "fromcurrent": True}])]
                )]
            ),
            frames=[
                go.Frame(
                    data=[go.Scatter(x=fd[0], y=fd[1], mode="markers", marker=dict(size=fd[3], color=fd[2]))],
                    traces=[0]
                ) for fd in frames_data
            ]
        )

        st.plotly_chart(fig_anim, use_container_width=True)

# =========================================================
# TAB 2: 맞춤형 설치 조건 도출기 (정부/기업용)
# =========================================================
with tab2:
    st.subheader("🎯 맞춤형 설치 조건 및 경제성/포집성 산출 도출기")
    st.write("설치 대상 해역의 예산 및 차단막 규격을 지정하여 최적의 설계 견적을 산출합니다.")

    t2_col1, t2_col2 = st.columns(2, gap="large")
    with t2_col1:
        target_budget = st.slider("사업 수용 가능 예산 (백만원)", 10, 500, 120, 10)
        boom_length = st.slider("차단막(Boom) 총 설치 길이 (m)", 50, 1000, 200, 25)
        deployment_months = st.number_input("운영 기간 (개월)", min_value=1, max_value=60, value=12)

    with t2_col2:
        wave_height = st.selectbox("해당 해역 평균 유의파고", ["0.5m 이하 (정온 해역)", "0.5m ~ 1.5m (보통 해역)", "1.5m 이상 (거친 해역)"])
        sea_bottom = st.selectbox("해저 지질 형태", ["사질 (모래)", "펄 (점토)", "암반 (바위)"])
        maintenance_freq = st.radio("유지보수 / 쓰레기 수거 주기", ["주 1회", "격주 1회", "월 1회"], horizontal=True)

    curr_net_speed = st.session_state.get("net_speed", 1.78)
    estimated_cost = boom_length * 0.42 + (deployment_months * 1.5)
    monthly_retrieval_ton = (boom_length * 0.18) * (curr_net_speed * 1.2)
    
    st.markdown("---")
    st.markdown("##### 📋 엔지니어링 설계를 위한 산출 요약")
    
    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.metric("예상 총 투입 비용", f"{estimated_cost:.1f} 백만원")
    r_col2.metric("월간 예상 미세플라스틱 포집량", f"{monthly_retrieval_ton:.1f} 톤/월")
    r_col3.metric("앵커링 소요 인장 하중", f"{boom_length * curr_net_speed * 0.85:.1f} kN")

    st.success("✅ **추천 앵커 타입**: " + ("플루크 앵커 (Fluke Anchor)" if sea_bottom=="사질 (모래)" else "중력식 콘크리트 앙카"))

# =========================================================
# TAB 3: 현장 설치 및 운영 주의사항
# =========================================================
with tab3:
    st.subheader("⚠️ 현장 설치 및 해양 운영 안전 주의사항")
    
    with st.expander("1. 🐋 해양 생물 혼획(Bycatch) 방지 조치", expanded=True):
        st.markdown("""
        - 차단막 하부 스커트(Skirt) 망목(Mesh) 크기를 최소 2mm 이하로 유지하여 치어 얽힘 방지.
        - 음향 퇴치 장치(Pinger)를 50m 간격으로 부착하여 어류 및 해양 포유류의 접근 예방.
        """)
        
    with st.expander("2. 🌀 태풍 및 기상 악화 시 안전 비상 프로토콜"):
        st.markdown("""
        - 유의파고 2.5m 이상 발효 시 **자동 해제(Quick Release) 핀** 작동 준비.
        - 파랑 하중으로 인한 펜스 파손을 막기 위해 차단막 유효 각도를 일시적으로 15° 이하로 조류 방향과 평행하게 접음.
        """)
        
    with st.expander("3. 🦠 생물 부착(Biofouling) 관리 및 주기적 세척"):
        st.markdown("""
        - 수온 20°C 이상 환경에서는 2주 내 차단막 표면 따개비 부착으로 중량 35% 증가.
        - 무독성 친환경 방오 코팅(Silicone-based anti-fouling) 필름 적용 권장.
        """)
        
    with st.expander("4. 🚢 해상 항법 안전 및 항로 표지 규정"):
        st.markdown("""
        - 해양경찰 및 항만청 신고 필수 (특수표지 부표 양쪽 끝단 설치).
        - 야간 자발광 LED 황색 등화(4초 1섬광) 및 AIS 해상 위치 발신기 설치.
        """)
