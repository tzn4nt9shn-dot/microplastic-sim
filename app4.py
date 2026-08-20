import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import time

st.set_page_config(page_title="해양 미세플라스틱 포집 최적화 솔루션", page_icon="🌊", layout="wide")

st.title("🌊 GIS 기반 해양 미세플라스틱 포집 최적화 솔루션 (v4.0)")
st.markdown("대화형 해지도 상호작용, 유속 조건별 최적 V자 펜스 수치 해석 및 리스크 방지 모듈이 연동된 통합 의사결정 지원 시스템입니다.")

# ==========================================
# 0. 데이터베이스 정의
# ==========================================
REGIONS_DB = {
    "한국 남해안 (대한해협 연안류)": {
        "lat": 34.5, "lon": 128.75, "u_x": 0.50, "u_y": 0.10, 
        "range": "0.30~0.80 m/s", "depth_avg": "80m", "desc": "연안 어업 및 항로 밀집 구역"
    },
    "쿠로시오 해류 (Kuroshio)": {
        "lat": 29.5, "lon": 131.0, "u_x": 1.80, "u_y": 0.35, 
        "range": "1.00~2.50 m/s", "depth_avg": "1,000m+", "desc": "강한 강풍과 고유속 난류 구역"
    },
    "태평양 쓰레기 지대 (GPGP)": {
        "lat": 30.0, "lon": -145.0, "u_x": 0.15, "u_y": -0.05, 
        "range": "0.05~0.20 m/s", "depth_avg": "4,000m+", "desc": "환류 체류 구역 (미세플라스틱 고농도)"
    },
    "캘리포니아 해류 (California)": {
        "lat": 37.5, "lon": -120.0, "u_x": 0.25, "u_y": -0.10, 
        "range": "0.10~0.40 m/s", "depth_avg": "500m", "desc": "북태평양 침적 연안 구역"
    }
}

PLASTIC_SPECS = {
    "PP (폴리프로필렌)": {"density": 885.0, "sinks": False, "desc": "표층 부유성 / 밀도 0.885 g/cm³"},
    "LDPE (저밀도 폴리에틸렌)": {"density": 910.0, "sinks": False, "desc": "표층 부유성 / 밀도 0.910 g/cm³"},
    "HDPE (고밀도 폴리에틸렌)": {"density": 960.0, "sinks": False, "desc": "약부유성 / 밀도 0.960 g/cm³"},
    "PET (페트)": {"density": 1395.0, "sinks": True, "desc": "침강성 (표층 펜스 불가) / 밀도 1.395 g/cm³"}
}

# ==========================================
# 1. 물리 연산 및 수치해석 엔진
# ==========================================
def calculate_optimal_setup(current_speed, plastic_density):
    RHO_WATER = 1025.0
    angles = list(range(10, 95, 5))
    efficiencies = []
    
    if plastic_density > RHO_WATER:
        return 0, 0.0, "⚠️ PET는 침강성 물질로 표층 펜스 포집이 불가능합니다.", angles, [0.0]*len(angles)

    reserve_buoyancy = (RHO_WATER - plastic_density) / RHO_WATER
    submergence_eff = np.clip(reserve_buoyancy / 0.1366, 0.3, 1.0)

    best_angle = 30
    max_score = -1.0
    
    for cand_angle in angles:
        rad = np.radians(cand_angle / 2.0)
        slide_eff = np.cos(rad)
        vortex_penalty = max(0.0, (current_speed - 0.2) * np.sin(rad) * 0.4)
        area_weight = 0.7 + 0.3 * np.sin(rad)
        
        score = ((100.0 * submergence_eff * slide_eff) - (vortex_penalty * 25.0)) * area_weight
        eff_val = min(98.0, max(10.0, score))
        efficiencies.append(round(eff_val, 1))
        
        if score > max_score:
            max_score = score
            best_angle = cand_angle

    pred_rate = min(98.0, max(15.0, max_score))
    advice = f"유속({current_speed:.2f}m/s) 및 부력 수치 기준, 와류 유실 방지와 포집 면적을 동시에 충족하는 최적 각도는 {best_angle}°입니다."
    return best_angle, round(pred_rate, 1), advice, angles, efficiencies

def run_simulation(u_x, u_y, angle_deg, spec, n_particles=100):
    RHO_WATER = 1025.0
    if spec["sinks"]:
        return None, 0.0, True

    reserve_buoyancy = (RHO_WATER - spec["density"]) / RHO_WATER
    submergence_factor = np.clip(reserve_buoyancy / 0.1366, 0.3, 1.0)
    rad = np.radians(angle_deg / 2.0)
    angle_efficiency = np.cos(rad)

    abs_ux = max(0.05, abs(u_x))
    total_time_needed = (10.0 / abs_ux) + (8.0 / abs_ux) + 10.0
    dt = 0.15
    max_steps = int(total_time_needed / dt)
    
    pos = np.zeros((n_particles, 2))
    pos[:, 0] = 0.0
    pos[:, 1] = np.linspace(-3.5, 3.5, n_particles)
    
    escape_prob = 1.0 - (submergence_factor * angle_efficiency)
    is_underflow_escaped = np.random.rand(n_particles) < escape_prob
    
    apex_x, apex_y = 10.0, 0.0
    theta = np.radians(angle_deg / 2.0)
    t_upper = np.array([np.cos(theta), -np.sin(theta)])
    t_lower = np.array([np.cos(theta), np.sin(theta)])
    
    trajectories = [pos.copy()]
    captured_mask = np.zeros(n_particles, dtype=bool)
    escaped_mask = np.zeros(n_particles, dtype=bool)
    u_ocean = np.array([u_x, u_y])
    
    for step in range(max_steps):
        current_pos = pos.copy()
        for i in range(n_particles):
            if captured_mask[i] or escaped_mask[i]:
                continue
            
            px, py = current_pos[i]
            if np.hypot(px - apex_x, py - apex_y) < 0.6:
                captured_mask[i] = True
                continue
            
            is_above = (py > 0)
            wall_x_threshold = apex_x - (abs(py - apex_y)) * np.tan(np.pi/2 - theta)
            
            if px >= wall_x_threshold and px <= apex_x + 1.0:
                if is_underflow_escaped[i]:
                    v_eff = u_ocean
                else:
                    t_vec = t_upper if is_above else t_lower
                    v_t = np.dot(u_ocean, t_vec)
                    v_eff = max(0.05, v_t) * t_vec
            else:
                v_eff = u_ocean
                
            pos[i] += v_eff * dt
            if pos[i, 0] > 15.0 or abs(pos[i, 1]) > 8.0:
                escaped_mask[i] = True

        trajectories.append(pos.copy())
        if np.all(captured_mask | escaped_mask):
            break
            
    trajectories = np.array(trajectories)
    capture_rate = (np.sum(captured_mask) / n_particles) * 100.0
    return trajectories, capture_rate, False

# ==========================================
# 2. 탭 구성
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "🗺️ 대화형 Folium 해지도 연동 & 분석", 
    "🎯 맞춤형 설치 조건 도출기 (정부/기업용)", 
    "⚠️ 현장 설치 및 운영 주의사항"
])

# ------------------------------------------
# TAB 1: 대화형 지도 클릭 & 시각화 컨텐츠
# ------------------------------------------
with tab1:
    st.subheader("📍 해 지도 상호작용: 빨간점 클릭 또는 해역 클릭")
    st.markdown("지도의 **빨간점(주요 해류)**을 클릭하거나 **해양 임의 위치**를 클릭하면 해당 해역의 유속 파라미터를 읽어와 **최적 조건 분석 그래프** 및 **파티클 동적 추적 시뮬레이션**을 실행합니다.")
    
    col_map, col_ctrl = st.columns([1.3, 1])
    
    # 세션 상태 초기화
    if "clicked_lat" not in st.session_state:
        st.session_state["clicked_lat"] = 34.5
        st.session_state["clicked_lon"] = 128.75
        st.session_state["selected_name"] = "한국 남해안 (대한해협 연안류)"
        st.session_state["custom_ux"] = 0.50
        st.session_state["custom_uy"] = 0.10

    with col_map:
        # Folium 지도 생성
        m = folium.Map(location=[20, 10], zoom_start=2, tiles="OpenStreetMap")
        
        # 주요 해류 마커 추가
        for region_name, info in REGIONS_DB.items():
            folium.CircleMarker(
                location=[info["lat"], info["lon"]],
                radius=9,
                popup=f"<b>{region_name}</b><br>유속: {info['u_x']} m/s",
                tooltip=f"클릭하여 선택: {region_name}",
                color="crimson",
                fill=True,
                fill_color="red",
                fill_opacity=0.8
            ).add_to(m)

        # 지도 렌더링 및 클릭 이벤트 수집
        map_output = st_folium(m, width="100%", height=420, key="interactive_ocean_map")
        
        # 클릭 이벤트 처리
        if map_output and map_output.get("last_clicked"):
            lat = map_output["last_clicked"]["lat"]
            lon = map_output["last_clicked"]["lng"]
            
            # 클릭 위치가 기존 마커 근처인지 판별
            matched_region = None
            for r_name, r_info in REGIONS_DB.items():
                if np.hypot(lat - r_info["lat"], lon - r_info["lon"]) < 3.0:
                    matched_region = r_name
                    break
            
            if matched_region:
                st.session_state["selected_name"] = matched_region
                st.session_state["clicked_lat"] = REGIONS_DB[matched_region]["lat"]
                st.session_state["clicked_lon"] = REGIONS_DB[matched_region]["lon"]
                st.session_state["custom_ux"] = REGIONS_DB[matched_region]["u_x"]
                st.session_state["custom_uy"] = REGIONS_DB[matched_region]["u_y"]
            else:
                st.session_state["selected_name"] = f"임의 선택 해역 (위도 {lat:.2f}°, 경도 {lon:.2f}°)"
                st.session_state["clicked_lat"] = round(lat, 2)
                st.session_state["clicked_lon"] = round(lon, 2)

    with col_ctrl:
        st.success(f"📍 **선택된 위치:** `{st.session_state['selected_name']}`")
        
        # 유속 설정 (임의 위치 클릭 시 조정 가능)
        if "임의 선택 해역" in st.session_state["selected_name"]:
            st.info("💡 클릭하신 좌표의 추정 유속을 입력하거나 세부 설정하세요.")
            ux_in = st.number_input("동서 방향 유속 u (m/s)", value=float(st.session_state["custom_ux"]), step=0.05)
            uy_in = st.number_input("남북 방향 유속 v (m/s)", value=float(st.session_state["custom_uy"]), step=0.05)
        else:
            r_info = REGIONS_DB[st.session_state["selected_name"]]
            ux_in, uy_in = r_info["u_x"], r_info["u_y"]
            st.write(f"• **대표 유속:** {r_info['range']}")
            st.write(f"• **평균 수심:** {r_info['depth_avg']} ({r_info['desc']})")
        
        target_p = st.selectbox("포집 대상 플라스틱 재질", list(PLASTIC_SPECS.keys()))
        p_spec = PLASTIC_SPECS[target_p]
        
        # 계산 실행
        curr_speed = np.hypot(ux_in, uy_in)
        opt_angle, pred_eff, reasoning, ang_list, eff_list = calculate_optimal_setup(curr_speed, p_spec["density"])
        
        st.divider()
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("권장 최적 V-각도", f"{opt_angle}°")
        m_col2.metric("예상 포집 효율", f"{pred_eff} %")
        
        run_sim_btn = st.button("🚀 선택 해역 시뮬레이션 및 분석 실행", use_container_width=True, type="primary")

    # ------------------------------------------
    # 클릭 결과 시각화 컨텐츠 (1 & 2)
    # ------------------------------------------
    st.divider()
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("### 📊 [시각화 1] 각도별 포집 효율 스위프 곡선")
        fig_curve, ax_c = plt.subplots(figsize=(6, 4))
        ax_c.plot(ang_list, eff_list, color='#1f77b4', lw=2.5, marker='o', ms=4, label='Capture Efficiency (%)')
        
        # 최적 각도 피크 강조
        if opt_angle in ang_list:
            opt_idx = ang_list.index(opt_angle)
            ax_c.plot(opt_angle, eff_list[opt_idx], 'ro', ms=10, label=f'Optimal Peak ({opt_angle}°)')
            ax_c.axvline(opt_angle, color='crimson', linestyle='--', alpha=0.7)
            
        ax_c.set_xlabel("Fence Angle (degrees)", fontsize=10)
        ax_c.set_ylabel("Expected Efficiency (%)", fontsize=10)
        ax_c.set_title(f"Angle vs Efficiency Trade-off Curve ({curr_speed:.2f} m/s)", fontsize=11)
        ax_c.grid(True, linestyle='--', alpha=0.5)
        ax_c.legend(loc='lower right')
        st.pyplot(fig_curve)
        plt.close(fig_curve)

    with v_col2:
        st.markdown("### 🌀 [시각화 2] 파티클 동적 추적 시뮬레이션")
        plot_spot = st.empty()
        
        if run_sim_btn:
            trajs, rate, is_sunk = run_simulation(ux_in, uy_in, opt_angle, p_spec)
            if is_sunk:
                st.error("⚠️ PET는 침강 물질로 표층 펜스 시뮬레이션이 불가능합니다.")
            else:
                total_steps = len(trajs)
                frame_stride = max(1, total_steps // 30)
                for f_idx in range(0, total_steps, frame_stride):
                    fig_sim, ax_s = plt.subplots(figsize=(6, 4))
                    apex_x, apex_y = 10.0, 0.0
                    rad = np.radians(opt_angle / 2.0)
                    wing_len = 6.0
                    
                    # 펜스 구조물
                    ax_s.plot([apex_x - wing_len*np.cos(rad), apex_x], [apex_y + wing_len*np.sin(rad), apex_y], 'k-', lw=3.5)
                    ax_s.plot([apex_x - wing_len*np.cos(rad), apex_x], [apex_y - wing_len*np.sin(rad), apex_y], 'k-', lw=3.5)
                    ax_s.plot(apex_x, apex_y, 'ro', ms=10)
                    
                    # 파티클 위치
                    curr_pos = trajs[f_idx]
                    ax_s.scatter(curr_pos[:, 0], curr_pos[:, 1], c='dodgerblue', alpha=0.7, s=18)
                    
                    ax_s.set_xlim(-1, 14)
                    ax_s.set_ylim(-6, 6)
                    ax_s.set_title(f"Dynamic Particle Flow (Angle: {opt_angle}°)")
                    ax_s.grid(True, linestyle='--', alpha=0.4)
                    
                    plot_spot.pyplot(fig_sim)
                    plt.close(fig_sim)
                    time.sleep(0.01)
        else:
            st.info("👆 오른쪽 위 '시뮬레이션 및 분석 실행' 버튼을 누르면 동적 입자 흐름이 모의 가동됩니다.")

# ------------------------------------------
# TAB 2: 맞춤형 설치 조건 도출기 (임의 수치)
# ------------------------------------------
with tab2:
    st.subheader("🛠️ 특정 해역 수치 직접 입력 기반 포집기 설계 도출")
    st.markdown("정부기관 또는 환경기업이 임의의 해역 유속 데이터를 입력하여 **최적 구조물 설계 파라미터**를 산출하는 계산 모듈입니다.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        custom_ux2 = st.number_input("동서 유속 u (m/s)", value=0.45, step=0.05, key="t2_ux")
        custom_uy2 = st.number_input("남북 유속 v (m/s)", value=0.15, step=0.05, key="t2_uy")
    with col_in2:
        custom_plastic2 = st.selectbox("주요 타겟 플라스틱", list(PLASTIC_SPECS.keys()), key="tab2_p2")
        custom_spec2 = PLASTIC_SPECS[custom_plastic2]
    with col_in3:
        fence_depth = st.slider("표층 펜스 유효 수심 (m)", 0.3, 1.5, 0.5, step=0.1)

    total_speed = np.hypot(custom_ux2, custom_uy2)
    b_ang, p_eff, reason, _, _ = calculate_optimal_setup(total_speed, custom_spec2["density"])
    
    st.divider()
    st.markdown("### 📋 정부·기업 제출용 최적 설계 가이드라인")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("입력 합성 유속", f"{total_speed:.2f} m/s")
    m2.metric("권장 최적 V자 각도", f"{b_ang}°")
    m3.metric("예상 포집 효율", f"{p_eff} %")
    m4.metric("추천 계류 닻 하중", f"{round(total_speed*1.6 + 2.0, 1)} 톤")

    st.info(f"**📌 과학적 도출 근거:** {reason}")
    
    st.markdown("**💡 현장 엔지니어가 설정 가능한 최적 운영 조건:**")
    compass_deg = int(np.degrees(np.arctan2(custom_uy2, custom_ux2)))
    st.write(f"1. **앵커 계류 나침반 방향:** 유속 입사 방향에 직각을 이루도록 V자 펜스 중앙축을 `{compass_deg}°` 방향으로 배치.")
    st.write(f"2. **포켓 수거 주기:** 유속 `{total_speed:.2f}m/s` 조건에서 와류 과부하 방지를 위해 **주 {int(max(1, 7 - total_speed*2))}회** 중앙 회수통 비움 필수.")
    st.write(f"3. **펜스 喫水(Draft) 설정:** HDPE 수집 비율이 높다면 펜스 수심을 `{fence_depth + 0.2:.1f}m`로 확장 권장.")

# ------------------------------------------
# TAB 3: 현장 설치 및 운영 주의사항
# ------------------------------------------
with tab3:
    st.subheader("⚠️ 해양 포집기 현장 설치 및 실전 운영 위험 관리 수칙")
    st.markdown("실제 바다 환경에 무동력 포집기를 장기 계류할 때 반드시 준수해야 하는 **환경·기술적 리스크 대응 수칙**입니다.")
    
    st.warning("**1. 해양 생물 포획 및 얽힘 (Bycatch & Entrapment Risk)**")
    st.write("- **위험성:** 어류, 바다거북 등 해양 생물이 V자 포켓 안으로 유입되어 탈출하지 못할 위험.")
    st.write("- **대응책:** 포켓 상부에 탈출용 망(Escape Hatch) 설치 및 수중 음파 퇴치기(Pingers) 부착 필수.")
    
    st.warning("**2. 태풍 및 고파도 시 구조물 유실 (Extreme Storm Overtopping)**")
    st.write("- **위험성:** 파고 2.0m 이상의 너울성 파도 발생 시 계류 로프 절단 및 펜스 유실 위험.")
    st.write("- **대응책:** 기상 악화 시 펜스가 바닥으로 접히거나 잠기는 **수중 침강식 발라스트(Submersible Ballast)** 연동.")
    
    st.warning("**3. 생물 부착(Biofouling)으로 인한 부력 저하**")
    st.write("- **위험성:** 따개비, 해조류 부착으로 펜스 중량이 증가하여 수중으로 침강, 미세플라스틱 유실 발생.")
    st.write("- **대응책:** 친환경 무독성 방오 도료(Silicone Antifouling Coating) 도포 및 월 1회 표면 세척 작업 필요.")
    
    st.warning("**4. 해상 교통 및 선박 충돌 (Navigation Hazards)**")
    st.write("- **위험성:** 야간 또는 안개 발생 시 어선 및 항해 선박과 포집 구조물의 충돌 위험.")
    st.write("- **대응책:** AIS(자동선박식별장치) 발신기, 태양광 LED 등부표 및 레이더 반사판(Radar Reflector) 의무 부착.")
