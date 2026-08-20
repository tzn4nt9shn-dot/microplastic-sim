import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time

st.set_page_config(page_title="해양 미세플라스틱 포집 최적화 솔루션", page_icon="🌊", layout="wide")

st.title("🌊 정부·기업 맞춤형 해양 미세플라스틱 포집 최적화 솔루션 (v3.0)")
st.markdown("GIS 해지도 연동, 유속별 최적 펜스 설계 도출 및 현장 설치·운영 리스크 검토 모듈이 통합된 결정 지원 시스템입니다.")

# ==========================================
# 0. 해역 관측 GIS 데이터베이스
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
    "PP (폴리프로필렌)": {"density": 885.0, "sinks": False, "range": "0.85~0.92 g/cm³"},
    "LDPE (저밀도 폴리에틸렌)": {"density": 910.0, "sinks": False, "range": "0.89~0.93 g/cm³"},
    "HDPE (고밀도 폴리에틸렌)": {"density": 960.0, "sinks": False, "range": "0.94~0.98 g/cm³"},
    "PET (페트)": {"density": 1395.0, "sinks": True, "range": "1.38~1.41 g/cm³"}
}

# ==========================================
# 1. 최적 각도 및 시뮬레이션 물리 연산 엔진
# ==========================================
def calculate_optimal_setup(current_speed, plastic_density):
    RHO_WATER = 1025.0
    if plastic_density > RHO_WATER:
        return 0, 0.0, "⚠️ PET는 침강성 물질로 표층 펜스 포집이 불가능합니다 (수중 저층망 필요)."

    reserve_buoyancy = (RHO_WATER - plastic_density) / RHO_WATER
    submergence_eff = np.clip(reserve_buoyancy / 0.1366, 0.3, 1.0)

    best_angle = 30
    max_score = -1.0
    
    for cand_angle in range(15, 95, 5):
        rad = np.radians(cand_angle / 2.0)
        slide_eff = np.cos(rad)
        vortex_penalty = max(0.0, (current_speed - 0.2) * np.sin(rad) * 0.4)
        area_weight = 0.7 + 0.3 * np.sin(rad)
        
        score = ((100.0 * submergence_eff * slide_eff) - (vortex_penalty * 25.0)) * area_weight
        if score > max_score:
            max_score = final_score = score
            best_angle = cand_angle

    pred_rate = min(98.0, max(15.0, (submergence_eff * np.cos(np.radians(best_angle/2.0)) - (current_speed*0.1)) * 100))
    advice = f"유속({current_speed}m/s)과 부력을 고려할 때 와류 유실을 줄이고 포집 면적을 확보하는 최적 각도는 {best_angle}°입니다."
    return best_angle, round(pred_rate, 1), advice

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
    dt = 0.1
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
    "🗺️ GIS 지도 연동 & 주요 해역 시뮬레이션", 
    "🎯 맞춤형 설치 조건 도출기 (정부/기업용)", 
    "⚠️ 현장 설치 및 운영 주의사항"
])

# ------------------------------------------
# TAB 1: GIS 지도 상호작용 & 주요 해역 시뮬레이션
# ------------------------------------------
with tab1:
    st.subheader("📍 관측 해역 GIS 지도 선택 및 실시간 동적 추적")
    
    # Plotly 해양 지도 시각화
    map_df = pd.DataFrame([
        {"해역명": k, "lat": v["lat"], "lon": v["lon"], "대표유속": f"{v['u_x']} m/s", "설명": v["desc"]}
        for k, v in REGIONS_DB.items()
    ])
    
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="해역명", hover_data=["대표유속", "설명"],
        color_discrete_sequence=["crimson"], size_max=15, zoom=1, height=350
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    col_sel, col_sim = st.columns([1, 2])
    
    with col_sel:
        selected_region_name = st.selectbox("📌 지도상 모니터링 해역 선택", list(REGIONS_DB.keys()))
        reg_info = REGIONS_DB[selected_region_name]
        
        st.info(f"**좌표:** 위도 {reg_info['lat']}°, 경도 {reg_info['lon']}°\n\n"
                f"**평균 유속:** {reg_info['range']} (대표: {reg_info['u_x']} m/s)\n\n"
                f"**수심 및 특징:** {reg_info['depth_avg']} | {reg_info['desc']}")
        
        sel_plastic = st.selectbox("포집 대상 재질 선택", list(PLASTIC_SPECS.keys()), key="tab1_p")
        sel_spec = PLASTIC_SPECS[sel_plastic]
        
        # 최적 각도 자동 도출
        spd = np.hypot(reg_info["u_x"], reg_info["u_y"])
        opt_ang, pred_r, adv = calculate_optimal_setup(spd, sel_spec["density"])
        
        st.success(f"💡 **추천 최적 V-각도:** `{opt_ang}°`\n\n**예상 포집 효율:** `{pred_r}%`")
        
        run_sim_btn = st.button("🚀 선택 해역 시뮬레이션 가동", use_container_width=True)

    with col_sim:
        plot_spot = st.empty()
        if run_sim_btn:
            trajs, rate, is_sunk = run_simulation(reg_info["u_x"], reg_info["u_y"], opt_ang, sel_spec)
            if is_sunk:
                st.error("⚠️ PET는 침강 물질로 표층 펜스로 포집할 수 없습니다.")
            else:
                total_steps = len(trajs)
                frame_stride = max(1, total_steps // 35)
                for f_idx in range(0, total_steps, frame_stride):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    apex_x, apex_y = 10.0, 0.0
                    rad = np.radians(opt_ang / 2.0)
                    wing_len = 6.0
                    
                    ax.plot([apex_x - wing_len*np.cos(rad), apex_x], [apex_y + wing_len*np.sin(rad), apex_y], 'k-', lw=3.5)
                    ax.plot([apex_x - wing_len*np.cos(rad), apex_x], [apex_y - wing_len*np.sin(rad), apex_y], 'k-', lw=3.5)
                    ax.plot(apex_x, apex_y, 'ro', ms=12)
                    
                    curr_pos = trajs[f_idx]
                    ax.scatter(curr_pos[:, 0], curr_pos[:, 1], c='dodgerblue', alpha=0.7, s=20)
                    
                    ax.set_xlim(-1, 14)
                    ax.set_ylim(-6, 6)
                    ax.set_title(f"Dynamic Tracking: {selected_region_name} (Angle: {opt_ang}°)")
                    ax.grid(True, linestyle='--', alpha=0.4)
                    
                    plot_spot.pyplot(fig)
                    plt.close(fig)
                    time.sleep(0.02)

# ------------------------------------------
# TAB 2: 맞춤형 설치 조건 도출기 (임의 해류 입력)
# ------------------------------------------
with tab2:
    st.subheader("🛠️ 특정 해역 유속 수치 입력 기반 포집기 설계 도출")
    st.markdown("정부기관 또는 환경기업이 임의의 해역 유속 데이터를 입력하여 **최적의 구조물 설계 파라미터**를 시뮬레이션하는 도구입니다.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        custom_ux = st.number_input("동서 유속 u (m/s)", value=0.40, step=0.05)
        custom_uy = st.number_input("남북 유속 v (m/s)", value=0.10, step=0.05)
    with col_in2:
        custom_plastic = st.selectbox("주요 수집 타겟 플라스틱", list(PLASTIC_SPECS.keys()), key="tab2_p")
        custom_spec = PLASTIC_SPECS[custom_plastic]
    with col_in3:
        fence_depth = st.slider("표층 펜스 유효 수심 (m)", 0.3, 1.5, 0.5, step=0.1)
        target_capacity = st.number_input("목표 일일 처리 유량 (천 톤)", value=50)

    total_speed = np.hypot(custom_ux, custom_uy)
    best_ang, pred_eff, reasoning = calculate_optimal_setup(total_speed, custom_spec["density"])
    
    st.divider()
    st.markdown("### 📋 기관 제안용 최적 설계 가이드라인")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("입력 합성 유속", f"{total_speed:.2f} m/s")
    m2.metric("권장 최적 V자 각도 (θ)", f"{best_ang}°")
    m3.metric("예상 포집 효율", f"{pred_eff} %")
    m4.metric("추천 계류 닻 하중", f"{round(total_speed*1.5 + 2.0, 1)} 톤")

    st.info(f"**📌 과학적 도출 근거:** {reasoning}")
    
    st.markdown("**💡 인간이 제어/조절 가능한 최적 운영 조건:**")
    st.write(f"1. **앵커 계류 각도:** 유속 방향과 V자 펜스 중심축 축선이 일치하도록 `{int(np.degrees(np.arctan2(custom_uy, custom_ux)))}°` 방향으로 나침반 배치가 필요합니다.")
    st.write(f"2. **포켓 수거 주기:** 유속 `{total_speed:.2f}m/s` 기준 포켓 적재량 과부하 방지를 위해 **주 {int(max(1, 7 - total_speed*2))}회** 중앙 회수함을 비워야 합니다.")
    st.write(f"3. **펜스 喫水(Draft) 조절:** HDPE 수집 비율이 높을 경우 펜스 유효 수심을 `{fence_depth + 0.2:.1f}m`로 확장 권장합니다.")

# ------------------------------------------
# TAB 3: 현장 설치 및 운영 주의사항
# ------------------------------------------
with tab3:
    st.subheader("⚠️ 해양 포집기 현장 설치 및 실전 운영 위험 관리 수칙")
    st.markdown("실제 바다 환경에 무동력 포집기를 수개월 이상 계류할 때 반드시 고려해야 하는 **환경·기술적 리스크 대응 가이드**입니다.")
    
    st.warning("**1. 해양 생물 포획 및 얽힘 (Bycatch & Entrapment Risk)**")
    st.write("- **위험성:** 어류, 어패류, 바다거북 등 해양 생물이 V자 포켓 안으로 유입되어 탈출하지 못할 위험.")
    st.write("- **대응책:** 포켓 상부에 탈출용 망(Escape Hatch) 설치 및 수중 음파 퇴치기(Pingers) 부착 필수.")
    
    st.warning("**2. 태풍 및 태풍급 고파도 시 구조물 파손 (Extreme Storm Overtopping)**")
    st.write("- **위험성:** 파고 $2.0\text{ m}$ 이상의 너울성 파도 발생 시 계류 로프 절단 및 펜스 유실 위험.")
    st.write("- **대응책:** 기상 악화 경보 시 펜스가 바닥으로 접히거나 자동으로 잠기는 **수중 침강식 발라스트(Submersible Ballast)** 연동.")
    
    st.warning("**3. 생물 부착(Biofouling)으로 인한 부력 저하**")
    st.write("- **위험성:** 따개비, 해조류 부착으로 펜스 중량이 증가하여 펜스가 수중으로 침강, 미세플라스틱 유실 발생.")
    st.write("- **대응책:** 친환경 무독성 방오 도료(Silicone Antifouling Coating) 도포 및 월 1회 표면 세척 작업 필요.")
    
    st.warning("**4. 해상 교통 및 선박 충돌 (Navigation Hazards)**")
    st.write("- **위험성:** 야간 또는 안개 발생 시 어선 및 항해 선박과 포집 구조물의 충돌 위험.")
    st.write("- **대응책:** AIS(자동선박식별장치) 발신기, 태양광 LED 등부표 및 레이더 반사판(Radar Reflector) 의무 부착.")