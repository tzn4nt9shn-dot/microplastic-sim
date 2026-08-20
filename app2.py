import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ==========================================
# 0. 웹 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="해류 이용 미세플라스틱 포집 시뮬레이터",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 GIS 기반 무동력 미세플라스틱 포집 시뮬레이터 (v2.1)")
st.markdown("유속에 따른 적응형 시뮬레이션 타임스텝 및 실시간 포집 애니메이션 엔진이 적용되었습니다.")

# ==========================================
# 1. 플라스틱 물성 세분화 (밀도 범위 반영)
# ==========================================
PLASTIC_SPECS = {
    "PP (폴리프로필렌)": {"density": 885.0, "sinks": False, "range": "0.85~0.92 g/cm³"},
    "LDPE (저밀도 폴리에틸렌)": {"density": 910.0, "sinks": False, "range": "0.89~0.93 g/cm³"},
    "HDPE (고밀도 폴리에틸렌)": {"density": 960.0, "sinks": False, "range": "0.94~0.98 g/cm³"},
    "PET (페트)": {"density": 1395.0, "sinks": True, "range": "1.38~1.41 g/cm³"}
}

# ==========================================
# 2. 사이드바 컨트롤러
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 변수 설정")

region = st.sidebar.selectbox("대상 해역 선택 (GIS)", ["GPGP (완만)", "남해안 (연안)", "크로시오 (난류)", "직접 입력"])
if region == "GPGP (완만)":
    u_x, u_y = 0.15, -0.05
elif region == "남해안 (연안)":
    u_x, u_y = 0.25, 0.10
elif region == "크로시오 (난류)":
    u_x, u_y = 0.75, 0.35
else:
    u_x = st.sidebar.slider("동서 유속 u (m/s)", 0.05, 1.0, 0.3)
    u_y = st.sidebar.slider("남북 유속 v (m/s)", -0.5, 0.5, 0.0)

fence_angle_deg = st.sidebar.selectbox("V자 펜스 각도 (θ)", [30, 45, 60, 90])
particle_size_mm = st.sidebar.selectbox("입자 크기 (d)", [0.5, 5.0])

plastic_choice = st.sidebar.selectbox("플라스틱 세부 재질", list(PLASTIC_SPECS.keys()))
selected_spec = PLASTIC_SPECS[plastic_choice]
st.sidebar.info(f"선택 재질 밀도 범위: {selected_spec['range']}")

animate_speed = st.sidebar.slider("애니메이션 재생 속도 (초)", 0.01, 0.1, 0.03)
num_particles = st.sidebar.slider("입자 개수", 50, 300, 100, step=50)

# ==========================================
# 3. 고도화된 적응형 유체 및 슬라이딩 엔진
# ==========================================
def run_simulation_engine(u_x, u_y, angle_deg, spec, size_mm, n_particles):
    if spec["sinks"]:
        return None, 0.0, 0, n_particles, True

    # 유속에 비례하여 필요한 총 연산 시간을 자동 산출 (적응형 타임스텝)
    abs_ux = max(0.05, abs(u_x))
    total_time_needed = (10.0 / abs_ux) + (8.0 / abs_ux) + 10.0  # 도달 시간 + 슬라이딩 시간
    
    dt = 0.1  # 0.1초 단위 안정 연산
    max_steps = int(total_time_needed / dt)
    
    pos = np.zeros((n_particles, 2))
    pos[:, 0] = 0.0
    pos[:, 1] = np.linspace(-3.5, 3.5, n_particles)
    
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
            
            # 중앙 포켓 포집 판정 (0.6m 이내 접근)
            if np.hypot(px - apex_x, py - apex_y) < 0.6:
                captured_mask[i] = True
                continue
            
            # V자 펜스 벽면 충돌 및 미끄러짐 유속 적용
            is_above = (py > 0)
            wall_x_threshold = apex_x - (abs(py - apex_y)) * np.tan(np.pi/2 - theta)
            
            if px >= wall_x_threshold and px <= apex_x + 1.0:
                t_vec = t_upper if is_above else t_lower
                v_t = np.dot(u_ocean, t_vec)
                v_eff = max(0.05, v_t) * t_vec  # 벽을 따라 포켓으로 유도
            else:
                v_eff = u_ocean
                
            pos[i] += v_eff * dt
            
            # 이탈 판정
            if pos[i, 0] > 15.0 or abs(pos[i, 1]) > 8.0:
                escaped_mask[i] = True

        trajectories.append(pos.copy())
        if np.all(captured_mask | escaped_mask):
            break
            
    trajectories = np.array(trajectories)
    capture_rate = (np.sum(captured_mask) / n_particles) * 100.0
    return trajectories, capture_rate, np.sum(captured_mask), n_particles, False

# ==========================================
# 4. 웹 UI 및 실시간 시각화
# ==========================================
tab1, tab2 = st.tabs(["🎥 실시간 동적 포집 시뮬레이션", "📊 36개 시나리오 통합 비교"])

with tab1:
    col_ctrl, col_vis = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("실행 변수 검토")
        st.write(f"- **해역 유속:** u={u_x}m/s, v={u_y}m/s")
        st.write(f"- **펜스 각도:** {fence_angle_deg}°")
        st.write(f"- **선택 재질:** {plastic_choice}")
        
        run_btn = st.button("🚀 시뮬레이션 및 애니메이션 시작", use_container_width=True)

    with col_vis:
        st.subheader("📍 입자 이동 및 펜스 충돌 실시간 시각화")
        plot_spot = st.empty()

        if run_btn:
            trajs, rate, cap_cnt, total_cnt, is_sunk = run_simulation_engine(
                u_x, u_y, fence_angle_deg, selected_spec, particle_size_mm, num_particles
            )
            
            if is_sunk:
                st.error("⚠️ [PET 침강 경고] PET는 밀도(1.38~1.41 g/cm³)가 해수보다 높아 바닥으로 침강합니다.")
                st.warning("설계도 기준 유효 높이(h=0.3~0.5m)의 표층 V자 펜스로는 포집할 수 없습니다. (포집 효율: 0.0%)")
            else:
                total_steps = len(trajs)
                frame_stride = max(1, total_steps // 40)  # 속도에 관계없이 약 40프레임으로 시각화 균일화
                
                for f_idx in range(0, total_steps, frame_stride):
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    
                    apex_x, apex_y = 10.0, 0.0
                    rad = np.radians(fence_angle_deg / 2.0)
                    wing_len = 6.0
                    
                    # 펜스 그리기
                    ax.plot([apex_x - wing_len*np.cos(rad), apex_x], [apex_y + wing_len*np.sin(rad), apex_y], 'k-', lw=3.5, label="V-Fence")
                    ax.plot([apex_x - wing_len*np.cos(rad), apex_x], [apex_y - wing_len*np.sin(rad), apex_y], 'k-', lw=3.5)
                    ax.plot(apex_x, apex_y, 'ro', ms=12, label="Pocket")
                    
                    # 현재 프레임 입자 그리기
                    curr_pos = trajs[f_idx]
                    ax.scatter(curr_pos[:, 0], curr_pos[:, 1], c='dodgerblue', alpha=0.7, s=25, label="Microplastics")
                    
                    # 유속 화살표
                    ax.quiver(1, 4.5, u_x, u_y, scale=3, color='navy', width=0.008)
                    ax.text(1, 5.2, f"Current ({u_x}m/s)", color='navy', fontweight='bold')
                    
                    ax.set_xlim(-1, 14)
                    ax.set_ylim(-6, 6)
                    ax.set_xlabel("X Distance (m)")
                    ax.set_ylabel("Y Distance (m)")
                    ax.set_title(f"Dynamic Particle Tracking (Progress: {int(f_idx/total_steps*100)}%)")
                    ax.grid(True, linestyle='--', alpha=0.4)
                    ax.legend(loc="upper left")
                    
                    plot_spot.pyplot(fig)
                    plt.close(fig)
                    time.sleep(animate_speed)
                
                with col_ctrl:
                    st.success("시뮬레이션 완료!")
                    st.metric("최종 포집 효율", f"{rate:.1f} %", f"{cap_cnt} / {total_cnt} 개 성공")

with tab2:
    st.subheader("🔥 플라스틱 재질별 종합 비교 데이터")
    if st.button("전체 시나리오 일괄 연산 실행"):
        res_list = []
        for reg_name, (ux, uy) in [("GPGP", (0.15, -0.05)), ("SouthSea", (0.25, 0.10)), ("Kuroshio", (0.75, 0.35))]:
            for ang in [30, 45, 60, 90]:
                for p_name, p_spec in PLASTIC_SPECS.items():
                    _, r, _, _, sunk = run_simulation_engine(ux, uy, ang, p_spec, 0.5, 80)
                    res_list.append({
                        "Region": reg_name,
                        "Angle": f"{ang}°",
                        "Plastic": p_name.split()[0],
                        "Capture_Rate(%)": 0.0 if sunk else round(r, 1)
                    })
        df_all = pd.DataFrame(res_list)
        st.dataframe(df_all, use_container_width=True)
        
        fig_hm, ax_hm = plt.subplots(figsize=(7, 4))
        piv = df_all.pivot_table(index="Plastic", columns="Angle", values="Capture_Rate(%)")
        sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax_hm)
        ax_hm.set_title("Plastic Type vs Fence Angle Capture Rate (%)")
        st.pyplot(fig_hm)