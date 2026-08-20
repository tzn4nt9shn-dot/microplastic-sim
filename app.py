import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 웹 페이지 기본 설정 및 디자인
# ==========================================
st.set_page_config(
    page_title="해류 이용 미세플라스틱 포집 시뮬레이터",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 GIS 기반 무동력 미세플라스틱 포집 펜스 시뮬레이터")
st.markdown("팀원의 유체역학 및 펜스 충돌 수식을 기반으로 구동되는 실시간 웹 시뮬레이터입니다.")

# ==========================================
# 1. 팀원의 물리 상수 및 기본 데이터 정의 (이미지 1번 항목)
# ==========================================
RHO_WATER = 1025.0  # 해수 밀도 (kg/m^3)
C_D = 1.0           # 항력 계수 (고정)
PLASTIC_DENSITIES = {
    "PE": 920.0,    # PE 밀도 (kg/m^3)
    "PP": 900.0,    # PP 밀도 (kg/m^3)
    "PET": 1380.0   # PET 밀도 (kg/m^3)
}

# ==========================================
# 2. 사이드바 (사용자가 조작하는 웹 컨트롤러)
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 변수 설정")

# (1) 해역 및 해류 유속 선택
region = st.sidebar.selectbox("대상 해역 선택 (GIS)", ["GPGP (완만)", "남해안 (연안)", "크로시오 (난류)", "직접 입력"])
if region == "GPGP (완만)":
    u_x, u_y = 0.15, -0.05
elif region == "남해안 (연안)":
    u_x, u_y = 0.25, 0.10
elif region == "크로시오 (난류)":
    u_x, u_y = 0.75, 0.35
else:
    u_x = st.sidebar.slider("동서 유속 u (m/s)", -1.0, 1.0, 0.3)
    u_y = st.sidebar.slider("남북 유속 v (m/s)", -1.0, 1.0, 0.0)

# (2) 펜스 각도 설정 (이미지 11번 항목)
fence_angle_deg = st.sidebar.selectbox("V자 펜스 각도 (θ)", [30, 45, 60, 90])

# (3) 플라스틱 물성치 선택
plastic_type = st.sidebar.selectbox("플라스틱 재질", ["PE", "PP", "PET"])
particle_size_mm = st.sidebar.selectbox("입자 크기 (d)", [0.5, 5.0])

# (4) 입자 수 및 계산 설정
num_particles = st.sidebar.slider("입자 개수", 100, 1000, 300, step=100)
dt = 0.01          # 시간 간격 Δt (s) (이미지 1번 항목)
max_steps = 1500    # 총 시뮬레이션 스텝 수

# ==========================================
# 3. 핵심 유체역학 및 충돌 물해석 엔진 (이미지 3~16번 수식 구현)
# ==========================================
def run_simulation(u_x, u_y, angle_deg, p_type, size_mm, n_particles):
    # --- A. 입자 물리량 계산 (이미지 5, 6번 항목) ---
    d = size_mm / 1000.0                       # mm -> m 변환
    area = np.pi * (d / 2.0)**2                # 입자 투영면적 A
    rho_p = PLASTIC_DENSITIES[p_type]          # 플라스틱 밀도
    mass = rho_p * (4.0 / 3.0) * np.pi * (d / 2.0)**3  # 입자 질량 m
    
    # --- B. 입자 초기 위치 설정 ---
    # x = 0 위치에서 y축 -5m ~ 5m 사이에 균등 배치
    pos = np.zeros((n_particles, 2))
    pos[:, 0] = 0.0
    pos[:, 1] = np.linspace(-4.5, 4.5, n_particles)
    
    vel = np.zeros((n_particles, 2))
    vel[:, 0] = u_x
    vel[:, 1] = u_y
    
    # --- C. V자 펜스 기하학적 정의 (이미지 10~16번 항목) ---
    apex_x, apex_y = 10.0, 0.0  # V자 펜스 꼭짓점(중앙 포켓) 위치
    theta = np.radians(angle_deg / 2.0)  # V자 반각
    fence_length = 8.0          # 펜스 날개 길이
    
    # 좌측 날개(상단) 접선/법선 벡터
    t_upper = np.array([np.cos(theta), -np.sin(theta)])  # 포켓 방향 접선
    n_upper = np.array([-np.sin(theta), -np.cos(theta)]) # 유수 방향 법선
    
    # 우측 날개(하단) 접선/법선 벡터
    t_lower = np.array([np.cos(theta), np.sin(theta)])
    n_lower = np.array([-np.sin(theta), np.cos(theta)])
    
    # 시뮬레이션 기록용
    trajectories = [pos.copy()]
    captured_mask = np.zeros(n_particles, dtype=bool)
    escaped_mask = np.zeros(n_particles, dtype=bool)
    
    u_ocean = np.array([u_x, u_y])
    
    for step in range(max_steps):
        # 1. 상대속도 계산 (이미지 3번: v_r = u - v_p)
        v_r = u_ocean - vel
        v_r_mag = np.linalg.norm(v_r, axis=1, keepdims=True) + 1e-8
        
        # 2. 항력 계산 (이미지 4번: F_d = 0.5 * rho * C_d * A * v_r^2)
        f_drag = 0.5 * RHO_WATER * C_D * area * v_r_mag * v_r
        
        # 3. 가속도 및 속도/위치 업데이트 (이미지 7, 8, 9번: Euler 기법)
        accel = f_drag / mass
        vel += accel * dt
        pos += vel * dt
        
        # 4. 펜스 충돌 및 슬라이딩 처리 (이미지 10~16번 항목)
        for i in range(n_particles):
            if captured_mask[i] or escaped_mask[i]:
                continue
            
            px, py = pos[i]
            
            # 중앙 포켓 도착 검증 (포집 성공)
            if np.hypot(px - apex_x, py - apex_y) < 0.6:
                captured_mask[i] = True
                vel[i] = [0, 0]
                continue
            
            # 상단 펜스 날개 충돌 체크 및 미끄러짐 (이미지 14, 15번 항목: v_n = 0)
            if py > 0 and px >= (apex_x - (py - apex_y) * np.tan(np.pi/2 - theta)):
                # 속도를 법선/접선으로 분해 후 법선 성분 제거
                v_t = np.dot(vel[i], t_upper)
                vel[i] = max(0, v_t) * t_upper  # 포켓 방향으로만 이동
                
            # 하단 펜스 날개 충돌 체크 및 미끄러짐
            elif py < 0 and px >= (apex_x - (apex_y - py) * np.tan(np.pi/2 - theta)):
                v_t = np.dot(vel[i], t_lower)
                vel[i] = max(0, v_t) * t_lower
                
            # 영역 이탈 체크 (포집 실패)
            if px > 15.0 or py > 8.0 or py < -8.0:
                escaped_mask[i] = True

        trajectories.append(pos.copy())
        
        # 모든 입자의 이동이 끝난 경우 일찍 종료
        if np.all(captured_mask | escaped_mask):
            break
            
    trajectories = np.array(trajectories)
    capture_rate = (np.sum(captured_mask) / n_particles) * 100.0
    return trajectories, capture_rate, np.sum(captured_mask), n_particles

# ==========================================
# 4. 웹 메인 화면 구성 및 결과 출력
# ==========================================
tab1, tab2 = st.tabs(["📊 단일 시뮬레이션 구동", "🔥 36가지 시나리오 배치 분석"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("실행 조건 요약")
        st.write(f"- **해류 유속:** u={u_x}m/s, v={u_y}m/s")
        st.write(f"- **펜스 각도:** {fence_angle_deg}°")
        st.write(f"- **플라스틱 재질:** {plastic_type} ({PLASTIC_DENSITIES[plastic_type]} kg/m³)")
        st.write(f"- **입자 크기:** {particle_size_mm} mm")
        
        start_btn = st.button("🚀 시뮬레이션 실행", use_container_width=True)
        
    if start_btn or 'trajectories' in st.session_state:
        trajectories, cap_rate, cap_cnt, total_cnt = run_simulation(
            u_x, u_y, fence_angle_deg, plastic_type, particle_size_mm, num_particles
        )
        st.session_state['trajectories'] = trajectories
        
        with col1:
            st.success("계산 완료!")
            st.metric("최종 포집 효율 (η)", f"{cap_rate:.1f} %", f"{cap_cnt} / {total_cnt} 개 포집")

        with col2:
            st.subheader("📍 입자 궤적 및 V자 펜스 시각화")
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # V자 펜스 그리기
            apex_x, apex_y = 10.0, 0.0
            rad = np.radians(fence_angle_deg / 2.0)
            wing_len = 6.0
            
            x_left = [apex_x - wing_len * np.cos(rad), apex_x]
            y_left = [apex_y + wing_len * np.sin(rad), apex_y]
            x_right = [apex_x - wing_len * np.cos(rad), apex_x]
            y_right = [apex_y - wing_len * np.sin(rad), apex_y]
            
            ax.plot(x_left, y_left, 'k-', lw=3, label="V-Fence Wall")
            ax.plot(x_right, y_right, 'k-', lw=3)
            ax.plot(apex_x, apex_y, 'ro', ms=10, label="Central Pocket")
            
            # 입자 궤적 그리기 (최종 위치 기준)
            last_pos = trajectories[-1]
            ax.scatter(last_pos[:, 0], last_pos[:, 1], c='dodgerblue', alpha=0.6, s=15, label="Particles")
            
            # 해류 화살표 표시
            ax.quiver(1, 4, u_x, u_y, scale=3, color='blue', alpha=0.5)
            ax.text(1, 4.5, "Ocean Current Vector", color='blue')

            ax.set_xlim(-1, 15)
            ax.set_ylim(-6, 6)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Particle Trajectories (Angle: {fence_angle_deg}°, Current: {u_x}m/s)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc="upper left")
            
            st.pyplot(fig)

with tab2:
    st.subheader("🧪 36가지 시나리오 자동 실험 및 분석 데이터")
    st.write("버튼을 누르면 모든 조건 조합(유속 3개 × 각도 4개 × 재질 3개)의 포집률을 일괄 계산합니다.")
    
    if st.button("🔥 36개 전체 시나리오 배치 실행"):
        results = []
        with st.spinner("36개 시나리오를 연산 중입니다..."):
            for sp_name, (ux, uy) in [("GPGP", (0.15, -0.05)), ("SouthSea", (0.25, 0.10)), ("Kuroshio", (0.75, 0.35))]:
                for ang in [30, 45, 60, 90]:
                    for ptype in ["PE", "PP", "PET"]:
                        _, rate, _, _ = run_simulation(ux, uy, ang, ptype, 0.5, 200)
                        results.append({
                            "Region": sp_name,
                            "Angle": f"{ang}°",
                            "Plastic": ptype,
                            "Capture_Rate(%)": round(rate, 1)
                        })
        
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)
        
        # 히트맵 시각화
        st.subheader("📊 해역 및 펜스 각도별 평균 포집 효율 히트맵")
        pivot_df = df_res.pivot_table(index="Angle", columns="Region", values="Capture_Rate(%)")
        
        fig_hp, ax_hp = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax_hp, cbar_kws={'label': 'Capture Rate (%)'})
        ax_hp.set_title("Fence Angle vs Region Capture Efficiency")
        st.pyplot(fig_hp)