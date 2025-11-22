import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bio-Scrubber CFD", layout="wide")

# --- ENGINEERING CONSTANTS ---
TANK_DIAMETER_MM = 700
TANK_AREA_M2 = np.pi * (TANK_DIAMETER_MM / 1000 / 2)**2

# --- PHYSICS ENGINE (Cached for Speed) ---
@st.cache_data
def run_simulation(u_in, width=100, height=350, steps=1000):
    # Setup Geometry
    ny, nx = height, width
    mask = np.ones((ny, nx))
    
    # Geometry: Packing + Sump
    packing_top = 15
    packing_bottom = int(ny * 0.65)
    period = 20
    wall_thickness = 6
    
    for i in range(nx):
        if (i % period) < wall_thickness:
            mask[packing_top:packing_bottom, i] = 0
    for j in range(packing_top, packing_bottom, 45):
        mask[j:j+3, :] = 1
        if (j % 90) == 0:
            mask[j, int(nx/3):int(2*nx/3)] = 0
            
    mask[:, 0] = 0
    mask[:, -1] = 0
    mask[packing_bottom:, 1:-1] = 1
    mask[0:5, :] = 1
    mask[-5:, :] = 1

    # LBM Initialization
    tau = 0.9
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    
    fin = np.zeros((9, ny, nx))
    rho = np.ones((ny, nx))
    u = np.zeros((2, ny, nx))
    u[1, :, :] = u_in # Initial down flow
    
    # Init Equilibrium
    for i in range(9):
        cu = 3 * (cx[i]*u[0] + cy[i]*u[1])
        fin[i] = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*(u[0]**2 + u[1]**2))
        
    # Simulation Loop (No graphics, just math)
    for _ in range(steps):
        # Macroscopic
        rho = np.sum(fin, axis=0)
        u = np.zeros((2, ny, nx))
        for i in range(9):
            u[0] += cx[i] * fin[i]
            u[1] += cy[i] * fin[i]
        mask_fluid = mask > 0
        u[:, mask_fluid] /= rho[mask_fluid]
        u[:, ~mask_fluid] = 0
        
        # Collision
        fout = np.zeros_like(fin)
        for i in range(9):
            cu = 3 * (cx[i]*u[0] + cy[i]*u[1])
            feq = rho * w[i] * (1 + cu + 0.5*cu**2 - 1.5*(u[0]**2 + u[1]**2))
            fout[i] = fin[i] - (fin[i] - feq) / tau
            fout[i, mask==0] = fin[noslip[i], mask==0]
            
        # Streaming
        for i in range(9):
            fin[i] = np.roll(fout[i], cx[i], axis=1)
            fin[i] = np.roll(fin[i], cy[i], axis=0)
            
        # Inlet
        for i in range(9):
             cu_in = 3 * (cy[i]*u_in)
             feq_in = w[i] * (1 + cu_in + 0.5*cu_in**2 - 1.5*u_in**2)
             fin[i, 0, :] = feq_in
        fin[:, -1, :] = fin[:, -2, :]

    velocity = np.sqrt(u[0]**2 + u[1]**2)
    velocity[mask == 0] = np.nan
    return velocity

# --- LAYOUT ---
st.title("Bio-Scrubber CFD Analysis by MST")
st.markdown("**H2S Load 25,000 ppm**")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Control Panel")
    s_h2s = st.slider("H2S Concentration (ppm)", 1000, 40000, 25000, 1000)
    s_flow = st.slider("Gas Velocity (m/s)", 0.005, 0.040, 0.018, 0.001)
    s_temp = st.slider("Temperature (°C)", 20, 60, 30)
    
    # Calculation
    q_nm3h = s_flow * TANK_AREA_M2 * 3600
    st.info(f"Calculated Flow Rate: **{q_nm3h:.2f} Nm³/h**")
    
    # Safety Logic
    load_factor = q_nm3h * s_h2s
    limit_factor = 25 * 25000
    if load_factor > limit_factor * 1.2:
        st.error("CRITICAL: Retention Time Too Short!")
    elif load_factor > limit_factor:
        st.warning("WARNING: Efficiency Dropping")
    else:
        st.success("SYSTEM OPTIMAL")

    run_btn = st.button("Run Simulation", type="primary")

with col2:
    if run_btn:
        with st.spinner('Simulating Fluid Dynamics...'):
            # Run Physics
            velocity_field = run_simulation(s_flow)
            
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.set_facecolor('#FF8C00')
            img = ax.imshow(velocity_field, cmap='jet', vmin=0, vmax=0.03, origin='upper')
            
            # Decorations
            ax.text(50, 15, "▼ INLET ▼", ha='center', color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
            ax.text(50, 280, "SUMP ZONE", ha='center', color='white', fontweight='bold', bbox=dict(facecolor='blue', alpha=0.3))
            ax.axis('off')
            
            cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Velocity (m/s)')
            
            st.pyplot(fig)
    else:
        st.info("Adjust parameters on the left and click 'Run Simulation'")
