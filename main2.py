import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# ------------------------------------------- #
# Masses
m1 = m2 = m3 = 1.0

# Initial positions
p1_0 = [1.0, 0.0, 0.0]
p2_0 = [-0.5,  np.sqrt(3)/2, 0.0]
p3_0 = [-0.5, -np.sqrt(3)/2, 0.0]

# Initial velocities
omega = 0.8
v1_0 = [0, omega, 0]
v2_0 = [0.7*omega, -0.35*omega, 0]
v3_0 = [-0.7*omega, -0.35*omega, 0]

y0 = np.array([p1_0, p2_0, p3_0, v1_0, v2_0, v3_0]).ravel()

# ------------------------------------------- #
# ODE system

def system_odes(t, S, m1, m2, m3):
    p1, p2, p3 = S[0:3], S[3:6], S[6:9]
    v1, v2, v3 = S[9:12], S[12:15], S[15:18]

    a1 = m2*(p2 - p1)/np.linalg.norm(p2 - p1)**3 + m3*(p3 - p1)/np.linalg.norm(p3 - p1)**3
    a2 = m1*(p1 - p2)/np.linalg.norm(p1 - p2)**3 + m3*(p3 - p2)/np.linalg.norm(p3 - p2)**3
    a3 = m1*(p1 - p3)/np.linalg.norm(p1 - p3)**3 + m2*(p2 - p3)/np.linalg.norm(p2 - p3)**3

    return np.concatenate([v1, v2, v3, a1, a2, a3])

# ------------------------------------------- #
# Solve system

t_eval = np.linspace(0, 7, 1800)
sol = solve_ivp(system_odes, (0, 7), y0, args=(m1, m2, m3), t_eval=t_eval)

p1 = sol.y[0:3]
p2 = sol.y[3:6]
p3 = sol.y[6:9]

# ------------------------------------------- #
# Animation figure

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="3d")

# Set bounds
xyz = sol.y[0:9]
min_val = np.min(xyz) - 0.2
max_val = np.max(xyz) + 0.2
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_zlim(min_val, max_val)

ax.set_title("Three-Body Simulation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

line1, = ax.plot([], [], [], "g", lw=1)
dot1,  = ax.plot([], [], [], "go")

line2, = ax.plot([], [], [], "r", lw=1)
dot2,  = ax.plot([], [], [], "ro")

line3, = ax.plot([], [], [], "b", lw=1)
dot3,  = ax.plot([], [], [], "bo")

# ------------------------------------------- #
# Animation update function
# IMPORTANT: No blit + ALWAYS use sequences for dot.set_data()

def update(i):
    line1.set_data(p1[0,:i], p1[1,:i])
    line1.set_3d_properties(p1[2,:i])
    dot1.set_data([p1[0,i]], [p1[1,i]])
    dot1.set_3d_properties([p1[2,i]])

    line2.set_data(p2[0,:i], p2[1,:i])
    line2.set_3d_properties(p2[2,:i])
    dot2.set_data([p2[0,i]], [p2[1,i]])
    dot2.set_3d_properties([p2[2,i]])

    line3.set_data(p3[0,:i], p3[1,:i])
    line3.set_3d_properties(p3[2,:i])
    dot3.set_data([p3[0,i]], [p3[1,i]])
    dot3.set_3d_properties([p3[2,i]])

    return line1, dot1, line2, dot2, line3, dot3

# ------------------------------------------- #
anim = FuncAnimation(fig, update, frames=len(t_eval), interval=10, blit=False)

# SAVE GIF (works on macOS)
anim.save("three_body.gif", dpi=100, writer=PillowWriter(fps=30))

print("Saved as three_body.gif")
