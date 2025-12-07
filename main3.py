import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import time

# ================================================================
#  MASS & INITIAL CONDITIONS (2D)
# ================================================================

m1, m2, m3 = 1.0, 2.0, 3.0

# Positions  (x, y)
p1_0 = [1.0, 0.0]
p2_0 = [-0.5,  np.sqrt(3)/2]
p3_0 = [-0.5, -np.sqrt(3)/2]

# Velocities (vx, vy)
omega = 0.8
v1_0 = [0.0,         omega]
v2_0 = [0.7*omega,  -0.35*omega]
v3_0 = [-0.7*omega, -0.35*omega]

# Combined state vector (12 variables)
# [x1,y1, x2,y2, x3,y3, vx1,vy1, vx2,vy2, vx3,vy3]
y0 = np.array([p1_0, p2_0, p3_0, v1_0, v2_0, v3_0]).ravel()


# ================================================================
#  SYSTEM OF ODES  (2D GRAVITY)
# ================================================================

def system_odes(t, S, m1, m2, m3):
    # Position vectors
    p1 = S[0:2]
    p2 = S[2:4]
    p3 = S[4:6]

    # Velocities
    v1 = S[6:8]
    v2 = S[8:10]
    v3 = S[10:12]

    # Accelerations
    def accel(pi, pj, mk):
        r = pj - pi
        dist = np.linalg.norm(r)
        return mk * r / dist**3

    a1 = accel(p1, p2, m2) + accel(p1, p3, m3)
    a2 = accel(p2, p1, m1) + accel(p2, p3, m3)
    a3 = accel(p3, p1, m1) + accel(p3, p2, m2)

    return np.array([v1, v2, v3, a1, a2, a3]).ravel()


# ================================================================
#  SOLVE SYSTEM
# ================================================================

time_s, time_e = 0, 20
t_points = np.linspace(time_s, time_e, 2001)

t1 = time.time()
sol = solve_ivp(
    fun=system_odes,
    t_span=(time_s, time_e),
    y0=y0,
    t_eval=t_points,
    args=(m1, m2, m3)
)
t2 = time.time()

print(f"Solved in: {t2-t1:.3f} seconds")

# Extract solutions
x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[2], sol.y[3]
x3, y3 = sol.y[4], sol.y[5]


# ================================================================
#  STATIC 2D TRAJECTORY PLOT
# ================================================================

plt.figure(figsize=(7,7))
plt.plot(x1, y1, 'g', label="Body 1")
plt.plot(x2, y2, 'r', label="Body 2")
plt.plot(x3, y3, 'b', label="Body 3")

plt.scatter([x1[-1]], [y1[-1]], c='g')
plt.scatter([x2[-1]], [y2[-1]], c='r')
plt.scatter([x3[-1]], [y3[-1]], c='b')

plt.title("2D Three-Body Simulation")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


# ================================================================
#  ANIMATION (2D)
# ================================================================

fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(min(x1.min(),x2.min(),x3.min())-1,
            max(x1.max(),x2.max(),x3.max())+1)
ax.set_ylim(min(y1.min(),y2.min(),y3.min())-1,
            max(y1.max(),y2.max(),y3.max())+1)
ax.set_title("2D Three-Body Animation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

line1, = ax.plot([], [], 'g', linewidth=1)
dot1, = ax.plot([], [], 'o', color='green')

line2, = ax.plot([], [], 'r', linewidth=1)
dot2, = ax.plot([], [], 'o', color='red')

line3, = ax.plot([], [], 'b', linewidth=1)
dot3, = ax.plot([], [], 'o', color='blue')

def update(frame):
    line1.set_data(x1[:frame], y1[:frame])
    dot1.set_data(x1[frame], y1[frame])

    line2.set_data(x2[:frame], y2[:frame])
    dot2.set_data(x2[frame], y2[frame])

    line3.set_data(x3[:frame], y3[:frame])
    dot3.set_data(x3[frame], y3[frame])

    return line1, dot1, line2, dot2, line3, dot3

anim = FuncAnimation(
    fig, update, frames=len(t_points),
    interval=20, blit=True
)

print("2D GIF saved as: three_body_2D.gif")
plt.show()
