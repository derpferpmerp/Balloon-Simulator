import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyatmos import coesa76
from numpy import sign, pi as PI, sqrt
from aquarel import load_theme
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from tqdm.rich import tqdm


# Constants
g = 9.81                               # [m/s^2] Acceleration due to gravity
gas_constant = 2077                    # Specific Gas Constant for Helium
drag_coeff = 0.5                       # Drag Coefficient for Balloon
helium_mass = 1.0                      # [kg] Mass of Helium inside Balloon
skin_mass = 2.5                        # [kg] Mass of the skin of the balloon
total_mass = helium_mass + skin_mass   # [kg] Combined mass of the system
burst_radius = 8                       # [m] Radius at which the Balloon bursts
time_step = 1                          # [s] Time step at which to simulate
initial_height = 0                     # [m] Starting height of the balloon
initial_radius = 0                     # [m] Starting radius of the balloon
launch_time = datetime(2025, 1, 1, 9, 0, 0)   # Launch datetime
t_end = 50000                          # [s] Maximum time step (forceful stop)
DEBUG = False

# Wind Bands
first_range   = [0, 5000]              # [m] Height range of the 1st band of wind
second_range  = [5000, 10000]          # [m] Height range of the 2nd band of wind
third_range   = [10000, 1e9]           # [m] Height range of the final band of wind
first_vector  = [-5, 0, 7]            # [km/hr] 1st Wind Band Direction Vector [x, y, z]
second_vector = [0, 5, -20]           # [km/hr] 2nd Wind Band Direction Vector [x, y, z]
third_vector  = [10, 0, 0]            # [km/hr] 3rd Wind Band Direction Vector [x, y, z]
first_band  = [[v/3.6 for v in first_vector],  first_range]   # 1st Vector Converted to [m/s]
second_band = [[v/3.6 for v in second_vector], second_range]  # 2nd Vector Converted to [m/s]
third_band  = [[v/3.6 for v in third_vector],  third_range]   # 3rd Vector Converted to [m/s]

def mag(X, Y, Z):
    return sqrt(X**2 + Y**2 + Z**2)

class System(object):
    def __init__(self, x, y, z, vx, vy, vz):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.r = initial_radius
        self.burst_time = 1e9
        self.endpoint = [0, 0, 0, 0]
        self.burst_occurred = False

    def get_atmosphere(self, z):
        atm = coesa76(z / 1000)
        air_density  = atm.rho[0]    # [kg/m^3]
        air_temp     = atm.T[0]      # [K]
        air_pressure = atm.P[0]      # [Pa]
        V = (helium_mass * gas_constant * air_temp) / air_pressure   # [m^3]
        self.r = pow(3*V / (4*PI), 1/3)                               # [m]
        return air_density, V

    def get_wind(self, z):
        if first_range[0] <= z < first_range[1]:
            return first_band[0]
        elif second_range[0] <= z < second_range[1]:
            return second_band[0]
        else:
            return third_band[0]

    def _drag(self, air_density, rel_v):
        A = PI * self.r**2
        return -0.5 * sign(rel_v) * drag_coeff * air_density * A * rel_v**2 / total_mass

    def vertical_acceleration(self, air_density, V, vz, wind_vz):
        buoyancy = air_density * V * g / total_mass
        return buoyancy + self._drag(air_density, vz - wind_vz) - g

    def balloon_dynamics(self, t, state):
        if self.endpoint != [0, 0, 0, 0]:
            return [0, 0, 0, 0, 0, 0]

        self.x, self.y, self.z, self.vx, self.vy, self.vz = state

        air_density, V = self.get_atmosphere(self.z)
        wind = self.get_wind(self.z)

        ax = self._drag(air_density, self.vx - wind[0])
        ay = self._drag(air_density, self.vy - wind[1])
        az = self.vertical_acceleration(air_density, V, self.vz, wind[2])

        if self.r > burst_radius and t < self.burst_time:
            burst_clock = launch_time + timedelta(seconds=float(t))
            print(f"\nAt {burst_clock.strftime('%H:%M:%S')}, The Balloon burst at ({self.x/1000:.3f}km, {self.y/1000:.3f}km, {self.z/1000:.3f}km) at a lateral distance {mag(self.x, self.y, 0)/1000:.3f}km")
            self.burst_time = t
            self.burst_occurred = True
            self.endpoint = [self.x, self.y, self.z]

        if int(t) % 10 == 0 and abs(t - int(t)) < 0.01 and DEBUG:
            print(f"Done for {t:.0f}s")

        return [self.vx, self.vy, self.vz, ax, ay, az]


# Solve System Using our Conditions

state0 = [0, 0, initial_height, 0, 0, 0]
system = System(*state0)

with tqdm(total=t_end, desc="Simulating", unit="s") as pbar:
    last_t = [0]
    original_dynamics = system.balloon_dynamics

    def tracked_dynamics(t, state):
        pbar.update(t - last_t[0])
        last_t[0] = t
        return original_dynamics(t, state)

    sol = solve_ivp(tracked_dynamics, [0, t_end], state0, method='RK45', max_step=time_step)

# Make curves end when balloon bursts, or use full trajectory if no burst
burst_index = len(sol.t) - 1
for i, el in enumerate(sol.t):
    if el >= system.burst_time and burst_index == len(sol.t) - 1:
        burst_index = i - 1
        break

if not system.burst_occurred:
    end_clock = launch_time + timedelta(seconds=float(sol.t[-1]))
    print(f"Balloon did not burst. Simulation ended at {end_clock.strftime('%H:%M:%S')} at ({sol.y[0][-1]/1000:.3f}km, {sol.y[1][-1]/1000:.3f}km, {sol.y[2][-1]/1000:.3f}km)")

SOL = []
for i in range(0, 6):
    SOL.append(sol.y[i][:burst_index])
SOL.append(sol.t[:burst_index])

# Convert simulation time (seconds) to datetime objects
timestamps = [launch_time + timedelta(seconds=float(t)) for t in SOL[6]]

theme = load_theme("boxy_dark")
theme.apply()

fig = plt.figure(figsize=(12, 10))

# Top-left: 3D Flight Depiction
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.xaxis.pane.set_facecolor('black')
ax1.yaxis.pane.set_facecolor('black')
ax1.zaxis.pane.set_facecolor('black')
ax1.minorticks_off()
ax1.plot(SOL[0], SOL[1], SOL[2], color='#00FFFF')
ax1.scatter([SOL[0][0]], [SOL[1][0]], [SOL[2][0]], color='#00FF00')
if system.burst_occurred:
    ax1.scatter(*system.endpoint, color='#FF0000')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_zlabel('Z Position (m)')
ax1.set_title('3D Flight Depiction')

# Top-right: Lateral Trajectory
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(SOL[0], SOL[1], color='#FF6600', zorder=0)
ax2.scatter(SOL[0][0], SOL[1][0], color='#00FF00', zorder=1)
ax2.scatter(SOL[0][-1], SOL[1][-1], color='#FF0000' if system.burst_occurred else '#FFFF00', zorder=1)
ax2.set_xlabel('X Position (m)')
ax2.set_ylabel('Y Position (m)')
ax2.set_title('Lateral Flight Depiction')

# Bottom-left: Z-Velocity over time
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(timestamps, SOL[5], color='#00FF66')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax3.set_xlabel('Time')
ax3.set_ylabel('Z Velocity (m/s)')
ax3.set_yscale('linear')
ax3.set_title('Z Velocity Over Time')

# Bottom-right: Z Position over time
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(timestamps, SOL[2], color='#4488FF')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax4.set_xlabel('Time')
ax4.set_ylabel('Z Position (m)')
ax4.set_title('Z Position Over Time')

# Adjust layout and show
plt.tight_layout()
plt.show()
fig.savefig("fig.png", dpi=400)