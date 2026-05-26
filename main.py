import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyatmos import coesa76
from numpy import sign, pi as PI, sqrt

# Constants
burst_time = 1e9                       # Placeholder
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
DEBUG = False

# Wind Bands
first_range = [0, 5000]                  # [m] Height range of the 1st band of wind
second_range = [5000, 10000]             # [m] Height range of the 2nd band of wind
third_range = [10000, 1e9]               # [m] Height range of the final band of wind
first_vector = [-5,0]                    # [km/hr] 1st Wind Band Direction Vector
second_vector = [0,5]                    # [km/hr] 1st Wind Band Direction Vector
third_vector = [10,0]                    # [km/hr] 1st Wind Band Direction Vector
first_band = [[first_vector[0]/3.6, first_vector[1]/3.6], first_range]      # 1st Vector Converted to [m/s]
second_band = [[second_vector[0]/3.6, second_vector[1]/3.6], second_range]  # 2nd Vector Converted to [m/s]
third_band = [[third_vector[0]/3.6, third_vector[1]/3.6], third_range]      # 3rd Vector Converted to [m/s]

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
		self.burst_time = 1e9         # Placeholder
		self.endpoint = [0,0,0,0]     # Placeholder

	def calculate(self, z, vz):
		# Get atmospheric conditions
		atm = coesa76(z/1000)     # Divide z [m] by 1000 to get input in [km]
		air_density = atm.rho[0]  # Air density (kg/m^3)
		air_temp = atm.T[0]       # Temperature (K)
		air_pressure = atm.P[0]   # Pressure (Pa)

		# Compute volume and buoyant force
		V = (helium_mass*gas_constant*air_temp)/air_pressure   #[m^3]
		self.r = pow(3*V/(4*PI), 1/3)                          #[m]
		buoyancy = air_density * V * g / total_mass            #[m/s^2]

		drag = -0.5*sign(vz)*drag_coeff*air_density*(PI*self.r**2)*(vz**2) / total_mass  #[m/s^2], Always opposes z velocity

		return buoyancy + drag - g    # [m/s^2] Drag is added because it opposes the sign of v_z by design

	def balloon_dynamics(self, t, state):
		if self.endpoint != [0,0,0,0]:
			return [0,0,0,0,0,0]
		self.x, self.y, self.z, self.vx, self.vy, self.vz = state
		ax, ay = [0,0]

		# Handle x-y movement depending on wind band
		if first_range[0] <= self.z < first_range[1]:
			self.vx, self.vy = first_band[0]  # [m/s]
		elif second_range[0] <= self.z < second_range[1]:
			self.vx, self.vy = second_band[0] # [m/s]
		else:
			self.vx, self.vy = third_band[0]  # [m/s]

		az = self.calculate(self.z, self.vz)
		if self.r > burst_radius and t < self.burst_time:
			print(f"After {t/3600:.3f} hours, The Balloon burst at ({self.x/1000:.3f}km, {self.y/1000:.3f}km, {self.z/1000:.3f}km) at a lateral distance {mag(self.x,self.y,0)/1000:.3f}km")
			self.burst_time = t
			self.endpoint = [self.x,self.y,self.z]
		if int(t) % 10 == 0 and abs(t-int(t)) < 0.01 and DEBUG:
			print(f"Done for {t:.0f}s")
		return [self.vx, self.vy, self.vz, ax, ay, az]


# Solve System Using our Conditions
state0 = [0, 0, initial_height, 0, 0, 0]
system = System(*state0)
sol = solve_ivp(system.balloon_dynamics, [0, 5000], state0, method='RK45', max_step=time_step)

# Make curves end when balloon bursts
burst_index = 0
for i, el in enumerate(sol.t):
	if el >= system.burst_time and burst_index == 0:
		burst_index = i - 1
		break
SOL = []
for i in range(0,6):
	SOL.append(sol.y[i][:burst_index])
SOL.append(sol.t[:burst_index])


fig = plt.figure(figsize=(12, 10))

# Top-left: 3D Flight Depiction
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(SOL[0], SOL[1], SOL[2])
ax1.scatter([SOL[0][0]], [SOL[1][0]], [SOL[2][0]],color='green')
ax1.scatter(*system.endpoint,color='red')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_zlabel('Z Position (m)')
ax1.set_title('3D Flight Depiction')

# Top-right: Lateral Trajectory
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(SOL[0], SOL[1], color='orange', zorder=0)
ax2.scatter(SOL[0][0], SOL[1][0], color='green', zorder=1)
ax2.scatter(SOL[0][-1], SOL[1][-1], color='red', zorder=1)
ax2.set_xlabel('X Position (m)')
ax2.set_ylabel('Y Position (m)')
ax2.set_title('Lateral Flight Depiction')

# Bottom-left: Z-Velocity over time
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(SOL[6], SOL[5], color='green')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Z Velocity (m/s)')
ax3.set_title('Z Velocity Over Time')

# Bottom-right: Z Position over time
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(SOL[6], SOL[2], color='b')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Z Position (m)')
ax4.set_title('Z Position Over Time')

# Adjust layout and show
plt.tight_layout()
plt.show()