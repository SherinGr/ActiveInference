"""
Created on Mon Mar 11 15:22:08 2019

@author: Sherin

CONTENTS:
In this file we simulate the control of a planar 2 DOF arm using Active Inference (AI).
We will assume that all model parameters etc. are known, the only task is to study the
performance of the filter and controller that AI provides. The main difficulty in this 
simulation is that the system is nonlinear.

NOTE:
In a future version, the idea is to replace the models by GPR's that can be learned on-
line. A second idea is to perform control in the workspace. In that scenario we would 
like the end effector to track a dynamic reference. This will be achieved by adding a 
virtual spring and damper between the current and desired end-effector position in the
generative model.

"""

import sys
import numpy as np
import sympy as sym
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as la

sys.path.insert(0, r"C:\Users\sheri\Desktop\Sherins Map All Things\Research\Active Inference\SSAI\active_inference")
from ssai import makeNoise
from numpy import sin, cos


class DoubleLink:
    """ Class for a 2-link 'robot arm' """

    params = collections.namedtuple('par', 'l1 l2 m1 m2 iz1 iz2')
    njoints = 2

    def __init__(self, l1, l2, m1, m2, iz1, iz2):
        """ Define 2 link robot arm parameters: """
        self.par = self.params(l1=l1, l2=l2, m1=m1, m2=m2, iz1=iz1, iz2=iz2)

    def dynamics(self, x, tau, *noise):
        """ Calculate the derivative of the state (angles,velocities):
        
            x   = [q1 q2 dq1 dq2]   current state in generalized coordinates
            t   = time
            tau = [tau1 tau2]       joint torques 
            noise is optional.
        """
        # Extract state
        q1, q2, dq1, dq2 = x

        # Parameters:
        par = self.par
        l1 = par.l1  # first link length [m]
        l2 = par.l2  # second link length [m]

        m1 = par.m1  # first link mass [kg]
        m2 = par.m2  # second link mass [kg]

        i1z = par.iz1  # first link inertia [kgm^2]
        i2z = par.iz2  # second link inertia [kgm^2]

        g = 9.81  # gravity constant [m/s^2]

        # Goniometric functions
        # s1  = sin(q1)
        # c1  = cos(q1)
        s2 = sin(q2)
        c2 = cos(q2)
        # s12 = sin(q1+q2)
        # c12 = cos(q1+q2)

        # Elements of the Inertia Matrix M
        m_11 = i1z + i2z + (l1 / 2) ** 2 * m1 + m2 * (l1 ** 2 + (l2 / 2) ** 2 + 2 * l1 * l2 / 2 * c2)
        m_12 = i2z + m2 * ((l2 / 2) ** 2 + l1 * l2 / 2 * c2)
        m_22 = i2z + m2 * (l2 / 2) ** 2
        m = np.ndarray([[m_11, m_12], [m_12, m_22]])

        # Coriolis matrix:
        c_11 = -(l1 * dq2 * s2 * (l2 / 2 * m2))
        c_12 = -(l1 * (dq2 + dq1) * s2 * (l2 / 2 * m2))
        c_21 = m2 * l1 * l2 / 2 * s2 * dq1
        c_22 = 0
        c = np.ndarray([[c_11, c_12], [c_21, c_22]])

        # Gravity matrix:
        g_1 = 0  # m1*l1/2*c1 + m2*(l2/2*c12 + l1*c1)
        g_2 = 0  # m2*l2/2*c12
        g_ = g * np.ndarray([[g_1], [g_2]])

        # Damping matrix
        d1, d2 = 30, 1
        d = np.ndarray([[d1 * dq1], [d2 * dq2]])

        # Dynamics update Computation of acceleration
        # Torque vector
        tau = np.ndarray(tau).T

        # The dynamics:
        if not noise:
            dq = np.ndarray([[dq1], [dq2]])
            ddq = np.linalg.inv(m).dot(tau - c.dot(dq) - d - g_)
        else:
            dq = np.ndarray([[dq1], [dq2]]) + noise[0:self.njoints]
            ddq = np.linalg.inv(m).dot(tau - c.dot(dq) - d - g_) + noise[self.njoints:]

        # Derivative of current state
        dx = (dq1, dq2, float(ddq[0]), float(ddq[-1]))

        return dx

    def joint_mapping(self, x):
        """ Map from joint angles to joint coordinates: """
        q1, q2, _, _ = x

        par = self.par
        x1 = par.l1 * cos(q1)
        y1 = par.l1 * sin(q1)
        x2 = x1 + par.l2 * cos(q1 + q2)
        y2 = y1 + par.l2 * sin(q1 + q2)

        return x1, y1, x2, y2

    def jacobian(self, x):
        """ arm jacobian """
        par = self.par
        q1, q2, _, _ = x
        dxq1 = -par.l1 * sin(q1) - par.l2 * sin(q1 + q2)
        dxq2 = -par.l1 * sin(q1 + q2)
        dyq1 = par.l1 * cos(q1) + par.l2 * cos(q1 + q2)
        dyq2 = par.l2 * cos(q1 + q2)

        j = np.ndarray([[dxq1, dxq2], [dyq1, dyq2]])

        return j

    def measure(self, x):
        """ produce observation """
        _, _, x_eef, y_eef = self.joint_mapping(x)
        q1, q2, _, _ = x

        return q1, q2, x_eef, y_eef


class ActiveInferenceAgent:
    """ Agent that can perform Active Inference on a 2-link planar arm. It assumes that the end effector is connected
        to the reference with a spring and damper.
    """

    # Active Inference for double link pendulum
    def __init__(self, arm: DoubleLink, kp, kd, cw, cz, a_mu, a_u):
        self.model = arm  # assuming that the system is known

        self.a_mu = a_mu  # perception learning rate (>0)
        self.a_u = a_u  # action learning rate (>0)
        self.kp = kp
        self.kd = kd

        # Turn coraviance arrays into matrices and invert them to precision matrices:
        if np.size(np.shape(cw)) == 1 or min(np.shape(cw)) == 1:
            cw = np.diag(cw)
        if np.size(np.shape(cz)) == 1 or min(np.shape(cz)) == 1:
            cz = np.diag(cz)

        self.Piz = la.inv(cz)
        self.Piw = la.inv(cw)

    def perception(self, mu, y, t):
        """
        The generative model that the agent employs assumes computed torque 
        control and a spring and damper attatched between the reference and end
        effector (so the torques are computed from workspace coordinates):
            
        q'  = q'
        q'' = M^(-1) tau_des
        
        where tau_des = J^T f_ext, where f_ext is the spring and damper force
        between the end effector and the reference.
        
        """
        mu_q, mu_dq = mu

        f_mu = self.belief_dynamics(mu_q, t)
        g_mu = self.model.measure(mu)

        dmdq = []
        dTdq = []

        # TODO: Define these variables properly
        dfdmu = np.ndarray([])  # [0 I; dM^(-1)dq 0]
        dgdmu = np.concatenate(np.ones([1, self.model.njoints]), dTdq, axis=1)  # [1 ; T(q)]

        dmu_q = mu_dq - dfdmu.dot(self.Piw).dot(mu_dq - f_mu) - dgdmu.dot(self.Piz).dot(y - g_mu)
        dmu_dq = - self.Piw.dot(mu_dq - f_mu)

        return tuple([np.squeeze(np.array(dmu_q)), np.squeeze(np.array(dmu_dq))])

    def action(self, mu, y):

        j = self.model.jacobian(mu)  # forward model (state dependent)
        g_mu = self.model.measure(mu)

        du = -self.a_u * j.T.dot(self.Piz).dot(y - g_mu)
        return du

    def virtual_pd_torques(self, x, t):
        """ Find current joint torques due to spring and damper """
        # Desired end effector coordinates:
        x_ref, y_ref, dx_ref, dy_ref = self.eef_reference(t)

        # Current end effector coordinates:
        _, _, x_eef, y_eef = self.model.joint_mapping(x)
        # Current end effector velocities:
        q = np.ndarray([x[0:self.model.njoints]]).T
        dx_eef, dy_eef = tuple(np.squeeze(np.array(self.model.jacobian(x).dot(q))))

        # Current errors:
        e = np.ndarray([[x_ref - x_eef], [y_ref - y_eef]])
        de = np.ndarray([[dx_ref - dx_eef], [dy_ref - dy_eef]])

        # Find virtual spring forces:
        f = self.kp * e + self.kd * de

        # Transform to torques:
        j = self.model.jacobian(x)
        tau = j.T.dot(f)

        return tuple(np.squeeze(np.array(tau)))

    @staticmethod
    def eef_reference(t):
        """ Retrieve x and y reference and velocities at time t """
        # Current reference is a Lissajous curve
        d = np.pi / 2
        x_ref = 0.8 * sin(1 * t + d)
        y_ref = 0.2 * sin(2 * t) + 0.8
        dx_ref = 0.8 * cos(t)
        dy_ref = 0.8 * cos(t)

        return x_ref, y_ref, dx_ref, dy_ref

    def belief_dynamics(self, x, t):
        """ Return state derivative according to model belief """
        # Get virtual spring torques:
        tau = self.virtual_pd_torques(x, t)
        # Calculate motion:
        dx = self.model.dynamics(x, tau)

        return dx


""" Simulate Active Inference for the robot arm, with spring and damper """
# Declare our robot arm, dave:
l1, l2, m1, m2, iz1, iz2 = 0.6, 0.4, 2.0, 1.3, 1, 1
dave = DoubleLink(l1, l2, m1, m2, iz1, iz2)

# Declare our intelligent agent, aibot:
# note: aibot does not use generalised motions
kp, kd = 300, 0
a_mu, a_u = 1, 1
cw = np.array([10, 10])
cz = np.array([10, 10, 10, 10])
aibot = ActiveInferenceAgent(dave, kp, kd, cw, cz, a_mu, a_u)

# Simulate the dynamics:
dt = 0.05
T = 10
t = np.arange(0, T + dt, dt)

gamma = 32

# Construct noise signals:
w = makeNoise(cw, gamma, t)
z = makeNoise(cz, gamma, t)

# Initial state [rad] and [rad/s]:
q1, q2, dq1, dq2 = 0, np.pi / 6, 0, 0
x0 = q1, q2, dq1, dq2

u = []
x = []
mu = []

# Simulation using Euler discretisation:
for i in range(np.size(t)):
    # Generative process:
    dx = dave.dynamics(x, t[i], u[:, i], w[:, i])
    y = dave.measure(x) + z[:, i]

    dmu = aibot.perception(mu[:, i], y, t[i])
    du = aibot.action(mu[:, i], y)

# TODO: store values along simulation

# Animate movement
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
ax.set_aspect('equal')
ax.grid()

ref, = ax.plot([], [], 'ro', ms=6)
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    ref.set_data([], [])
    time_text.set_text('')
    return line, ref, time_text


def animate(i):
    x1, y1, x2, y2 = dave.joint_mapping(tuple(x[i, :]))

    x_ref, y_ref, _, _ = aibot.eef_reference(t[i])

    thisx = [0, x1, x2]
    thisy = [0, y1, y2]

    line.set_data(thisx, thisy)
    ref.set_data(x_ref, y_ref)

    time_text.set_text(time_template % (i * dt))

    return line, ref, time_text


ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),
                              interval=25, blit=True, init_func=init)
plt.show()
