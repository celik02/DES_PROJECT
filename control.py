
import casadi as cd
import numpy as np
# from circle_fit import taubinSVD
from scipy.integrate import odeint,solve_ivp,ode
import cvxpy as cp

lf = 1.0
lr = 1.0
a0 = 0
a1 = 0
a2 = 0
Mass = 1
dt = 0.15
def dynamics(t, x):
    dx = [0] * 6
    beta = 0
    dx[0] = x[3] * np.cos(x[2] + beta)
    dx[1] = x[3] * np.sin(x[2] + beta)
    dx[2] = (x[3] * x[5]) / (lf + lr)
    dx[3] = -1/Mass * (a0*np.sign(x[3])+ a1*x[3] + a2*x[3]**2) + 1 / Mass * x[4]
    dx[4] = 0
    dx[5] = 0
    return dx


def dynamics1(t, x):
    dx = [0] * 6
    beta = 0
    dx[0] = x[3] * np.cos(x[2])
    dx[1] = x[3] * np.sin(x[2])
    dx[2] = (x[3] * x[5]) / (lf + lr)
    dx[3] = x[4]
    dx[4] = 0
    dx[5] = 0
    return dx
def motion(dynamics, state, Input, timespan, teval):
    y0 = np.append(state, Input)
    sol = solve_ivp(dynamics, timespan, y0, method = 'DOP853', t_eval=[teval], atol=1e-6)
    x = np.reshape(sol.y[0:len(state)], len(state))
    return x

def rk4( t, state, Input, n):
    state = np.append(state, Input)
    # Calculating step size
    # x0 = np.append(state, Input)
    h = np.array([(t[-1] - t[0]) / n])
    t0 = t[0]
    for i in range(n):
        k1 = np.array(dynamics(t0, state))
        k2 = np.array(dynamics((t0 + h / 2), (state + h * k1 / 2)))
        k3 = np.array(dynamics((t0 + h / 2), (state + h * k2 / 2)))
        k4 = np.array(dynamics((t0 + h), (state + h * k3)))
        k = np.array(h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        # k = np.array(h * (k1))
        xn = state + k
        state = xn
        t0 = t0 + h

    return xn[0:4]



def mpc_exec(Psi_ref, states, states_ip, index, radius):
    dt = 0.1
    statesnumber = 4
    Inputnumber = 2
    N = 2
    params = 20 * [0.5]
    x0,y0,psi0,v0 = states
    opti = cd.Opti()
    umin = -0.6 * 9.81 * Mass
    umax = 0.4 * 9.81 * Mass

    steermin = -0.6
    steermax = 0.6

    X = opti.variable(statesnumber, N + 1)
    u = opti.variable(Inputnumber + 2, N)
    s = opti.variable(4, N)


    curr_states = states_ip
    for k in range(0,N):
        a = 2
        b = 3
        xip, yip, psi_ip, vip = curr_states
        b3 = (X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2 - 4**2
        lfb3 = (2 * ((X[0, k] - xip) ** 2 / a + (X[1, k] - yip) ** 2 / b - X[3, k] ** 2 - 4**2) +
                (X[3, k] * cd.cos(X[2, k]) * (2 * X[0, k] - 2 * xip)) / a -
                (vip * cd.cos(psi_ip) * (2 * X[0, k] - 2 * xip)) / a +
                (X[3, k] * cd.sin(X[2, k]) * (2 * X[1, k] - 2 * yip)) / b -
                (vip * cd.sin(psi_ip) * (2 * X[1, k] - 2 * yip)) / b) + 2 * (X[3, k]) * (+1/Mass * (a0*cd.sign(X[3, k])+ a1*X[3, k] + a2*X[3, k]**2))
        lgb3u = -2 * X[3, k] * 1 / Mass
        lgb3delta = 0

        opti.subject_to(lfb3 + lgb3u * u[0, k] + lgb3delta * u[1, k] + s[2, k] >= 0)
        curr_states = motion(dynamics, curr_states, [0, 0], [0, dt], dt)



    for k in range(0, N):
        b = X[3, k] - 0
        lfb = params[14] * b
        lfb = 1 * b
        lgu = 1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

        b = -X[3, k] + 50
        lfb = params[15] * b
        lfb = 1 * b
        lgu = -1
        opti.subject_to(lfb + lgu * u[0, k] >= 0)

    # Define the cost function
    cost = 0
    diag_elements_u = [params[16], params[17], 1 * params[18], params[19]]
    u_ref = np.zeros((Inputnumber + 2, N))

    normalization_factor = [max(-umin, umax), max(steermax, -steermin), 100,
                            0.4]
    for i in range(Inputnumber + 2):
        for h in range(N):
            cost += 0.5 * diag_elements_u[i] * ((u[i, h] - u_ref[i][h]) / normalization_factor[i]) ** 2
            cost += 10 ** 8 * (s[0, h]) ** 2
            cost += 10 ** 8 * (s[1, h]) ** 2
            cost += 10 ** 8 * (s[2, h]) ** 2
            cost += 10 ** 8 * (s[3, h]) ** 2

    opti.subject_to(umin <= u[0, :])
    opti.subject_to(u[0, :] <= umax)
    eps3 = 1
    for k in range(0, N):
        phi = 1
        v_ref = 18
        free_driving_threshold = phi * v_ref
        relative_distance = np.sqrt((states_ip[0] - x0) ** 2 + (states_ip[1] - y0) ** 2)
        if relative_distance >= free_driving_threshold:
            v_des = v_ref
        else:
            v_des = max(0, (v_ref - 0)/(free_driving_threshold - 4)*(relative_distance - 4) + 0)

        v_des = min(v_des, np.sqrt(radius * .5 * 400)/3.6)
        V = (X[3, k] - v_des) ** 2

        lfV =  eps3 * V + 2 * (X[3, k] - v_des) * (-1/Mass * (a0*cd.sign(X[3, k])+ a1*X[3, k] + a2*X[3, k]**2))
        lgu = 2 * (X[3, k] - v_des) * 1/Mass
        opti.subject_to(lfV + lgu * u[0, k] - u[2, k] <= 0)

        V = (X[2, k] - Psi_ref[0]) ** 2
        lfV = 10 * eps3 * V
        lgdelta = 2 * (X[2, k] - Psi_ref[0]) * X[3, k] / (lr + lf)
        opti.subject_to(lfV + lgdelta * u[1, k] - u[3, k] <= 0)

    opti.subject_to(steermin <= u[1, :])
    opti.subject_to(u[1, :] <= steermax)

    opti.subject_to(X[:, 0] == states)  # initialize states
    timespan = [0, dt]

    for h in range(N):  # initial guess
        opti.set_initial(X[:, h], [x0, y0, Psi_ref[0], v0])

    for k in range(N):
        state = []
        Input = []
        for j in range(statesnumber):
            state.append(X[j, k])
        for j in range(0, Inputnumber):
            Input.append(u[j, k])
        state = rk4(timespan, state, Input, 1)
        for j in range(statesnumber):
            opti.subject_to(X[j, k + 1] == state[j])


    opts = {}
    opts['print_time'] = False
    opts['ipopt.print_level'] = False
    opti.solver('ipopt', opts)
    opti.minimize(cost)
    sol = opti.solve()

    if N > 1:
        acc = 1 / Mass * sol.value(u)[0, 0]
        model_acc = -1 / Mass * (a0 * np.sign(v0) + a1 * v0 + a2* v0 ** 2) + 1 / Mass * sol.value(u)[0, 0]
        return "solution found", [acc, sol.value(u)[1, 0]], sol.value(X)[:,1], model_acc
    else:
        if np.any(sol.value(s) > 0.1):  # or LeftCBF <= 0 or RightCBF <= 0:
            s_vars = [0.1] * 4
            acc = -6
            model_acc = -1 / Mass * (0.1 * np.sign(v0) + 5 * v0 + 0.25 * v0 ** 2) + 1 / Mass * acc
            next_states = motion(dynamics, states, [acc, 0.0], [0, dt], dt)
            return "No solution found", np.array([acc, 0.0]), next_states, model_acc
        else:
            acc = 1 / Mass * sol.value(u[0])
            model_acc = -1 / Mass * (a0 * np.sign(v0) + a1 * v0 + a2 * v0 ** 2) + 1 / Mass * acc
            return "solution found", sol.value(u[:2]),sol.value(X)[:,1], model_acc



