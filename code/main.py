"""
Running this file:
  - checks the prediction matrices of Question 1,
  - simulates 200 steps with mpcfree, with and without noise (Question 4),
  - saves the data and the figures in ../results/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # save figures to files, without opening a window
import matplotlib.pyplot as plt

# All outputs (data and figures) go to ../results/, created if needed.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Dynamics: z_{t+1} = M z_t + B u_t + sigma * e_t
# The state is z_t = (x_t, v_t) in R^4:
#   entries 0, 1 : position x_t in the plane,
#   entries 2, 3 : velocity v_t.
# The control u_t in R^2 is an acceleration: it only acts on the velocity.
# ---------------------------------------------------------------------
M = np.block([[np.eye(2), np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
B = np.vstack([np.zeros((2, 2)), np.eye(2)])


def prediction_matrices(T):
    """
    Question 1: build A (4T x 2T) and C (4T x 4) such that z = A u + C z_t,
    where u stacks the T future controls and z the T predicted states.

    We use M = N kron I2 and B = b kron I2, with N = [[1, 1], [0, 1]] and
    b = [0, 1]. By the mixed-product property, A = At kron I2 and
    C = Ct kron I2, where At and Ct are the small matrices built below.
    """
    At = np.zeros((2*T, T))
    Ct = np.zeros((2*T, 2))
    for k in range(1, T + 1):          # block row k  <->  predicted state z_{t+k}
        Ct[2*(k-1):2*k, :] = [[1, k], [0, 1]]      # N^k
        for j in range(k):             # block column j  <->  control u_{t+j}
            At[2*(k-1):2*k, j] = [k - 1 - j, 1]    # N^(k-1-j) b
        # Columns j >= k stay zero: future controls do not affect z_{t+k}.
    I2 = np.eye(2)
    return np.kron(At, I2), np.kron(Ct, I2)


def check_prediction_matrices(T=5, seed=0):
    """
    Question 1, check: for random z_t and u, compare A u + C z_t with the
    states obtained by applying z_{s+1} = M z_s + B u_s step by step.
    Returns the norm of the difference (should be at machine precision).
    """
    rng = np.random.default_rng(seed)
    A, C = prediction_matrices(T)
    zt = rng.standard_normal(4)
    u = rng.standard_normal(2*T)
    z, Z = zt, []
    for i in range(T):
        z = M @ z + B @ u[2*i:2*i+2]   # u[2i:2i+2] is the control u_{t+i}
        Z.append(z)
    return np.linalg.norm(np.concatenate(Z) - (A @ u + C @ zt))


def mpcfree(zt, T, A=None, C=None, return_plan=False):
    """
    Question 3: unconstrained MPC controller.

    Inputs:
      zt          current state in R^4
      T           horizon (T >= 1)
      A, C        (optional) prediction matrices of Question 1. They depend
                  on T only, not on zt, so they can be built once and passed
                  in, instead of being rebuilt at every step of Question 4.
                  If not given, they are built here.
      return_plan (optional) if True, also return the full optimal plan u*.

    Output:
      ut          first optimal control in R^2 (the only one applied)
      u_star      full optimal plan in R^{2T}, only if return_plan is True
    """
    if A is None or C is None:
        A, C = prediction_matrices(T)
    # Question 2: the unique minimizer of 1/2 ||A u + C zt||^2 is obtained
    # by one direct least-squares solve (equivalent to one Newton step).
    u_star = np.linalg.lstsq(A, -C @ zt, rcond=None)[0]
    ut = u_star[:2]                    # first control u_t = first 2 entries
    return (ut, u_star) if return_plan else ut


def simulate(controller, z0, T, sigma, n_steps=200, seed=0):
    """
    Question 4: closed-loop MPC simulation over n_steps steps.

    At each step:
      1. compute u_t from the current (observed) state z_t,
      2. draw the noise e_t ~ N(0, I4), AFTER u_t has been chosen,
      3. apply the true dynamics z_{t+1} = M z_t + B u_t + sigma * e_t.

    Returns:
      Z  array (n_steps+1) x 4, row t = state z_t
         (columns 0, 1 = position x_t ; columns 2, 3 = velocity v_t)
      U  array n_steps x 2, row t = applied control u_t
    """
    rng = np.random.default_rng(seed)  # fixed seed: reproducible noise
    A, C = prediction_matrices(T)      # built once, outside the loop
    Z = np.zeros((n_steps + 1, 4))
    U = np.zeros((n_steps, 2))
    Z[0] = z0
    for t in range(n_steps):
        U[t] = controller(Z[t], T, A, C)
        e = rng.standard_normal(4)
        Z[t + 1] = M @ Z[t] + B @ U[t] + sigma * e
    return Z, U


if __name__ == "__main__":
    # ---------------- Question 1: check ----------------
    print("Q1 check, ||z_sim - (A u + C z_t)|| =", check_prediction_matrices())

    # ---------------- Question 4: parameters of the statement ----------------
    z0 = np.array([1.0, 0.0, 0.0, 0.25])     # x0 = (1, 0), v0 = (0, 0.25)
    T, sigma, n_steps = 10, 0.01, 200

    # Run requested by the statement (sigma = 0.01), and a noiseless run
    # (sigma = 0) used only as a reference, to separate the effect of the
    # controller from the effect of the noise.
    Z_noisy, U_noisy = simulate(mpcfree, z0, T, sigma, n_steps)
    Z_clean, U_clean = simulate(mpcfree, z0, T, 0.0, n_steps)

    # Save the raw data (states and controls of both runs).
    np.savez(os.path.join(RESULTS_DIR, "q4_mpcfree_run.npz"),
             Z_noisy=Z_noisy, U_noisy=U_noisy, Z_clean=Z_clean, U_clean=U_clean)

    # Quantities used in the plots and in the report.
    t_z = np.arange(n_steps + 1)                 # time steps of the states
    nz_noisy = np.linalg.norm(Z_noisy, axis=1)   # ||z_t||, one value per step
    nz_clean = np.linalg.norm(Z_clean, axis=1)
    nu_noisy = np.abs(U_noisy).max(axis=1)       # largest entry of u_t

    # ---------------- Plot 1: distance to the target state ----------------
    # Log scale: ||z_t|| varies over many orders of magnitude.
    # The noiseless curve reaches exactly 0, so 1e-16 is added to avoid log(0).
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(t_z, nz_noisy, label=r"$\sigma = 0.01$")
    ax.semilogy(t_z, nz_clean + 1e-16, "--", label=r"$\sigma = 0$")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"$\|z_t\|$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "q4_state_norm.pdf"))
    plt.close(fig)

    # ---------------- Plot 2: trajectory of the robot in the plane ----------------
    # The position x_t is given by the first two columns of Z:
    # Z[:, 0] = first coordinate, Z[:, 1] = second coordinate.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(Z_noisy[:, 0], Z_noisy[:, 1], "-", lw=0.8, label=r"$\sigma = 0.01$")
    ax.plot(Z_clean[:, 0], Z_clean[:, 1], "--", lw=1.2, label=r"$\sigma = 0$")
    ax.plot(*z0[:2], "ko", ms=5, label=r"start $x_0$")   # initial position
    ax.plot(0, 0, "k+", ms=10, mew=1.5, label="origin")    # target position
    ax.set_xlabel(r"$(x_t)_1$")
    ax.set_ylabel(r"$(x_t)_2$")
    ax.set_aspect("equal", adjustable="datalim")         # same scale on both axes
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    # Inset: zoom around the origin for t >= 20, once the initial transient is
    # over. At the scale of the full trajectory, the noise-driven motion would
    # be invisible.
    axins = ax.inset_axes([0.06, 0.56, 0.38, 0.38])
    axins.plot(Z_noisy[20:, 0], Z_noisy[20:, 1], "-", lw=0.6)
    axins.plot(0, 0, "k+", ms=8, mew=1.2)
    axins.set_title(r"zoom, $t \geq 20$", fontsize=7)
    axins.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "q4_trajectory.pdf"))
    plt.close(fig)

    # ---------------- Numbers quoted in the report ----------------
    print("||z_t|| (noisy) at t = 0, 1, 2, 5, 10:", np.round(nz_noisy[[0, 1, 2, 5, 10]], 4))
    print("||z_t|| (clean) at t = 0, 1, 2, 5, 10:", np.round(nz_clean[[0, 1, 2, 5, 10]], 6))
    print("mean / max ||z_t|| (noisy) for t >= 20:",
          round(nz_noisy[20:].mean(), 4), "/", round(nz_noisy[20:].max(), 4))
    print("max |u_t| at t = 0 (noisy):", round(nu_noisy[0], 4))
    print("max |x_t| (noisy) for t >= 20:", round(np.abs(Z_noisy[20:, :2]).max(), 4))
    print("median of max|u_t| (noisy) for t >= 20:", round(np.median(nu_noisy[20:]), 4))