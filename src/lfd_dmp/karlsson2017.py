
#%%
import numpy as np
from dmpbbo.dmp.Dmp import *
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem

class FunctionApproximatorkarlsson:
    def __init__(self, n_kernel, alpha_x):
        
        self.n_kernel = n_kernel
        
        # Centers (c) and width (1/d) of the guassians
        # c shape (:,1)
        ct = np.atleast_2d(np.linspace(0, 1, n_kernel)).T
        cx = np.exp(-alpha_x*3*ct)
        c = cx
        self.c = cx
        # d shape (:,1)
        d = np.power(np.diff(c,axis=0)*0.55,2)
        d = 1/np.append(d, d[-1])
        self.d = np.atleast_2d(d).T
        
    
    def train(self,inputs,targets, s):
        """
        train function approximator with the demonstration trajectory

        input : input x's of shape (:,1) or (:,)
        target : target f values associated with the inputs, shape (:,1)
        s :  the function's scaling term (not present in target values, but necessary fro training), shape (:,1)
        """
        
        inputs = np.squeeze(inputs)
        self.inputs = inputs
        self.targets = targets
        self.s = s

        p = inputs.shape[0]
        c = self.c
        d = self.d
        n_kernel = self.n_kernel
        
        psi = np.zeros((p,n_kernel))
        
        # Create the gaussians with c's and d's and evaluate them at inputs' value
        for t in range(0, p):
            psi[t,:] = np.squeeze(np.exp(-0.5*np.power(((inputs[t]-c)),2) * d))
            
        w = np.zeros((n_kernel,1))
        
        # Formula exactly like mentioned in the paper
        for i in range(0, n_kernel):
            gamma_i = np.diag(psi[:,i])
            w[i] = np.transpose(s) @ gamma_i @ targets @ np.linalg.inv(np.transpose(s) @ gamma_i @ s)
            
        self.w = w
        
        
    def predict(self, input):
        c = self.c
        d = self.d
        w = self.w
        
        psi = np.exp(-0.5*np.power(((input-c)),2) * d)
        f = np.sum(np.transpose(w) @ psi) / (np.sum(psi+ 10 ** -10))
        return f # scalar


class DMPkarlsson:
    
    def __init__(self, y0, g, tau, alpha_z, beta_z, alpha_x, n_kernel, kc, alpha_e, kp, kv):
        
        # number of DOFs, int
        self.dim = y0.size
        
        # Initial state, shape (dim,)
        self.y0 = y0
        
        # Goal state, shape (dim,)
        self.g = g
        
        # Timing of the trajectory, float64 (I have no idea why /3, but it's according to paper)
        self.tau = tau/3

        # Mass spring damper system parameters
        self.alpha_z = alpha_z
        self.beta_z = beta_z

        # Exponential phase system parameter
        self.alpha_x = alpha_x
        
        # Number of kernel for function approximator
        self.n_kernel = n_kernel

        # Perturbation control parameters
        self.kc = kc
        self.alpha_e = alpha_e
        
        # Control Params
        self.kp = kp
        self.kv = kv

        # Initialize karlsson function approximators as many as the dimension value
        self.function_approximators = []
        
        for i in range(0, self.dim):
            self.function_approximators.append(FunctionApproximatorkarlsson(self.n_kernel, self.alpha_x))

    @classmethod
    def from_traj(cls,trajectory, alpha_e, alpha_z, alpha_x,
                beta_z, kc, kp, kv, n_kernel):

        tau = trajectory.ts_[-1]
        y0 = trajectory.ys_[0,:]
        g = trajectory.ys_[-1,:]
        dmp = cls(y0, g, tau, alpha_z, beta_z, alpha_x, n_kernel,kc,alpha_e, kp, kv)
        dmp.train(trajectory)
      
        return dmp


    
    def train(self,trajectory):
        """
        Train DMP from the dmpbbo's trajectory object class
        """

        self.p = trajectory.ts_.size       
        
        self.f_target = np.power(self.tau,2)*trajectory.ydds_ - self.alpha_z*(self.beta_z*(self.g-trajectory.ys_) -self.tau*trajectory.yds_)
        
        # Initialize exponential phase system
        self.phase_system = ExponentialSystem(self.tau, 1,0,self.alpha_x)
        
        ## X integration, Analytical solution
        (self.xs,self.xds) = self.phase_system.analyticalSolution(trajectory.ts_)

        ### X integration matlab code style
        
        # self.xs = np.zeros((self.p,1))
        # self.xs[0] = 1
        
        # for t in range(1, self.p):
        #     dt = 1/250
        #     xdot = -self.alpha_x*self.xs[t-1]/self.tau
        #     self.xs[t] = self.xs[t-1] + xdot*dt
        
        # ## X integration, a more modern style
        # self.xs = np.zeros((self.p,1))
        # self.xds = np.zeros((self.p,1))
        
        # (self.xs[0],self.xds[0]) = self.phase_system.integrateStart()
        
        # for t in range(1, self.p):
        #     dt = 1/250
        #     (self.xs[t],self.xds[t]) = self.phase_system.integrateStep(dt,self.xs[t-1])
        
        # The target f scaling term
        self.s = self.xs*(self.g-self.y0)
        
        # Train function approximators
        for dd in range(0, self.dim):
            targets = self.f_target[:,dd][:,np.newaxis]
            ss = self.s[:,dd][:,np.newaxis]
            self.function_approximators[dd].train(self.xs,targets,ss)

    def integrateStart(self):
        """
        Start integration of DMP equation without perturbation
        """

        # reset the phase system, (:,) and (:,)
        (self.x,self.xd) = self.phase_system.integrateStart()

        # yd of the dmp, (:,)
        self.yd = np.zeros(self.dim)
        # y of dmp, (:,)
        self.y = self.y0
        #ydd of dmp (:,)
        self.ydd = np.zeros(self.dim)
        # Keep track of the timing
        self.t = 0
        
        return (self.x,self.y,self.yd,self.ydd)
        
    def set_tau(self, tau):
        # Again I have no idea why. It's according to the paper
        self.tau = tau/3
        self.phase_system.set_tau(self.tau)

    def get_tau(self):
        return (self.tau * 3)
        
    def set_initial_state(self, y0):
        self.y0 = y0
        
    def set_attractor_state(self,g):
        self.g = g
        
    def integrateStep(self, t):
        """
        Integrate the DMP equation without perturbation

        t : the target time, in seconds
        """
        # Step delta time
        self.dt = t - self.t
        self.t = t
        
        # Calculate yd
        self.dmp2acc()
        self.yd = self.yd + self.ydd*self.dt
        self.y = self.y + self.yd*self.dt
        (self.x,self.xd) = self.phase_system.integrateStep(self.dt,self.x)
        
        return (self.x,self.y,self.yd,self.ydd)
    
    def predict_f(self):
        self.f = np.zeros(self.dim)
        for i in range(0, self.dim):
            self.f[i] = self.function_approximators[i].predict(self.x)
            self.f[i] = self.f[i] * self.x * (self.g-self.y0)[i]

    def dmp2acc(self):
        self.predict_f()
        self.ydd = 1/(self.tau ** 2) * (self.alpha_z*(self.beta_z*(self.g-self.y) - self.tau*self.yd) + self.f)

    def get_ydd_a_lowgain_ff(self):
        self.ydd_a = self.kp*(self.y-self.y_a) + self.kv*(self.yd-self.yd_a) + self.ydd
    
    def dmp2vel_acc_ss(self):
        self.predict_f()
        self.z_dot = 1/self.tau_adapt * (self.alpha_z*(self.beta_z*(self.g-self.y) - self.z) + self.f)
        self.z = self.z + self.dt*self.z_dot
        self.yd = self.z/self.tau_adapt
        self.ydd = (self.z_dot*self.tau_adapt-self.tau*self.z*2*self.kc*self.e*(self.alpha_e*(self.y_a-self.y-self.e)))/self.tau_adapt**2

    
    def controlStart(self):

        # reset the phase system, (:,) and (:,)
        (self.x,self.xd) = self.phase_system.integrateStart()
        # DMP's yd (:,)
        self.yd = np.zeros(self.dim)
        # DMP's ydd (:,)
        self.ydd = np.zeros(self.dim)
        # DMP's y (:,)
        self.y = self.y0
        # Keep track of time
        self.t = 0
        # Actual y (:,)
        self.y_a = self.y0
        # Actual yd (:,)
        self.yd_a = np.zeros(self.dim)
        
        # Actual ydd (:,)
        self.ydd_a = np.zeros(self.dim)
        
        # Error between DMP and actual (:,)
        self.e = np.zeros(self.dim)
        # Z value in DMP equation (:,)
        self.z = np.zeros(self.dim)
        
        return (self.x, self.y, self.y_a)
    
    def controlStep(self,t,y_a_new,yd_a_new):
        """
        Calculate new control command based on the current time, position, and velocity

        t : current time value in seconds
        y_a_new : current actual position
        yd_a_new : current actual velocity
        """

        self.dt = t - self.t
        self.t = t
        self.yd_a = yd_a_new
        
        # Calculate the adapted value of Tau based on the last error value
        # (Return the maximum tau value along DOFs in order to make sure system is matched with the slowest DOF)
        self.tau_adapt = np.amax(self.tau * (1+(self.kc*(self.e**2))))

        # Calculate yd and ydd based on the last perturbations happened
        self.dmp2vel_acc_ss()

        # Calculate new y_a_dd command based on last position, the new velocity feedback, and calculated dmp yd and ydd
        self.get_ydd_a_lowgain_ff()

        self.y = self.y + self.yd*self.dt
        self.e_dot = self.alpha_e*(y_a_new-self.y-self.e)
        self.e = self.e + self.e_dot*self.dt
        
        # Update phase system's timing with the adapted tau
        self.phase_system.set_tau(self.tau_adapt)
        
        (self.x,self.xd) = self.phase_system.integrateStep(self.dt,self.x)
        self.y_a = y_a_new
        
        return (self.ydd_a, self.x, self.y, self.yd, self.ydd, self.tau_adapt)

    def statesAsTrajectory(self,ts, ys, yds, ydds):
        return Trajectory(ts,ys,yds,ydds)
