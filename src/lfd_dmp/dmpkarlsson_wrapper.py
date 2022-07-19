
import rospy
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lfd_interface.srv import TrainDemonstration, TrainDemonstrationRequest, TrainDemonstrationResponse
from lfd_interface.srv import PlanLFD, PlanLFDRequest, PlanLFDResponse

from lfd_dmp.karlsson2017 import DMPkarlsson
from dmpbbo.dmp.Trajectory import *
from lfd_dmp.dmp_wrapper import DMPWrapper

class DMPkarlssonService(DMPWrapper):

    def __init__(self):
        super().__init__()

        self.alpha_e = 5
        self.alpha_z = 25
        self.alpha_x = 1
        self.beta_z = 6.25
        self.kc = 10000
        self.kp = 25
        self.kv = 10
        self.n_kernel = 15

    def train(self, trajectory):
        dmp = DMPkarlsson.from_traj(trajectory, self.alpha_e, self.alpha_z,
                                    self.alpha_x, self.beta_z, self.kc, self.kp, 
                                    self.kv, self.n_kernel)
        return dmp

    def plan(self, ts):
        n_time_steps = ts.size
        ys = np.zeros((n_time_steps, self.dmp.dim))
        yds = np.zeros((n_time_steps, self.dmp.dim))
        ydds = np.zeros((n_time_steps, self.dmp.dim))

        (_, ys[0], yds[0], ydds[0]) = self.dmp.integrateStart()

        for i in range(1, n_time_steps):
            (_, ys[i], yds[i], ydds[i]) = self.dmp.integrateStep(ts[i])

        return self.dmp.statesAsTrajectory(ts,ys,yds,ydds)