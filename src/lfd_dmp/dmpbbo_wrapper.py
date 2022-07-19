
import rospy
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lfd_interface.srv import TrainDemonstration, TrainDemonstrationRequest, TrainDemonstrationResponse
from lfd_interface.srv import PlanLFD, PlanLFDRequest, PlanLFDResponse

from dmpbbo.dmp.Dmp import *
from dmpbbo.functionapproximators.FunctionApproximatorLWR import *
from lfd_dmp.dmp_wrapper import DMPWrapper

class DMPBBOService(DMPWrapper):

    def __init__(self):
        super().__init__()
        self.num_bases = 10

    def train(self, trajectory):
        n_dim = trajectory.dim()

        # Setup Function Approximators
        function_apps = []
        for i in range(0,n_dim):
            function_apps.append(FunctionApproximatorLWR(self.num_bases))

        # Setup DMP
        name='DmpBbo'
        dmp_type='IJSPEERT_2002_MOVEMENT'
        # dmp_type='KULVICIUS_2012_JOINING'

        dmp = Dmp.from_traj(trajectory, function_apps, name, dmp_type)

        return dmp

    def plan(self, ts):
        n_time_steps = ts.size
        dt = ts[1]
        xs_step = np.zeros([n_time_steps,self.dmp.dim_])
        xds_step = np.zeros([n_time_steps,self.dmp.dim_])

        (x,xd) = self.dmp.integrateStart()

        xs_step[0,:] = x
        xds_step[0,:] = xd
        for tt in range(1,n_time_steps):
            (xs_step[tt,:],xds_step[tt,:]) = self.dmp.integrateStep(dt,xs_step[tt-1,:])

        return self.dmp.statesAsTrajectory(ts,xs_step,xds_step)
