
import rospy
import numpy as np



from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorRBFN import FunctionApproximatorRBFN
from dmpbbo.functionapproximators.FunctionApproximatorLWR import FunctionApproximatorLWR
from dmpbbo.functionapproximators.FunctionApproximatorWLS import FunctionApproximatorWLS

from lfd_dmp.dmp_wrapper import DMPWrapper

class DMPBBOService(DMPWrapper):

    def __init__(self):
        super().__init__()
        self.num_bases = 20

    def train(self, trajectory):

        # function_apps = [FunctionApproximatorRBFN(self.num_bases, 0.7) for _ in range(trajectory.dim)]
        function_apps = [FunctionApproximatorLWR(self.num_bases, 0.5) for _ in range(trajectory.dim)]
        # function_apps = [FunctionApproximatorWLS() for _ in range(trajectory.dim)]
        # Setup DMP
        name='DmpBbo'
        dmp_type='IJSPEERT_2002_MOVEMENT'
        # dmp_type='KULVICIUS_2012_JOINING'

        dmp = Dmp.from_traj(trajectory, function_apps, dmp_type="IJSPEERT_2002_MOVEMENT")

        return dmp

    def plan(self, ts):
        n_time_steps = ts.size
        dt = ts[1]

        (x,xd) = self.dmp.integrate_start()

        xs_step = np.zeros([n_time_steps,x.shape[0]])
        xds_step = np.zeros([n_time_steps,x.shape[0]])

        xs_step[0,:] = x
        xds_step[0,:] = xd
        for tt in range(1,n_time_steps):
            (xs_step[tt,:],xds_step[tt,:]) = self.dmp.integrate_step(dt,xs_step[tt-1,:])

        return self.dmp.states_as_trajectory(ts,xs_step,xds_step)
