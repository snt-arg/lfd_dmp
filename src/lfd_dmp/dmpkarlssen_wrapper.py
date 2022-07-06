
import rospy
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lfd_interface.srv import TrainDemonstration, TrainDemonstrationRequest, TrainDemonstrationResponse
from lfd_interface.srv import PlanLFD, PlanLFDRequest, PlanLFDResponse

from lfd_dmp.karlssen2017 import DMPKarlssen
from dmpbbo.dmp.Trajectory import *

class DMPKarlssenService:

    def __init__(self):
        #DMP Parameters
        self.num_bases = 10
        self.dt = 0.2
        self.trained_dmps = {}

        self.server_train_demonstration = rospy.Service("train_demonstration",TrainDemonstration, self.cb_train_demonstration)
        self.server_plan_dmp = rospy.Service("plan_lfd",PlanLFD, self.cb_plan_dmp)

    def cb_train_demonstration(self, req: TrainDemonstrationRequest):
        self.joint_names = req.demonstration.joint_trajectory.joint_names
        # Setup Trajectory
        n_time_steps = len(req.demonstration.joint_trajectory.points)
        n_dim = len(req.demonstration.joint_trajectory.joint_names)

        path = np.zeros([n_time_steps,n_dim])
        ts = np.zeros(n_time_steps)

        for (i,point) in enumerate(req.demonstration.joint_trajectory.points):
            path[i,:] = point.positions
            ts[i] = point.time_from_start.to_sec()

        traj = Trajectory(ts, path)

        dmp = DMPKarlssen.from_traj(traj)

        # Store Trained DMP
        self.trained_dmps[req.demonstration.name] = dmp

        rospy.loginfo("dmpbbo training completed successfully")
        return TrainDemonstrationResponse(True)


    def cb_plan_dmp(self, req : PlanLFDRequest):
        self.dmp = self.trained_dmps[req.name]
        tau = self.dmp.tau * 3.5
        start = req.start.positions
        goal = req.goal.positions

        self.dmp.set_tau(tau)
        self.dmp.set_initial_state(np.array(start))
        self.dmp.set_attractor_state(np.array(goal))

        n_time_steps = int(np.ceil(tau/self.dt) + 1)
        ts = np.linspace(0,tau,n_time_steps)
        
        ys = np.zeros((n_time_steps, self.dmp.dim))
        yds = np.zeros((n_time_steps, self.dmp.dim))
        ydds = np.zeros((n_time_steps, self.dmp.dim))

        (_, ys[0], yds[0], ydds[0]) = self.dmp.integrateStart(tau)

        for i in range(1, n_time_steps):
            (_, ys[i], yds[i], ydds[i]) = self.dmp.integrateStep(ts[i])

        traj_reproduced = self.dmp.statesAsTrajectory(ts,ys,yds,ydds)

        plan_path = JointTrajectory()
        plan_path.header.frame_id = "world"

        for i in range(0,n_time_steps):
            pt = JointTrajectoryPoint()
            pt.positions = traj_reproduced.ys_[i,:]
            pt.velocities = traj_reproduced.yds_[i,:]
            pt.accelerations = traj_reproduced.ydds_[i,:]
            pt.time_from_start = rospy.Duration.from_sec(traj_reproduced.ts_[i])
            plan_path.points.append(pt)

        plan_path.joint_names = self.joint_names

        return PlanLFDResponse(plan_path)