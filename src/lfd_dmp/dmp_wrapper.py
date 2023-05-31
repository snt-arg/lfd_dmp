
import rospy
import numpy as np
import matplotlib.pyplot as plt

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lfd_interface.srv import TrainDemonstration, TrainDemonstrationRequest, TrainDemonstrationResponse
from lfd_interface.srv import PlanLFD, PlanLFDRequest, PlanLFDResponse


from dmpbbo.dmps.Trajectory import Trajectory

class DMPWrapper:

    def __init__(self):
        #DMP Parameters
        self.dt = 0.05
        self.trained_dmps = {}
        self.trained_demos = {}

        self.server_train_demonstration = rospy.Service("train_demonstration",TrainDemonstration, self.cb_train_demonstration)
        self.server_plan_dmp = rospy.Service("plan_lfd",PlanLFD, self.cb_plan_dmp)

    def create_trajectory(self, demonstration):
        n_time_steps = len(demonstration.joint_trajectory.points)
        n_dim = len(demonstration.joint_trajectory.joint_names)

        path = np.zeros([n_time_steps,n_dim])
        ts = np.zeros(n_time_steps)

        for (i,point) in enumerate(demonstration.joint_trajectory.points):
            path[i,:] = point.positions
            ts[i] = point.time_from_start.to_sec()

        return Trajectory(ts, path)

    def train(self, trajectory):
        raise NotImplementedError()

    def cb_train_demonstration(self, req: TrainDemonstrationRequest):
        self.joint_names = req.demonstration.joint_trajectory.joint_names
        traj = self.create_trajectory(req.demonstration)

        dmp = self.train(traj)

        # Store Trained DMP
        self.trained_dmps[req.demonstration.name] = dmp
        self.trained_demos[req.demonstration.name] = traj

        rospy.loginfo("dmp training completed successfully")
        return TrainDemonstrationResponse(True)

    def plan(self, ts):
        raise NotImplementedError()

    def init_dmp(self, name, start, goal, tau):
        self.dmp = self.trained_dmps[name]
        self.dmp.tau = tau
        if len(start) != 0:
            self.dmp.y_init = np.array(start)
        if len(goal) != 0:
            self.dmp.y_attr = np.array(goal) 

    def cb_plan_dmp(self, req : PlanLFDRequest):
        self.init_dmp(req.plan.name, req.plan.start.positions, req.plan.goal.positions, req.plan.tau)
        tau = req.plan.tau

        n_time_steps = int(np.ceil(tau/self.dt) + 1)
        ts = np.linspace(0,tau,n_time_steps)

        traj_reproduced = self.plan(ts)

        self.plot(self.trained_demos[req.plan.name], traj_reproduced)

        plan_path = JointTrajectory()
        plan_path.header.frame_id = "world"

        for i in range(0,n_time_steps):
            pt = JointTrajectoryPoint()
            pt.positions = traj_reproduced.ys[i,:]
            pt.velocities = traj_reproduced.yds[i,:]
            pt.accelerations = traj_reproduced.ydds[i,:]
            pt.time_from_start = rospy.Duration.from_sec(traj_reproduced.ts[i])
            plan_path.points.append(pt)

        plan_path.joint_names = self.joint_names

        return PlanLFDResponse(plan_path)
    
    def plot(self, demo, plan):
        lines, axs = demo.plot()
        plt.setp(lines, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
        plt.setp(lines, label="demonstration")

        lines, axs = plan.plot(axs)
        plt.setp(lines, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5))
        plt.setp(lines, label="reproduced")

        plt.legend()
        t = f"Comparison between demonstration and reproduced"
        plt.gcf().canvas.set_window_title(t)
        plt.show()
