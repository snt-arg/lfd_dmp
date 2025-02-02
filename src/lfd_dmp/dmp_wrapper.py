
import rospy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from copy import deepcopy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lfd_interface.srv import TrainDemonstration, TrainDemonstrationRequest, TrainDemonstrationResponse
from lfd_interface.srv import PlanLFD, PlanLFDRequest, PlanLFDResponse


from dmpbbo.dmps.Trajectory import Trajectory

class DMPWrapper:

    def __init__(self, robot_ns):
        #DMP Parameters
        self.n_time_steps = 100
        self.trained_dmps = {}
        self.trained_demos = {}

        self.server_train_demonstration = rospy.Service(f"{robot_ns}/train_demonstration",TrainDemonstration, self.cb_train_demonstration)
        self.server_plan_dmp = rospy.Service(f"{robot_ns}/plan_lfd",PlanLFD, self.cb_plan_dmp)

    def create_trajectory(self, demonstration):
        n_time_steps = len(demonstration.joint_trajectory.points)
        n_dim = len(demonstration.joint_trajectory.joint_names)

        ys = np.zeros([n_time_steps,n_dim])
        yds = np.zeros([n_time_steps,n_dim])
        ydds = np.zeros([n_time_steps,n_dim])
        ts = np.zeros(n_time_steps)

        for (i,point) in enumerate(demonstration.joint_trajectory.points):
            ys[i,:] = point.positions
            yds[i,:] = point.velocities or None
            ydds[i,:] = point.accelerations or None
            ts[i] = point.time_from_start.to_sec()

        if np.isnan(yds).any(): 
            yds = None
        if np.isnan(ydds).any(): 
            ydds = None

        return Trajectory(ts, ys, yds, ydds)

    def train(self, trajectory):
        raise NotImplementedError()

    def cb_train_demonstration(self, req: TrainDemonstrationRequest):
        self.joint_names = req.demonstration.joint_trajectory.joint_names
        traj = self.create_trajectory(req.demonstration)

        dmp = self.train(traj)

        demo_alias = req.demonstration.trajectory_type + req.demonstration.name
        # Store Trained DMP
        self.trained_dmps[demo_alias] = dmp
        self.trained_demos[demo_alias] = traj

        rospy.loginfo("dmp training completed successfully")
        return TrainDemonstrationResponse(True)

    def plan(self, ts):
        raise NotImplementedError()

    def init_dmp(self, alias, start, goal, tau_scale):
        self.dmp = deepcopy(self.trained_dmps[alias])
        self.dmp.tau *= tau_scale
        if len(start) != 0:
            self.dmp.y_init = np.array(start)
        if len(goal) != 0:
            self.dmp.y_attr = np.array(goal)
        return self.dmp.tau

    def cb_plan_dmp(self, req : PlanLFDRequest):
        demo_alias = req.plan.trajectory_type + req.plan.name
        tau = self.init_dmp(demo_alias, req.plan.start.positions, req.plan.goal.positions, req.plan.tau)

        plan_tau = tau
        # plan_tau = tau+ 0.18
        n_time_steps = self.n_time_steps
        ts = np.linspace(0,plan_tau,n_time_steps)

        traj_reproduced = self.plan(ts)

        self.plot(self.trained_demos[demo_alias], traj_reproduced)

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

        # plan_path.points[0].accelerations[1:] = [0.0,0.0,0.0,0.0,0.0,0.0]

        with open("/tmp/{}.pickle".format(req.plan.name), 'wb') as file:
            pickle.dump(plan_path,file)

        return PlanLFDResponse(plan_path)
    
    def plot(self, demo, plan):
        lines, axs = demo.plot()
        plt.setp(lines, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
        # plt.setp(lines, label="demonstration")

        lines, axs = plan.plot(axs)
        plt.setp(lines, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5))
        # plt.setp(lines, label="reproduced")
        plt.tight_layout()

        plt.legend()
        t = f"Comparison between demonstration and reproduced"
        plt.gcf().canvas.manager.set_window_title(t)

        current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = '/tmp/' + f'output_dmp_{current_datetime}.svg'
        plt.savefig(filename, format='svg')
        # plt.show()
