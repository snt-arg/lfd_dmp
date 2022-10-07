
import rospy
import actionlib
import numpy as np

from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState

from lfd_dmp.karlsson2017 import DMPkarlsson
from dmpbbo.dmp.Trajectory import *
from lfd_dmp.dmp_wrapper import DMPWrapper

from lfd_interface.msg import ControlLFDAction, ControlLFDResult

from arm_controllers.msg import ControlDataMsg
from lfd_dmp.msg import DMPDataMsg


class DMPkarlssonService(DMPWrapper):

    def __init__(self, dmpconfig):
        super().__init__()

        self.alpha_e = dmpconfig["alpha_e"]
        self.alpha_z = dmpconfig["alpha_z"]
        self.alpha_x = dmpconfig["alpha_x"]
        self.beta_z = dmpconfig["beta_z"]
        self.kc = np.array(dmpconfig["kc"])
        self.kp = np.array(dmpconfig["kp"])
        self.kv = np.array(dmpconfig["kv"])
        self.ki = np.array(dmpconfig["ki"])
        self.ki_limMaxInt = np.array(dmpconfig["ki_limMaxInt"])
        self.ki_limMinInt = np.array(dmpconfig["ki_limMinInt"])
        self.n_kernel = dmpconfig["n_kernel"]

        self.controller = DMPkarlssonController(self)


        self.as_control = actionlib.SimpleActionServer("control_lfd", ControlLFDAction, auto_start = False)
        self.as_control.register_goal_callback(self.acb_control_goal)
        self.as_control.register_preempt_callback(self.acb_control_preempt)
        self.as_control.start()

    def acb_control_goal(self):
        self.goal = self.as_control.accept_new_goal()
        self.init_dmp(self.goal.plan.name, self.goal.plan.start.positions, self.goal.plan.goal.positions, self.goal.plan.tau)
        self.controller.start_control()


    def acb_control_preempt(self):
        self.as_control.set_preempted()


    def set_goal_reached(self):
        if not self.as_control.is_active():
            return
        
        result = ControlLFDResult()
        result.success = True
        self.as_control.set_succeeded(result)
        rospy.loginfo("Target reached successfully!")

    def train(self, trajectory):
        dmp = DMPkarlsson.from_traj(trajectory, self.alpha_e, self.alpha_z,
                                    self.alpha_x, self.beta_z, self.kc, self.kp, 
                                    self.kv, self.n_kernel, self.ki, self.ki_limMinInt, self.ki_limMaxInt)
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


class DMPkarlssonController:

    def __init__(self, dmp_service : DMPkarlssonService):
        
        self.dmp_service = dmp_service

        self.pub_control = rospy.Publisher("control_command", ControlDataMsg, queue_size=1)
        self.pub_dmp = rospy.Publisher("dmp_feedback", DMPDataMsg, queue_size=1)

        self.sub_controller = None
        self.x = 1
    
    def start_control(self):
        self.dmp_service.dmp.controlStart()
        self.t_ref = rospy.get_time()
        if self.sub_controller is None:
            self.sub_controller = rospy.Subscriber("/joint_states", JointState, self.cb_control_loop, queue_size=1)
    
    def check_target_reached(self):
        if self.x < 0.003:
            self.dmp_service.set_goal_reached()

    def cb_control_loop(self, statemsg : JointState):
        t = rospy.get_time() - self.t_ref
        dmp_data, control_data = self.dmp_service.dmp.controlStep(t,statemsg.position,statemsg.velocity)

        now = rospy.Time.now()
        control_data.header.stamp = now
        dmp_data.header.stamp = now
        
        self.pub_control.publish(control_data)
        self.pub_dmp.publish(dmp_data)

        self.x = dmp_data.x.data
        self.check_target_reached()