
import rospy
import actionlib
import numpy as np

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from lfd_dmp.karlsson2017 import DMPkarlsson
from dmpbbo.dmp.Trajectory import *
from lfd_dmp.dmp_wrapper import DMPWrapper

from lfd_interface.srv import PlanLFD, PlanLFDRequest, PlanLFDResponse
from lfd_interface.msg import ControlLFDAction, ControlLFDResult

class DMPkarlssonService(DMPWrapper):

    def __init__(self):
        super().__init__()

        self.alpha_e = 5
        self.alpha_z = 25
        self.alpha_x = 1
        self.beta_z = 6.25
        self.kc = 10000
        self.kp = 3000
        self.kv = 2 * np.sqrt(self.kp)
        self.n_kernel = 15

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


class DMPkarlssonController:

    def __init__(self, dmp_service : DMPkarlssonService):
        
        self.dmp_service = dmp_service
        self.pub_ydd = rospy.Publisher("ydd_control", Float64MultiArray, queue_size=1)
        self.sub_controller = None
        self.x = 1
    
    def start_control(self):
        self.dmp_service.dmp.controlStart()
        self.t_ref = rospy.get_time()
        if self.sub_controller is None:
            self.sub_controller = rospy.Subscriber("/joint_states", JointState, self.cb_control_loop, queue_size=1)
    
    def check_target_reached(self):
        if self.x < 0.001:
            self.dmp_service.set_goal_reached()

    def cb_control_loop(self, statemsg : JointState):
        t = rospy.get_time() - self.t_ref
        ydd_a,self.x,_,_ = self.dmp_service.dmp.controlStep(t,statemsg.position,statemsg.velocity)
        u = Float64MultiArray()
        u.data = ydd_a
        self.pub_ydd.publish(u)
        self.check_target_reached()