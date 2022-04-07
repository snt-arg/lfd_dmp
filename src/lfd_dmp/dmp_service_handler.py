
import roslib; 
roslib.load_manifest('dmp')
import rospy

import numpy as np

from dmp.srv import *
from dmp.msg import *

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lfd_interface.srv import TrainDemonstration, TrainDemonstrationRequest, TrainDemonstrationResponse
from lfd_interface.srv import PlanLFD, PlanLFDRequest, PlanLFDResponse

class DMPService:

    def __init__(self):
        #DMP Parameters
        self.K = 200
        self.D = 2.0 * np.sqrt(self.K)
        self.num_bases = 8
        self.dt_default = 1.0

        self.integrate_iter = 5     #dt is rather large, so this is > 1  
        self.seg_length = 20        #-1 for planning until convergence to goal
        self.goal_thresh = 0.2      # to be duplicated as a list with a length of DOF

        self.trained_dmps = {}

        self.server_train_demonstration = rospy.Service("train_demonstration",TrainDemonstration, self.cb_train_demonstration)
        self.server_plan_dmp = rospy.Service("plan_lfd",PlanLFD, self.cb_plan_dmp)

    def cb_train_demonstration(self, req: TrainDemonstrationRequest):
        print(req.demonstration.joint_trajectory)
        joint_path = []

        self.joint_names = req.demonstration.joint_trajectory.joint_names
        self.dof = len(req.demonstration.joint_trajectory.points[0].positions)

        for waypoint in req.demonstration.joint_trajectory.points:
            joint_path.append(list(waypoint.positions))

        train_resp = self.train_dmp(joint_path)

        #add the dmp to the list of trained dmps for further use
        self.trained_dmps[req.demonstration.name] = train_resp

        rospy.loginfo("dmp training completed successfully")

        return TrainDemonstrationResponse(True)
    
    #Set a DMP as active for planning
    def activate_dmp(self, dmp_list):
        try:
            sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
            sad(dmp_list)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)

    
    def train_dmp(self, joint_path):
        demo_trajectory = DMPTraj()

        for i , waypoint in enumerate(joint_path):
            dmp_point = DMPPoint()
            dmp_point.positions = waypoint
            demo_trajectory.points.append(dmp_point)
            demo_trajectory.times.append(self.dt_default * i)
        
        k_gains = [self.K] * self.dof
        d_gains = [self.D] * self.dof

        rospy.loginfo("Starting DMP Training")
        rospy.wait_for_service('learn_dmp_from_demo')

        try:
            lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
            resp = lfd(demo_trajectory, k_gains, d_gains, self.num_bases)

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)
        
        rospy.loginfo("DMP Training Finished")   
                    
        return resp     


    def cb_plan_dmp(self, req : PlanLFDRequest):
        rospy.loginfo("entered plan callback")

        self.activate_dmp(self.trained_dmps[req.name].dmp_list)


        start = req.start.positions
        goal = req.goal.positions
        tau = self.trained_dmps[req.name].tau

        #print(start)
        #print(goal)

        planning_resp = self.plan_dmp(start,goal,tau)
        #print (planning_resp)
        plan_path = JointTrajectory()

        for point in planning_resp.plan.points:
            pt = JointTrajectoryPoint()
            pt.positions = point.positions
            pt.velocities = point.velocities
            plan_path.points.append(pt)
        
        plan_path.joint_names = self.joint_names

        print(plan_path)
        return PlanLFDResponse(plan_path)


    def plan_dmp(self, start, goal, tau):
        rospy.loginfo("DMP Planning Started")
        rospy.wait_for_service('get_dmp_plan')

        x_dot_0 = [0] * self.dof
        t_0 = 0
        goal_thresh = [self.goal_thresh] * self.dof

        try:
            gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
            resp = gdp(start, x_dot_0, t_0, goal, goal_thresh, 
                    self.seg_length, tau, self.dt_default, self.integrate_iter)
        except rospy.ServiceException as e:
            print ("Service call failed: %s"%e)
        print ("DMP planning done")   
                
        return resp 