
import roslib; 
roslib.load_manifest('dmp')
import rospy

import numpy as np

from dmp.srv import *
from dmp.msg import *

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from lfd_interface.srv import TrainDemonstration, TrainDemonstrationRequest, TrainDemonstrationResponse

class DMPService:

    def __init__(self):
        self.server_train_demonstration = rospy.Service("train_demonstration",TrainDemonstration, self.cb_train_demonstration)
    
    #Learn a DMP from demonstration data
    def makeLFDRequest(self, dims, traj, dt, K_gain, D_gain, num_bases):
        demotraj = DMPTraj()
            
        for i in range(len(traj)):
            pt = DMPPoint()
            pt.positions = traj[i]
            demotraj.points.append(pt)
            demotraj.times.append(dt*i)
                
        k_gains = [K_gain]*dims
        d_gains = [D_gain]*dims
            
        print ("Starting LfD...")
        rospy.wait_for_service('learn_dmp_from_demo')
        try:
            lfd = rospy.ServiceProxy('learn_dmp_from_demo', LearnDMPFromDemo)
            resp = lfd(demotraj, k_gains, d_gains, num_bases)
        except rospy.ServiceException as e:
            print ("Service call failed: %s"%e)
        print ("LfD done")    
                
        return resp
    
    #Set a DMP as active for planning
    def makeSetActiveRequest(self, dmp_list):
        try:
            sad = rospy.ServiceProxy('set_active_dmp', SetActiveDMP)
            sad(dmp_list)
        except rospy.ServiceException as e:
            print ("Service call failed: %s"%e)
    
    #Generate a plan from a DMP
    def makePlanRequest(self, x_0, x_dot_0, t_0, goal, goal_thresh, 
                        seg_length, tau, dt, integrate_iter):
        print ("Starting DMP planning...")
        rospy.wait_for_service('get_dmp_plan')
        try:
            gdp = rospy.ServiceProxy('get_dmp_plan', GetDMPPlan)
            resp = gdp(x_0, x_dot_0, t_0, goal, goal_thresh, 
                    seg_length, tau, dt, integrate_iter)
        except rospy.ServiceException as e:
            print ("Service call failed: %s"%e)
        print ("DMP planning done")   
                
        return resp 
    
    def cb_train_demonstration(self,req : TrainDemonstrationRequest):
        joint_trajectory = []
        for point in req.demonstration.joint_trajectory.points:
            joint_trajectory.append(list(point.positions))

        
        dims = 7                
        dt = 1.0                
        K = 200                 
        D = 2.0 * np.sqrt(K)      
        num_bases = 8          

        resp = self.makeLFDRequest(dims, joint_trajectory, dt, K, D, num_bases)

        #Set it as the active DMP
        self.makeSetActiveRequest(resp.dmp_list)
        rospy.loginfo("dmp training completed successfully")

        return TrainDemonstrationResponse(True)
        






