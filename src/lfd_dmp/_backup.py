
import roslib; 
roslib.load_manifest('dmp')
import rospy

import numpy as np

from dmp.srv import *
from dmp.msg import *

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class DMPService:

    def __init__(self):
        rospy.Subscriber("/lfd_bringup/teach_trajectory_joint", RobotTrajectory , self.joint_traj_callback)
        rospy.Subscriber("/lfd_bringup/plan_start_goal_joint", RobotTrajectory , self.plan_joint_callback)
        self.dmp_ready = False
    
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
    
    def plan_joint_callback(self, msg):
        if self.dmp_ready==False:
            return
        
        rospy.loginfo("entered plan callback")
        x_0 = list(msg.joint_trajectory.points[0].positions)
        x_dot_0 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]   
        t_0 = 0                
        goal = list(msg.joint_trajectory.points[1].positions)
        goal_thresh = [0.05,0.05]
        seg_length = 20          
        tau = 10       
        dt = 1.0
        integrate_iter = 5
        plan = self.makePlanRequest(x_0, x_dot_0, t_0, goal, goal_thresh, 
                            seg_length, tau, dt, integrate_iter)

        #TODO the topic namespace (lfd_bringup) is fragile and prune to bugs, how to overcome it?
        pub = rospy.Publisher('/lfd_bringup/plan_trajectory_joint', RobotTrajectory, queue_size=1, latch=True)
        plan_traj = RobotTrajectory()
        for point in plan.plan.points:
            traj_point = JointTrajectoryPoint()
            traj_point.positions = tuple(point.positions)
            plan_traj.joint_trajectory.points.append(traj_point)
        
        plan_traj.joint_trajectory.joint_names = msg.joint_trajectory.joint_names # ("panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7")
        pub.publish(plan_traj)


    def joint_traj_callback(self, msg):
        print ("entered teach callback")
        joint_datapoints = msg.joint_trajectory.points
        joint_traj = []
        for point in joint_datapoints:
            joint_traj.append(list(point.positions))
        

        #Create a DMP from a 2-D trajectory
        dims = 7                
        dt = 1.0                
        K = 200                 
        D = 2.0 * np.sqrt(K)      
        num_bases = 8          

        resp = self.makeLFDRequest(dims, joint_traj, dt, K, D, num_bases)

        #Set it as the active DMP
        self.makeSetActiveRequest(resp.dmp_list)
        self.dmp_ready = True
        rospy.loginfo("dmp training completed successfully")




