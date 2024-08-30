#!/usr/bin/env python3
import roslib; 
roslib.load_manifest('dmp')
import rospy 
from dmp.srv import *
from dmp.msg import *
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from dynamic_reconfigure.server import Server
from lfd_dmp.cfg import DMPKarlssonConfig

from lfd_dmp.dmp_service_handler import DMPService
from lfd_dmp.dmpbbo_wrapper import DMPBBOService
from lfd_dmp.dmpkarlsson_wrapper import DMPkarlssonService

if __name__ == '__main__':

    rospy.init_node('dmp_bringup_node')
    
    dmp_method = rospy.get_param("~dmp_method")
    training_mode = rospy.get_param("~training_mode")
    num_kernels = rospy.get_param("~num_kernels")
    robot_ns = rospy.get_param("~robot_ns")
    try:
        dmp_config = rospy.get_param("~dmpconfig")
    except:
        rospy.logwarn("No Config file found. Continuing...")

    if dmp_method == 'dmpros':
        rospy.loginfo("Executing DMP ROS Method")
        dmp_service = DMPService()
    elif dmp_method == 'dmpbbo':
        rospy.loginfo("Executing DMP BBO Method")
        dmp_service = DMPBBOService(training_mode=training_mode, num_kernels=num_kernels,
                                    robot_ns=robot_ns)
    elif dmp_method == 'dmpkarlsson':
        rospy.loginfo("Executing DMP Karlsson Method")
        dmp_service = DMPkarlssonService(dmp_config)
        srv = Server(DMPKarlssonConfig, dmp_service.cb_dyn_reconfig)
    
    rospy.spin()



    