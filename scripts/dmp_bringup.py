#!/usr/bin/env python3
import roslib; 
roslib.load_manifest('dmp')
import rospy 
import copy
import numpy as np
from dmp.srv import *
from dmp.msg import *
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from lfd_dmp.dmp_service_handler import DMPService


if __name__ == '__main__':

    rospy.init_node('dmp_tutorial_node')

    dmp_service = DMPService()

    rospy.spin()



    