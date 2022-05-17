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

from lfd_dmp.dmpbbo_wrapper import DMPBBOService


if __name__ == '__main__':

    rospy.init_node('dmpbbo_node')

    dmp_service = DMPBBOService()

    rospy.spin()



    