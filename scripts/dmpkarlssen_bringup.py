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

from lfd_dmp.dmpkarlssen_wrapper import DMPKarlssenService


if __name__ == '__main__':

    rospy.init_node('dmpkarlssen_node')

    dmp_service = DMPKarlssenService()

    rospy.spin()



    