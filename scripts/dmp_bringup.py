#!/usr/bin/env python3
import rospy 
from lfd_dmp.dmpbbo_wrapper import DMPBBOService

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

    if dmp_method == 'dmpbbo':
        rospy.loginfo("Executing DMP BBO Method")
        dmp_service = DMPBBOService(training_mode=training_mode, num_kernels=num_kernels,
                                    robot_ns=robot_ns)
    
    rospy.spin()



    