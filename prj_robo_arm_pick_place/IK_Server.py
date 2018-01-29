#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

#================================================================================
# required modules
#================================================================================
import numpy as np
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
from time import time

#================================================================================
# define useful functions
#================================================================================
# Convert radians to degrees
def rtd(rad):
    return rad * 180 / np.pi

# convert degrees to radians
def dtr(deg):
    return deg * np.pi / 180

# generalized homogeneous transformation matrix, given parameters: alpha, a, d, theta and dh
def ghmt(alpha, a, d, theta, dh):
    HTM =  Matrix([[            cos(theta),           -sin(theta),           0,             a],
                   [ sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                   [ sin(theta)*sin(alpha), cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                   [                     0,                      0,          0,             1]])
    return HTM.subs(dh)


# define rotation matrixes
def R_x(q):
    RM = Matrix([[ 1,      0,       0],
                 [ 0, cos(q), -sin(q)],
                 [ 0, sin(q),  cos(q)]])
    return  RM

def R_y(q):
    RM =  Matrix([[  cos(q), 0, sin(q)],
                  [       0, 1,      0],
                  [ -sin(q), 0, cos(q)]])
    return RM

def R_z(q):
    RM =  Matrix([[ cos(q), -sin(q), 0],
                  [ sin(q),  cos(q), 0],
                  [      0,       0, 1]])
    return RM

#========================================================================================
# define DH parameter table 
#========================================================================================
# angles between X axis
theta1, theta2, theta3, theta4, theta5, theta6, theta7 = symbols('theta1:8')
# distances between X axis
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
# distances between Z axis
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
# angles between Z-axes
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')


# DH parameters Table
DHT = { alpha0:     0,  a0:      0, d1:  0.75, theta1:      theta1,
        alpha1: -pi/2,  a1:   0.35, d2:     0, theta2: theta2-pi/2, 
        alpha2:     0,  a2:   1.25, d3:     0, theta3:      theta3,
        alpha3: -pi/2,  a3: -0.054, d4:  1.50, theta4:      theta4,
        alpha4:  pi/2,  a4:      0, d5:     0, theta5:      theta5,
        alpha5: -pi/2,  a5:      0, d6:     0, theta6:      theta6,
        alpha6:     0,  a6:      0, d7: 0.303, theta7:           0
      }


#========================================================================================
# Individual Transforms
#========================================================================================
#T0_1 = ghmt(alpha0, a0, d1, theta1, DHT)
#T1_2 = ghmt(alpha1, a1, d2, theta2, DHT)
#T2_3 = ghmt(alpha2, a2, d3, theta3, DHT)
#T3_4 = ghmt(alpha3, a3, d4, theta4, DHT)
#T4_5 = ghmt(alpha4, a4, d5, theta5, DHT)
#T5_6 = ghmt(alpha5, a5, d6, theta6, DHT)
#T6_7 = ghmt(alpha6, a6, d7, theta7, DHT)

# Total tranform from base ling to the end effector
#T0_E = simplify(T0_1 * T1_2  * T2_3 * T3_4 * T4_5 * T5_6 * T6_7)

#========================================================================================
# From URDF
#========================================================================================
# orientation difference between definition of gripper link in URDF file and the DHT.
# (rotation around Z axis by 180 deg and X axis by -90 deg)
R_corr = simplify(R_z(np.pi) * R_y(-np.pi/2))

# Joint distances 
J3_4 = 0.96
J4_5 = 0.54

#================================================================================
# Main function which calculates the IK and FK
#================================================================================
def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        # Initialize service response
        joint_trajectory_list = []  # Stores joint angle values calculated
        for x in xrange(0, len(req.poses)):
            start_time = time()
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()
            
            #=================================================================
            # End Effector location and orientation, from the request.
            #=================================================================
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z
            ee_pos = np.array((px ,py, pz))
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                 req.poses[x].orientation.z, req.poses[x].orientation.w])
            
            
            # end-effector rotation matrix and position
            ee_rot = R_z(yaw) * R_y(pitch) * R_x(roll)

            # correction of the rotation matrix
            ee_rot_corrected = ee_rot * R_corr

            # having position vector and rotation matrix, we can build the Homogenous transform which represent
            # the transform between end-effector and base frame.
            # in order to decouple the IK problem we need as next to calculate the wrist center position. 

            #=================================================================
            # Wrist  position
            #=================================================================
            # The wrist is there ti allow the end effector to assume just any orientation in the space. 
            # considering that there is only a displacement of the grip along the z-axis therefore we can take
            # the Z unit vector description in the rotation matrix and multiply time the displacement
            wc_x = px - DHT[d7] * ee_rot_corrected[0, 2]
            wc_y = py - DHT[d7] * ee_rot_corrected[1, 2]
            wc_z = pz - DHT[d7] * ee_rot_corrected[2, 2]
            wc_pos = np.array((wc_x ,wc_y, wc_z))
           

            #=================================================================
            # arm variables
            #=================================================================
            # theta 1 is simple as the wrist center is already refferede to the base link.
            theta1 = atan2(wc_y, wc_x).evalf()

            # limitation as per urdf file
            theta1 = np.clip(theta1, dtr(-185), dtr(185))
            

            # position of joint-2. given theta1
            xj2 = DHT[a1] * cos(theta1)
            yj2 = DHT[a1] * sin(theta1)
            zj2 = DHT[d1]
            j2_pos = np.array((xj2 ,yj2, zj2))

            # here we are using the triangle indicates as from the lesson. 
            # distances between joint 2 and the wrist center, calculates distance in space
            l2 = np.linalg.norm(wc_pos-j2_pos)
            
             # x, y, and z distances between joint 2 and wrist center
            wc2x2 = wc_x - xj2
            wc2y2 = wc_y - yj2
            wc2z2 = wc_z - zj2

            # distance between joint 3 and the wrist center
            x34 = np.hypot(J3_4, DHT[a3])            
            phi1 = pi - atan2(abs(DHT[a3]), x34)
            l3 = sqrt(J3_4**2 + J4_5**2 - 2 * J3_4 * J4_5 * cos(phi1))


            # Determine the angle for joint 2
            #dzwc = sqrt(wc2x2**2 + wc2y2**2)
            dzwc = np.hypot(wc2x2, wc2y2)
            phi3 = atan2(wc2z2, dzwc)
            cos_phi4 = (l3**2 - l2**2 - DHT[a2]**2) / (-2 * l2 * DHT[a2])
            if abs(cos_phi4) > 1:
                cos_phi4 = 1
            phi4 = atan2(sqrt(1 - cos_phi4**2), cos_phi4)
            # joint 2
            theta2 = (pi/2 - (phi3 + phi4)).evalf()
            theta2 = np.clip(theta2, dtr(-45), dtr(85))



            # Determine the angle for joint 3
            cos_phi2 = (l2**2 - l3**2 - DHT[a2]**2) / (-2 * l3 * DHT[a2])
            if abs(cos_phi2) > 1:
                cos_phi2 = 1
                
            phi2 = atan2(sqrt(1 - cos_phi2**2), cos_phi2)          
            # joint 3
            theta3 = (pi/2 - phi2).evalf()
            theta3 = np.clip(theta3, dtr(-210), dtr(155-90))

            #=================================================================
            # Inverse Orientation, 
            #=================================================================
            # Individual transformation matrices
            T0_1 = ghmt(alpha0, a0, d1, theta1, DHT)
            T1_2 = ghmt(alpha1, a1, d2, theta2, DHT)
            T2_3 = ghmt(alpha2, a2, d3, theta3, DHT)

            # transform between the base link to the wrist center
            T0_3 = simplify(T0_1 * T1_2 * T2_3).evalf(subs={theta1: theta1, theta2: theta2, theta3: theta3})
            
            # Rotation matrix for the spherical wrist
            R0_3 = T0_3[0:3, 0:3]

            # Calculate the rotation matrix of the spherical wrist joints, use inverse or transpose as the
            # properties or rotation matrix do allow for this. 
            R3_6 = R0_3.T * ee_rot_corrected
            R3_6_np = np.array(R3_6).astype(np.float64)

            # Convert the rotation matrix to Euler angles using tf
            alpha, beta, gamma = tf.transformations.euler_from_matrix(R3_6_np, axes='rxyz')   # xyx, yzx, xyz
            
            theta4 = alpha
            theta5 = beta
            theta6 = gamma

            theta4 = np.pi/2 + theta4
            theta5 = np.pi/2 - theta5

            
            # Populate response for the IK request
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)
            print("\nTotal runtime to calculate joint angles %04.4f seconds.\n"%(time() - start_time))

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


#================================================================================
# ROS Service for calculating the IK
#================================================================================
def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()

