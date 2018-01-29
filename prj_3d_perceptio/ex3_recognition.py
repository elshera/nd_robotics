#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    rospy.loginfo("Received new cloud data!\n")

    #---------------------------------------
    # Convert ROS msg to PCL data
    #---------------------------------------
    cloud = ros_to_pcl(pcl_msg)


    #---------------------------------------
    # Voxel Grid Downsampling
    #---------------------------------------
    vox = cloud.make_voxel_grid_filter()
    # voxel size
    LEAF_SIZE = 0.01   #  cm seems to be a good choice to not miss features
    # Set the voxel size in the filter object
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()


    #---------------------------------------
    # PassThrough Filter
    #---------------------------------------
    # create filter object
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.76
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    # Apply filter to the voxel grid, resultant point cloud. 
    cloud_filtered = passthrough.filter()


    #---------------------------------------
    # RANSAC Plane Segmentation
    #---------------------------------------
    # create segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model Experiment with different values for max_distance for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()


    #---------------------------------------
    # Extract table and objects
    #---------------------------------------
    pcl_table   = cloud_filtered.extract(inliers, negative=False)
    pcl_objects = cloud_filtered.extract(inliers, negative=True)


    #---------------------------------------
    # Euclidean Clustering
    #---------------------------------------
    white_cloud = XYZRGB_to_XYZ(pcl_objects)
    tree = white_cloud.make_kdtree()


    #---------------------------------------
    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    #---------------------------------------
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(3000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()


    #---------------------------------------
    # Assign a random color to each isolated object in the scene
    #---------------------------------------
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):    
            color_cluster_point_list.append([white_cloud[indice][0],
            	                            white_cloud[indice][1],             
    										white_cloud[indice][2], 
    									    rgb_to_float(cluster_color[j])])


    # Create new cloud containing all clusters, each with unique color
    pcl_obj_cluster = pcl.PointCloud_PointXYZRGB()
    pcl_obj_cluster.from_list(color_cluster_point_list)


    #---------------------------------------
    # Convert PCL data to ROS messages
    #---------------------------------------
    pcl_table_ros         = pcl_to_ros(pcl_table)
    pcl_objects_ros       = pcl_to_ros(pcl_objects)
    ros_obj_cluster_ros   = pcl_to_ros(pcl_obj_cluster)


    #---------------------------------------
    # Publish ROS messages
    #---------------------------------------
    pcl_objects_pub.publish(pcl_objects_ros)
    pcl_table_pub.publish(pcl_table_ros)
    pcl_obj_cluster_pub.publish(ros_obj_cluster_ros)


    #---------------------------------------
    # Classify the clusters! (loop through each detected cluster one at a time)
    #---------------------------------------
    detected_objects_labels = []
	detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        
        # Store the object's cloud in this list
		object_cluster = []


		# Create an individual cluster just for the object being processed
        for i, pts in enumerate(pts_list):
            # Retrieve cloud values for the x, y, z, rgb object
            object_cluster.append([cloud_objects[pts][0],
                                   cloud_objects[pts][1],
                                   cloud_objects[pts][2],
                                   cloud_objects[pts][3]])
            
        # Grab the points for the cluster
        pcl_cluster = pcl.PointCloud_PointXYZRGB()
		pcl_cluster.from_list(object_cluster)

        # Convert the cluster from pcl to ROS using helper function
        ros_cloud = pcl_to_ros(pcl_cluster)

        # Extract histogram features (similar to capture_features.py)
        histogram_bins = 64
        chists  = compute_color_histograms(ros_cloud, nbins=histogram_bins, using_hsv=True)
        normals = get_normals(ros_cloud)
        nhists  = compute_normal_histograms(normals, nbins=histogram_bins)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        pcl_object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do       = DetectedObject()
        do.label = label
        do.cloud = ros_cloud
        detected_objects.append(do)

    #---------------------------------------
    # Publish the list of detected objects
    #---------------------------------------
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
	pcl_detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':
    #---------------------------------------
    # ROS node initialization
    #---------------------------------------
    rospy.init_node('recognition')


    #---------------------------------------
    # Create Subscribers
    #---------------------------------------
    subs = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size=1)


    #---------------------------------------
    # Create Publishers
    #---------------------------------------
    pcl_objects_pub          = rospy.Publisher('/pcl_object',      PointCloud2, queue_size=1)
    pcl_table_pub            = rospy.Publisher('/pcl_table',       PointCloud2, queue_size=1)
    pcl_obj_cluster_pub      = rospy.Publisher('/pcl_obj_cluster', PointCloud2, queue_size=1)
    pcl_detected_objects_pub = rospy.Publisher('/pcl_det_objects', PointCloud2, queue_size=1)
    pcl_object_markers_pub   = rospy.Publisher('/object_markers',  Marker, queue_size=1)


    #---------------------------------------
    # TODO: Load Model From disk
    #---------------------------------------
    model   = pickle.load(open('model.sav', 'rb'))
    clf     = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
	scaler  = model['scaler']


    #---------------------------------------
    # Initialize color_list
    #---------------------------------------
    get_color_list.color_list = []


    #---------------------------------------
    # Spin while node is not shutdown
    #---------------------------------------
    while not rospy.is_shutdown():
        rospy.spin()

