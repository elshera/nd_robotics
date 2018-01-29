#!/usr/bin/env python

# Import modules
from pcl_helper import *

# prefiltering operation of the cloud, as per the exrecise 1
def pcl_filter(pcl_msg):
    # ROS callback function for the Point Cloud Subscriber. Takes a point cloud and
    # performs object recognition on it. 
    rospy.loginfo("Received new cloud data!\n")

    #---------------------------------------
    # TODO: Convert ROS msg to PCL data
    #---------------------------------------
    cloud = ros_to_pcl(pcl_msg)
    # save the original cloud
    pcl.save(cloud, 'cloud_original.pcd')

    #---------------------------------------
    # TODO: Statistical Outlier Filtering. We need to remove as much noise as possible so it make sense to apply a filtering before.
    #---------------------------------------
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)
    # Any point with a mean distance larger than global will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(0.1)
    cloud_filtered = outlier_filter.filter()
    # save filter for checking/debug purpose
    pcl.save(cloud_filtered, 'cloud_filtered.pcd')

    #---------------------------------------
    # TODO: Voxel Grid Downsampling, reduce computation of the scene
    #---------------------------------------
    vox = cloud_filtered.make_voxel_grid_filter()
    # voxel size
    LEAF_SIZE = 0.005  #  5 mm seems to be a good choice to not miss features
    # Set the voxel size in the filter object
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_voxeled = vox.filter()
    # save filter for checking/debug purpose
    pcl.save(cloud_voxeled, 'cloud_voxeled.pcd')

    #---------------------------------------
    # TODO: PassThrough Filter
    #---------------------------------------
    # create filter object, on z axis
    passthrough_z = cloud_voxeled.make_passthrough_filter()
    passthrough_z.set_filter_field_name('z')
    passthrough_z.set_filter_limits(0.75, 1.1)
    cloud_passthrough = passthrough_z.filter()

    # save filter for checking/debug purpose
    pcl.save(cloud_passthrough, 'cloud_passthrough.pcd')
    return cloud_passthrough


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    
    cloud_filtered = pcl_filter(pcl_msg)

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
    # Assign a color corresponding to each segmented object in scene
    #---------------------------------------
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
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

   

if __name__ == '__main__':
    #---------------------------------------
    # ROS node initialization
    #---------------------------------------
    rospy.init_node('segmentation')

    #---------------------------------------
    # Create Subscribers
    #---------------------------------------
    subs = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size=1)

    #---------------------------------------
    # Create Publishers
    #---------------------------------------
    pcl_objects_pub      = rospy.Publisher('/pcl_object',      PointCloud2, queue_size=1)
    pcl_table_pub        = rospy.Publisher('/pcl_table',       PointCloud2, queue_size=1)
    pcl_obj_cluster_pub  = rospy.Publisher('/pcl_obj_cluster', PointCloud2, queue_size=1)

    #---------------------------------------
    # Initialize color_list
    #---------------------------------------
    get_color_list.color_list = []

    #---------------------------------------
    # Spin while node is not shutdown
    #---------------------------------------
    while not rospy.is_shutdown():
        rospy.spin()
