# Import PCL module
import pcl



#------------------------------------------------------------
# Load Point Cloud file
#-----------------------------------------------------------
cloud = pcl.load_XYZRGB('tabletop.pcd')
pcl.save(cloud, 'cloud_original.pcd')
print(cloud.size)



#------------------------------------------------------------
# Voxel Grid filter
#------------------------------------------------------------
# Create a VoxelGrid object for our input point cloud
vox = cloud.make_voxel_grid_filter()

# voxel size
LEAF_SIZE = 0.01   #  cm seems to be a good choice to not miss features

# Set the voxel size in the filter object
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# Call the filter function to obtain the resultant downsampled point cloud
cloud_voxeled = vox.filter()
print(cloud_voxeled.size)

# save the data out for visualization
filename = 'cloud_voxeled.pcd'
pcl.save(cloud_voxeled, filename)



#------------------------------------------------------------
# PassThrough filter
#------------------------------------------------------------
# create filter object
passthrough = cloud_voxeled.make_passthrough_filter()

# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passthrough.set_filter_limits(axis_min, axis_max)

# Apply filter to the voxel grid, resultant point cloud. 
cloud_voxeled = passthrough.filter()
print(cloud_voxeled.size)

# save file for inspection
filename = 'cloud_passthrough.pcd'
pcl.save(cloud_voxeled, filename)



#------------------------------------------------------------
# RANSAC plane segmentation
#------------------------------------------------------------
# create segmentation object
seg = cloud_voxeled.make_segmenter()

# Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Max distance for a point to be considered fitting the model
# Experiment with different values for max_distance for segmenting the table
max_distance = 0.01
seg.set_distance_threshold(max_distance)
# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()



#------------------------------------------------------------
# Extract inliers
#------------------------------------------------------------
table_cloud = cloud_voxeled.extract(inliers, negative=False)
print(table_cloud.size)
# save file for inspection
filename = 'clod_table.pcd'
pcl.save(table_cloud, filename)




#------------------------------------------------------------
# Extract outliers
#------------------------------------------------------------
# Much like the previous filters, we start by creating a filter object: 
object_cloud = cloud_voxeled.extract(inliers, negative=True)
print(object_cloud.size)
# Save pcd for tabletop objects
filename = 'cloud_object.pcd'
pcl.save(object_cloud, filename)



#------------------------------------------------------------
# Appliy statistical filter
#------------------------------------------------------------
outlier_filter = object_cloud.make_statistical_outlier_filter()
# Set the number of neighboring points to analyze for any given point
outlier_filter.set_mean_k(50)
# Any point with a mean distance larger than global will be considered out
outlier_filter.set_std_dev_mul_thresh(1.0)
cloud_filtered = outlier_filter.filter()
# Save pcd for statistical filter
filename = 'cloud_filtered.pcd'
pcl.save(object_cloud, filename)

