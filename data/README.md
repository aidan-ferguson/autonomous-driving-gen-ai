This folder contains all the scripts used in the creation of the datasets - as well as all other resources

The purpose of this dataset is to generate a set of pairs of corresponding frames. Where the first frame in the pair is the image from the simulator and the second image is the real world camera image.

The process of doing this involves:
- Playing the bag in R-VIZ with LiDAR merging on for a higher resolution and moving the focal point of the view to the base of all the cones - then recording the position and colour of each cone
- Then step through the bag so the car has moved and record all the cone positions again - we can then manually line up multiple ground truth cone annotations and get the full track
- Doing this for every LiDAR frame will allow us to also reconstruct ground truth poses

Notes:
- The ground truth annotations are in the reference frame of the velodyne LiDAR (when the LiDAR was switched on)
- TODO: coordinate system
- The ground truth positions are taken from the base of the centre of the cone (they can be translated depending on their colour after the fact)