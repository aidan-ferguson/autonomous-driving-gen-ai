from rosbag2_py import StorageOptions, ConverterOptions, SequentialReader
from sensor_msgs.msg import Image
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
import numpy as np
import cv2, imageio, os, argparse


def create_image_folder(folder_path: str) -> None:
  """
  Create a folder to contain the images if the user selects the option to export as image files
  """
  if not os.path.exists(folder_path):
    os.mkdir(folder_path)


def read_all_images(bag_directory: str, image_topic: str, skip_frames: int, start_frame: int) -> list[Image]:
  """
  Goes through a rosbag and reads every single image from the image topic. Returning it as a list of sensor_msgs/Image
  """
  storage_id = "sqlite3"
  serialization_fmt = "cdr"
  image_list = []
  co = ConverterOptions(serialization_fmt, serialization_fmt)
  so = StorageOptions(bag_directory, storage_id)
  bag = SequentialReader()
  bag.open(so, co)
  message_idx = 0
  while bag.has_next():
    topic_name, bin_data, _ = bag.read_next()
    if topic_name == image_topic:
        image = deserialize_message(bin_data, Image)
        image_list.append(image)
  
  image_list.sort(key=lambda msg: (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))
  print(f"Read a total of {len(image_list)} images from bag.")
  image_list = image_list[start_frame::skip_frames]
  print(f"Post-filtering there are {len(image_list)} images left")
  return image_list


def estimate_fps(images: list[Image]) -> float:
  intervals = []
  for i in range(len(images) - 1):
    prev = images[i].header.stamp
    next = images[i+1].header.stamp
    next_seconds = next.sec + next.nanosec * 1e-9
    prev_seconds = prev.sec + prev.nanosec * 1e-9
    interval = next_seconds - prev_seconds
    intervals.append(interval)
  avg_interval = np.mean(intervals)
  return 1.0 / avg_interval


def save_images_to_folder(folder_path: str, images: list[Image]) -> None:
  """
  Write a list of 'png' images to a specified folder
  """
  framerate = estimate_fps(images)
  print(f"Estimating a framerate of {framerate} fps.")
  for idx, image in enumerate(images):
    full_path = os.path.join(folder_path, str(idx) + ".png")
    cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")

    cv2.imwrite(full_path, cv_image)
  print("Finished writing images")


def make_mp4_from_images(output_path: str, images: list[Image]):
  """
  Converts an array of sensor_msgs/Image into an mp4.
  """
  framerate = estimate_fps(images)
  print(f"Estimating a framerate of {framerate} fps.")
  writer = imageio.v2.get_writer(output_path, fps=framerate)
  for image in images:
    cv_image = CvBridge().imgmsg_to_cv2(image, "rgb8")
    writer.append_data(cv_image)
  writer.close()
  print(f"Finished writing video to {output_path}.")


def main():
  # Parse arguments
  parser = argparse.ArgumentParser(prog='Video Enhancer', description='Exports videos/images from ros2 bags')
  parser.add_argument('bag_directory')
  parser.add_argument('image_topic_name')
  parser.add_argument('--format', choices=['video', 'images'], default='video')
  parser.add_argument('--output_video', default='output.mp4')
  parser.add_argument('--output_image_dir', default='images')
  parser.add_argument('--skip_frames', default="1")
  parser.add_argument('--start_frame', default="0")

  args = parser.parse_args()

  bag_directory = args.bag_directory
  image_topic = args.image_topic_name
  output_path = args.output_video if args.format == "video" else args.output_image_dir
  skip_frames = max(1, int(args.skip_frames))
  start_frame = max(0, int(args.start_frame))

  print(f"{bag_directory=} {image_topic=}")
  images = read_all_images(bag_directory, image_topic, skip_frames, start_frame)

  if args.format == "video":
    make_mp4_from_images(output_path, images)
  else:
    create_image_folder(output_path)
    save_images_to_folder(output_path, images)


if __name__ == "__main__":
  main()
