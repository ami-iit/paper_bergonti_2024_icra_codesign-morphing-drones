cmake_minimum_required(VERSION 3.0.2)
project(ros_muav)

find_package(catkin REQUIRED COMPONENTS
  joint_state_publisher
  joint_state_publisher_gui
  robot_state_publisher
  rviz
  xacro
  rospy
  jsk_rviz_plugins
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
