# UGProject
Endoscopy Image Analysis Web-platform

EzEndo is a web-based platform designed to empower healthcare professionals and researchers by providing a seamless way to analyze endoscopy images. Users can upload endoscopy images onto our platform to be processed and analyzed by our advanced AI algorithms to identify if their image is normal or abnormal, and provide insights into various similar endoscopy cases. With EzEndo, we aim to enhance diagnostic accuracy, speed up the analysis process, and contribute to the advancement of gastrointestinal health care.

Traceback (most recent call last):
  File "/uolstore/home/users/sc22yyt/ros2_ws/install/lab3/lib/lab3/tracesquare", line 33, in <module>
    sys.exit(load_entry_point('lab3==0.0.0', 'console_scripts', 'tracesquare')())
  File "/uolstore/home/users/sc22yyt/ros2_ws/install/lab3/lib/python3.10/site-packages/lab3/exercise2.py", line 66, in main
    trace_square.drive_square()
  File "/uolstore/home/users/sc22yyt/ros2_ws/install/lab3/lib/python3.10/site-packages/lab3/exercise2.py", line 50, in drive_square
    self.walk_forward(distance, speed)
  File "/uolstore/home/users/sc22yyt/ros2_ws/install/lab3/lib/python3.10/site-packages/lab3/exercise2.py", line 25, in walk_forward
    desired_velocity.linear.x = 0
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/geometry_msgs/msg/_vector3.py", line 135, in x
    assert \
AssertionError: The 'x' field must be of type 'float'
terminate called without an active exception
