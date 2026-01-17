#!/usr/bin/env python3
# coding=utf-8

import rospy
import math
from sensor_msgs.msg import JointState
import tf
import tf.transformations as tft
from scout_gazebo.msg import LidarAngle  # 导入自定义消息

class LidarAngleTracker:
    def __init__(self):
        rospy.init_node('lidar_angle_tracker')
        
        # 初始化变量
        self.initial_angle = None
        self.current_angle = None
        self.normalized_angle = None  # 标准化到[0, 2π)
        self.relative_angle_deg = None  # 相对角度(度)
        
        # 创建发布器用于输出相对角度
        self.angle_pub = rospy.Publisher('/velodyne_relative_angle', JointState, queue_size=10)

        # 创建发布器用于发布自定义消息
        self.lidar_angle_pub = rospy.Publisher('/velodyne_angle_deg', LidarAngle, queue_size=10)
        
        # 订阅joint_states话题
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)

        
        # 创建TF监听器用于验证
        self.tf_listener = tf.TransformListener()
        
        # 主循环频率提高到400Hz
        self.rate = rospy.Rate(400)  # 从10Hz提高到400Hz
        
        rospy.loginfo("雷达角度跟踪器已启动，等待joint_states数据...")
    
    def joint_states_callback(self, msg):
        # 查找velodyne_spin_joint的索引
        if 'velodyne_spin_joint' in msg.name:
            idx = msg.name.index('velodyne_spin_joint')
            self.current_angle = msg.position[idx]
            
            # 如果是第一次收到数据，记录初始角度
            if self.initial_angle is None:
                self.initial_angle = self.current_angle
                rospy.loginfo("初始角度: %.2f弧度", self.initial_angle)
            
            # 计算相对角度（弧度）
            relative_angle = self.current_angle - self.initial_angle
            
            # 标准化到[0, 2π)
            self.normalized_angle = relative_angle % (2 * math.pi)
            
            # 转换为度
            self.relative_angle_deg = self.normalized_angle * 180.0 / math.pi
            
            # 发布相对角度信息
            angle_msg = JointState()
            angle_msg.header.stamp = rospy.Time.now()
            angle_msg.name = ['velodyne_relative_angle']
            angle_msg.position = [self.normalized_angle]
            self.angle_pub.publish(angle_msg)

            # 发布自定义消息
            lidar_angle_msg = LidarAngle()
            lidar_angle_msg.header.stamp = rospy.Time.now()
            lidar_angle_msg.angle = self.relative_angle_deg
            self.lidar_angle_pub.publish(lidar_angle_msg)
    
    def print_angle_info(self):
        if self.current_angle is not None and self.initial_angle is not None:
            # 获取TF中的实际旋转
            try:
                (trans, rot) = self.tf_listener.lookupTransform('velodyne_mount', 'velodyne', rospy.Time(0))
                # 这里可以解析rot四元数，但在这个简单案例中，我们直接用joint_state的信息
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            
            # 仅每10次循环打印一次信息(每秒40次)，避免日志过多
            if hasattr(self, 'print_counter'):
                self.print_counter += 1
            else:
                self.print_counter = 0
                
            if self.print_counter % 10 == 0:
                # 显示角度信息
                rospy.loginfo("当前角度: %.2f弧度, 初始角度: %.2f弧度", 
                             self.current_angle, self.initial_angle)
                rospy.loginfo("相对角度: %.2f弧度 (%.2f度)", 
                             self.normalized_angle, self.relative_angle_deg)
                rospy.loginfo("已旋转圈数: %.2f圈", 
                             (self.current_angle - self.initial_angle) / (2 * math.pi))
                rospy.loginfo("----------------------------")
    
    def run(self):
        while not rospy.is_shutdown():
            self.print_angle_info()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        tracker = LidarAngleTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass