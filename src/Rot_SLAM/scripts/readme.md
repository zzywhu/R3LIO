# scripts
## batch-process-datasets.sh
该脚本用于数据批量处理和测试。
### Usage
#### 1. 准备测试数据集
按照如下目录结构放置待测试数据集：
```
.
./das_hesai_2022-11-01-10-48
./das_hesai_2022-11-01-10-48/Daslidar.pcap
./das_hesai_2022-11-01-10-48/imu_data.txt
./das_hesai_2022-11-01-10-48/motor_angle.txt
./das_hesai_2022-11-01-10-29-noloop
./das_hesai_2022-11-01-10-29-noloop/Daslidar.pcap
./das_hesai_2022-11-01-10-29-noloop/imu_data.txt
./das_hesai_2022-11-01-10-29-noloop/imu_data (copy).csv
./das_hesai_2022-11-01-10-29-noloop/motor_angle.txt
./das_hesai_2022-11-01-10-36-noloop
./das_hesai_2022-11-01-10-36-noloop/Daslidar.pcap
./das_hesai_2022-11-01-10-36-noloop/imu_data.txt
./das_hesai_2022-11-01-10-36-noloop/motor_angle.txt
```
规则是每组数据放到一个文件夹里。
#### 2. 准备配置文件
修改配置文件，主要配置项有：
1. imu_file_path：路径改为./imu_data.txt；
2. raster_file_path：路径改为./motor_angle.txt；
3. pcap_file_path：路径改为./Daslidar.pcap；
4. lidar_correction_file_path：路径修改为绝对路径，例如`/path/to/PandarXT-16.csv`；
5. 其它配置项自行配置即可。
#### 3. 启动批量处理脚本
进入数据目录，运行批量测试脚本：
`bash /opt/my-workspace/dasslam_dev/scripts/batch-process-datasets.sh /opt/my-workspace/dasslam_dev/build/DasSlamTest /opt/my-workspace/dasslam_dev/config/das_pcap.yaml `。
