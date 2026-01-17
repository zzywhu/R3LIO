# todo

PATH_TO_DASSLAM_TEST_EXE=$1
CONFIG_FILE_PATH=$2

CUR_DIR=$(PWD)

for e in $(ls ${CUR_DIR})
do
  cd ${CUR_DIR}/$e
  mkdir PCD/ result/
  ${PATH_TO_DASSLAM_TEST_EXE} ${CONFIG_FILE_PATH}
done