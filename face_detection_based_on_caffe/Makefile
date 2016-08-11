caffe_path=/home/caffe/caffe  #change to your path
#using opencv 2.xx and opencv 3.1 on tested
all:
	#g++ face_detect.cpp -o face_detect -DCPU_ONLY -DHAVE_CAFFE -std=c++0x -Wall -I/usr/local/include/opencv -I/usr/local/include -I/home/caffe/caffe/include -L/home/caffe/caffe/build/lib -lcaffe -lprotobuf -lglog -lpthread -lboost_system -lboost_filesystem -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_calib3d
	
	g++ face_detect.cpp -o face_detect -DCPU_ONLY -std=c++0x -Wall -I/usr/local/include/opencv -I/usr/local/include -I/home/caffe/caffe/include -lprotobuf -lglog -lpthread -lboost_system -lboost_filesystem -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_calib3d /home/caffe/caffe/build/lib/libcaffe.so -lprotobuf -lglog -lrt -Wl,-rpath,/usr/local/lib:/home/caffe/caffe/build/lib
clean:
	rm -f face_detect
