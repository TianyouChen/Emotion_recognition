    表情识别v1.0，实现对人脸表情：生气、厌恶、恐惧、开心、伤心、惊讶、中性7种表情识别，训练数据用的是fer2013，其中训练集图片28709张，验证集3589张，测试集3589张。
    表情分类模型是基于caffe_ssd框架下做的，首先git clone在github上面的caffe_ssd源码，git clone https://github.com/weiliu89/caffe/tree/ssd，然后按照wiki教程编译caffe_ssd即可在下面利用已经训练好的表情分类模型对人脸表情进行分类，测试脚本在scripts下,在终端输入python facial_test_fuck.py --caffe_root /home/ubuntu/caffe/ --INPUT_ROOT /home/ubuntu/caffe/faceR/ --TEST_ROOT /home/ubuntu/caffe/faceR/data/test/ 即可测试模型在测试集下的每一类准确率以及每类的准确率，其中caffe_root路径为caffe_ssd源码的路径，INPUT_ROOT为模型文件和deploy测试文件的保存路径，TEST_ROOT为测试集存放路径，测试集文件夹test下包含7个文件夹分别存放7类表情测试图片。

    仓库的data文件夹存放了表情识别训练网络的数据输入，lmdb和均值文件
    models文件夹下存放了训练网络、网络训练配置solver文件、deploy测试文件和训练好的caffemodel，目前7分类表情验证准确率64%
    scripts下是表情识别测试代码，对每一类表情做准确率测试，最后也会统计在所有表情上的综合准确率，acc_result.txt保存了测试准确率结果。

