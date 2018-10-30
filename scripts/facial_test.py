import numpy as np
import os
import sys
import argparse
import glob
import time
import skimage
import argparse
import time


def test(caffe_root,INPUT_ROOT,TEST_ROOT):
    
    os.chdir(caffe_root)
    sys.path.insert(0,os.path.join(caffe_root,'python'))
    import caffe
    MODEL_FILE = INPUT_ROOT+'220000.caffemodel'
    DEPLOY_FILE = INPUT_ROOT+'deploy_resnet_facial.prototxt'


    caffe.set_mode_cpu()
    net = caffe.Net(DEPLOY_FILE, MODEL_FILE, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(1, 3, 40, 40)
    count=0.0
    acc=0
    count_all = 0
    num_all = 0.0
    dir_list = os.listdir(TEST_ROOT)
    for i in range(0,len(dir_list)):
        path = os.path.join(TEST_ROOT,dir_list[i])
        file_list = os.listdir(path)
        acc0 = 0.0
        count0 = 0
        for j in range(0,len(file_list)):
            img_dir = os.path.join(path,file_list[j])
            img = caffe.io.load_image(img_dir)
            net.blobs['data'].data[...] = transformer.preprocess('data', img)
            labels=['anger','disgust','fear','happy','sad','surprised','normal']
            T1 = time.time()
            out = net.forward()
            T2 = time.time()
            #print(T2-T1)
            pridects = out['prob'][0]
            #print pridects
            pridect = pridects.argmax()
            #print pridect
            acc=acc+1
            acc0=acc0+1
            if pridect == int(dir_list[i]):
                count=count+1
                count0=count0+1
            #print(float(j)/len(file_list))
            #print(len(file_list))
            #print(path+img_dir)
            #print labels[pridects.argmax()]
        #print(count0)
        acc1=count0/acc0
        print labels[i]
        #print(acc,count,acc1)
        print "acc[%d]=%f"%(i,acc1)
    count_all=count+count_all
           
    num_all=acc+num_all
    #print(count_all)

    #print(num_all)
    print(count_all/num_all)
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--caffe_root",help="give your caffe_root")
    parser.add_argument("--INPUT_ROOT",help="give input_dir")
    parser.add_argument("--TEST_ROOT",help="give test_dir")
    return parser.parse_args()

def main(args):
    test(args.caffe_root,args.INPUT_ROOT,args.TEST_ROOT)

if __name__ == "__main__":
    main(parse_args())
