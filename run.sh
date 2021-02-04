export LD_LIBRARY_PATH=build/
# Alexnet
# ./build/examples/graph_alexnet --layout=NCHW --target=NEON --threads=1 --type=F32 --data=data_model/public/alexnet --image=data/images/227x227.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 
# ./build/examples/graph_alexnet --layout=NCHW --target=NEON --threads=4 --type=F32 --data=data_model/public/alexnet --image=data/images/227x227.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# Resnet50
# ./build/examples/graph_resnet50 --layout=NCHW --target=NEON --threads=1 --type=F32 --data=data_model/public --image=data/images/227x227.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 
# ./build/examples/graph_resnet50 --layout=NCHW --target=NEON --threads=4 --type=QASYMM8 --data=data_model/public --image=data/images/227x227.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# VGG16
# ./build/examples/graph_vgg16 --layout=NCHW --target=NEON --threads=1 --type=F32 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 
# ./build/examples/graph_vgg16 --layout=NCHW --target=NEON --threads=4 --type=F32 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# VGG19
# ./build/examples/graph_vgg19 --layout=NCHW --target=NEON --threads=1 --type=F32 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 
# ./build/examples/graph_vgg19 --layout=NCHW --target=NEON --threads=4 --type=F32 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# Squeeznet
# ./build/examples/graph_squeezenet_v1_1 --layout=NCHW --target=NEON --threads=1 --type=F32 --data=data_model/public --image=data/images/227x227.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 
./build/examples/graph_squeezenet_v1_1 --layout=NCHW --target=NEON --threads=4 --type=F32 --data=data_model/public --image=data/images/227x227.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# Mobilenet-v1
# ./build/examples/graph_mobilenet --fast-math --layout=NCHW --target=NEON --threads=1 --type=F32 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 
# ./build/examples/graph_mobilenet --fast-math --layout=NCHW --target=NEON --threads=4 --type=QASYMM8 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# Mobilenet-v1-quantization
# ./build/examples/graph_mobilenet --fast-math --layout=NCHW --target=NEON --threads=1 --type=QASYMM8 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# inception_v3-quantization
# ./build/examples/graph_inception_v3 --fast-math --layout=NCHW --target=NEON --threads=1 --type=QASYMM8 --data=data_model/public --image=data/images/299x299.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# inception_v4
# ./build/examples/graph_inception_v4 --fast-math --layout=NCHW --target=NEON --threads=1 --type=QASYMM8 --data=data_model/public --image=data/images/299x299.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# Mobilenet-v2-quantization
# ./build/examples/graph_mobilenet_v2 --fast-math --layout=NCHW --target=NEON --threads=1 --type=QASYMM8 --data=data_model/public --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# Densenet
./build/examples/graph_squeezenet --layout=NHWC --target=NEON --threads=4 --type=F32 --data=data_model/densenet_assets --image=data/images/224x224.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 


######################################################################################################################################################################################

# --enable-cl-cache \
# --no-enable-cl-cache \

# --enable-tuner \
# --no-enable-tuner \

# --fast-math \
# --no-fast-math \

# layout = {NCHW,NHWC,} 
# target = {NEON,CL,GC,} 
# type   = {QASYMM8,F16,F32,} 

# scons Werror=1 debug=0 asserts=0 neon=1 opencl=0 examples=1 os=linux arch=armv7a -j12

# cd ~/Documents/source/open_model_zoo/tools/downloader && \
# python3 downloader.py --name resnext* --output_dir ~/Documents/source/ComputeLibrary/data_model/ && \
# cd ~/Documents/source/ComputeLibrary

# python3 ../../../../scripts/tensorflow_data_extractor.py
# python3 ../../../../scripts/tf_frozen_model_extractor.py
# python3 ../../../../scripts/caffe_data_extractor.py

# exec bash \
# NET_DIR=loss3 && echo $NET_DIR
# sudo mkdir  data_model/public/googlenet-v1/cnn_data/googlenet_model/$NET_DIR && \
# sudo mv data_model/public/googlenet-v1/cnn_data/googlenet_model/$NET_DIR\_* data_model/public/googlenet-v1/cnn_data/googlenet_model/$NET_DIR/
