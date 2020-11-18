export LD_LIBRARY_PATH=build/

./build/examples/graph_googlenet --layout=NCHW --target=NEON --threads=4 --type=F32 --data=data_model/public/googlenet-v1 --image=data/images/test.ppm --labels=data/imagenet1000_clsidx_to_labels.txt 

# --enable-cl-cache \
# --no-enable-cl-cache \

# --enable-tuner \
# --no-enable-tuner \

# --fast-math \
# --no-fast-math \

# layout = {NCHW,NHWC,} 
# target = {NEON,CL,GC,} 
# type   = {QASYMM8,F16,F32,} 

# scons Werror=1 debug=0 asserts=0 neon=1 opencl=0 examples=1 os=linux arch=armv7a -j1
# python3 downloader.py --name resnet-50* --output_dir ~/Documents/source/ComputeLibrary/data_model/

# exec bash \
# NET_DIR=loss3 && echo $NET_DIR
# sudo mkdir  data_model/public/googlenet-v1/cnn_data/googlenet_model/$NET_DIR && \
# sudo mv data_model/public/googlenet-v1/cnn_data/googlenet_model/$NET_DIR\_* data_model/public/googlenet-v1/cnn_data/googlenet_model/$NET_DIR/
