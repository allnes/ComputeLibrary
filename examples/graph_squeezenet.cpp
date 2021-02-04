/*
 * Copyright (c) 2017-2020 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdisabled-optimization"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API */
class GraphSqueezenetExample : public Example
{
public:
    GraphSqueezenetExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "SqueezeNetV1")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 103.94f, 116.78f, 123.68f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb, false, 1. / 58.8235294);

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U, 3U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

#if 1
graph << common_params.target
      << common_params.fast_math_hint
      << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor)))
      << ConvolutionLayer(
          7U, 7U, 64U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(2, 2, 3, 3))
      .set_name("conv1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv1_scale_b.npy"))
      .set_name("conv1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
      << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, operation_layout, PadStrideInfo(2, 2, 0, 0))).set_name("pool1");
SubStream left_0(graph);
SubStream right_0(graph);
right_0 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_1/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x1_scale_b.npy"))
      .set_name("conv2_1/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_1/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv2_1/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_1/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x2_scale_b.npy"))
      .set_name("conv2_1/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_1/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_1_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv2_1/x2");
graph 
      << ConcatLayer(std::move(left_0), std::move(right_0)).set_name("concat_2_1");
SubStream left_1(graph);
SubStream right_1(graph);
right_1 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-05)
      .set_name("conv2_2/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x1_scale_b.npy"))
      .set_name("conv2_2/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_2/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv2_2/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-05)
      .set_name("conv2_2/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x2_scale_b.npy"))
      .set_name("conv2_2/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_2/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_2_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv2_2/x2");
graph 
      << ConcatLayer(std::move(left_1), std::move(right_1)).set_name("concat_2_2");
SubStream left_2(graph);
SubStream right_2(graph);
right_2 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_3/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x1_scale_b.npy"))
      .set_name("conv2_3/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_3/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv2_3/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_3/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x2_scale_b.npy"))
      .set_name("conv2_3/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_3/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_3_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv2_3/x2");
graph 
      << ConcatLayer(std::move(left_2), std::move(right_2)).set_name("concat_2_3");
SubStream left_3(graph);
SubStream right_3(graph);
right_3 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_4/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x1_scale_b.npy"))
      .set_name("conv2_4/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_4/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv2_4/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_4/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x2_scale_b.npy"))
      .set_name("conv2_4/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_4/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_4_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv2_4/x2");
graph 
      << ConcatLayer(std::move(left_3), std::move(right_3)).set_name("concat_2_4");
SubStream left_4(graph);
SubStream right_4(graph);
right_4 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_5/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x1_scale_b.npy"))
      .set_name("conv2_5/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_5/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv2_5/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_5/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x2_scale_b.npy"))
      .set_name("conv2_5/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_5/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_5_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv2_5/x2");
graph 
      << ConcatLayer(std::move(left_4), std::move(right_4)).set_name("concat_2_5");
SubStream left_5(graph);
SubStream right_5(graph);
right_5 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_6/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x1_scale_b.npy"))
      .set_name("conv2_6/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_6/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv2_6/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_6/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x2_scale_b.npy"))
      .set_name("conv2_6/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_6/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_6_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv2_6/x2");
graph 
      << ConcatLayer(std::move(left_5), std::move(right_5)).set_name("concat_2_6")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_blk_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_blk_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv2_blk/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_blk_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_blk_scale_b.npy"))
      .set_name("conv2_blk/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2_blk")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv2_blk_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv2_blk")
      << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 2, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool2");
	  
SubStream left_6(graph);
SubStream right_6(graph);
right_6 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_1/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x1_scale_b.npy"))
      .set_name("conv3_1/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_1/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_1/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_1/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x2_scale_b.npy"))
      .set_name("conv3_1/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_1/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_1_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_1/x2");
graph 
      << ConcatLayer(std::move(left_6), std::move(right_6)).set_name("concat_3_1");
SubStream left_7(graph);
SubStream right_7(graph);
right_7 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_2/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x1_scale_b.npy"))
      .set_name("conv3_2/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_2/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_2/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_2/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x2_scale_b.npy"))
      .set_name("conv3_2/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_2/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_2_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_2/x2");
graph 
      << ConcatLayer(std::move(left_7), std::move(right_7)).set_name("concat_3_2");
SubStream left_8(graph);
SubStream right_8(graph);
right_8 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_3/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x1_scale_b.npy"))
      .set_name("conv3_3/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_3/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_3/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_3/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x2_scale_b.npy"))
      .set_name("conv3_3/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_3/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_3_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_3/x2");
graph 
      << ConcatLayer(std::move(left_8), std::move(right_8)).set_name("concat_3_3");
SubStream left_9(graph);
SubStream right_9(graph);
right_9 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_4/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x1_scale_b.npy"))
      .set_name("conv3_4/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_4/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_4/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_4/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x2_scale_b.npy"))
      .set_name("conv3_4/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_4/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_4_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_4/x2");
graph 
      << ConcatLayer(std::move(left_9), std::move(right_9)).set_name("concat_3_4");
SubStream left_10(graph);
SubStream right_10(graph);
right_10 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_5/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x1_scale_b.npy"))
      .set_name("conv3_5/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_5/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_5/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_5/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x2_scale_b.npy"))
      .set_name("conv3_5/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_5/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_5_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_5/x2");
graph 
      << ConcatLayer(std::move(left_10), std::move(right_10)).set_name("concat_3_5");
SubStream left_11(graph);
SubStream right_11(graph);
right_11 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_6/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x1_scale_b.npy"))
      .set_name("conv3_6/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_6/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_6/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_6/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x2_scale_b.npy"))
      .set_name("conv3_6/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_6/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_6_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_6/x2");
graph 
      << ConcatLayer(std::move(left_11), std::move(right_11)).set_name("concat_3_6");
SubStream left_12(graph);
SubStream right_12(graph);
right_12 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_7/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x1_scale_b.npy"))
      .set_name("conv3_7/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_7/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_7/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_7/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x2_scale_b.npy"))
      .set_name("conv3_7/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_7/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_7_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_7/x2");
graph 
      << ConcatLayer(std::move(left_12), std::move(right_12)).set_name("concat_3_7");
SubStream left_13(graph);
SubStream right_13(graph);
right_13 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_8/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x1_scale_b.npy"))
      .set_name("conv3_8/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_8/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_8/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_8/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x2_scale_b.npy"))
      .set_name("conv3_8/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_8/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_8_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_8/x2");
graph 
      << ConcatLayer(std::move(left_13), std::move(right_13)).set_name("concat_3_8");
SubStream left_14(graph);
SubStream right_14(graph);
right_14 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_9/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x1_scale_b.npy"))
      .set_name("conv3_9/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_9/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_9/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_9/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x2_scale_b.npy"))
      .set_name("conv3_9/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_9/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_9_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_9/x2");
graph 
      << ConcatLayer(std::move(left_14), std::move(right_14)).set_name("concat_3_9");
SubStream left_15(graph);
SubStream right_15(graph);
right_15 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_10/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x1_scale_b.npy"))
      .set_name("conv3_10/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_10/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_10/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_10/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x2_scale_b.npy"))
      .set_name("conv3_10/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_10/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_10_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_10/x2");
graph 
      << ConcatLayer(std::move(left_15), std::move(right_15)).set_name("concat_3_10");
SubStream left_16(graph);
SubStream right_16(graph);
right_16 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_11/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x1_scale_b.npy"))
      .set_name("conv3_11/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_11/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_11/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_11/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x2_scale_b.npy"))
      .set_name("conv3_11/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_11/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_11_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_11/x2");
graph 
      << ConcatLayer(std::move(left_16), std::move(right_16)).set_name("concat_3_11");
SubStream left_17(graph);
SubStream right_17(graph);
right_17 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_12/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x1_scale_b.npy"))
      .set_name("conv3_12/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_12/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_12/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_12/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x2_scale_b.npy"))
      .set_name("conv3_12/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_12/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_12_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv3_12/x2");
graph 
      << ConcatLayer(std::move(left_17), std::move(right_17)).set_name("concat_3_12")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_blk_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_blk_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv3_blk/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_blk_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_blk_scale_b.npy"))
      .set_name("conv3_blk/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3_blk")
      << ConvolutionLayer(
          1U, 1U, 256U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv3_blk_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv3_blk")
      << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 2, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool3");
SubStream left_18(graph);
SubStream right_18(graph);
right_18 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_1/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x1_scale_b.npy"))
      .set_name("conv4_1/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_1/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_1/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_1/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x2_scale_b.npy"))
      .set_name("conv4_1/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_1/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_1_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_1/x2");
graph 
      << ConcatLayer(std::move(left_18), std::move(right_18)).set_name("concat_4_1");
SubStream left_19(graph);
SubStream right_19(graph);
right_19 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_2/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x1_scale_b.npy"))
      .set_name("conv4_2/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_2/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_2/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_2/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x2_scale_b.npy"))
      .set_name("conv4_2/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_2/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_2_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_2/x2");
graph 
      << ConcatLayer(std::move(left_19), std::move(right_19)).set_name("concat_4_2");
SubStream left_20(graph);
SubStream right_20(graph);
right_20 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_3/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x1_scale_b.npy"))
      .set_name("conv4_3/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_3/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_3/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_3/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x2_scale_b.npy"))
      .set_name("conv4_3/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_3/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_3_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_3/x2");
graph 
      << ConcatLayer(std::move(left_20), std::move(right_20)).set_name("concat_4_3");
SubStream left_21(graph);
SubStream right_21(graph);
right_21 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_4/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x1_scale_b.npy"))
      .set_name("conv4_4/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_4/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_4/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_4/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x2_scale_b.npy"))
      .set_name("conv4_4/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_4/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_4_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_4/x2");
graph 
      << ConcatLayer(std::move(left_21), std::move(right_21)).set_name("concat_4_4");
SubStream left_22(graph);
SubStream right_22(graph);
right_22 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_5/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x1_scale_b.npy"))
      .set_name("conv4_5/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_5/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_5/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_5/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x2_scale_b.npy"))
      .set_name("conv4_5/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_5/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_5_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_5/x2");
graph 
      << ConcatLayer(std::move(left_22), std::move(right_22)).set_name("concat_4_5");
SubStream left_23(graph);
SubStream right_23(graph);
right_23 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_6/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x1_scale_b.npy"))
      .set_name("conv4_6/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_6/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_6/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_6/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x2_scale_b.npy"))
      .set_name("conv4_6/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_6/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_6_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_6/x2");
graph 
      << ConcatLayer(std::move(left_23), std::move(right_23)).set_name("concat_4_6");
SubStream left_24(graph);
SubStream right_24(graph);
right_24 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_7/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x1_scale_b.npy"))
      .set_name("conv4_7/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_7/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_7/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_7/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x2_scale_b.npy"))
      .set_name("conv4_7/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_7/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_7_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_7/x2");
graph 
      << ConcatLayer(std::move(left_24), std::move(right_24)).set_name("concat_4_7");
SubStream left_25(graph);
SubStream right_25(graph);
right_25 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_8/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x1_scale_b.npy"))
      .set_name("conv4_8/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_8/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_8/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_8/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x2_scale_b.npy"))
      .set_name("conv4_8/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_8/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_8_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_8/x2");
graph 
      << ConcatLayer(std::move(left_25), std::move(right_25)).set_name("concat_4_8");
SubStream left_26(graph);
SubStream right_26(graph);
right_26 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_9/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x1_scale_b.npy"))
      .set_name("conv4_9/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_9/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_9/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_9/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x2_scale_b.npy"))
      .set_name("conv4_9/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_9/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_9_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_9/x2");
graph 
      << ConcatLayer(std::move(left_26), std::move(right_26)).set_name("concat_4_9");
SubStream left_27(graph);
SubStream right_27(graph);
right_27 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_10/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x1_scale_b.npy"))
      .set_name("conv4_10/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_10/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_10/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_10/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x2_scale_b.npy"))
      .set_name("conv4_10/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_10/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_10_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_10/x2");
graph 
      << ConcatLayer(std::move(left_27), std::move(right_27)).set_name("concat_4_10");
SubStream left_28(graph);
SubStream right_28(graph);
right_28 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_11/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x1_scale_b.npy"))
      .set_name("conv4_11/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_11/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_11/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_11/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x2_scale_b.npy"))
      .set_name("conv4_11/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_11/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_11_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_11/x2");
graph 
      << ConcatLayer(std::move(left_28), std::move(right_28)).set_name("concat_4_11");
SubStream left_29(graph);
SubStream right_29(graph);
right_29 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_12/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x1_scale_b.npy"))
      .set_name("conv4_12/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_12/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_12/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_12/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x2_scale_b.npy"))
      .set_name("conv4_12/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_12/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_12_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_12/x2");
graph 
      << ConcatLayer(std::move(left_29), std::move(right_29)).set_name("concat_4_12");
SubStream left_30(graph);
SubStream right_30(graph);
right_30 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_13/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x1_scale_b.npy"))
      .set_name("conv4_13/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_13/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_13/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_13/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x2_scale_b.npy"))
      .set_name("conv4_13/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_13/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_13_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_13/x2");
graph 
      << ConcatLayer(std::move(left_30), std::move(right_30)).set_name("concat_4_13");
SubStream left_31(graph);
SubStream right_31(graph);
right_31 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_14/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x1_scale_b.npy"))
      .set_name("conv4_14/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_14/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_14/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_14/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x2_scale_b.npy"))
      .set_name("conv4_14/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_14/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_14_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_14/x2");
graph 
      << ConcatLayer(std::move(left_31), std::move(right_31)).set_name("concat_4_14");
SubStream left_32(graph);
SubStream right_32(graph);
right_32 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_15/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x1_scale_b.npy"))
      .set_name("conv4_15/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_15/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_15/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_15/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x2_scale_b.npy"))
      .set_name("conv4_15/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_15/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_15_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_15/x2");
graph 
      << ConcatLayer(std::move(left_32), std::move(right_32)).set_name("concat_4_15");
SubStream left_33(graph);
SubStream right_33(graph);
right_33 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_16/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x1_scale_b.npy"))
      .set_name("conv4_16/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_16/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_16/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_16/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x2_scale_b.npy"))
      .set_name("conv4_16/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_16/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_16_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_16/x2");
graph 
      << ConcatLayer(std::move(left_33), std::move(right_33)).set_name("concat_4_16");
SubStream left_34(graph);
SubStream right_34(graph);
right_34 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_17/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x1_scale_b.npy"))
      .set_name("conv4_17/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_17/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_17/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_17/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x2_scale_b.npy"))
      .set_name("conv4_17/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_17/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_17_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_17/x2");
graph 
      << ConcatLayer(std::move(left_34), std::move(right_34)).set_name("concat_4_17");
SubStream left_35(graph);
SubStream right_35(graph);
right_35 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_18/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x1_scale_b.npy"))
      .set_name("conv4_18/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_18/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_18/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_18/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x2_scale_b.npy"))
      .set_name("conv4_18/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_18/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_18_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_18/x2");
graph 
      << ConcatLayer(std::move(left_35), std::move(right_35)).set_name("concat_4_18");
SubStream left_36(graph);
SubStream right_36(graph);
right_36 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_19/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x1_scale_b.npy"))
      .set_name("conv4_19/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_19/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_19/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_19/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x2_scale_b.npy"))
      .set_name("conv4_19/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_19/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_19_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_19/x2");
graph 
      << ConcatLayer(std::move(left_36), std::move(right_36)).set_name("concat_4_19");
SubStream left_37(graph);
SubStream right_37(graph);
right_37 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_20/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x1_scale_b.npy"))
      .set_name("conv4_20/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_20/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_20/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_20/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x2_scale_b.npy"))
      .set_name("conv4_20/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_20/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_20_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_20/x2");
graph 
      << ConcatLayer(std::move(left_37), std::move(right_37)).set_name("concat_4_20");
SubStream left_38(graph);
SubStream right_38(graph);
right_38 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_21/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x1_scale_b.npy"))
      .set_name("conv4_21/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_21/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_21/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_21/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x2_scale_b.npy"))
      .set_name("conv4_21/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_21/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_21_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_21/x2");
graph 
      << ConcatLayer(std::move(left_38), std::move(right_38)).set_name("concat_4_21");
SubStream left_39(graph);
SubStream right_39(graph);
right_39 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_22/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x1_scale_b.npy"))
      .set_name("conv4_22/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_22/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_22/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_22/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x2_scale_b.npy"))
      .set_name("conv4_22/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_22/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_22_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_22/x2");
graph 
      << ConcatLayer(std::move(left_39), std::move(right_39)).set_name("concat_4_22");
SubStream left_40(graph);
SubStream right_40(graph);
right_40 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_23/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x1_scale_b.npy"))
      .set_name("conv4_23/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_23/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_23/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_23/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x2_scale_b.npy"))
      .set_name("conv4_23/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_23/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_23_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_23/x2");
graph 
      << ConcatLayer(std::move(left_40), std::move(right_40)).set_name("concat_4_23");
SubStream left_41(graph);
SubStream right_41(graph);
right_41 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_24/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x1_scale_b.npy"))
      .set_name("conv4_24/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_24/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_24/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_24/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x2_scale_b.npy"))
      .set_name("conv4_24/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_24/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_24_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv4_24/x2");
graph 
      << ConcatLayer(std::move(left_41), std::move(right_41)).set_name("concat_4_24")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_blk_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_blk_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv4_blk/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_blk_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_blk_scale_b.npy"))
      .set_name("conv4_blk/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4_blk")
      << ConvolutionLayer(
          1U, 1U, 512U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv4_blk_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv4_blk")
      << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 2, operation_layout, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL))).set_name("pool4");
SubStream left_42(graph);
SubStream right_42(graph);
right_42 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_1/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x1_scale_b.npy"))
      .set_name("conv5_1/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_1/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_1/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_1/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x2_scale_b.npy"))
      .set_name("conv5_1/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_1/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_1_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_1/x2");
graph 
      << ConcatLayer(std::move(left_42), std::move(right_42)).set_name("concat_5_1");
SubStream left_43(graph);
SubStream right_43(graph);
right_43 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_2/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x1_scale_b.npy"))
      .set_name("conv5_2/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_2/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_2/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_2/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x2_scale_b.npy"))
      .set_name("conv5_2/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_2/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_2_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_2/x2");
graph 
      << ConcatLayer(std::move(left_43), std::move(right_43)).set_name("concat_5_2");
SubStream left_44(graph);
SubStream right_44(graph);
right_44 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_3/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x1_scale_b.npy"))
      .set_name("conv5_3/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_3/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_3/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_3/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x2_scale_b.npy"))
      .set_name("conv5_3/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_3/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_3_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_3/x2");
graph 
      << ConcatLayer(std::move(left_44), std::move(right_44)).set_name("concat_5_3");
SubStream left_45(graph);
SubStream right_45(graph);
right_45 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_4/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x1_scale_b.npy"))
      .set_name("conv5_4/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_4/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_4/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_4/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x2_scale_b.npy"))
      .set_name("conv5_4/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_4/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_4_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_4/x2");
graph 
      << ConcatLayer(std::move(left_45), std::move(right_45)).set_name("concat_5_4");
SubStream left_46(graph);
SubStream right_46(graph);
right_46 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_5/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x1_scale_b.npy"))
      .set_name("conv5_5/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_5/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_5/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_5/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x2_scale_b.npy"))
      .set_name("conv5_5/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_5/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_5_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_5/x2");
graph 
      << ConcatLayer(std::move(left_46), std::move(right_46)).set_name("concat_5_5");
SubStream left_47(graph);
SubStream right_47(graph);
right_47 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_6/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x1_scale_b.npy"))
      .set_name("conv5_6/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_6/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_6/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_6/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x2_scale_b.npy"))
      .set_name("conv5_6/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_6/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_6_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_6/x2");
graph 
      << ConcatLayer(std::move(left_47), std::move(right_47)).set_name("concat_5_6");
SubStream left_48(graph);
SubStream right_48(graph);
right_48 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_7/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x1_scale_b.npy"))
      .set_name("conv5_7/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_7/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_7/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_7/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x2_scale_b.npy"))
      .set_name("conv5_7/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_7/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_7_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_7/x2");
graph 
      << ConcatLayer(std::move(left_48), std::move(right_48)).set_name("concat_5_7");
SubStream left_49(graph);
SubStream right_49(graph);
right_49 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_8/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x1_scale_b.npy"))
      .set_name("conv5_8/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_8/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_8/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_8/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x2_scale_b.npy"))
      .set_name("conv5_8/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_8/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_8_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_8/x2");
graph 
      << ConcatLayer(std::move(left_49), std::move(right_49)).set_name("concat_5_8");
SubStream left_50(graph);
SubStream right_50(graph);
right_50 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_9/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x1_scale_b.npy"))
      .set_name("conv5_9/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_9/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_9/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_9/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x2_scale_b.npy"))
      .set_name("conv5_9/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_9/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_9_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_9/x2");
graph 
      << ConcatLayer(std::move(left_50), std::move(right_50)).set_name("concat_5_9");
SubStream left_51(graph);
SubStream right_51(graph);
right_51 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_10/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x1_scale_b.npy"))
      .set_name("conv5_10/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_10/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_10/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_10/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x2_scale_b.npy"))
      .set_name("conv5_10/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_10/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_10_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_10/x2");
graph 
      << ConcatLayer(std::move(left_51), std::move(right_51)).set_name("concat_5_10");
SubStream left_52(graph);
SubStream right_52(graph);
right_52 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_11/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x1_scale_b.npy"))
      .set_name("conv5_11/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_11/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_11/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_11/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x2_scale_b.npy"))
      .set_name("conv5_11/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_11/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_11_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_11/x2");
graph 
      << ConcatLayer(std::move(left_52), std::move(right_52)).set_name("concat_5_11");
SubStream left_53(graph);
SubStream right_53(graph);
right_53 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_12/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x1_scale_b.npy"))
      .set_name("conv5_12/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_12/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_12/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_12/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x2_scale_b.npy"))
      .set_name("conv5_12/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_12/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_12_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_12/x2");
graph 
      << ConcatLayer(std::move(left_53), std::move(right_53)).set_name("concat_5_12");
SubStream left_54(graph);
SubStream right_54(graph);
right_54 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_13/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x1_scale_b.npy"))
      .set_name("conv5_13/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_13/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_13/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_13/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x2_scale_b.npy"))
      .set_name("conv5_13/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_13/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_13_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_13/x2");
graph 
      << ConcatLayer(std::move(left_54), std::move(right_54)).set_name("concat_5_13");
SubStream left_55(graph);
SubStream right_55(graph);
right_55 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_14/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x1_scale_b.npy"))
      .set_name("conv5_14/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_14/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_14/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_14/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x2_scale_b.npy"))
      .set_name("conv5_14/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_14/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_14_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_14/x2");
graph 
      << ConcatLayer(std::move(left_55), std::move(right_55)).set_name("concat_5_14");
SubStream left_56(graph);
SubStream right_56(graph);
right_56 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_15/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x1_scale_b.npy"))
      .set_name("conv5_15/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_15/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_15/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_15/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x2_scale_b.npy"))
      .set_name("conv5_15/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_15/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_15_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_15/x2");
graph 
      << ConcatLayer(std::move(left_56), std::move(right_56)).set_name("concat_5_15");
SubStream left_57(graph);
SubStream right_57(graph);
right_57 
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x1_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x1_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_16/x1/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x1_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x1_scale_b.npy"))
      .set_name("conv5_16/x1/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_16/x1")
      << ConvolutionLayer(
          1U, 1U, 128U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x1_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("conv5_16/x1")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x2_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x2_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_16/x2/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x2_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x2_scale_b.npy"))
      .set_name("conv5_16/x2/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_16/x2")
      << ConvolutionLayer(
          3U, 3U, 32U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_16_x2_w.npy", weights_layout),
          std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
          PadStrideInfo(1, 1, 1, 1))
      .set_name("conv5_16/x2");
graph 
      << ConcatLayer(std::move(left_57), std::move(right_57)).set_name("concat_5_16")
      << BatchNormalizationLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_blk_bn_w.npy"),
                                 get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_blk_bn_b.npy"),
                                 get_random_accessor(1.f, 1.f),
                                 get_random_accessor(1.f, 1.f),
                                 1e-5)
      .set_name("conv5_blk/bn")
      << ScaleLayer(get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_blk_scale_w.npy"),
                    get_weights_accessor(data_path, "/cnn_data/densenet_assets/conv5_blk_scale_b.npy"))
      .set_name("conv5_blk/scale")
      << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5_blk")
      << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, operation_layout)).set_name("pool5")
      << ConvolutionLayer(
          1U, 1U, 1000U,
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/fc6_w.npy", weights_layout),
          get_weights_accessor(data_path, "/cnn_data/densenet_assets/fc6_b.npy"),
          PadStrideInfo(1, 1, 0, 0))
      .set_name("fc6")
	  << SoftmaxLayer().set_name("prob")
      << OutputLayer(get_output_accessor(common_params, 5));

#endif



        // Finalize graph
        GraphConfig config;
        config.num_threads      = common_params.threads;
        config.use_tuner        = common_params.enable_tuner;
        config.tuner_mode       = common_params.tuner_mode;
        config.tuner_file       = common_params.tuner_file;
        config.convert_to_uint8 = (common_params.data_type == DataType::QASYMM8);

        graph.finalize(common_params.target, config);

        return true;
    }
    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

};

/** Main program for Squeezenet v1.0
 *
 * Model is based on:
 *      https://arxiv.org/abs/1602.07360
 *      "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
 *      Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
 *
 * Provenance: https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphSqueezenetExample>(argc, argv);
}

#pragma GCC diagnostic pop
