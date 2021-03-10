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
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCBatchNormalizationLayer.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/RandomBatchNormalizationLayerDataset.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BatchNormalizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> tolerance_f(0.00001f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr AbsoluteTolerance<float> tolerance_f16(0.01f);  /**< Tolerance value for comparing reference's output against implementation's output for DataType::F16 */
const auto                         act_infos = framework::dataset::make("ActivationInfo",
{
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f),
    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 8.f, 2.f),
});

const auto data_GB = combine(framework::dataset::make("UseGamma", { false, true }),
                             framework::dataset::make("UseBeta", { false, true }));
const auto data_f16 = combine(combine(combine(data_GB, act_infos), framework::dataset::make("DataType", DataType::F16)),
                              framework::dataset::make("DataLayout", { DataLayout::NCHW }));
const auto data_f32 = combine(combine(combine(data_GB, act_infos), framework::dataset::make("DataType", DataType::F32)),
                              framework::dataset::make("DataLayout", { DataLayout::NCHW }));
} // namespace

TEST_SUITE(GC)
TEST_SUITE(BatchNormalizationLayer)

template <typename T>
using GCBatchNormalizationLayerFixture = BatchNormalizationLayerValidationFixture<GCTensor, GCAccessor, GCBatchNormalizationLayer, T>;

TEST_SUITE(Float)
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(Random, GCBatchNormalizationLayerFixture<half>, framework::DatasetMode::NIGHTLY, combine(datasets::SmallRandomBatchNormalizationLayerDataset(), data_f16))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f16, 0);
}
TEST_SUITE_END()

TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(Random, GCBatchNormalizationLayerFixture<float>, framework::DatasetMode::NIGHTLY, combine(datasets::LargeRandomBatchNormalizationLayerDataset(), data_f32))
{
    // Validate output
    validate(GCAccessor(_target), _reference, tolerance_f, 0);
}
TEST_SUITE_END()
TEST_SUITE_END()

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
