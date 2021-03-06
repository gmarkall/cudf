/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>

namespace cudf {
namespace experimental {

/// Factory to create a SUM aggregation
std::unique_ptr<aggregation> make_sum_aggregation() {
  return std::make_unique<aggregation>(aggregation::SUM);
}
/// Factory to create a MIN aggregation
std::unique_ptr<aggregation> make_min_aggregation() {
  return std::make_unique<aggregation>(aggregation::MIN);
}
/// Factory to create a MAX aggregation
std::unique_ptr<aggregation> make_max_aggregation() {
  return std::make_unique<aggregation>(aggregation::MAX);
}
/// Factory to create a COUNT aggregation
std::unique_ptr<aggregation> make_count_aggregation() {
  return std::make_unique<aggregation>(aggregation::COUNT);
}
/// Factory to create a MEAN aggregation
std::unique_ptr<aggregation> make_mean_aggregation() {
  return std::make_unique<aggregation>(aggregation::MEAN);
}
/// Factory to create a MEDIAN aggregation
std::unique_ptr<aggregation> make_median_aggregation() {
  // TODO I think this should just return a quantile_aggregation?
  return std::make_unique<aggregation>(aggregation::MEDIAN);
}
/// Factory to create a QUANTILE aggregation
std::unique_ptr<aggregation> make_quantile_aggregation(
    std::vector<double> const& q, interpolation i) {
  aggregation* a = new detail::quantile_aggregation{q, i};
  return std::unique_ptr<aggregation>(a);
}

namespace detail {
namespace {
template <typename SourceType>
struct dispatch_aggregation_kind {
  template <aggregation::Kind k>
  constexpr data_type operator()() const noexcept {
    return data_type{
        cudf::experimental::type_to_id<target_type_t<SourceType, k>>()};
  }
};

struct dispatch_source_type {
  template <typename SourceType>
  constexpr data_type operator()(aggregation::Kind k) const noexcept {
    return aggregation_dispatcher(k, dispatch_aggregation_kind<SourceType>{});
  }
};
}  // namespace

// Return target data_type for the given source_type and aggregation
data_type target_type(data_type source_type, aggregation::Kind k) {
  return cudf::experimental::type_dispatcher(source_type, dispatch_source_type{}, k);
}
}  // namespace detail
}  // namespace experimental
}  // namespace cudf
