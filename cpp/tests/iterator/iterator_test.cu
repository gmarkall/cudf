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

#include <cudf/detail/iterator.cuh>                // include iterator header
#include <iterator/transform_unary_functions.cuh>  //for meanvar

#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <bitset>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.hpp>
#include <utilities/legacy/device_operators.cuh>

#include <thrust/equal.h>
#include <thrust/transform.h>

// for reduction tests
#include <cub/device/device_reduce.cuh>
#include <thrust/device_vector.h>

// ---------------------------------------------------------------------------

template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

bool random_bool()
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<int> uniform{0, 1};

  return static_cast<bool>(uniform(engine));
}

template <typename T>
std::ostream& operator<<(std::ostream& os, cudf::meanvar<T> const& rhs)
{
  return os << "[" << rhs.value <<
               ", " << rhs.value_squared << 
               ", " << rhs.count << "] ";
};

auto strings_to_string_views(std::vector<std::string>& input_strings) {
  auto all_valid =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  std::vector<char> chars;
  std::vector<int32_t> offsets;
  std::tie(chars, offsets) = 
    cudf::test::detail::make_chars_and_offsets(
      input_strings.begin(), input_strings.end(), all_valid);
  thrust::device_vector<char> dev_chars(chars);
  char* c_start = thrust::raw_pointer_cast(dev_chars.data());

 // calculate the expected value by CPU. (but contains device pointers)
  std::vector<cudf::string_view> replaced_array(input_strings.size());
  std::transform(thrust::counting_iterator<size_t>(0),
                 thrust::counting_iterator<size_t>(replaced_array.size()),
                 replaced_array.begin(), [c_start, offsets](auto i) {
                   return cudf::string_view(c_start + offsets[i],
                                            offsets[i + 1] - offsets[i]);
                 });
  return std::make_tuple(std::move(dev_chars), replaced_array);
}

// ---------------------------------------------------------------------------

template <typename T>
struct IteratorTest : public GdfTest
{
  // iterator test case which uses cub
  template <typename InputIterator, typename T_output>
  void iterator_test_cub(T_output expected, InputIterator d_in, int num_items)
  {
    T_output init{0};
    thrust::device_vector<T_output> dev_result(1, init);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in,
                              dev_result.begin(), num_items,
                              cudf::DeviceMin{},
                              init);
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in,
                              dev_result.begin(), num_items,
                              cudf::DeviceMin{},
                              init);

    evaluate(expected, dev_result, "cub test");
  }

  // iterator test case which uses thrust
  template <typename InputIterator, typename T_output>
  void iterator_test_thrust(std::vector<T_output>& expected, InputIterator d_in, int num_items) 
  {
    InputIterator d_in_last = d_in + num_items;
    EXPECT_EQ(thrust::distance(d_in, d_in_last), num_items);
    thrust::device_vector<T_output> dev_expected(expected);

    // Can't use this because time_point make_pair bug in libcudacxx
    // bool result = thrust::equal(thrust::device, d_in, d_in_last, dev_expected.begin());
    bool result = thrust::transform_reduce(thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(d_in, dev_expected.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_in_last, dev_expected.end())),
        [] __device__(auto it) { return (thrust::get<0>(it)) == T_output(thrust::get<1>(it)); },
        true,
        thrust::logical_and<bool>());
    #ifndef NDEBUG
    thrust::device_vector<bool> vec(expected.size(), false);
    thrust::transform(thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(d_in, dev_expected.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_in_last, dev_expected.end())),
        vec.begin(),
        [] __device__(auto it) { return (thrust::get<0>(it)) == T_output(thrust::get<1>(it)); }
        );
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<bool>(std::cout, " "));
    std::cout<<std::endl;
    #endif

    EXPECT_TRUE(result) << "thrust test";
  }

  template <typename T_output>
  void evaluate(T_output expected, thrust::device_vector<T_output>& dev_result, const char* msg = nullptr) 
  {
    thrust::host_vector<T_output> hos_result(dev_result);

    EXPECT_EQ(expected, hos_result[0]) << msg;
    std::cout << "Done: expected <" << msg << "> = "
      //<< hos_result[0] //TODO uncomment after time_point ostream operator<< 
      << std::endl;
  }

  template <typename T_output>
  void values_equal_test(std::vector<T_output>& expected, const cudf::column_device_view& col)
  {
    if (col.nullable()) {
      auto it_dev = cudf::experimental::detail::make_null_replacement_iterator(col, T_output{0});
      iterator_test_thrust(expected, it_dev, col.size());
    } else {
      auto it_dev = col.begin<T_output>();
      iterator_test_thrust(expected, it_dev, col.size());
    }
  }
};

// using TestingTypes = cudf::test::NumericTypes;
using TestingTypes = cudf::test::AllTypes;

TYPED_TEST_CASE(IteratorTest, TestingTypes);

// tests for non-null iterator (pointer of device array)
TYPED_TEST(IteratorTest, non_null_iterator) {
  using T = TypeParam;
  std::vector<T> hos_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  thrust::device_vector<T> dev_array(hos_array);

  // calculate the expected value by CPU.
  std::vector<T> replaced_array(hos_array);

  // driven by iterator as a pointer of device array.
  // FIXME: compilation error for cudf::experimental::bool8
  // auto it_dev = dev_array.begin();
  // this->iterator_test_thrust(replaced_array, it_dev, dev_array.size());
  // this->iterator_test_cub(expected_value, it_dev, dev_array.size());

  // test column input
  cudf::test::fixed_width_column_wrapper<T> w_col(hos_array.begin(),
                                                  hos_array.end());
  this->values_equal_test(replaced_array, *cudf::column_device_view::create(w_col));
}

// Tests for null input iterator (column with null bitmap)
// Actually, we can use cub for reduction with nulls without creating custom
// kernel or multiple steps. We may accelarate the reduction for a column using
// cub
TYPED_TEST(IteratorTest, null_iterator) {
  using T = TypeParam;
  T init = T{0};
  // data and valid arrays
  std::vector<T> hos({0, 6, 0, -14, 13, 64, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});

  // create a column with bool vector
  cudf::test::fixed_width_column_wrapper<T> w_col(hos.begin(), hos.end(),
                                                  host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  std::vector<T> replaced_array(hos.size());
  std::transform(hos.begin(), hos.end(), host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T expected_value =
      *std::min_element(replaced_array.begin(), replaced_array.end());
  // TODO uncomment after time_point ostream operator<<
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev = cudf::experimental::detail::make_null_replacement_iterator(*d_col, T{0});
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
  this->values_equal_test(replaced_array, *d_col);
}

// Tests up cast reduction with null iterator.
// The up cast iterator will be created by transform_iterator and
// cudf::experimental::detail::make_null_replacement_iterator(col, T{0})
TYPED_TEST(IteratorTest, null_iterator_upcast) {
  const int column_size{1000};
  using T = int8_t;
  using T_upcast = int64_t;
  T init{0};

  // data and valid arrays
  std::vector<T> hos(column_size);
  std::generate(hos.begin(), hos.end(),
                []() { return static_cast<T>(random_int<T>(-128, 127)); });
  std::vector<bool> host_bools(column_size);
  std::generate(host_bools.begin(), host_bools.end(),
                []() { return static_cast<bool>(random_bool()); });

  cudf::test::fixed_width_column_wrapper<T> w_col(hos.begin(), hos.end(),
                                                  host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  std::vector<T> replaced_array(d_col->size());
  std::transform(hos.begin(), hos.end(), host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T_upcast expected_value =
      *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev = cudf::experimental::detail::make_null_replacement_iterator(*d_col, T{0});
  auto it_dev_upcast =
      thrust::make_transform_iterator(it_dev, thrust::identity<T_upcast>());
  this->iterator_test_thrust(replaced_array, it_dev_upcast, d_col->size());
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
}

// Tests for square input iterator using helper strcut
// `cudf::transformer_squared<T, T_upcast>` The up cast iterator will be created
// by make_transform_iterator(
//        cudf::experimental::detail::make_null_replacement_iterator(col, T{0}), 
//        cudf::detail::transformer_squared<T_upcast>)
TYPED_TEST(IteratorTest, null_iterator_square) {
  const int column_size{1000};
  using T = int8_t;
  using T_upcast = int64_t;
  T init{0};
  cudf::transformer_squared<T_upcast> transformer{};

  // data and valid arrays
  std::vector<T> hos(column_size);
  std::generate(hos.begin(), hos.end(),
                []() { return static_cast<T>(random_int(-128, 128)); });
  std::vector<bool> host_bools(column_size);
  std::generate(host_bools.begin(), host_bools.end(),
                []() { return static_cast<bool>(random_bool()); });

  cudf::test::fixed_width_column_wrapper<T> w_col(hos.begin(), hos.end(),
                                                  host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  std::vector<T_upcast> replaced_array(d_col->size());
  std::transform(hos.begin(), hos.end(), host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x * x : init; });
  T_upcast expected_value =
      *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev = cudf::experimental::detail::make_null_replacement_iterator(*d_col, T{0});
  auto it_dev_upcast = thrust::make_transform_iterator(it_dev, thrust::identity<T_upcast>());
  auto it_dev_squared = thrust::make_transform_iterator(it_dev_upcast, transformer);
  this->iterator_test_thrust(replaced_array, it_dev_squared, d_col->size());
  this->iterator_test_cub(expected_value, it_dev_squared, d_col->size());
}

TYPED_TEST(IteratorTest, large_size_reduction) {
  using T = TypeParam;

  const int column_size{1000000};
  const T init{0};

  // data and valid arrays
  std::vector<T> hos(column_size);
  std::generate(hos.begin(), hos.end(),
                []() { return static_cast<T>(random_int(-128, 128)); });
  std::vector<bool> host_bools(column_size);
  std::generate(host_bools.begin(), host_bools.end(),
                []() { return static_cast<bool>(random_bool()); });

  cudf::test::fixed_width_column_wrapper<TypeParam> w_col(
      hos.begin(), hos.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate by cudf::reduce
  std::vector<T> replaced_array(d_col->size());
  std::transform(hos.begin(), hos.end(), host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev = cudf::experimental::detail::make_null_replacement_iterator(*d_col, init);
  this->iterator_test_thrust(replaced_array, it_dev, d_col->size());
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
}

/*
// TODO: make_pair_iterator and enable this test
// TODO: enable this test also at __CUDACC_DEBUG__
// This test causes fatal compilation error only at device debug mode.
// Workaround: exclude this test only at device debug mode.
#if !defined(__CUDACC_DEBUG__)
// Test for mixed output value using `ColumnOutputMix`
// It computes `count`, `sum`, `sum_of_squares` at a single reduction call.
// It wpuld be useful for `var`, `std` operation
TYPED_TEST(IteratorTest, mean_var_output)
{
    using T = int32_t;
    using T_upcast = int64_t;
    using T_output = cudf::meanvar<T_upcast>;
    cudf::transformer_meanvar<T_upcast> transformer{};

    const int column_size{5000};
    const T_upcast init{0};

    // data and valid arrays
    std::vector<T>  hos(column_size);
    std::generate(hos.begin(), hos.end(),
        []()  { return static_cast<T>(random_int(-128, 128)); });

    std::vector<bool> host_bools(column_size);
    std::generate(host_bools.begin(), host_bools.end(),
        []() { return static_cast<bool>( random_bool() ); } );

    cudf::test::fixed_width_column_wrapper<TypeParam> w_col(hos.begin(),
hos.end(), host_bools.begin()); auto d_col =
cudf::column_device_view::create(w_col);

    // calculate expected values by CPU
    T_output expected_value;

    expected_value.count = d_col->size() - d_col->null_count();

    std::vector<T> replaced_array(d_col->size());
    std::transform(hos.begin(), hos.end(), host_bools.begin(),
        replaced_array.begin(), [&](T x, bool b) { return (b)? x : init; } );

    expected_value.count = d_col->size() - d_col->null_count();
    expected_value.value = std::accumulate(replaced_array.begin(),
replaced_array.end(), T_upcast{0}); expected_value.value_squared =
std::accumulate(replaced_array.begin(), replaced_array.end(), T_upcast{0},
        [](T acc, T i) { return acc + i * i; });

    std::cout << "expected <mixed_output> = " << expected_value << std::endl;

    // GPU test
    // TODO: make_pair_iterator
    auto it_dev = cudf::experimental::detail::make_null_replacement_iterator(*d_col, init);
    //auto it_dev = cudf::make_pair_iterator<true, T>
    //    (static_cast<T*>( w_col.get()->data ), w_col.get()->valid, init);
    auto it_dev_squared = thrust::make_transform_iterator(it_dev, transformer);
    this->iterator_test_thrust(replaced_array, it_dev_squared, d_col->size());
    //this->iterator_test_cub(expected_value, it_dev_squared, d_col->size());
}
#endif
*/

TYPED_TEST(IteratorTest, error_handling) {
  using T = TypeParam;
  std::vector<T> hos_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});

  cudf::test::fixed_width_column_wrapper<T> w_col_no_null(hos_array.begin(),
                                                          hos_array.end());
  cudf::test::fixed_width_column_wrapper<T> w_col_null(hos_array.begin(),
                                                       hos_array.end(),
                                                       host_bools.begin());
  auto d_col_no_null = cudf::column_device_view::create(w_col_no_null);
  auto d_col_null = cudf::column_device_view::create(w_col_null);

  // expects error: data type mismatch
  if (!(std::is_same<T, double>::value)) {
    CUDF_EXPECT_THROW_MESSAGE((d_col_null->begin<double>()),
                              "the data type mismatch");
  }
  // expects error: data type mismatch
  if (!(std::is_same<T, float>::value)) {
    CUDF_EXPECT_THROW_MESSAGE((cudf::experimental::detail::make_null_replacement_iterator(*d_col_null, float{0})),
                              "the data type mismatch");
  }

  CUDF_EXPECT_THROW_MESSAGE((cudf::experimental::detail::make_null_replacement_iterator(*d_col_no_null, T{0})),
                            "Unexpected non-nullable column.");

  CUDF_EXPECT_THROW_MESSAGE((d_col_null->begin<T>()),
                            "Unexpected column with nulls.");
}

struct StringIteratorTest :  public IteratorTest<cudf::string_view> { 
};

TEST_F(StringIteratorTest, string_view_null_iterator ) {
  using T = cudf::string_view;
  // T init = T{"", 0};
  std::string zero("zero");
  // the char data has to be in GPU
  thrust::device_vector<char> initmsg(zero.begin(), zero.end());
  T init = T{initmsg.data().get(), int(initmsg.size())};

  // data and valid arrays
  std::vector<std::string> hos({"one", "two", "three", "four", "five", "six", "eight", "nine"});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});

  // replace nulls in CPU
  std::vector<std::string> replaced_strings(hos.size());
  std::transform(hos.begin(), hos.end(), host_bools.begin(),
                 replaced_strings.begin(),
                 [zero](auto s, auto b) { return b ? s : zero; });

  thrust::device_vector<char> dev_chars;
  std::vector<T> replaced_array(hos.size());
  std::tie(dev_chars, replaced_array) = strings_to_string_views(replaced_strings);

  // create a column with bool vector
  cudf::test::strings_column_wrapper w_col(hos.begin(), hos.end(),
                                           host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);
 
  // GPU test
  auto it_dev = cudf::experimental::detail::make_null_replacement_iterator(*d_col, init);
  this->iterator_test_thrust(replaced_array, it_dev, hos.size());
  // this->values_equal_test(replaced_array, *d_col); //string_view{0} is invalid
}

TEST_F(StringIteratorTest, string_view_no_null_iterator ) {
  using T = cudf::string_view;
  // T init = T{"", 0};
  std::string zero("zero");
  // the char data has to be in GPU
  thrust::device_vector<char> initmsg(zero.begin(), zero.end());
  T init = T{initmsg.data().get(), int(initmsg.size())};

  // data array
  std::vector<std::string> hos({"one", "two", "three", "four", "five", "six", "eight", "nine"});

  thrust::device_vector<char> dev_chars;
  std::vector<T> all_array(hos.size());
  std::tie(dev_chars, all_array) = strings_to_string_views(hos);

  // create a column with bool vector
  cudf::test::strings_column_wrapper w_col(hos.begin(), hos.end());
  auto d_col = cudf::column_device_view::create(w_col);
 
  // GPU test
  auto it_dev = d_col->begin<T>();
  this->iterator_test_thrust(all_array, it_dev, hos.size());
}
