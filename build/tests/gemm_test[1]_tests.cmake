add_test([=[GemmNaiveTest.BasicCase]=]  /mnt/windows/Media/Programs/SberConvolution_of_4D_tensors/build/tests/gemm_test [==[--gtest_filter=GemmNaiveTest.BasicCase]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[GemmNaiveTest.BasicCase]=]  PROPERTIES WORKING_DIRECTORY /mnt/windows/Media/Programs/SberConvolution_of_4D_tensors/build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  gemm_test_TESTS GemmNaiveTest.BasicCase)
