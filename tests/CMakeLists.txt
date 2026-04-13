enable_testing()

find_package(GTest REQUIRED)

include(GoogleTest)


add_executable(dim_test unit/dim_test.cpp)

target_link_libraries(dim_test PRIVATE dim GTest::gtest_main)

gtest_discover_tests(dim_test)


add_executable(tensor_test unit/tensor_test.cpp)

target_link_libraries(tensor_test PRIVATE tensor GTest::gtest_main)

gtest_discover_tests(tensor_test)


add_executable(gemm_test unit/gemm_test.cpp)

target_link_libraries(gemm_test PRIVATE gemm GTest::gtest_main)

gtest_discover_tests(gemm_test)


add_executable(convolution_test unit/convolution_test.cpp)

target_link_libraries(convolution_test PRIVATE convolution GTest::gtest_main)

gtest_discover_tests(convolution_test)


find_program(BASH_EXECUTABLE bash REQUIRED)
file (GLOB testfiles "e2e/io/*.dat")
foreach (file ${testfiles})
    add_test(NAME "${file}_Naive" 
        COMMAND bash -c "${CMAKE_SOURCE_DIR}/tests/e2e/run.sh ${file} \"./main -n\""
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        
    )
    set_tests_properties("${file}_Naive" PROPERTIES DEPENDS main)

    add_test(NAME "${file}_Im2col" 
        COMMAND bash -c "${CMAKE_SOURCE_DIR}/tests/e2e/run.sh ${file} \"./main -im\""
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        
    )
    set_tests_properties("${file}_Im2col" PROPERTIES DEPENDS main)
endforeach()
