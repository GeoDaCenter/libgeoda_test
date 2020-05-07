
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################
# Add test cpp file
set (TEST_SOURCE_FILES
        ./test_weights.cpp
        ./test_localSA.cpp
        ./test_clustering.cpp
        ./test_algorithms.cpp)

add_executable( runUnitTests ${TEST_SOURCE_FILES} )

set_target_properties(runUnitTests PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS YES)

target_include_directories(runUnitTests PUBLIC
        ${gtest_SOURCE_DIR}/include
        ${gmock_SOURCE_DIR}/include
        ${libgeoda_INCLUDE_DIRS}
        )
# Link test executable against gtest & gtest_main
target_link_libraries(runUnitTests
        gtest
        gtest_main
        geoda
        )

add_test( runUnitTests runUnitTests )
