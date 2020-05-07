#############################################################
#
# libgeoda_dylib library
#
#############################################################
cmake_minimum_required(VERSION 2.8.10)

project(geoda)

# Remove 'lib' prefix for shared libraries on Windows
if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()

add_library(${PROJECT_NAME} SHARED
        ${GEODA_SOURCE_FILES}
        ${CMAKE_SOURCE_DIR}/libgeoda/libgeoda.cpp
        )

# Add install target for libgeoda_dylib
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

#  install header files
foreach ( file ${GEODA_INCLUDE_FILES} )
    get_filename_component( dir ${file} DIRECTORY )
    install( FILES ../libgeoda/${file} DESTINATION include/libgeoda/${dir} )
endforeach()

# PRIVATE - use only for this specified build target
# PUBLIC - use it for specified target and for targets which links with this project
# INTERFACE -- use it only for targets which links with the current project

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_SOURCE_DIR}/../eigen3
        ${CMAKE_SOURCE_DIR}/../boost_static/include
        )

# Link test executable against gtest & gtest_main
target_link_libraries(${PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/../boost_static/lib/${DEP_OS_SUB}/libboost_date_time.a
        ${CMAKE_SOURCE_DIR}/../boost_static/lib/${DEP_OS_SUB}/libboost_system.a
        ${CMAKE_SOURCE_DIR}/../boost_static/lib/${DEP_OS_SUB}/libboost_thread.a
)

# Expose libgeoda_dylib's public includes (including the ones of C's dependencies transitively)
# to other subprojects through cache variable.
set(lib${PROJECT_NAME}_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

