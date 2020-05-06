#############################################################
#
# libgeoda library
#
#############################################################
add_library(geoda STATIC
        ${GEODA_SOURCE_FILES}
        ${CMAKE_SOURCE_DIR}/src/libgeoda.cpp
        )

# Add install target for libgeoda
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include)

# PRIVATE - use only for this specified build target
# PUBLIC - use it for specified target and for targets which links with this project
# INTERFACE -- use it only for targets which links with the current project

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (MINGW)
        target_include_directories(geoda PUBLIC ${CMAKE_SOURCE_DIR}/../boost_static/include)
        target_include_directories(geoda PUBLIC ${CMAKE_SOURCE_DIR}/../eigen3)
        target_include_directories(geoda PUBLIC ${PROJECT_SOURCE_DIR}/src)

else()
        target_include_directories(geoda PUBLIC ${CMAKE_SOURCE_DIR}/../eigen3)
        target_include_directories(geoda PUBLIC ${CMAKE_SOURCE_DIR}/../boost_static/include)
        target_include_directories(geoda PUBLIC ${PROJECT_SOURCE_DIR}/src)
endif()


#  copy files
foreach(hfile ${GEODA_INCLUDE_FILES})
    get_filename_component(barename ${hfile} NAME)
    add_custom_command(
            TARGET geoda POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy ${hfile} ${DEP_LIBRARY_PATH}/libgeoda_static/include/${barename})
endforeach()

add_custom_command(
        TARGET geoda POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy $<TARGET_FILE:geoda>
        ${DEP_LIBRARY_PATH}/libgeoda_static/lib/${DEP_OS_SUB}/libgeoda.a)

