cmake_minimum_required( VERSION 3.4 )
project( rgbd_stream_converter )

# libcgt
set( LIBCGT_DIR ../.. )
include_directories( ${LIBCGT_DIR}/build/lib/include )
include_directories( ${LIBCGT_DIR}/build/lib/include/core )
link_directories( ${LIBCGT_DIR}/build/lib/lib )

file( GLOB_RECURSE HEADERS src/*.h src/*.inl )
file( GLOB_RECURSE SOURCES src/*.cpp )

add_executable( rgbd_stream_converter ${HEADERS} ${SOURCES} )
set_property( TARGET rgbd_stream_converter PROPERTY CXX_STANDARD 11 )

target_link_libraries( rgbd_stream_converter
    debug cgt_cored
    debug cgt_camera_wrappersd
    optimized cgt_core
    optimized cgt_camera_wrappers
)
