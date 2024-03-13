cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(DispToH VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the library
add_library(disp_to_h SHARED disp_to_h.c rpc.o iio.o)

# Link against necessary libraries
target_link_libraries(disp_to_h z tiff png jpeg m)

# Set the destination directory
set(DEST_DIR ${CMAKE_SOURCE_DIR}/../lib)

# Add the custom command to copy the .so file to the destination directory
add_custom_command(TARGET disp_to_h POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:disp_to_h> ${DEST_DIR}/disp_to_h.so)
message(STATUS "Post build command: ${CMAKE_COMMAND} -E copy $<TARGET_FILE:disp_to_h> ${DEST_DIR}/disp_to_h.so")