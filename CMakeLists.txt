cmake_minimum_required(VERSION 3.16)
project(vivid_plexus)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED VIVID_SRC_DIR)
  set(VIVID_SRC_DIR "${CMAKE_SOURCE_DIR}/../vivid")
endif()
if(NOT DEFINED VIVID_BUILD_DIR)
  set(VIVID_BUILD_DIR "${VIVID_SRC_DIR}/build")
endif()
if(NOT DEFINED VIVID_PLUGIN_SUFFIX)
  set(VIVID_PLUGIN_SUFFIX .dylib)
endif()

add_library(vivid_operator_api INTERFACE)
target_include_directories(vivid_operator_api INTERFACE
  ${VIVID_SRC_DIR}/src
)

# Prefer the host runtime's libwgpu_native to avoid loading a second WebGPU
# runtime in package plugins.
set(_wgpu_include "")
set(_wgpu_lib_dir "")

if(EXISTS "${VIVID_SRC_DIR}/include/webgpu/webgpu.h")
  set(_wgpu_include "${VIVID_SRC_DIR}/include")
endif()

if(EXISTS "${VIVID_BUILD_DIR}/vivid.app/Contents/MacOS/libwgpu_native.dylib")
  set(_wgpu_lib_dir "${VIVID_BUILD_DIR}/vivid.app/Contents/MacOS")
elseif(EXISTS "${VIVID_BUILD_DIR}/libwgpu_native.dylib")
  set(_wgpu_lib_dir "${VIVID_BUILD_DIR}")
endif()

# Fallback for unusual build trees
if(_wgpu_include STREQUAL "" OR _wgpu_lib_dir STREQUAL "")
  set(_deps_dir "${VIVID_BUILD_DIR}/_deps")
  if(IS_DIRECTORY "${_deps_dir}")
    file(GLOB _dep_entries "${_deps_dir}/*")
    foreach(_entry ${_dep_entries})
      get_filename_component(_entry_name "${_entry}" NAME)
      if(_entry_name MATCHES "wgpu.*-src")
        if(_wgpu_include STREQUAL "" AND EXISTS "${_entry}/include/webgpu/webgpu.h")
          set(_wgpu_include "${_entry}/include")
        endif()
        if(_wgpu_lib_dir STREQUAL "" AND IS_DIRECTORY "${_entry}/lib")
          set(_wgpu_lib_dir "${_entry}/lib")
        endif()
      endif()
    endforeach()
  endif()
endif()

add_library(webgpu INTERFACE)
if(NOT _wgpu_include STREQUAL "")
  target_include_directories(webgpu INTERFACE "${_wgpu_include}")
endif()
if(NOT _wgpu_lib_dir STREQUAL "")
  if(EXISTS "${_wgpu_lib_dir}/libwgpu_native.dylib")
    set(WGPU_NATIVE_LIB "${_wgpu_lib_dir}/libwgpu_native.dylib")
  else()
    unset(WGPU_NATIVE_LIB CACHE)
    find_library(WGPU_NATIVE_LIB wgpu_native PATHS "${_wgpu_lib_dir}" NO_DEFAULT_PATH)
  endif()
  if(WGPU_NATIVE_LIB)
    target_link_libraries(webgpu INTERFACE "${WGPU_NATIVE_LIB}")
  endif()
endif()
function(add_plexus_operator name source)
  add_library(${name} MODULE ${source})
  target_link_libraries(${name} PRIVATE vivid_operator_api webgpu)
  set_target_properties(${name} PROPERTIES
    PREFIX ""
    SUFFIX "${VIVID_PLUGIN_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endfunction()

add_plexus_operator(plexus src/plexus.cpp)
add_plexus_operator(plexus_synth src/plexus_synth.cpp)

add_custom_target(operators DEPENDS plexus plexus_synth)

add_executable(test_package_manifest tests/test_package_manifest.cpp)
target_compile_features(test_package_manifest PRIVATE cxx_std_17)
