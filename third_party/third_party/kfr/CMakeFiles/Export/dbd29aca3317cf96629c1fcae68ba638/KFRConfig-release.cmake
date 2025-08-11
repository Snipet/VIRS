#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kfr_dsp_sse2" for configuration "Release"
set_property(TARGET kfr_dsp_sse2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kfr_dsp_sse2 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkfr_dsp_sse2.a"
  )

list(APPEND _cmake_import_check_targets kfr_dsp_sse2 )
list(APPEND _cmake_import_check_files_for_kfr_dsp_sse2 "${_IMPORT_PREFIX}/lib/libkfr_dsp_sse2.a" )

# Import target "kfr_dsp_sse41" for configuration "Release"
set_property(TARGET kfr_dsp_sse41 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kfr_dsp_sse41 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkfr_dsp_sse41.a"
  )

list(APPEND _cmake_import_check_targets kfr_dsp_sse41 )
list(APPEND _cmake_import_check_files_for_kfr_dsp_sse41 "${_IMPORT_PREFIX}/lib/libkfr_dsp_sse41.a" )

# Import target "kfr_dsp_avx" for configuration "Release"
set_property(TARGET kfr_dsp_avx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kfr_dsp_avx PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkfr_dsp_avx.a"
  )

list(APPEND _cmake_import_check_targets kfr_dsp_avx )
list(APPEND _cmake_import_check_files_for_kfr_dsp_avx "${_IMPORT_PREFIX}/lib/libkfr_dsp_avx.a" )

# Import target "kfr_dsp_avx2" for configuration "Release"
set_property(TARGET kfr_dsp_avx2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kfr_dsp_avx2 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkfr_dsp_avx2.a"
  )

list(APPEND _cmake_import_check_targets kfr_dsp_avx2 )
list(APPEND _cmake_import_check_files_for_kfr_dsp_avx2 "${_IMPORT_PREFIX}/lib/libkfr_dsp_avx2.a" )

# Import target "kfr_dsp_avx512" for configuration "Release"
set_property(TARGET kfr_dsp_avx512 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kfr_dsp_avx512 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkfr_dsp_avx512.a"
  )

list(APPEND _cmake_import_check_targets kfr_dsp_avx512 )
list(APPEND _cmake_import_check_files_for_kfr_dsp_avx512 "${_IMPORT_PREFIX}/lib/libkfr_dsp_avx512.a" )

# Import target "kfr_io" for configuration "Release"
set_property(TARGET kfr_io APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kfr_io PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkfr_io.a"
  )

list(APPEND _cmake_import_check_targets kfr_io )
list(APPEND _cmake_import_check_files_for_kfr_io "${_IMPORT_PREFIX}/lib/libkfr_io.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
