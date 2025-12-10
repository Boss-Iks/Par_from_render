set(COMMON_SRC_FILES   
  "${CMAKE_SOURCE_DIR}/common/src/rayos.cpp"
)

set(CURRENT_DIR_SRC_FILES     
)

add_unit_test_target(
  TARGET_NAME utcommon
  SOURCE_FILES ${COMMON_SRC_FILES} ${CURRENT_DIR_SRC_FILES}
  LIBRARY_FILTER common
  COVERAGE_DIR coverage-common
  LIBRARY_TO_LINK common
)


add_test(NAME utcommon COMMAND $<TARGET_FILE:utcommon>)


target_include_directories(utcommon PRIVATE  
  ${CMAKE_SOURCE_DIR}/soa/src
)