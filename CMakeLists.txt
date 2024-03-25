cmake_minimum_required(VERSION 3.24)
project(BRIEF CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# Put binaries and libraries in the same location.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Always export compile_commands.json for lsp like clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED PATHS /home/jun/Documents/brief/opt)

add_executable(brief main.cpp compute_brief_cpu.cpp compute_brief_cuda.cpp compute_brief_cuda.cu)

target_link_libraries(brief PRIVATE ${OpenCV_LIBS})

