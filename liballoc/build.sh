mkdir -p build
cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
cd ..
cmake --build build --target deploy -j 10 -v
