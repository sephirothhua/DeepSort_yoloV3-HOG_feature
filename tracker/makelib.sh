cp -f ./src/api/deepsort.h ./include/
mkdir build && cd build

cmake ../src -DCMAKE_BUILD_TYPE=Debug
make -j8

mv libdeepsort.so ../


