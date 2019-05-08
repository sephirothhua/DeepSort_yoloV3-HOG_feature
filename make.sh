cd darknet
make -j8

cd ..
cd tracker
sh makelib.sh

cd ..

cd code
make
mv my_demo ../

cd ..
mv ./darknet/libdarknet.so ./
mv ./tracker/libdeepsort.so ./
