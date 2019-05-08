cd darknet
make clean
rm -r backup
rm -r obj
rm -r result

cd ../tracker
sh clean.sh


cd ..
rm my_demo
rm ./code/*.o
rm *.so
