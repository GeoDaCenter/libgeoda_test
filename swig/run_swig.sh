echo "create R wrappers..."
cp ./libgeoda_src/libgeoda.i rgeoda.i
sed -i.bu 's/module libgeoda/module rgeoda/g' rgeoda.i 
swig -c++ -r -I./libgeoda_src -o rgeoda.cpp rgeoda.i
