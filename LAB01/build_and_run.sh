echo "starting building"
nvcc transf.cu -o transf
echo "build finished"
echo "launching"
./transf