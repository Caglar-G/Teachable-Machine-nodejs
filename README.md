# Teachable-Machine-nodejs
Using the model developed in Teachable-Machine with tensorflow in node js

## Rebuild the package on Raspberry Pi
To use this package on Raspberry Pi, you need to rebuild the node native addon with the following command after you installed the package:

    npm rebuild @tensorflow/tfjs-node --build-from-source


## ToDo
1. add description to readme
2. add labels dynamically
2. image resize or crop for teachable machine model