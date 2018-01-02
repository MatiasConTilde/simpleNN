# simpleNN
Processing library for making a very basic and simple neural network

## Usage
#### Create a new instance
```java
import simpleNN.*;
import Jama.*;

int[] layerSizes = {3, 4, 2};
double learningRate = 0.1;
Network nn = new Network(layerSizes, learningRate);
```

#### Feed forward
```java
double[] inputValues = {1, 2, 3};
Matrix results = nn.test(inputValues);
```

#### Training
```java
double[] inputValues = {1, 2, 3};
double[] desiredValues = {1, 2};
nn.train(inputValues, desiredValues);
```

## Compiling
```bash
cd Processing/libraries/
git clone https://github.com/MatiasConTilde/simpleNN.git
cd simpleNN
mkdir library
javac -cp src/simpleNN.jar -d library src/simpleNN/*
cd library
jar -cf simpleNN.jar simpleNN
```

## Thanks
Heavily inspired by https://github.com/shiffman/Neural-Network-p5/

Using Jama matrix library http://math.nist.gov/javanumerics/jama/
