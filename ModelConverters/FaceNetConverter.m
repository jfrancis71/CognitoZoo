(* ::Package:: *)

(*
   The weights for this model come from training in a neural network library called CognitoNet which is now retired.
   
   That training session used images from the Face Scrub data set:
   http: http://vintage.winklerbros.net/facescrub.html
   H.-W. Ng, S. Winkler.
   A data-driven approach to cleaning large face datasets.
   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.
   
   Example usage: HighlightImage[img,Rectangle@@@CZDetectFaces[img]]
   
   You need to download the following two files and install them somewhere on youe search path.
   FaceNet2Convolve.json from https://drive.google.com/file/d/0Bzhe0pgVZtNUMFhfcGJwRE9sRWc/view?usp=sharing
   GenderNet.json from https://drive.google.com/file/d/0Bzhe0pgVZtNUaDY5ZzFiN2ZfTFU/view?usp=sharing
*)




(* Private Implementation Code *)

<<CZUtils.m


CZFaceParameters = Import["CZModels/FaceNet.json"];
CZconv1=ConvolutionLayer[32,{5,5},"Biases"-> CZFaceParameters[[1,1]],"Weights"-> Transpose[CZFaceParameters[[1,2]],{3,4,2,1}]];
CZconv2=ConvolutionLayer[32,{5,5},"Biases"-> CZFaceParameters[[2,1]],"Weights"->Transpose[CZFaceParameters[[2,2]],{3,4,2,1}]];
CZconv3=ConvolutionLayer[64,{5,5},"Biases"-> CZFaceParameters[[3,1]],"Weights"->Transpose[CZFaceParameters[[3,2]],{3,4,2,1}]];
CZconv4=ConvolutionLayer[1,{1,1},"Biases"-> {CZFaceParameters[[4,1]]},"Weights"->Transpose[CZFaceParameters[[4,2]],{3,4,2,1}]];


CZFaceNet=NetChain[{
   CZconv1,ElementwiseLayer[Tanh],PoolingLayer[2,"Stride"->2],
   CZconv2,ElementwiseLayer[Tanh],PoolingLayer[2,"Stride"->2],
   CZconv3,ElementwiseLayer[Tanh],
   CZconv4,ElementwiseLayer[LogisticSigmoid]}];


CZGenderParameters = Import["CZModels/GenderNet.json"];
CZGconv1=ConvolutionLayer[32,{5,5},"Biases"-> CZGenderParameters[[1,1]],"Weights"->Transpose[ CZGenderParameters[[1,2]],{3,4,2,1}],"PaddingSize"->2];
CZGconv2=ConvolutionLayer[32,{5,5},"Biases"-> CZGenderParameters[[2,1]],"Weights"->Transpose[ CZGenderParameters[[2,2]],{3,4,2,1}],"PaddingSize"->2];
CZGconv3=ConvolutionLayer[64,{5,5},"Biases"-> CZGenderParameters[[3,1]],"Weights"->Transpose[ CZGenderParameters[[3,2]],{3,4,2,1}],"PaddingSize"->2];
CZGlinear1=DotPlusLayer[1,"Biases"->{CZGenderParameters[[4,1]]},"Weights"->Transpose[CZGenderParameters[[4,2]]]];


CZGenderNet = NetChain[{
   CZGconv1,Tanh,PoolingLayer[2,"Stride"->2],
   CZGconv2,Tanh,PoolingLayer[2,"Stride"->2],
   CZGconv3,Tanh,PoolingLayer[2,"Stride"->2],
   FlattenLayer[],
   CZGlinear1,
   LogisticSigmoid}
];
