(* ::Package:: *)

(*
   The weights for this model come from training in a neural network library called CognitoNet which is now retired.
   
   That training session used images from the Face Scrub data set:
   http: http://vintage.winklerbros.net/facescrub.html
   H.-W. Ng, S. Winkler.
   A data-driven approach to cleaning large face datasets.
   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.
*)


CNFaceModel=Import["https://sites.google.com/site/julianwfrancis/dummy/FaceNet2Convolve.wdx"];


CNconv1=ConvolutionLayer[32,{5,5},"Biases"-> CNFaceModel[[1,1,All,1]],"Weights"-> Transpose[{CNFaceModel[[1,1,All,2]]},{2,1,3,4}]];


CNconv2=ConvolutionLayer[32,{5,5},"Biases"-> CNFaceModel[[4,1,All,1]],"Weights"->CNFaceModel[[4,1,All,2]]];


CNconv3=ConvolutionLayer[64,{5,5},"Biases"-> CNFaceModel[[7,1,All,1]],"Weights"->CNFaceModel[[7,1,All,2]]];


CNconv4=ConvolutionLayer[1,{1,1},"Biases"-> {CNFaceModel[[9,1]]},"Weights"->{CNFaceModel[[9,2]]}];


CNFaceNet=NetChain[{
   CNconv1,ElementwiseLayer[Tanh],PoolingLayer[2,"Stride"->2],
   CNconv2,ElementwiseLayer[Tanh],PoolingLayer[2,"Stride"->2],
   CNconv3,ElementwiseLayer[Tanh],
   CNconv4,ElementwiseLayer[LogisticSigmoid]}];


Options[ CNSingleScaleDetectObjects ] = {
   Threshold->0.997,
};
CNSingleScaleDetectObjects[image_?ImageQ, net_, opts:OptionsPattern[]] := If[Min[ImageDimensions[image]]>=32,
   map=(net@{ColorConvert[image,"GrayScale"]//ImageData})[[1]];
   extractPositions=Position[map,x_/;x>OptionValue[Threshold]];
   origCoords=Map[({4*#[[2]] + (16-4),ImageDimensions[image][[2]]-4*#[[1]]+4-16})&,extractPositions];
   Map[{{#[[1]]-15,#[[2]]-15},{#[[1]]+16,#[[2]]+16}}&,origCoords],
   {}]


Options[ CNMultiScaleDetectObjects ] = Options[ CNSingleScaleDetectObjects ];
CNMultiScaleDetectObjects[image_?ImageQ, net_, opts:OptionsPattern[] ] :=
   Flatten[Table[(ImageDimensions[image][[1]]/(32*1.2^sc))*CNSingleScaleDetectObjects[ImageResize[image,32*1.2^sc], net, opts],{sc,0,Log[Min[ImageDimensions[image][[1]],800]/32]/Log[1.2]}],1]


Options[ CNDetectFaces ] = Options[ CNSingleScaleDetectObjects ];
CNDetectFaces[image_?ImageQ, opts:OptionsPattern[]] := 
 CNMultiScaleDetectObjects[image, CNFaceNet, opts];
