(* ::Package:: *)

(*
   The weights for this model come from training in a neural network library called CognitoNet which is now retired.
   
   That training session used images from the Face Scrub data set:
   http: http://vintage.winklerbros.net/facescrub.html
   H.-W. Ng, S. Winkler.
   A data-driven approach to cleaning large face datasets.
   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.
   
   Example usage: HighlightImage[img,Rectangle@@@CZDetectFaces[img]]
*)


<<CZUtils.m


CZFaceModel=Import["https://sites.google.com/site/julianwfrancis/dummy/FaceNet2Convolve.wdx"];
CZconv1=ConvolutionLayer[32,{5,5},"Biases"-> CZFaceModel[[1,1,All,1]],"Weights"-> Transpose[{CZFaceModel[[1,1,All,2]]},{2,1,3,4}]];
CZconv2=ConvolutionLayer[32,{5,5},"Biases"-> CZFaceModel[[4,1,All,1]],"Weights"->CZFaceModel[[4,1,All,2]]];
CZconv3=ConvolutionLayer[64,{5,5},"Biases"-> CZFaceModel[[7,1,All,1]],"Weights"->CZFaceModel[[7,1,All,2]]];
CZconv4=ConvolutionLayer[1,{1,1},"Biases"-> {CZFaceModel[[9,1]]},"Weights"->{CZFaceModel[[9,2]]}];


CZFaceNet=NetChain[{
   CZconv1,ElementwiseLayer[Tanh],PoolingLayer[2,"Stride"->2],
   CZconv2,ElementwiseLayer[Tanh],PoolingLayer[2,"Stride"->2],
   CZconv3,ElementwiseLayer[Tanh],
   CZconv4,ElementwiseLayer[LogisticSigmoid]}];


Options[ CZSingleScaleDetectObjects ] = {
   Threshold->0.997
};
(* Conceptually it is a sliding window (32x32) object detector running at a single scale.
   In practice it is implemented convolutionally ( for performance reasons ) so the net
   should be fully convolutional, ie no fully connected layers.
   The net output should be a 2D array of numbers indicating a metric for likelihood of object being present.
   The net filter should accept an array of real numbers (ie this works on greyscale images). You can supply a
   colour image as input to the function, but this is just converted to greyscale before being fed to the neural net
   Note the geometric factor 4 in mapping from the output array to the input array, this is because we have
   downsampled twice in the neural network, so there is a coupling from this algorithm to the architecture
   of the neural net supplied.
   The rational for using a greyscale neural net is that I am particularly fascinated by shape (much more
   than colour), so wanted to look at performance driven by that factor alone. A commercial use might take
   a different view.
*)
CZSingleScaleDetectObjects[image_?ImageQ, net_, opts:OptionsPattern[]] := If[Min[ImageDimensions[image]]<32,{},
   map=(net@{ColorConvert[image,"GrayScale"]//ImageData})[[1]];
   extractPositions=Position[map,x_/;x>OptionValue[Threshold]];
   origCoords=Map[{Extract[map,#],4*#[[2]] + (16-4),ImageDimensions[image][[2]]-4*#[[1]]+4-16}&,extractPositions];
   Map[{#[[1]],{#[[2]]-15,#[[3]]-15},{#[[2]]+16,#[[3]]+16}}&,origCoords],
   ]


(* Like MapAt, but will work if an empty list is present  *)
CNMapAt[f_,{},spec_] := {}
CNMapAt[f_,dat_,spec_] := MapAt[f,dat,spec]


Options[ CZMultiScaleDetectObjects ] = Options[ CZSingleScaleDetectObjects ];
(* Implements a sliding window object detector at multiple scales.
   The function resamples the image at scales ranging from a minimum width of 32 up to 800 at 20% scale increments.
   The maximum width of 800 was chosen for 2 reasons: to limit inference run time and to limit the number of likely
   false positives / image, implying the detector's limit is to recognise faces larger than 32/800 (4%) of the image width.
   Note that if for example you had high resolution images with faces in the far distance and wanted to detect those and were
   willing to accept false positives within the image, you might reconsider that tradeoff.
   However, the main use case was possibly high resolution images where faces are not too distant with objective of limiting
   false positives across the image as a whole.
*)
CZMultiScaleDetectObjects[image_?ImageQ, net_, opts:OptionsPattern[] ] :=
   Flatten[Table[
      CNMapAt[(#*ImageDimensions[image][[1]]/(32*1.2^sc))&,
         CZSingleScaleDetectObjects[ImageResize[image,32*1.2^sc], net, opts],{All,2;;3}],
      {sc,0,Log[Min[ImageDimensions[image][[1]],800]/32]/Log[1.2]}],1]


Options[ CZDetectFaces ] = Options[ CZSingleScaleDetectObjects ];
(* Works like FindFaces, ie returns { {{x1,y1},{x2,y2}},... }
   On the Caltech 1999 face dataset, we achieve a recognition rate of around 92% with
   an average of 14% of false positives/image.
   The Caltech dataset has 450 images where most faces are quite close to camera,
   where images are of size 896x592. Most of these images are of good quality, but some
   are challenging, eg. cartoon, significant obscuring of face or poor lighting conditions.
   Reference comparison, FindFances achieves 99.6% recognition, but 56% average false positive rate/image
*)
CZDetectFaces::usage="
CZDetectFaces[img,options] returns {{xmin1,ymin1},{xmax1,ymax1},...
Options are: Threshold

Example usage: HighlightImage[img,Rectangle@@@CZDetectFaces[img]].
";
CZDetectFaces[image_?ImageQ, opts:OptionsPattern[]] := 
   CZDeleteOverlappingWindows[ CZMultiScaleDetectObjects[image, CZFaceNet, opts] ];
