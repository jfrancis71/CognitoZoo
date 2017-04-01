(* ::Package:: *)

(*
   Implements YOLO

   You will need to download the YOLO weight file and install it on a Mathematica search path, eg your home directory.
    File available here:
      https://drive.google.com/file/d/0Bzhe0pgVZtNUVmJlTjYwNzFWSm8/view?usp=sharing

   Credit:
      The following tensorflow code was used as a reference implementation:
      Jinyoung Choi
      https://github.com/gliese581gg/YOLO_tensorflow
      
      That model was originally from Darknet, Joseph Redmon:
      https://pjreddie.com/darknet/yolo/
      
      Citation:
      @misc{darknet13,
      author =   {Joseph Redmon},
      title =    {Darknet: Open Source Neural Networks in C},
      howpublished = {\url{http://pjreddie.com/darknet/}},
      year = {2013--2016}
      }
      
      Weights were initially from Jinyoung Choi's tensorflow checkpoint file which I believe had been converted from
      the Darknet YOLO model.
*)


yolofile="yolonet.hdf5";


conv1W=Import[yolofile,{"Datasets","/conv1W"}];
conv1b=Import[yolofile,{"Datasets","/conv1b"}];


conv2W=Import[yolofile,{"Datasets","/conv2W"}];
conv2b=Import[yolofile,{"Datasets","/conv2b"}];


conv3W=Import[yolofile,{"Datasets","/conv3W"}];
conv3b=Import[yolofile,{"Datasets","/conv3b"}];


conv4W=Import[yolofile,{"Datasets","/conv4W"}];
conv4b=Import[yolofile,{"Datasets","/conv4b"}];


conv5W=Import[yolofile,{"Datasets","/conv5W"}];
conv5b=Import[yolofile,{"Datasets","/conv5b"}];


conv6W=Import[yolofile,{"Datasets","/conv6W"}];
conv6b=Import[yolofile,{"Datasets","/conv6b"}];


conv7W=Import[yolofile,{"Datasets","/conv7W"}];
conv7b=Import[yolofile,{"Datasets","/conv7b"}];


conv8W=Import[yolofile,{"Datasets","/conv8W"}];
conv8b=Import[yolofile,{"Datasets","/conv8b"}];


conv9W=Import[yolofile,{"Datasets","/conv9W"}];
conv9b=Import[yolofile,{"Datasets","/conv9b"}];


fc10W=Import[yolofile,{"Datasets","/fc10W"}];
fc10b=Import[yolofile,{"Datasets","/fc10b"}];


fc11W=Import[yolofile,{"Datasets","/fc11W"}];
fc11b=Import[yolofile,{"Datasets","/fc11b"}];


fc12W=Import[yolofile,{"Datasets","/fc12W"}];
fc12b=Import[yolofile,{"Datasets","/fc12b"}];


CZConv1=ConvolutionLayer[16,{3,3},"Biases"->conv1b,"Weights"->Transpose[conv1W,{3,4,2,1}],"PaddingSize"->1];


CZConv2=ConvolutionLayer[32,{3,3},"Biases"->conv2b,"Weights"->Transpose[conv2W,{3,4,2,1}],"PaddingSize"->1];


CZConv3=ConvolutionLayer[64,{3,3},"Biases"->conv3b,"Weights"->Transpose[conv3W,{3,4,2,1}],"PaddingSize"->1];


CZConv4=ConvolutionLayer[128,{3,3},"Biases"->conv4b,"Weights"->Transpose[conv4W,{3,4,2,1}],"PaddingSize"->1];


CZConv5=ConvolutionLayer[256,{3,3},"Biases"->conv5b,"Weights"->Transpose[conv5W,{3,4,2,1}],"PaddingSize"->1];


CZConv6=ConvolutionLayer[512,{3,3},"Biases"->conv6b,"Weights"->Transpose[conv6W,{3,4,2,1}],"PaddingSize"->1];


CZConv7=ConvolutionLayer[1024,{3,3},"Biases"->conv7b,"Weights"->Transpose[conv7W,{3,4,2,1}],"PaddingSize"->1];


CZConv8=ConvolutionLayer[1024,{3,3},"Biases"->conv8b,"Weights"->Transpose[conv8W,{3,4,2,1}],"PaddingSize"->1];


CZConv9=ConvolutionLayer[1024,{3,3},"Biases"->conv9b,"Weights"->Transpose[conv9W,{3,4,2,1}],"PaddingSize"->1];


CZFc10=DotPlusLayer[256,"Biases"->fc10b,"Weights"->(fc10W//Transpose)];


CZFc11=DotPlusLayer[4096,"Biases"->fc11b,"Weights"->(fc11W//Transpose)];


CZFc12=DotPlusLayer[1470,"Biases"->fc12b,"Weights"->(fc12W//Transpose)];


classes={"aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};


CZGetOutput[boxes_, position_]:=
   boxes[[position[[2]],position[[3]],position[[1]]]]


CZGetCentre[boxes_, position_]:=
   (CZGetOutput[boxes,position][[1;;2]]+{position[[3]],7-position[[2]]}-{1,0})/7


CZGetSize[boxes_, position_]:=
   CZGetOutput[boxes, position][[3;;4]]*CZGetOutput[boxes, position][[3;;4]]


CZGetBoundingBox[boxes_, position_]:=
   {{CZGetCentre[boxes, position][[1]]-CZGetSize[boxes, position][[1]]/2,
     CZGetCentre[boxes, position][[2]]-CZGetSize[boxes, position][[2]]/2},
    {CZGetCentre[boxes, position][[1]]+CZGetSize[boxes, position][[1]]/2,
     CZGetCentre[boxes, position][[2]]+CZGetSize[boxes, position][[2]]/2}}


CZDetectObjects[img_] := (
   conv1=CZConv1@Transpose[ImageResize[img,{448,448}]//ImageData,{2,3,1}];
   lr1=UnitStep[conv1]*conv1 + (1-UnitStep[conv1])*conv1*0.1;
   maxpool1=NetChain[{PoolingLayer[{2,2},"Stride"->2]}]@lr1;
   conv2=CZConv2@maxpool1;
   lr2=UnitStep[conv2]*conv2 + (1-UnitStep[conv2])*conv2*0.1;
   maxpool2=NetChain[{PoolingLayer[{2,2},"Stride"->2]}]@lr2;
   conv3=CZConv3@maxpool2;
   lr3=UnitStep[conv3]*conv3 + (1-UnitStep[conv3])*conv3*0.1;
   maxpool3=NetChain[{PoolingLayer[{2,2},"Stride"->2]}]@lr3;
   conv4=CZConv4@maxpool3;
   lr4=UnitStep[conv4]*conv4 + (1-UnitStep[conv4])*conv4*0.1;
   maxpool4=NetChain[{PoolingLayer[{2,2},"Stride"->2]}]@lr4;
   conv5=CZConv5@maxpool4;
   lr5=UnitStep[conv5]*conv5 + (1-UnitStep[conv5])*conv5*0.1;
   maxpool5=NetChain[{PoolingLayer[{2,2},"Stride"->2]}]@lr5;
   conv6=CZConv6@maxpool5;
   lr6=UnitStep[conv6]*conv6 + (1-UnitStep[conv6])*conv6*0.1;
   maxpool6=NetChain[{PoolingLayer[{2,2},"Stride"->2]}]@lr6;
   conv7=CZConv7@maxpool6;
   lr7=UnitStep[conv7]*conv7 + (1-UnitStep[conv7])*conv7*0.1;
   conv8=CZConv8@lr7;
   lr8=UnitStep[conv8]*conv8 + (1-UnitStep[conv8])*conv8*0.1;
   conv9=CZConv9@lr8;
   lr9=UnitStep[conv9]*conv9 + (1-UnitStep[conv9])*conv9*0.1;
   fc10=CZFc10@Flatten[Transpose[lr9,{3,1,2}]];
   fc10=CZFc10@Flatten[lr9];
   lr10=UnitStep[fc10]*fc10 + (1-UnitStep[fc10])*fc10*0.1;
   fc11=CZFc11@Flatten[lr10];
   lr11=UnitStep[fc11]*fc11 + (1-UnitStep[fc11])*fc11*0.1;
   fc12=CZFc12@Flatten[lr11];
   classProbs=ArrayReshape[fc12[[1;;980]],{7,7,20}];
   scales=ArrayReshape[fc12[[981;;1078]],{7,7,2}];
   probs={scales[[All,All,1]]*classProbs,scales[[All,All,2]]*classProbs};
   boxes=ArrayReshape[fc12[[1079;;-1]],{7,7,2,4}];
   p1=Position[probs,x_/;x>.2];
   Map[{classes[[#[[4]]]],CZGetBoundingBox[boxes,#].DiagonalMatrix[ImageDimensions[img]]}&,p1]
)


CZDisplayObject[object_]:={Rectangle@@object[[2]],Text[Style[object[[1]],White,24],{20,20}+object[[2,1]]]}
