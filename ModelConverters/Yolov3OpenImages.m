(* ::Package:: *)

(* Yolo v3 Open Images *)


leakyReLU=ElementwiseLayer[Ramp[#]+Ramp[-#]*-.1&];


hdfFile="/Users/julian/yolov3/darknet/multset.h5";


YoloConvLayer[name_,filters_,filterSize_,stride_,side_]:=Module[
{weights=Import[hdfFile,{"Datasets",name<>"_weights"}],
biases=Import[hdfFile,{"Datasets",name<>"_biases"}],
rm=Import[hdfFile,{"Datasets",name<>"_rolling_mean"}],
rv=Import[hdfFile,{"Datasets",name<>"_rolling_variance"}],
scales=Import[hdfFile,{"Datasets",name<>"_scales"}]
},
NetChain[{
ConvolutionLayer[filters,{filterSize,filterSize},"Stride"->stride,"Weights"->weights,"Biases"->ConstantArray[0,filters],"PaddingSize"->If[filterSize==1,0,1]],
BatchNormalizationLayer["Epsilon"->.00001,"Input"->{filters,side,side},"Gamma"->scales,"Beta"->biases,"MovingMean"->rm,"MovingVariance"->rv],
leakyReLU
}]
];


(* ::Input:: *)
(*(* Residual block filters and resolution unchanged *)*)
(*SmallResidualBlock[ firstLayer_, filters_, size_ ] :=*)
(*NetGraph[{*)
(*YoloConvLayer["layer"<>ToString[firstLayer],filters/2,1, 1, size ],*)
(*YoloConvLayer["layer"<>ToString[firstLayer+1],filters,3, 1, size ],*)
(*ThreadingLayer[Plus]*)
(*},{1->2->3,NetPort["Input"]->3}]*)


(* ::Input:: *)
(*(* Halves the output resolution and doubles number of filters *)*)
(*LargeResidualBlock[ firstLayer_, inputFilters_, inputSize_ ] :=*)
(*NetGraph[{*)
(*YoloConvLayer["layer"<>ToString[firstLayer], inputFilters*2, 3, 2, inputSize/2] ,*)
(*YoloConvLayer["layer"<>ToString[firstLayer+1], inputFilters, 1, 1, inputSize/2] ,*)
(*YoloConvLayer["layer"<>ToString[firstLayer+2], inputFilters*2, 3, 1, inputSize/2] ,*)
(*ThreadingLayer[Plus]*)
(*},{1->2->3->4,1->4}]*)


(* ::Input:: *)
(*inp=Import["/Users/julian/yolov3/darknet/multset.h5",{"Datasets","/input"}];inp//Dimensions*)


(* ::Input:: *)
(*mylayer0=YoloConvLayer["layer0",32,3,1,608][inp];mylayer0//Dimensions*)


(* ::Input:: *)
(*mylayer4 = LargeResidualBlock[ 1, 32, 608][mylayer0];mylayer4//Dimensions*)


(* ::Input:: *)
(*mylayer8 = LargeResidualBlock[ 5, 64, 304 ][mylayer4]; mylayer8//Dimensions*)


(* ::Input:: *)
(*mylayer11=SmallResidualBlock[9, 128,152][mylayer8];mylayer11//Dimensions*)


(* ::Input:: *)
(*mylayer15 = LargeResidualBlock[ 12, 128, 152 ][mylayer11]; mylayer15//Dimensions*)


(* ::Input:: *)
(*mylayer18=SmallResidualBlock[16, 256,76][mylayer15];mylayer18//Dimensions*)


(* ::Input:: *)
(*mylayer21=SmallResidualBlock[19, 256,76][mylayer18];mylayer21//Dimensions*)


(* ::Input:: *)
(*mylayer24=SmallResidualBlock[22, 256,76][mylayer21];mylayer24//Dimensions*)


(* ::Input:: *)
(*mylayer27=SmallResidualBlock[25, 256,76][mylayer24];mylayer27//Dimensions*)


(* ::Input:: *)
(*mylayer30=SmallResidualBlock[28, 256,76][mylayer27];mylayer30//Dimensions*)


(* ::Input:: *)
(*mylayer33=SmallResidualBlock[31, 256,76][mylayer30];mylayer33//Dimensions*)


(* ::Input:: *)
(*mylayer36=SmallResidualBlock[34, 256,76][mylayer33];mylayer36//Dimensions*)


(* ::Input:: *)
(*mylayer40 = LargeResidualBlock[ 37, 256, 76 ][mylayer36]; mylayer40//Dimensions*)


(* ::Input:: *)
(*mylayer43=SmallResidualBlock[41, 512,38][mylayer40];mylayer43//Dimensions*)


(* ::Input:: *)
(*mylayer46=SmallResidualBlock[44, 512,38][mylayer43];mylayer46//Dimensions*)


(* ::Input:: *)
(*mylayer49=SmallResidualBlock[47, 512,38][mylayer46];mylayer49//Dimensions*)


(* ::Input:: *)
(*mylayer52=SmallResidualBlock[50, 512,38][mylayer49];mylayer52//Dimensions*)


(* ::Input:: *)
(*mylayer55=SmallResidualBlock[53, 512,38][mylayer52];mylayer55//Dimensions*)


(* ::Input:: *)
(*mylayer58=SmallResidualBlock[56, 512,38][mylayer55];mylayer58//Dimensions*)


(* ::Input:: *)
(*mylayer61=SmallResidualBlock[59, 512,38][mylayer58];mylayer61//Dimensions*)


(* ::Input:: *)
(*mylayer65= LargeResidualBlock[ 62, 512, 38 ][mylayer61]; mylayer65//Dimensions*)


(* ::Input:: *)
(*mylayer68=SmallResidualBlock[66, 1024,19][mylayer65];mylayer68//Dimensions*)


(* ::Input:: *)
(*mylayer71=SmallResidualBlock[69, 1024,19][mylayer68];mylayer71//Dimensions*)


(* ::Input:: *)
(*mylayer74=SmallResidualBlock[72, 1024,19][mylayer71];mylayer74//Dimensions*)


(* ::Input:: *)
(*mylayer76=SmallResidualBlock[72, 1024,19][mylayer74];mylayer76//Dimensions*)


(* ::Input:: *)
(*mylayer79=NetChain[{*)
(*YoloConvLayer["layer75",512,1, 1, 19 ],*)
(*YoloConvLayer["layer76",1024,3, 1, 19 ],*)
(*YoloConvLayer["layer77",512,1, 1, 19 ],*)
(*YoloConvLayer["layer78",1024,3, 1, 19 ],*)
(*YoloConvLayer["layer79",512,1, 1, 19 ]*)
(*}*)
(*][mylayer74];mylayer79//Dimensions*)


(* ::Input:: *)
(*mylayer81=NetChain[{*)
(*YoloConvLayer["layer75",512,1, 1, 19 ],*)
(*YoloConvLayer["layer76",1024,3, 1, 19 ],*)
(*YoloConvLayer["layer77",512,1, 1, 19 ],*)
(*YoloConvLayer["layer78",1024,3, 1, 19 ],*)
(*YoloConvLayer["layer79",512,1, 1, 19 ],*)
(*YoloConvLayer["layer80",1024,3, 1, 19 ],*)
(*ConvolutionLayer[1818,{1,1},"Weights"->Import[hdfFile,{"Datasets","layer81_weights"}],"Biases"->Import[hdfFile,{"Datasets","layer81_biases"}]]*)
(*}][mylayer74];mylayer81//Dimensions*)


(* ::Input:: *)
(*mylayer84=YoloConvLayer["layer84",256,1,1,19][mylayer79];mylayer84//Dimensions*)


(* ::Input:: *)
(*mylayer85=DeconvolutionLayer[256,{2,2},"Weights"->Table[If[j==i,1,0],{j,1,256},{i,1,256},{2},{2}],"Biases"->ConstantArray[0,256],"Stride"->2][mylayer84];mylayer85//Dimensions*)


(* ::Input:: *)
(*mylayer86=CatenateLayer[][{mylayer85,mylayer61}];mylayer86//Dimensions*)


(* ::Input:: *)
(*mylayer91=NetChain[{*)
(*YoloConvLayer["layer87",256,1, 1, 38 ],*)
(*YoloConvLayer["layer88",512,3, 1, 38 ],*)
(*YoloConvLayer["layer89",256,1, 1, 38 ],*)
(*YoloConvLayer["layer90",512,3, 1, 38 ],*)
(*YoloConvLayer["layer91",256,1, 1, 38 ]*)
(*}][mylayer86];mylayer91//Dimensions*)


(* ::Input:: *)
(*mylayer93 = NetChain[{ *)
(*YoloConvLayer["layer92",512,3, 1, 38 ],*)
(*ConvolutionLayer[1818,{1,1},"Weights"->Import[hdfFile,{"Datasets","layer93_weights"}],"Biases"->Import[hdfFile,{"Datasets","layer93_biases"}]]*)
(*}][mylayer91];mylayer93//Dimensions*)


(* ::Input:: *)
(*mylayer96=YoloConvLayer["layer96",128,1,1,38][mylayer91];mylayer96//Dimensions*)


(* ::Input:: *)
(*mylayer97=DeconvolutionLayer[128,{2,2},"Weights"->Table[If[j==i,1,0],{j,1,128},{i,1,128},{2},{2}],"Biases"->ConstantArray[0,128],"Stride"->2][mylayer96];mylayer97//Dimensions*)


(* ::Input:: *)
(*mylayer98=CatenateLayer[][{mylayer97,mylayer36}];mylayer98//Dimensions*)


(* ::Input:: *)
(*mylayer105=NetChain[{*)
(*YoloConvLayer["layer99",128,1, 1, 76 ],*)
(*YoloConvLayer["layer100",256,3, 1, 76 ],*)
(*YoloConvLayer["layer101",128,1, 1, 76 ],*)
(*YoloConvLayer["layer102",256,3, 1, 76 ],*)
(*YoloConvLayer["layer103",128,1, 1, 76 ],*)
(*YoloConvLayer["layer104",256,3, 1, 76 ],*)
(*ConvolutionLayer[1818,{1,1},"Weights"->Import[hdfFile,{"Datasets","layer105_weights"}],"Biases"->Import[hdfFile,{"Datasets","layer105_biases"}]]*)
(*}][mylayer98];mylayer105//Dimensions*)


(* ::Input:: *)
(*layer105=Import["/Users/julian/yolov3/darknet/multset.h5",{"Datasets","/layer105"}];layer105//Dimensions*)


(* ::Input:: *)
(*k=Abs[mylayer105-layer105];Max[k]*)


(* ::Input:: *)
(*Position[k,Max[k]]*)


(* ::Input:: *)
(*mylayer105[[1525,71,55]]*)


(* ::Input:: *)
(*layer105[[1525,71,55]]*)


(* ::Input:: *)
(*{Image[inp,Interleaving->False],(LogisticSigmoid@layer93[[1]])//Image}*)


(* ::Input:: *)
(*classes=Import["/Users/julian/yolov3/darknet/data/openimages.names","Table"][[All,1]];Length[classes]*)


(* ::Input:: *)
(*l=layer81[[5]];Position[l,Max[l]]*)


(* ::Input:: *)
(*q=layer81[[6;;601+5,10,12]];Length[q]*)


(* ::Input:: *)
(*Position[q,x_/;x>0]*)


(* ::Input:: *)
(*classes[[571]]*)


(* ::Input:: *)
(*classes*)
