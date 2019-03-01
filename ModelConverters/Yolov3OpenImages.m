(* ::Package:: *)

(* Yolo v3 Open Images *)


leakyReLU = ElementwiseLayer[Ramp[#]+Ramp[-#]*-.1&];


hdfFile = "/Users/julian/yolov3/darknet/Yolov3OpenImages.h5";


YoloConvLayer[ name_, filters_, filterSize_, stride_, side_ ] := Module[{
   weights = Import[hdfFile,{"Datasets",name<>"_weights"}],
   biases = Import[hdfFile,{"Datasets",name<>"_biases"}],
   rm = Import[hdfFile,{"Datasets",name<>"_rolling_mean"}],
   rv = Import[hdfFile,{"Datasets",name<>"_rolling_variance"}],
   scales = Import[hdfFile,{"Datasets",name<>"_scales"}]
},
   NetChain[{
      ConvolutionLayer[filters,{filterSize,filterSize},"Stride"->stride,"Weights"->weights,"Biases"->ConstantArray[0,filters],"PaddingSize"->If[filterSize==1,0,1]],
      BatchNormalizationLayer["Epsilon"->.00001,"Input"->{filters,side,side},"Gamma"->scales,"Beta"->biases,"MovingMean"->rm,"MovingVariance"->rv],
      leakyReLU
}]];


(* Residual block filters and resolution unchanged *)
SmallResidualBlock[ firstLayer_, filters_, size_ ] := NetGraph[{
   YoloConvLayer[ "layer"<>ToString[firstLayer], filters/2, 1, 1, size ],
   YoloConvLayer[ "layer"<>ToString[firstLayer+1], filters, 3, 1, size ],
   ThreadingLayer[ Plus ]},
   { 1->2->3, NetPort["Input"]->3 }];


(* Halves the output resolution and doubles number of filters *)
LargeResidualBlock[ firstLayer_, inputFilters_, inputSize_ ] := NetGraph[{
   YoloConvLayer[ "layer"<>ToString[firstLayer], inputFilters*2, 3, 2, inputSize/2 ],
   YoloConvLayer[ "layer"<>ToString[firstLayer+1], inputFilters, 1, 1, inputSize/2 ],
   YoloConvLayer[ "layer"<>ToString[firstLayer+2], inputFilters*2, 3, 1, inputSize/2 ],
   ThreadingLayer[ Plus ]},
   { 1->2->3->4, 1->4 }];


inp = Import["/Users/julian/yolov3/darknet/multset.h5",{"Datasets","/input"}];inp//Dimensions


{3,608,608}


"layer0"->"layer4"->"layer8"->"layer11"->"layer15"->"layer18"->"layer21"->"layer24"->"layer27"->
"layer30"->"layer33"->"layer36"->"layer40"->"layer43"->"layer46"->"layer49"->"layer52"->"layer55"->"layer58"->
"layer61"->"layer65"->"layer68"->"layer71"->"layer74"->"layer75"->"layer76"->"layer77"->"layer78"->"layer79"->"layer80"->"layer81"->"layer84"


mylayer84Net


yoloOpenImagesConvNet = NetGraph[{
   "layer0"->YoloConvLayer[ "layer0", 32, 3, 1, 608],
   "layer4"->LargeResidualBlock[ 1, 32, 608 ],
   "layer8"->LargeResidualBlock[ 5, 64, 304 ],
   "layer11"->SmallResidualBlock[ 9, 128, 152 ],
   "layer15"->LargeResidualBlock[ 12, 128, 152 ],
   "layer18"->SmallResidualBlock[ 16, 256, 76 ],
   "layer21"->SmallResidualBlock[ 19, 256, 76 ],
   "layer24"->SmallResidualBlock[ 22, 256, 76 ],
   "layer27"->SmallResidualBlock[ 25, 256, 76 ],
   "layer30"->SmallResidualBlock[ 28, 256, 76 ],
   "layer33"->SmallResidualBlock[ 31, 256, 76 ],
   "layer36"->SmallResidualBlock[ 34, 256, 76 ],
   "layer40"->LargeResidualBlock[ 37, 256, 76  ],
   "layer43"->SmallResidualBlock[ 41, 512, 38 ],
   "layer46"->SmallResidualBlock[ 44, 512, 38 ],
   "layer49"->SmallResidualBlock[ 47, 512, 38 ],
   "layer52"->SmallResidualBlock[ 50, 512, 38 ],
   "layer55"->SmallResidualBlock[ 53, 512, 38 ],
   "layer58"->SmallResidualBlock[ 56, 512, 38 ],
   "layer61"->SmallResidualBlock[ 59, 512, 38 ],
   "layer65"->LargeResidualBlock[ 62, 512, 38 ],
   "layer68"->SmallResidualBlock[ 66, 1024, 19 ],
   "layer71"->SmallResidualBlock[ 69, 1024, 19 ],
   "layer74"->SmallResidualBlock[ 72, 1024, 19 ],
   "layer75"->YoloConvLayer[ "layer75", 512, 1, 1, 19 ],
   "layer76"->YoloConvLayer[ "layer76", 1024, 3, 1, 19 ],
   "layer77"->YoloConvLayer[ "layer77", 512, 1, 1, 19 ],
   "layer78"->YoloConvLayer[ "layer78", 1024, 3, 1, 19 ],
   "layer79"->YoloConvLayer[ "layer79", 512, 1, 1, 19 ],
   "layer80"->YoloConvLayer[ "layer80", 1024, 3, 1, 19 ],
   "layer81"->ConvolutionLayer[ 1818, {1,1}, "Weights"->Import[hdfFile,{"Datasets","layer81_weights"}], "Biases"->Import[hdfFile,{"Datasets","layer81_biases"}]],
   "layer84"->YoloConvLayer[ "layer84", 256, 1, 1, 19],
   "layer85"->DeconvolutionLayer[ 256, {2,2}, "Weights"->Table[If[j==i,1,0],{j,1,256},{i,1,256},{2},{2}], "Biases"->ConstantArray[0,256],"Stride"->2],
   "layer86"->CatenateLayer[],
   "layer87"->YoloConvLayer["layer87", 256, 1, 1, 38 ],
   "layer88"->YoloConvLayer["layer88", 512, 3, 1, 38 ],
   "layer89"->YoloConvLayer["layer89", 256, 1, 1, 38 ],
   "layer90"->YoloConvLayer["layer90", 512, 3, 1, 38 ],
   "layer91"->YoloConvLayer["layer91", 256, 1, 1, 38 ],
   "layer92"->YoloConvLayer["layer92", 512, 3, 1, 38 ],
   "layer93"->ConvolutionLayer[ 1818, {1,1}, "Weights"->Import[hdfFile,{"Datasets","layer93_weights"}], "Biases"->Import[hdfFile,{"Datasets","layer93_biases"}]],
   "layer96"->YoloConvLayer[ "layer96", 128, 1, 1, 38],
   "layer97"->DeconvolutionLayer[ 128, {2,2}, "Weights"->Table[If[j==i,1,0],{j,1,128},{i,1,128},{2},{2}], "Biases"->ConstantArray[0,128],"Stride"->2],
   "layer98"->CatenateLayer[],
   "layer99"->YoloConvLayer[ "layer99", 128, 1, 1, 76 ],
   "layer100"->YoloConvLayer[ "layer100",256, 3, 1, 76 ],
   "layer101"->YoloConvLayer[ "layer101",128, 1, 1, 76 ],
   "layer102"->YoloConvLayer[ "layer102",256, 3, 1, 76 ],
   "layer103"->YoloConvLayer[ "layer103",128, 1, 1, 76 ],
   "layer104"->YoloConvLayer[ "layer104",256, 3, 1, 76 ],
   "layer105"->ConvolutionLayer[ 1818, {1,1}, "Weights"->Import[hdfFile,{"Datasets","layer105_weights"}], "Biases"->Import[hdfFile,{"Datasets","layer105_biases"}]]
},{
"layer0"->"layer4"->"layer8"->"layer11"->"layer15"->"layer18"->"layer21"->"layer24"->"layer27"->
"layer30"->"layer33"->"layer36"->"layer40"->"layer43"->"layer46"->"layer49"->"layer52"->"layer55"->"layer58"->
"layer61"->"layer65"->"layer68"->"layer71"->"layer74"->"layer75"->"layer76"->"layer77"->"layer78"->"layer79"->"layer80"->"layer81",
"layer79"->"layer84"->"layer85",
{"layer85","layer61"}->"layer86"->"layer87"->"layer88"->"layer89"->"layer90"->"layer91"->"layer92"->"layer93",
"layer91"->"layer96"->"layer97",
{"layer97","layer36"}->"layer98"->"layer99"->"layer100"->"layer101"->"layer102"->"layer103"->"layer104"->"layer105"
}];


mylayer81=NetChain[{
YoloConvLayer["layer75",512,1, 1, 19 ],
YoloConvLayer["layer76",1024,3, 1, 19 ],
YoloConvLayer["layer77",512,1, 1, 19 ],
YoloConvLayer["layer78",1024,3, 1, 19 ],
YoloConvLayer["layer79",512,1, 1, 19 ],
YoloConvLayer["layer80",1024,3, 1, 19 ],
ConvolutionLayer[1818,{1,1},"Weights"->Import[hdfFile,{"Datasets","layer81_weights"}],"Biases"->Import[hdfFile,{"Datasets","layer81_biases"}]]
}][mylayer74];mylayer81//Dimensions


{1818,19,19}


mylayer84=YoloConvLayer["layer84",256,1,1,19][mylayer79];mylayer84//Dimensions


{256,19,19}


mylayer85=DeconvolutionLayer[256,{2,2},"Weights"->Table[If[j==i,1,0],{j,1,256},{i,1,256},{2},{2}],"Biases"->ConstantArray[0,256],"Stride"->2][mylayer84];mylayer85//Dimensions


{256,38,38}


mylayer86=CatenateLayer[][{mylayer85,mylayer61}];mylayer86//Dimensions


{768,38,38}


mylayer91=NetChain[{
YoloConvLayer["layer87",256,1, 1, 38 ],
YoloConvLayer["layer88",512,3, 1, 38 ],
YoloConvLayer["layer89",256,1, 1, 38 ],
YoloConvLayer["layer90",512,3, 1, 38 ],
YoloConvLayer["layer91",256,1, 1, 38 ]
}][mylayer86];mylayer91//Dimensions


{256,38,38}


mylayer93 = NetChain[{ 
YoloConvLayer["layer92",512,3, 1, 38 ],
ConvolutionLayer[1818,{1,1},"Weights"->Import[hdfFile,{"Datasets","layer93_weights"}],"Biases"->Import[hdfFile,{"Datasets","layer93_biases"}]]
}][mylayer91];mylayer93//Dimensions


{1818,38,38}


mylayer96=YoloConvLayer["layer96",128,1,1,38][mylayer91];mylayer96//Dimensions


{128,38,38}


mylayer97=DeconvolutionLayer[128,{2,2},"Weights"->Table[If[j==i,1,0],{j,1,128},{i,1,128},{2},{2}],"Biases"->ConstantArray[0,128],"Stride"->2][mylayer96];mylayer97//Dimensions


{128,76,76}


mylayer98=CatenateLayer[][{mylayer97,mylayer36}];mylayer98//Dimensions


{384,76,76}


mylayer105=NetChain[{
YoloConvLayer["layer99",128,1, 1, 76 ],
YoloConvLayer["layer100",256,3, 1, 76 ],
YoloConvLayer["layer101",128,1, 1, 76 ],
YoloConvLayer["layer102",256,3, 1, 76 ],
YoloConvLayer["layer103",128,1, 1, 76 ],
YoloConvLayer["layer104",256,3, 1, 76 ],
ConvolutionLayer[1818,{1,1},"Weights"->Import[hdfFile,{"Datasets","layer105_weights"}],"Biases"->Import[hdfFile,{"Datasets","layer105_biases"}]]
}][mylayer98];mylayer105//Dimensions


{1818,76,76}


(NetTake[yolonet,{"layer0","layer105"}][inp] == mylayer105)//AbsoluteTiming


Max[Abs[de]]


layer81=Import["/Users/julian/yolov3/darknet/Yolov3OpenImages.h5",{"Datasets","/layer81"}];layer81//Dimensions


{1818,19,19}


k=Abs[mylayer105-layer105];Max[k]


0.02184581756591797`


Position[k,Max[k]]


{{1525,71,55}}


mylayer105[[1525,71,55]]


-11.484664916992188`


layer105[[1525,71,55]]


-11.506510734558105`


objMap1=LogisticSigmoid@Partition[mylayer81,606][[All,5]];objMap1//Dimensions


{3,19,19}


objMap2=LogisticSigmoid@Partition[mylayer93,606][[All,5]];objMap2//Dimensions


{3,38,38}


objMap3=LogisticSigmoid@Partition[mylayer105,606][[All,5]];objMap3//Dimensions


{3,76,76}


classMap1=LogisticSigmoid@Partition[mylayer81,606][[All,6;;]];classMap1//Dimensions


{3,601,19,19}


classMap2=LogisticSigmoid@Partition[mylayer93,606][[All,6;;]];classMap2//Dimensions


{3,601,38,38}


classMap3=LogisticSigmoid@Partition[mylayer105,606][[All,6;;]];classMap3//Dimensions


{3,601,76,76}


px1=Partition[mylayer81,606][[All,1]];px1//Dimensions


{3,19,19}


py1=Partition[mylayer81,606][[All,2]];py1//Dimensions


{3,19,19}


x1=Table[j,{3},{i,0,18},{j,0,18}];x1//Dimensions


{3,19,19}


y1=Table[i,{3},{i,0,18},{j,0,18}];y1//Dimensions


{3,19,19}


nx1=LogisticSigmoid@px1+x1;nx1//Dimensions


{3,19,19}


ny1=LogisticSigmoid@py1+y1;ny1//Dimensions


{3,19,19}


pw1=Partition[mylayer81,606][[All,3]];pw1//Dimensions


{3,19,19}


ph1=Partition[mylayer81,606][[All,4]];ph1//Dimensions


{3,19,19}


w1=Table[{116,122,124}[[n]],{n,1,3},{19},{19}];w1//Dimensions


{3,19,19}


h1=Table[{90,124,90}[[n]],{n,1,3},{19},{19}];h1//Dimensions


{3,19,19}


nw1=Exp[pw1]*w1/606;nw1//Dimensions


{3,19,19}


nh1=Exp[ph1]*h1/606;nh1//Dimensions


{3,19,19}


nh1[[1,9+1,7+1]]


(* ::Input:: *)
(*{Image[inp,Interleaving->False],(LogisticSigmoid@layer93[[1]])//Image}*)


(* ::Input:: *)
(*classes=Import["/Users/julian/yolov3/darknet/data/openimages.names","Table"][[All,1]];Length[classes]*)


(* ::Input:: *)
(*classes*)
