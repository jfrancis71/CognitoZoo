(* ::Package:: *)

(*
   Converts tiny Yolo v2 weights to Mathematica v11 neural net

   The code is based on the tiny YOLO model from Darknet, Joseph Redmon:
      https://pjreddie.com/darknet/yolo/
      
      Citation:
      @misc{darknet13,
      author =   {Joseph Redmon},
      title =    {Darknet: Open Source Neural Networks in C},
      howpublished = {\url{http://pjreddie.com/darknet/}},
      year = {2013--2016}
      }      
*)


file=OpenRead["~/Google Drive/Personal/Computer Science/CZModels/tiny-yolo-voc.weights",BinaryFormat->True];


BinaryReadList[file,"Integer32",4]; (*Some magic numbers indicating file format versioning *)


ReadConvolutionLayer[file_,{outputLayers_,inputLayers_,w_,h_}]:=(
convb=BinaryReadList[file,"Real32",outputLayers];
convScales=BinaryReadList[file,"Real32",outputLayers];
convRM=BinaryReadList[file,"Real32",outputLayers];
convRV=BinaryReadList[file,"Real32",outputLayers];
convW=ArrayReshape[BinaryReadList[file,"Real32",outputLayers*inputLayers*w*h],{outputLayers,inputLayers,w,h}];
{convb,convW,convScales,convRM,convRV})


(* Naming Convention:
Layers follow the same structure as tiny YOLO, but index starts from 1, so for example CZ layer 5 corresponds to tiny YOLO layer 4.
And the convolution layer includes the convolution, batch normalization and leaky RELU
*)


{conv1b,conv1W,conv1Scales,conv1RM,conv1RV}=ReadConvolutionLayer[file,{16,3,3,3}];


CZConv1=ConvolutionLayer[16,{3,3},"Biases"->Table[0,{16}],"Weights"->conv1W,"PaddingSize"->1];


CZBN1=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{16,416,416},"Scaling"->conv1Scales,"Biases"->conv1b,"MovingMean"->conv1RM,"MovingVariance"->conv1RV];


{conv3b,conv3W,conv3Scales,conv3RM,conv3RV}=ReadConvolutionLayer[file,{32,16,3,3}];


CZConv3=ConvolutionLayer[32,{3,3},"Biases"->Table[0,{32}],"Weights"->conv3W,"PaddingSize"->1];


CZBN3=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{32,208,208},"Scaling"->conv3Scales,"Biases"->conv3b,"MovingMean"->conv3RM,"MovingVariance"->conv3RV];


{conv5b,conv5W,conv5Scales,conv5RM,conv5RV}=ReadConvolutionLayer[file,{64,32,3,3}];


CZConv5=ConvolutionLayer[64,{3,3},"Biases"->Table[0,{64}],"Weights"->conv5W,"PaddingSize"->1];


CZBN5=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{64,104,104},"Scaling"->conv5Scales,"Biases"->conv5b,"MovingMean"->conv5RM,"MovingVariance"->conv5RV];


{conv7b,conv7W,conv7Scales,conv7RM,conv7RV}=ReadConvolutionLayer[file,{128,64,3,3}];


CZConv7=ConvolutionLayer[128,{3,3},"Biases"->Table[0,{128}],"Weights"->conv7W,"PaddingSize"->1];


CZBN7=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{128,52,52},"Scaling"->conv7Scales,"Biases"->conv7b,"MovingMean"->conv7RM,"MovingVariance"->conv7RV];


{conv9b,conv9W,conv9Scales,conv9RM,conv9RV}=ReadConvolutionLayer[file,{256,128,3,3}];


CZConv9=ConvolutionLayer[256,{3,3},"Biases"->Table[0,{256}],"Weights"->conv9W,"PaddingSize"->1];


CZBN9=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{256,26,26},"Scaling"->conv9Scales,"Biases"->conv9b,"MovingMean"->conv9RM,"MovingVariance"->conv9RV];


{conv11b,conv11W,conv11Scales,conv11RM,conv11RV}=ReadConvolutionLayer[file,{512,256,3,3}];


CZConv11=ConvolutionLayer[512,{3,3},"Biases"->Table[0,{512}],"Weights"->conv11W,"PaddingSize"->1];


CZBN11=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{512,13,13},"Scaling"->conv11Scales,"Biases"->conv11b,"MovingMean"->conv11RM,"MovingVariance"->conv11RV];


{conv13b,conv13W,conv13Scales,conv13RM,conv13RV}=ReadConvolutionLayer[file,{1024,512,3,3}];


CZConv13=ConvolutionLayer[1024,{3,3},"Biases"->Table[0,{1024}],"Weights"->conv13W,"PaddingSize"->1];


CZBN13=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{1024,13,13},"Scaling"->conv13Scales,"Biases"->conv13b,"MovingMean"->conv13RM,"MovingVariance"->conv13RV];


{conv14b,conv14W,conv14Scales,conv14RM,conv14RV}=ReadConvolutionLayer[file,{1024,1024,3,3}];


CZConv14=ConvolutionLayer[1024,{3,3},"Biases"->Table[0,{1024}],"Weights"->conv14W,"PaddingSize"->1];


CZBN14=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{1024,13,13},"Scaling"->conv14Scales,"Biases"->conv14b,"MovingMean"->conv14RM,"MovingVariance"->conv14RV];


conv15b=BinaryReadList[file,"Real32",125];
conv15W=ArrayReshape[BinaryReadList[file,"Real32",125*1024*1*1],{125,1024,1,1}];


CZConv15=ConvolutionLayer[125,{1,1},"Biases"->conv15b,"Weights"->conv15W];


leakyReLU = ElementwiseLayer[Ramp[#]+Ramp[-#]*-.1&];


trunkNet = NetChain[{
   CZConv1,CZBN1,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv3,CZBN3,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv5,CZBN5,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv7,CZBN7,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv9,CZBN9,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv11,CZBN11,leakyReLU,PaddingLayer[{{0,0},{0,1},{0,1}},-100],PoolingLayer[{2,2},"Stride"->1],
   CZConv13,CZBN13,leakyReLU,
   CZConv14,CZBN14,leakyReLU,
   CZConv15},
   "Input"->NetEncoder[{"Image",{416,416},ColorSpace->"RGB"}]];


biases={{1.08,1.19},{3.42,4.41},{6.63,11.38},{9.42,5.11},{16.62,10.52}};


BoxTransformationNet = NetGraph[{
   "cx"->{PartLayer[1],LogisticSigmoid,ConstantPlusLayer["Biases"->Table[x,{y,0,12},{x,0,12},{s,1,5}]],ElementwiseLayer[#/13.&]},
   "cy"->{PartLayer[2],LogisticSigmoid,ConstantPlusLayer["Biases"->Table[y,{y,0,12},{x,0,12},{s,1,5}]],ElementwiseLayer[#/13.&]},
   "width"->{PartLayer[3],ElementwiseLayer[Exp],ConstantTimesLayer["Scaling"->Table[biases[[s,1]]/13.,{y,0,12},{x,0,12},{s,1,5}]]},
   "height"->{PartLayer[4],ElementwiseLayer[Exp],ConstantTimesLayer["Scaling"->Table[biases[[s,2]]/13.,{y,0,12},{x,0,12},{s,1,5}]]},
   "minx"->ThreadingLayer[#1-#2/2&],
   "miny"->ThreadingLayer[1-#1-#2/2&],
   "maxx"->ThreadingLayer[#1+#2/2&],
   "maxy"->ThreadingLayer[1-#1+#2/2&],
   "cat"->CatenateLayer[],
   "reshape"->{ElementwiseLayer[#*416&],ReshapeLayer[{4,13,13,5}],TransposeLayer[{1<->4,1<->2,2<->3}],ReshapeLayer[{13,13,5,2,2}]}
},{
   {"cx","width"}->{"minx","maxx"},{"cy","height"}->{"miny","maxy"},
   {"minx","miny","maxx","maxy"}->"cat"->"reshape"}];


DecoderNet=NetGraph[{
"reshape"->{ReshapeLayer[{5,25,13,13}],TransposeLayer[]},
"objectness"->{PartLayer[5],LogisticSigmoid,TransposeLayer[{1<->3,1<->2}]},
"classes"->{PartLayer[6;;25],TransposeLayer[{1<->4,2<->3,1<->2}],SoftmaxLayer[]},
"boxes"->{PartLayer[1;;4],TransposeLayer[{2<->4,2<->3}],BoxTransformationNet}},{"reshape"->"objectness"->NetPort["Objectness"],
"reshape"->"classes"->NetPort["ClassProb"],
"reshape"->"boxes"->NetPort["Boxes"]
}];


TinyYoloNet = NetGraph[{
   "trunkNet"->trunkNet,
   "decoderNet"->DecoderNet,
   "flatObjectness"->FlattenLayer[],"flatClassProb"->FlattenLayer[2],"flatBoxes"->FlattenLayer[2]},{
   "trunkNet"->"decoderNet",
   NetPort[{"decoderNet","Objectness"}]->"flatObjectness"->NetPort["Objectness"],
   NetPort[{"decoderNet","ClassProb"}]->"flatClassProb"->NetPort["ClassProb"],
   NetPort[{"decoderNet","Boxes"}]->"flatBoxes"->NetPort["Boxes"]
   }];
