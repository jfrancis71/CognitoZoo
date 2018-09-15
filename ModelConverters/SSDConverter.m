(* ::Package:: *)

(*
   Credit:
   This implementation is based on Changan Wang's Tensorflow code:
      https://github.com/HiKapok/SSD.TensorFlow
         
   SSD VGG 300 is based on the following paper:
   https://arxiv.org/abs/1512.02325
   Title: SSD: Single Shot MultiBox Detector
   Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
   Cheng-Yang Fu, Alexander C. Berg

   The weight file used here was initialised using the VOC2007+2012 trained model
   referenced in the repository: https://github.com/HiKapok/SSD.TensorFlow
   (Downloaded 18 June 2018)
*)


SSDFileName="CZModels/SSD300VGGHikapokNet20180618.hdf";


anchorsx1 = Table[2*x/75,{y,1,38},{x,.5,37.5}];
anchorsy1 = Table[2*y/75,{y,.5,37.5},{x,1,38}];
anchorsw1 = {0.141,0.141,0.070};
anchorsh1 = {0.141,0.070,0.141};


anchorsx2 = Table[4*x/75,{y,1,19},{x,.5,18.5}];
anchorsy2 = Table[4*y/75,{y,.5,18.5},{x,1,19}];
anchorsw2 = {0.273,0.282,0.346,0.141,0.115};
anchorsh2 = {0.273,0.141,0.115,0.282,0.346};


anchorsx3 = Table[8*x/75,{y,1,10},{x,.5,9.5}];
anchorsy3 = Table[8*y/75,{y,.5,9.5},{x,1,10}];
anchorsw3 = {0.454,0.530,0.649,0.265,0.216};
anchorsh3 = {0.454,0.265,0.216,0.530,0.649};


anchorsx4 = Table[16*x/75,{y,1,5},{x,.5,4.5}];
anchorsy4 = Table[16*y/75,{y,.5,4.5},{x,1,5}];
anchorsw4 = {0.631,0.777,0.952,0.388,0.317};
anchorsh4 = {0.631,0.388,0.317,0.777,0.952};


anchorsx5 = Table[x/3,{y,1,3},{x,.5,2.5}];
anchorsy5 = Table[y/3,{y,.5,2.5},{x,1,3}];
anchorsw5 = {0.807,1.02,0.512};
anchorsh5 = {0.807,0.512,1.02};


anchorsx6 = Table[x/2,{y,1,1},{x,1,1}];
anchorsy6 = Table[y/2,{y,1,1},{x,1,1}];
anchorsw6 = {0.983,1.27,0.636};
anchorsh6 = {0.983,0.636,1.27};


(* Looks like the HiKapok version has BGR ordering for some obscure reason *)
conv1W1=Reverse[Transpose[Import[SSDFileName,{"Datasets","/conv1_1W"}],{3,4,2,1}],2]*256.;
conv1W2=Transpose[Import[SSDFileName,{"Datasets","/conv1_2W"}],{3,4,2,1}];
conv1B1=Import[SSDFileName,{"Datasets","/conv1_1B"}];
conv1B2=Import[SSDFileName,{"Datasets","/conv1_2B"}];

conv2W1=Transpose[Import[SSDFileName,{"Datasets","/conv2_1W"}],{3,4,2,1}];
conv2W2=Transpose[Import[SSDFileName,{"Datasets","/conv2_2W"}],{3,4,2,1}];
conv2B1=Import[SSDFileName,{"Datasets","/conv2_1B"}];
conv2B2=Import[SSDFileName,{"Datasets","/conv2_2B"}];

conv3W1=Transpose[Import[SSDFileName,{"Datasets","/conv3_1W"}],{3,4,2,1}];
conv3W2=Transpose[Import[SSDFileName,{"Datasets","/conv3_2W"}],{3,4,2,1}];
conv3W3=Transpose[Import[SSDFileName,{"Datasets","/conv3_3W"}],{3,4,2,1}];
conv3B1=Import[SSDFileName,{"Datasets","/conv3_1B"}];
conv3B2=Import[SSDFileName,{"Datasets","/conv3_2B"}];
conv3B3=Import[SSDFileName,{"Datasets","/conv3_3B"}];

conv4W1=Transpose[Import[SSDFileName,{"Datasets","/conv4_1W"}],{3,4,2,1}];
conv4W2=Transpose[Import[SSDFileName,{"Datasets","/conv4_2W"}],{3,4,2,1}];
conv4W3=Transpose[Import[SSDFileName,{"Datasets","/conv4_3W"}],{3,4,2,1}];
conv4B1=Import[SSDFileName,{"Datasets","/conv4_1B"}];
conv4B2=Import[SSDFileName,{"Datasets","/conv4_2B"}];
conv4B3=Import[SSDFileName,{"Datasets","/conv4_3B"}];


conv5W1=Transpose[Import[SSDFileName,{"Datasets","/conv5_1W"}],{3,4,2,1}];
conv5W2=Transpose[Import[SSDFileName,{"Datasets","/conv5_2W"}],{3,4,2,1}];
conv5W3=Transpose[Import[SSDFileName,{"Datasets","/conv5_3W"}],{3,4,2,1}];
conv5B1=Import[SSDFileName,{"Datasets","/conv5_1B"}];
conv5B2=Import[SSDFileName,{"Datasets","/conv5_2B"}];
conv5B3=Import[SSDFileName,{"Datasets","/conv5_3B"}];


conv6W=Transpose[Import[SSDFileName,{"Datasets","/conv6_W"}],{3,4,2,1}];
conv6B=Import[SSDFileName,{"Datasets","/conv6_B"}];


conv7W=Transpose[Import[SSDFileName,{"Datasets","/conv7_W"}],{3,4,2,1}];
conv7B=Import[SSDFileName,{"Datasets","/conv7_B"}];


conv8W1=Transpose[Import[SSDFileName,{"Datasets","/conv8_1W"}],{3,4,2,1}];
conv8W2=Transpose[Import[SSDFileName,{"Datasets","/conv8_2W"}],{3,4,2,1}];
conv8B1=Import[SSDFileName,{"Datasets","/conv8_1B"}];
conv8B2=Import[SSDFileName,{"Datasets","/conv8_2B"}];


conv9W1=Transpose[Import[SSDFileName,{"Datasets","/conv9_1W"}],{3,4,2,1}];
conv9W2=Transpose[Import[SSDFileName,{"Datasets","/conv9_2W"}],{3,4,2,1}];
conv9B1=Import[SSDFileName,{"Datasets","/conv9_1B"}];
conv9B2=Import[SSDFileName,{"Datasets","/conv9_2B"}];


conv10W1=Transpose[Import[SSDFileName,{"Datasets","/conv10_1W"}],{3,4,2,1}];
conv10W2=Transpose[Import[SSDFileName,{"Datasets","/conv10_2W"}],{3,4,2,1}];
conv10B1=Import[SSDFileName,{"Datasets","/conv10_1B"}];
conv10B2=Import[SSDFileName,{"Datasets","/conv10_2B"}];


conv11W1=Transpose[Import[SSDFileName,{"Datasets","/conv11_1W"}],{3,4,2,1}];
conv11W2=Transpose[Import[SSDFileName,{"Datasets","/conv11_2W"}],{3,4,2,1}];
conv11B1=Import[SSDFileName,{"Datasets","/conv11_1B"}];
conv11B2=Import[SSDFileName,{"Datasets","/conv11_2B"}];


block4ClassesW = Transpose[Import[SSDFileName,{"Datasets","/block4_classes_W"}],{3,4,2,1}];
block4ClassesB = Import[SSDFileName,{"Datasets","/block4_classes_B"}];
block4LocW = Transpose[Import[SSDFileName,{"Datasets","/block4_loc_W"}],{3,4,2,1}];
block4LocB = Import[SSDFileName,{"Datasets","/block4_loc_B"}];


block7ClassesW = Transpose[Import[SSDFileName,{"Datasets","/block7_classes_W"}],{3,4,2,1}];
block7ClassesB = Import[SSDFileName,{"Datasets","/block7_classes_B"}];
block7LocW = Transpose[Import[SSDFileName,{"Datasets","/block7_loc_W"}],{3,4,2,1}];
block7LocB = Import[SSDFileName,{"Datasets","/block7_loc_B"}];


block8ClassesW = Transpose[Import[SSDFileName,{"Datasets","/block8_classes_W"}],{3,4,2,1}];
block8ClassesB = Import[SSDFileName,{"Datasets","/block8_classes_B"}];
block8LocW = Transpose[Import[SSDFileName,{"Datasets","/block8_loc_W"}],{3,4,2,1}];
block8LocB = Import[SSDFileName,{"Datasets","/block8_loc_B"}];


block9ClassesW = Transpose[Import[SSDFileName,{"Datasets","/block9_classes_W"}],{3,4,2,1}];
block9ClassesB = Import[SSDFileName,{"Datasets","/block9_classes_B"}];
block9LocW = Transpose[Import[SSDFileName,{"Datasets","/block9_loc_W"}],{3,4,2,1}];
block9LocB = Import[SSDFileName,{"Datasets","/block9_loc_B"}];


block10ClassesW = Transpose[Import[SSDFileName,{"Datasets","/block10_classes_W"}],{3,4,2,1}];
block10ClassesB = Import[SSDFileName,{"Datasets","/block10_classes_B"}];
block10LocW = Transpose[Import[SSDFileName,{"Datasets","/block10_loc_W"}],{3,4,2,1}];
block10LocB = Import[SSDFileName,{"Datasets","/block10_loc_B"}];


block11ClassesW = Transpose[Import[SSDFileName,{"Datasets","/block11_classes_W"}],{3,4,2,1}];
block11ClassesB = Import[SSDFileName,{"Datasets","/block11_classes_B"}];
block11LocW = Transpose[Import[SSDFileName,{"Datasets","/block11_loc_W"}],{3,4,2,1}];
block11LocB = Import[SSDFileName,{"Datasets","/block11_loc_B"}];


blockNet4 = NetChain[{
   ConvolutionLayer[64,{3,3},"Biases"->conv1B1,"Weights"->conv1W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[64,{3,3},"Biases"->conv1B2,"Weights"->conv1W2,"PaddingSize"->1],Ramp,
   PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"Biases"->conv2B1,"Weights"->conv2W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[128,{3,3},"Biases"->conv2B2,"Weights"->conv2W2,"PaddingSize"->1],Ramp,
   PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"Biases"->conv3B1,"Weights"->conv3W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[256,{3,3},"Biases"->conv3B2,"Weights"->conv3W2,"PaddingSize"->1],Ramp,
   ConvolutionLayer[256,{3,3},"Biases"->conv3B3,"Weights"->conv3W3,"PaddingSize"->1],Ramp,
   PaddingLayer[{{0,0},{0,1},{0,1}}],PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[512,{3,3},"Biases"->conv4B1,"Weights"->conv4W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv4B2,"Weights"->conv4W2,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv4B3,"Weights"->conv4W3,"PaddingSize"->1],Ramp
   }];


blockNet7 = NetChain[{
   PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[512,{3,3},"Biases"->conv5B1,"Weights"->conv5W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv5B2,"Weights"->conv5W2,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv5B3,"Weights"->conv5W3,"PaddingSize"->1],Ramp,
   PoolingLayer[{3,3},"Stride"->1,"PaddingSize"->1],
   ConvolutionLayer[1024,{3,3},"Biases"->conv6B,"Weights"->conv6W,"PaddingSize"->6,"Dilation"->6],Ramp,
   ConvolutionLayer[1024,{1,1},"Biases"->conv7B,"Weights"->conv7W],Ramp
}];


blockNet8 = NetChain[{
   ConvolutionLayer[256, {1,1}, "Biases"->conv8B1,"Weights"->conv8W1],Ramp,
   PaddingLayer[{{0,0},{1,1},{1,1}}],
   ConvolutionLayer[512, {3,3}, "Biases"->conv8B2,"Weights"->conv8W2,"Stride"->2],Ramp
   }];


blockNet9 = NetChain[{
   ConvolutionLayer[128, {1,1}, "Biases"->conv9B1,"Weights"->conv9W1],Ramp,
   PaddingLayer[{{0,0},{0,1},{0,1}}],
   ConvolutionLayer[256, {3,3}, "Biases"->conv9B2,"Weights"->conv9W2,"Stride"->2],Ramp
   }];


blockNet10 = NetChain[{
   ConvolutionLayer[128, {1,1}, "Biases"->conv10B1,"Weights"->conv10W1],Ramp,
   ConvolutionLayer[256, {3,3}, "Biases"->conv10B2,"Weights"->conv10W2],Ramp
   }];


blockNet11 = NetChain[{
   ConvolutionLayer[128, {1,1}, "Biases"->conv11B1,"Weights"->conv11W1],Ramp,
   ConvolutionLayer[256, {3,3}, "Biases"->conv11B2,"Weights"->conv11W2],Ramp
   }];


ngamma4=Table[20,{c,1,512},{38},{38}];


MultiBoxNetLayer1=NetGraph[{
(* These layers up to the convolution compute the channed normalisation which is just
   used in layer1. Slightly convoluted code (forgive the pun) to use SoftmaxLayer for this.
*)
   TransposeLayer[1->3],
   ElementwiseLayer[Log[Max[#^2,10^-6]]&],SoftmaxLayer[],
   ElementwiseLayer[Sqrt],TransposeLayer[1->3],
   ConstantTimesLayer["Scaling"->ngamma4],
   ConvolutionLayer[63,{3,3},"Biases"->block4ClassesB,"Weights"->block4ClassesW,"PaddingSize"->1],
   ReshapeLayer[{3,21,38,38}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[12,{3,3},"Biases"->block4LocB,"Weights"->block4LocW,"PaddingSize"->1]
 },{1->2,2->3,3->4,4->5,5->6,
    6->7,7->8,8->9,9->10,10->11,
    6->12,
    11->NetPort["ObjMap1"],12->NetPort["SSDLocs1"]}];


MultiBoxNetLayer2=NetGraph[{
   ConvolutionLayer[105,{3,3},"Biases"->block7ClassesB,"Weights"->block7ClassesW,"PaddingSize"->1],
   ReshapeLayer[{5,21,19,19}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[20,{3,3},"Biases"->block7LocB,"Weights"->block7LocW,"PaddingSize"->1]
 },{1->2,2->3,3->4,4->5,5->NetPort["ObjMap2"],6->NetPort["SSDLocs2"]}];


MultiBoxNetLayer3=NetGraph[{
   ConvolutionLayer[105,{3,3},"Biases"->block8ClassesB,"Weights"->block8ClassesW,"PaddingSize"->1],
   ReshapeLayer[{5,21,10,10}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[20,{3,3},"Biases"->block8LocB,"Weights"->block8LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap3"],6->NetPort["SSDLocs3"]}];


MultiBoxNetLayer4=NetGraph[{
   ConvolutionLayer[105,{3,3},"Biases"->block9ClassesB,"Weights"->block9ClassesW,"PaddingSize"->1],
   ReshapeLayer[{5,21,5,5}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],   
   ConvolutionLayer[20,{3,3},"Biases"->block9LocB,"Weights"->block9LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap4"],6->NetPort["SSDLocs4"]}];


MultiBoxNetLayer5=NetGraph[{
   ConvolutionLayer[63,{3,3},"Biases"->block10ClassesB,"Weights"->block10ClassesW,"PaddingSize"->1],
   ReshapeLayer[{3,21,3,3}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[12,{3,3},"Biases"->block10LocB,"Weights"->block10LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap5"],6->NetPort["SSDLocs5"]}];


MultiBoxNetLayer6=NetGraph[{
   ConvolutionLayer[63,{3,3},"Biases"->block11ClassesB,"Weights"->block11ClassesW,"PaddingSize"->1],
   ReshapeLayer[{3,21,1,1}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[12,{3,3},"Biases"->block11LocB,"Weights"->block11LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap6"],6->NetPort["SSDLocs6"]}];


SSDNetMain=NetGraph[{
   blockNet4,
   blockNet7,
   blockNet8,
   blockNet9,
   blockNet10,
   blockNet11,
   MultiBoxNetLayer1,
   MultiBoxNetLayer2,
   MultiBoxNetLayer3,
   MultiBoxNetLayer4,
   MultiBoxNetLayer5,
   MultiBoxNetLayer6},
   {1->2->3->4->5->6,
   1->7,2->8,3->9,
   4->10,5->11,6->12}];


(*
   Output format is 4*Anchors*Height*Width in x,y,width,height order where x and y represent the centre of the rectangle
*)
MultiBoxLocationDecoder[ anchorsx_, anchorsy_, anchorsw_, anchorsh_ ] :=
   Module[{width=Dimensions[anchorsx][[1]],height=Dimensions[anchorsx][[2]],anchors=Length[anchorsh]},
      NetGraph[{
         ReshapeLayer[{Length[anchorsh],4,height,width}],
   
         PartLayer[{All,2}],
         ConstantTimesLayer["Scaling"->Table[0.1*anchorsw[[b]],{b,1,anchors},{height},{width}]],
         ConstantPlusLayer["Biases"->ConstantArray[anchorsx,{anchors}]],
   
         PartLayer[{All,1}],
         ConstantTimesLayer["Scaling"->Table[0.1*anchorsh[[b]],{b,1,anchors},{height},{width}]],
         ConstantPlusLayer["Biases"->ConstantArray[anchorsy,{anchors}]],

         PartLayer[{All,4}],
         ElementwiseLayer[Exp[#*0.2]&],
         ConstantTimesLayer["Scaling"->Table[anchorsw[[b]],{b,1,anchors},{height},{width}]],

         PartLayer[{All,3}],
         ElementwiseLayer[Exp[#*0.2]&],
         ConstantTimesLayer["Scaling"->Table[anchorsh[[b]],{b,1,anchors},{height},{width}]],

         CatenateLayer[],
         ReshapeLayer[{4,anchors,height,width}]
   },
   {1->{2,5,8,11},2->3->4,5->6->7,8->9->10,11->12->13,{4,7,10,13}->14->15}]
];


SSDNet = NetGraph[{
   SSDNetMain,
   MultiBoxLocationDecoder[anchorsx1,anchorsy1,anchorsw1,anchorsh1],
   MultiBoxLocationDecoder[anchorsx2,anchorsy2,anchorsw2,anchorsh2],
   MultiBoxLocationDecoder[anchorsx3,anchorsy3,anchorsw3,anchorsh3],
   MultiBoxLocationDecoder[anchorsx4,anchorsy4,anchorsw4,anchorsh4],
   MultiBoxLocationDecoder[anchorsx5,anchorsy5,anchorsw5,anchorsh5],
   MultiBoxLocationDecoder[anchorsx6,anchorsy6,anchorsw6,anchorsh6]
},
   {
   NetPort[{1,"SSDLocs1"}]->2->NetPort["Locs1"],
   NetPort[{1,"SSDLocs2"}]->3->NetPort["Locs2"],
   NetPort[{1,"SSDLocs3"}]->4->NetPort["Locs3"],
   NetPort[{1,"SSDLocs4"}]->5->NetPort["Locs4"],
   NetPort[{1,"SSDLocs5"}]->6->NetPort["Locs5"],
   NetPort[{1,"SSDLocs6"}]->7->NetPort["Locs6"]
   },
   "Input"->NetEncoder[{"Image",{300,300},"ColorSpace"->"RGB","MeanImage"->{123,117,104}/256.}]];


(* CloudExport[ SSDNet, "WLNET","SSD300VGGHikapokNet20180618.wlnet" ] *)
