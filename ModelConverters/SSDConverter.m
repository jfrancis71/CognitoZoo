(* ::Package:: *)

(*
   Credit:
   Wei Liu's CAFFE model was the reference model for this Mathematia implementation:
      https://github.com/weiliu89/caffe/tree/ssd (Downloaded 21/09/2018)
   and uses weights from VGG_VOC0712_SSD_300x300_iter_120000.caffemodel (Downloaded 20/09/2018)
         
   SSD VGG 300 is based on the following paper:
   https://arxiv.org/abs/1512.02325
   Title: SSD: Single Shot MultiBox Detector
   Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
   Cheng-Yang Fu, Alexander C. Berg
*)


SSDFileName="CZModels/SSDVGG300PascalVOCReference20180920.hdf";


anchorsx1 = Table[2*x/75,{y,1,38},{x,.5,37.5}];
anchorsy1 = Table[2*y/75,{y,.5,37.5},{x,1,38}];
anchorsw1 = {30/300,42/300,42/300,21/300};
anchorsh1 = {30/300,42/300,21/300,42/300};


anchorsx2 = Table[4*x/75,{y,1,19},{x,.5,18.5}];
anchorsy2 = Table[4*y/75,{y,.5,18.5},{x,1,19}];
anchorsw2 = {60/300,81/300,84/300,42/300,103/300,34/300};
anchorsh2 = {60/300,81/300,42/300,84/300,34/300,103/300};


anchorsx3 = Table[8*x/75,{y,1,10},{x,.5,9.5}];
anchorsy3 = Table[8*y/75,{y,.5,9.5},{x,1,10}];
anchorsw3 = {111,134,156,78,192,64}/300;
anchorsh3 = {111,134,78,156,64,192}/300;


anchorsx4 = Table[16*x/75,{y,1,5},{x,.5,4.5}];
anchorsy4 = Table[16*y/75,{y,.5,4.5},{x,1,5}];
anchorsw4 = {162,185,229,114,280,93}/300;
anchorsh4 = {162,185,114,229,93,280}/300;


anchorsx5 = Table[x/3,{y,1,3},{x,.5,2.5}];
anchorsy5 = Table[y/3,{y,.5,2.5},{x,1,3}];
anchorsw5 = {213,237,301,150}/300;
anchorsh5 = {213,237,150,301}/300;


anchorsx6 = Table[x/2,{y,1,1},{x,1,1}];
anchorsy6 = Table[y/2,{y,1,1},{x,1,1}];
anchorsw6 = {264,288,373,186}/300;
anchorsh6 = {264,288,186,373}/300;


(* Caffe has BGR ordering for some obscure reason, because of openCV? *)
conv1W1=Reverse[Import[SSDFileName,{"Datasets","/conv1_1W"}],2]*255;
conv1W2=Import[SSDFileName,{"Datasets","/conv1_2W"}];
conv1B1=Import[SSDFileName,{"Datasets","/conv1_1B"}];
conv1B2=Import[SSDFileName,{"Datasets","/conv1_2B"}];

conv2W1=Import[SSDFileName,{"Datasets","/conv2_1W"}];
conv2W2=Import[SSDFileName,{"Datasets","/conv2_2W"}];
conv2B1=Import[SSDFileName,{"Datasets","/conv2_1B"}];
conv2B2=Import[SSDFileName,{"Datasets","/conv2_2B"}];

conv3W1=Import[SSDFileName,{"Datasets","/conv3_1W"}];
conv3W2=Import[SSDFileName,{"Datasets","/conv3_2W"}];
conv3W3=Import[SSDFileName,{"Datasets","/conv3_3W"}];
conv3B1=Import[SSDFileName,{"Datasets","/conv3_1B"}];
conv3B2=Import[SSDFileName,{"Datasets","/conv3_2B"}];
conv3B3=Import[SSDFileName,{"Datasets","/conv3_3B"}];

conv4W1=Import[SSDFileName,{"Datasets","/conv4_1W"}];
conv4W2=Import[SSDFileName,{"Datasets","/conv4_2W"}];
conv4W3=Import[SSDFileName,{"Datasets","/conv4_3W"}];
conv4B1=Import[SSDFileName,{"Datasets","/conv4_1B"}];
conv4B2=Import[SSDFileName,{"Datasets","/conv4_2B"}];
conv4B3=Import[SSDFileName,{"Datasets","/conv4_3B"}];


conv5W1=Import[SSDFileName,{"Datasets","/conv5_1W"}];
conv5W2=Import[SSDFileName,{"Datasets","/conv5_2W"}];
conv5W3=Import[SSDFileName,{"Datasets","/conv5_3W"}];
conv5B1=Import[SSDFileName,{"Datasets","/conv5_1B"}];
conv5B2=Import[SSDFileName,{"Datasets","/conv5_2B"}];
conv5B3=Import[SSDFileName,{"Datasets","/conv5_3B"}];


conv6W=Import[SSDFileName,{"Datasets","/conv6_W"}];
conv6B=Import[SSDFileName,{"Datasets","/conv6_B"}];


conv7W=Import[SSDFileName,{"Datasets","/conv7_W"}];
conv7B=Import[SSDFileName,{"Datasets","/conv7_B"}];


conv8W1=Import[SSDFileName,{"Datasets","/conv8_1W"}];
conv8W2=Import[SSDFileName,{"Datasets","/conv8_2W"}];
conv8B1=Import[SSDFileName,{"Datasets","/conv8_1B"}];
conv8B2=Import[SSDFileName,{"Datasets","/conv8_2B"}];


conv9W1=Import[SSDFileName,{"Datasets","/conv9_1W"}];
conv9W2=Import[SSDFileName,{"Datasets","/conv9_2W"}];
conv9B1=Import[SSDFileName,{"Datasets","/conv9_1B"}];
conv9B2=Import[SSDFileName,{"Datasets","/conv9_2B"}];


conv10W1=Import[SSDFileName,{"Datasets","/conv10_1W"}];
conv10W2=Import[SSDFileName,{"Datasets","/conv10_2W"}];
conv10B1=Import[SSDFileName,{"Datasets","/conv10_1B"}];
conv10B2=Import[SSDFileName,{"Datasets","/conv10_2B"}];


conv11W1=Import[SSDFileName,{"Datasets","/conv11_1W"}];
conv11W2=Import[SSDFileName,{"Datasets","/conv11_2W"}];
conv11B1=Import[SSDFileName,{"Datasets","/conv11_1B"}];
conv11B2=Import[SSDFileName,{"Datasets","/conv11_2B"}];


block4ClassesW = Import[SSDFileName,{"Datasets","/block4_classes_W"}];
block4ClassesB = Import[SSDFileName,{"Datasets","/block4_classes_B"}];
block4LocW = Import[SSDFileName,{"Datasets","/block4_loc_W"}];
block4LocB = Import[SSDFileName,{"Datasets","/block4_loc_B"}];


block7ClassesW = Import[SSDFileName,{"Datasets","/block7_classes_W"}];
block7ClassesB = Import[SSDFileName,{"Datasets","/block7_classes_B"}];
block7LocW = Import[SSDFileName,{"Datasets","/block7_loc_W"}];
block7LocB = Import[SSDFileName,{"Datasets","/block7_loc_B"}];


block8ClassesW = Import[SSDFileName,{"Datasets","/block8_classes_W"}];
block8ClassesB = Import[SSDFileName,{"Datasets","/block8_classes_B"}];
block8LocW = Import[SSDFileName,{"Datasets","/block8_loc_W"}];
block8LocB = Import[SSDFileName,{"Datasets","/block8_loc_B"}];


block9ClassesW = Import[SSDFileName,{"Datasets","/block9_classes_W"}];
block9ClassesB = Import[SSDFileName,{"Datasets","/block9_classes_B"}];
block9LocW = Import[SSDFileName,{"Datasets","/block9_loc_W"}];
block9LocB = Import[SSDFileName,{"Datasets","/block9_loc_B"}];


block10ClassesW = Import[SSDFileName,{"Datasets","/block10_classes_W"}];
block10ClassesB = Import[SSDFileName,{"Datasets","/block10_classes_B"}];
block10LocW = Import[SSDFileName,{"Datasets","/block10_loc_W"}];
block10LocB = Import[SSDFileName,{"Datasets","/block10_loc_B"}];


block11ClassesW = Import[SSDFileName,{"Datasets","/block11_classes_W"}];
block11ClassesB = Import[SSDFileName,{"Datasets","/block11_classes_B"}];
block11LocW = Import[SSDFileName,{"Datasets","/block11_loc_W"}];
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
   ConvolutionLayer[256, {3,3}, "Biases"->conv9B2,"Weights"->conv9W2,"Stride"->2,"PaddingSize"->1],Ramp
   }];


blockNet10 = NetChain[{
   ConvolutionLayer[128, {1,1}, "Biases"->conv10B1,"Weights"->conv10W1],Ramp,
   ConvolutionLayer[256, {3,3}, "Biases"->conv10B2,"Weights"->conv10W2],Ramp
   }];


blockNet11 = NetChain[{
   ConvolutionLayer[128, {1,1}, "Biases"->conv11B1,"Weights"->conv11W1],Ramp,
   ConvolutionLayer[256, {3,3}, "Biases"->conv11B2,"Weights"->conv11W2],Ramp
   }];


ngamma4c = Import[SSDFileName,{"Datasets","/conv4_3_norm"}];
ngamma4=Table[ngamma4c[[c]],{c,1,512},{38},{38}];


MultiBoxNetLayer1=NetGraph[{
(* These layers up to the convolution compute the channed normalisation which is just
   used in layer1. Slightly convoluted code (forgive the pun) to use SoftmaxLayer for this.
*)
   TransposeLayer[1->3],
   ElementwiseLayer[Log[Max[#^2,10^-6]]&],SoftmaxLayer[],
   ElementwiseLayer[Sqrt],TransposeLayer[1->3],
   ConstantTimesLayer["Scaling"->ngamma4],
   ConvolutionLayer[84,{3,3},"Biases"->block4ClassesB,"Weights"->block4ClassesW,"PaddingSize"->1],
   ReshapeLayer[{4,21,38,38}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[16,{3,3},"Biases"->block4LocB,"Weights"->block4LocW,"PaddingSize"->1]
 },{1->2,2->3,3->4,4->5,5->6,
    6->7,7->8,8->9,9->10,10->11,
    6->12,
    11->NetPort["ObjMap1"],12->NetPort["SSDLocs1"]}];


MultiBoxNetLayer2=NetGraph[{
   ConvolutionLayer[126,{3,3},"Biases"->block7ClassesB,"Weights"->block7ClassesW,"PaddingSize"->1],
   ReshapeLayer[{6,21,19,19}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[24,{3,3},"Biases"->block7LocB,"Weights"->block7LocW,"PaddingSize"->1]
 },{1->2,2->3,3->4,4->5,5->NetPort["ObjMap2"],6->NetPort["SSDLocs2"]}];


MultiBoxNetLayer3=NetGraph[{
   ConvolutionLayer[126,{3,3},"Biases"->block8ClassesB,"Weights"->block8ClassesW,"PaddingSize"->1],
   ReshapeLayer[{6,21,10,10}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[24,{3,3},"Biases"->block8LocB,"Weights"->block8LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap3"],6->NetPort["SSDLocs3"]}];


MultiBoxNetLayer4=NetGraph[{
   ConvolutionLayer[126,{3,3},"Biases"->block9ClassesB,"Weights"->block9ClassesW,"PaddingSize"->1],
   ReshapeLayer[{6,21,5,5}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],   
   ConvolutionLayer[24,{3,3},"Biases"->block9LocB,"Weights"->block9LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap4"],6->NetPort["SSDLocs4"]}];


MultiBoxNetLayer5=NetGraph[{
   ConvolutionLayer[84,{3,3},"Biases"->block10ClassesB,"Weights"->block10ClassesW,"PaddingSize"->1],
   ReshapeLayer[{4,21,3,3}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[16,{3,3},"Biases"->block10LocB,"Weights"->block10LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap5"],6->NetPort["SSDLocs5"]}];


MultiBoxNetLayer6=NetGraph[{
   ConvolutionLayer[84,{3,3},"Biases"->block11ClassesB,"Weights"->block11ClassesW,"PaddingSize"->1],
   ReshapeLayer[{4,21,1,1}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[16,{3,3},"Biases"->block11LocB,"Weights"->block11LocW,"PaddingSize"->1]},
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
   
         PartLayer[{All,1}],
         ConstantTimesLayer["Scaling"->Table[0.1*anchorsw[[b]],{b,1,anchors},{height},{width}]],
         ConstantPlusLayer["Biases"->ConstantArray[anchorsx,{anchors}]],
   
         PartLayer[{All,2}],
         ConstantTimesLayer["Scaling"->Table[0.1*anchorsh[[b]],{b,1,anchors},{height},{width}]],
         ConstantPlusLayer["Biases"->ConstantArray[anchorsy,{anchors}]],

         PartLayer[{All,3}],
         ElementwiseLayer[Exp[#*0.2]&],
         ConstantTimesLayer["Scaling"->Table[anchorsw[[b]],{b,1,anchors},{height},{width}]],

         PartLayer[{All,4}],
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
   "Input"->NetEncoder[{"Image",{300,300},"ColorSpace"->"RGB","MeanImage"->{123,117,104}/255.}]];


(* Exporting code and setting permissions:
   CloudExport[ SSDNet, "WLNET","SSDVGG300PascalVOCReference20180920.wlnet" ]
   Options[CloudObject["SSDVGG300PascalVOCReference20180920.wlnet"],Permissions]
   SetOptions[CloudObject["SSDVGG300PascalVOCReference20180920.wlnet"], Permissions\[Rule]"Public"]   
*)
