(* ::Package:: *)

conv1W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv1_1W"}],{3,4,2,1}];
conv1W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv1_2W"}],{3,4,2,1}];

conv2W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv2_1W"}],{3,4,2,1}];
conv2W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv2_2W"}],{3,4,2,1}];

conv3W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv3_1W"}],{3,4,2,1}];
conv3W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv3_2W"}],{3,4,2,1}];
conv3W3=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv3_3W"}],{3,4,2,1}];

conv4W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv4_1W"}],{3,4,2,1}];
conv4W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv4_2W"}],{3,4,2,1}];
conv4W3=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv4_3W"}],{3,4,2,1}];

conv5W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv5_1W"}],{3,4,2,1}];
conv5W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv5_2W"}],{3,4,2,1}];
conv5W3=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv5_3W"}],{3,4,2,1}];

conv6W=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv6_W"}],{3,4,2,1}];
conv7W=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv7_W"}],{3,4,2,1}];

conv8W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv8_1W"}],{3,4,2,1}];
conv8W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv8_2W"}],{3,4,2,1}];

conv9W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv9_1W"}],{3,4,2,1}];
conv9W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv9_2W"}],{3,4,2,1}];

conv10W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv10_1W"}],{3,4,2,1}];
conv10W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv10_2W"}],{3,4,2,1}];

conv11W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv11_1W"}],{3,4,2,1}];
conv11W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv11_2W"}],{3,4,2,1}];


block4ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block4_classes_W"}],{3,4,2,1}];
block4ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block4_classes_B"}];
block4LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block4_loc_W"}],{3,4,2,1}];
block4LocB = Import["CZModels/ssd.hdf",{"Datasets","/block4_loc_B"}];


block7ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block7_classes_W"}],{3,4,2,1}];
block7ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block7_classes_B"}];
block7LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block7_loc_W"}],{3,4,2,1}];
block7LocB = Import["CZModels/ssd.hdf",{"Datasets","/block7_loc_B"}];


block8ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block8_classes_W"}],{3,4,2,1}];
block8ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block8_classes_B"}];
block8LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block8_loc_W"}],{3,4,2,1}];
block8LocB = Import["CZModels/ssd.hdf",{"Datasets","/block8_loc_B"}];


block9ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block9_classes_W"}],{3,4,2,1}];
block9ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block9_classes_B"}];
block9LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block9_loc_W"}],{3,4,2,1}];
block9LocB = Import["CZModels/ssd.hdf",{"Datasets","/block9_loc_B"}];


block10ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block10_classes_W"}],{3,4,2,1}];
block10ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block10_classes_B"}];
block10LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block10_loc_W"}],{3,4,2,1}];
block10LocB = Import["CZModels/ssd.hdf",{"Datasets","/block10_loc_B"}];


block11ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block11_classes_W"}],{3,4,2,1}];
block11ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block11_classes_B"}];
block11LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block11_loc_W"}],{3,4,2,1}];
block11LocB = Import["CZModels/ssd.hdf",{"Datasets","/block11_loc_B"}];


conv1B1=Import["CZModels/ssd.hdf",{"Datasets","/conv1_1B"}];
conv1B2=Import["CZModels/ssd.hdf",{"Datasets","/conv1_2B"}];

conv2B1=Import["CZModels/ssd.hdf",{"Datasets","/conv2_1B"}];
conv2B2=Import["CZModels/ssd.hdf",{"Datasets","/conv2_2B"}];

conv3B1=Import["CZModels/ssd.hdf",{"Datasets","/conv3_1B"}];
conv3B2=Import["CZModels/ssd.hdf",{"Datasets","/conv3_2B"}];
conv3B3=Import["CZModels/ssd.hdf",{"Datasets","/conv3_3B"}];

conv4B1=Import["CZModels/ssd.hdf",{"Datasets","/conv4_1B"}];
conv4B2=Import["CZModels/ssd.hdf",{"Datasets","/conv4_2B"}];
conv4B3=Import["CZModels/ssd.hdf",{"Datasets","/conv4_3B"}];

conv5B1=Import["CZModels/ssd.hdf",{"Datasets","/conv5_1B"}];
conv5B2=Import["CZModels/ssd.hdf",{"Datasets","/conv5_2B"}];
conv5B3=Import["CZModels/ssd.hdf",{"Datasets","/conv5_3B"}];

conv6B=Import["CZModels/ssd.hdf",{"Datasets","/conv6_B"}];

conv7B=Import["CZModels/ssd.hdf",{"Datasets","/conv7_B"}];

conv8B1=Import["CZModels/ssd.hdf",{"Datasets","/conv8_1B"}];
conv8B2=Import["CZModels/ssd.hdf",{"Datasets","/conv8_2B"}];

conv9B1=Import["CZModels/ssd.hdf",{"Datasets","/conv9_1B"}];
conv9B2=Import["CZModels/ssd.hdf",{"Datasets","/conv9_2B"}];

conv10B1=Import["CZModels/ssd.hdf",{"Datasets","/conv10_1B"}];
conv10B2=Import["CZModels/ssd.hdf",{"Datasets","/conv10_2B"}];

conv11B1=Import["CZModels/ssd.hdf",{"Datasets","/conv11_1B"}];
conv11B2=Import["CZModels/ssd.hdf",{"Datasets","/conv11_2B"}];



gamma4=Import["CZModels/ssd.hdf",{"Datasets","/block4_gamma"}];


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
   PaddingLayer[{{0,0},{1,1},{1,1}}],   
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


ngamma4=Table[gamma4[[c]],{c,1,512},{38},{38}];


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
    11->NetPort["ObjMap1"],12->NetPort["Locs1"]}];


MultiBoxNetLayer2=NetGraph[{
   ConvolutionLayer[126,{3,3},"Biases"->block7ClassesB,"Weights"->block7ClassesW,"PaddingSize"->1],
   ReshapeLayer[{6,21,19,19}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[24,{3,3},"Biases"->block7LocB,"Weights"->block7LocW,"PaddingSize"->1]
 },{1->2,2->3,3->4,4->5,5->NetPort["ObjMap2"],6->NetPort["Locs2"]}];


MultiBoxNetLayer3=NetGraph[{
   ConvolutionLayer[126,{3,3},"Biases"->block8ClassesB,"Weights"->block8ClassesW,"PaddingSize"->1],
   ReshapeLayer[{6,21,10,10}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[24,{3,3},"Biases"->block8LocB,"Weights"->block8LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap3"],6->NetPort["Locs3"]}];


MultiBoxNetLayer4=NetGraph[{
   ConvolutionLayer[126,{3,3},"Biases"->block9ClassesB,"Weights"->block9ClassesW,"PaddingSize"->1],
   ReshapeLayer[{6,21,5,5}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],   
   ConvolutionLayer[24,{3,3},"Biases"->block9LocB,"Weights"->block9LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap4"],6->NetPort["Locs4"]}];


MultiBoxNetLayer5=NetGraph[{
   ConvolutionLayer[84,{3,3},"Biases"->block10ClassesB,"Weights"->block10ClassesW,"PaddingSize"->1],
   ReshapeLayer[{4,21,3,3}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[16,{3,3},"Biases"->block10LocB,"Weights"->block10LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap5"],6->NetPort["Locs5"]}];


MultiBoxNetLayer6=NetGraph[{
   ConvolutionLayer[84,{3,3},"Biases"->block11ClassesB,"Weights"->block11ClassesW,"PaddingSize"->1],
   ReshapeLayer[{4,21,1,1}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[],
   ConvolutionLayer[16,{3,3},"Biases"->block11LocB,"Weights"->block11LocW,"PaddingSize"->1]},
   {1->2,2->3,3->4,4->5,5->NetPort["ObjMap6"],6->NetPort["Locs6"]}];


SSDNet=NetGraph[{
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
   4->10,5->11,6->12},
   "Input"->{3,300,300}];