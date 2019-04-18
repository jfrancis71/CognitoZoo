(* ::Package:: *)

(*


branch2cW//Dimensions


branch2aW=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2a_w"}];
branch2aBNS = Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2a_bn_s"}];
branch2aBNScaling = Table[ branch2aBNS[[c]], {c,1,64}, {224}, {288} ];
branch2aBNB=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2a_bn_b"}];
branch2aBNBiases = Table[ branch2aBNB[[c]], {c,1,64}, {224}, {288} ];


branch2bW=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2b_w"}];
branch2bBNS = Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2b_bn_s"}];
branch2bBNScaling = Table[ branch2bBNS[[c]], {c,1,64}, {224}, {288} ];
branch2bBNB=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2b_bn_b"}];
branch2bBNBiases = Table[ branch2bBNB[[c]], {c,1,64}, {224}, {288} ];


branch2cW=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2c_w"}];
branch2cBNS = Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2c_bn_s"}];
branch2cBNScaling = Table[ branch2cBNS[[c]], {c,1,256}, {224}, {288} ];
branch2cBNB=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2c_bn_b"}];
branch2cBNBiases = Table[ branch2cBNB[[c]], {c,1,256}, {224}, {288} ];


branch1W=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch1_w"}];
branch2cBNS = Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2c_bn_s"}];
branch2cBNScaling = Table[ branch2cBNS[[c]], {c,1,256}, {224}, {288} ];
branch2cBNB=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch2c_bn_b"}];
branch2cBNBiases = Table[ branch2cBNB[[c]], {c,1,256}, {224}, {288} ];


branch1BNS = Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch1_bn_s"}];
branch1BNScaling = Table[ branch1BNS[[c]], {c,1,256}, {224}, {288} ];
branch1BNB=Import["/home/julian/tmp/RetinaNet.hdf5",{"Datasets","res2_0_branch1_bn_b"}];
branch1BNBiases = Table[ branch1BNB[[c]], {c,1,256}, {224}, {288} ];
*)


impW[ hdfName_String ] := Import[ "/home/julian/detectron_mount/RetinaNetNew.hdf5", {"Datasets", hdfName} ];


impM[ hdfName_String, channels_, height_, width_ ] := Module[ {dat = Import[ "/home/julian/detectron_mount/RetinaNetNew.hdf5", {"Datasets", hdfName} ] }, Table[ dat[[c]], {c,1,channels}, {height}, {width} ] ];


BNConvolutionLayer[ outputChannels_, kernelSize_, stride_, paddingSize_, weightName_, scalingName_, biasesName_,  res_ ] :=
   NetChain[{ConvolutionLayer[ outputChannels, kernelSize, "Weights"->impW[weightName], "Biases"->ConstantArray[0,outputChannels], "Stride"->stride, "PaddingSize"->paddingSize ],
   ConstantTimesLayer[ "Scaling"->impM[scalingName,outputChannels,res[[1]],res[[2]]] ], ConstantPlusLayer[ "Biases"->impM[biasesName,outputChannels,res[[1]],res[[2]]] ],
   Ramp}
];


RetinaNet = NetGraph[{
   "conv1"->BNConvolutionLayer[ 64, {7,7}, 2, 3, "conv1_w", "res_conv1_bn_s","res_conv1_bn_b", {448, 576} ],
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],
   "res2_0_branch2a"->BNConvolutionLayer[ 64, {1,1}, 1, 0, "res2_0_branch2a_w", "res2_0_branch2a_bn_s","res2_0_branch2a_bn_b", {224, 288} ],
   "res2_0_branch2b"->BNConvolutionLayer[ 64, {3,3}, 1, 1, "res2_0_branch2b_w", "res2_0_branch2b_bn_s","res2_0_branch2b_bn_b", {224, 288} ],
   "res2_0_branch2c_bn"->BNConvolutionLayer[ 256, {1,1}, 1, 0, "res2_0_branch2c_w", "res2_0_branch2c_bn_s","res2_0_branch2c_bn_b", {224, 288} ]
(*
   "conv1"\[Rule]{ConvolutionLayer[ 64, {7,7}, "Weights"->impW["conv1_w"], "Biases"->ConstantArray[0,64], "Stride"->2, "PaddingSize"->3 ],
   ConstantTimesLayer[ "Scaling"->impM["res_conv1_bn_s",64,448,576] ], ConstantPlusLayer[ "Biases"->impM["res_conv1_bn_b",64,448,576] ],
   Ramp},
      BNConvolutionLayer[ 64, {7,7}, 2, 3, "conv1_w", "res_conv1_bn_s","res_conv1_bn_b", {448, 576} ] 
   ,
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],

   "res2_0_branch2a"->{ConvolutionLayer[ 64, {1,1}, "Weights"->branch2aW, "Biases"->ConstantArray[0,64] ],
   ConstantTimesLayer[ "Scaling"->branch2aBNScaling ], ConstantPlusLayer[ "Biases"->branch2aBNBiases ],
   Ramp},
   "res2_0_branch2b"->{ConvolutionLayer[ 64, {3,3}, "Weights"->branch2bW, "Biases"->ConstantArray[0,64], "PaddingSize"->1 ],
   ConstantTimesLayer[ "Scaling"->branch2bBNScaling ], ConstantPlusLayer[ "Biases"->branch2bBNBiases ],
   Ramp},
   "res2_0_branch2c_bn"->{ConvolutionLayer[ 256, {1,1}, "Weights"->branch2cW, "Biases"->ConstantArray[0,256] ],
      ConstantTimesLayer[ "Scaling"->branch2cBNScaling ], ConstantPlusLayer[ "Biases"->branch2cBNBiases ]},
   "res2_0_branch1_bn"->{ConvolutionLayer[ 256, {1,1}, "Weights"->branch1W, "Biases"->ConstantArray[0,256] ], ConstantTimesLayer[ "Scaling"->branch1BNScaling ], ConstantPlusLayer[ "Biases"->branch1BNBiases ]},
   "res2_0_branch2c_bn_sum"->TotalLayer[],
   "res2_0_branch2c_bn_sum_relu"->Ramp,
   
   "res2_1_branch2a"->{ConvolutionLayer[ 64, {1,1}, "Weights"->impW["res2_1_branch2a_w"], "Biases"->ConstantArray[0,64] ],
   ConstantTimesLayer[ "Scaling"->impM["res2_1_branch2a_bn_s",64,224,288] ], ConstantPlusLayer[ "Biases"->impM["res2_1_branch2a_bn_b",64,224,288] ],
   Ramp},
   "res2_1_branch2b"->{ConvolutionLayer[ 64, {3,3}, "Weights"->impW["res2_1_branch2b_w"], "Biases"->ConstantArray[0,64], "PaddingSize"->1 ],
   ConstantTimesLayer[ "Scaling"->impM["res2_1_branch2b_bn_s",64,224,288] ], ConstantPlusLayer[ "Biases"->impM["res2_1_branch2b_bn_b",64,224,288] ],
   Ramp},
   "res2_1_branch2c_bn"->{ConvolutionLayer[ 256, {1,1}, "Weights"->impW["res2_1_branch2c_w"], "Biases"->ConstantArray[0,256] ],
      ConstantTimesLayer[ "Scaling"->impM["res2_1_branch2c_bn_s",256,224,288] ], ConstantPlusLayer[ "Biases"->impM["res2_1_branch2c_bn_b",256,224,288] ]},
   "res2_1_branch2c_bn_sum"->TotalLayer[],
   "res2_1_branch2c_bn_sum_relu"->Ramp,
   
   "res2_2_branch2a"->{ConvolutionLayer[ 64, {1,1}, "Weights"->impW["res2_2_branch2a_w"], "Biases"->ConstantArray[0,64] ],
   ConstantTimesLayer[ "Scaling"->impM["res2_2_branch2a_bn_s",64,224,288] ], ConstantPlusLayer[ "Biases"->impM["res2_2_branch2a_bn_b",64,224,288] ],
   Ramp},
   "res2_2_branch2b"->{ConvolutionLayer[ 64, {3,3}, "Weights"->impW["res2_2_branch2b_w"], "Biases"->ConstantArray[0,64], "PaddingSize"->1 ],
   ConstantTimesLayer[ "Scaling"->impM["res2_2_branch2b_bn_s",64,224,288] ], ConstantPlusLayer[ "Biases"->impM["res2_2_branch2b_bn_b",64,224,288] ],
   Ramp},
   "res2_2_branch2c_bn"->{ConvolutionLayer[ 256, {1,1}, "Weights"->impW["res2_2_branch2c_w"], "Biases"->ConstantArray[0,256] ],
      ConstantTimesLayer[ "Scaling"->impM["res2_2_branch2c_bn_s",256,224,288] ], ConstantPlusLayer[ "Biases"->impM["res2_2_branch2c_bn_b",256,224,288] ]},
   "res2_2_sum"->TotalLayer[],
   "res2_2_sum_relu"->Ramp,

   "res3_0_branch2a"->{ConvolutionLayer[ 128, {1,1}, "Weights"->impW["res3_0_branch2a_w"], "Biases"->ConstantArray[0,128], "Stride"->2 ],
   ConstantTimesLayer[ "Scaling"->impM["res3_0_branch2a_bn_s",128,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_0_branch2a_bn_b",128,112,144] ],
   Ramp},
   "res3_0_branch2b"->{ConvolutionLayer[ 128, {3,3}, "Weights"->impW["res3_0_branch2b_w"], "Biases"->ConstantArray[0,128], "PaddingSize"->1 ],
   ConstantTimesLayer[ "Scaling"->impM["res3_0_branch2b_bn_s",128,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_0_branch2b_bn_b",128,112,144] ],
   Ramp},
   "res3_0_branch2c_bn"->{ConvolutionLayer[ 512, {1,1}, "Weights"->impW["res3_0_branch2c_w"], "Biases"->ConstantArray[0,512] ],
      ConstantTimesLayer[ "Scaling"->impM["res3_0_branch2c_bn_s",512,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_0_branch2c_bn_b",512,112,144] ]},
      
   "res3_0_branch1"->{ConvolutionLayer[ 512, {1,1}, "Weights"->impW["res3_0_branch1_w"], "Biases"->ConstantArray[0,512], "Stride"->2 ],
   ConstantTimesLayer[ "Scaling"->impM["res3_0_branch1_bn_s",512,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_0_branch1_bn_b",512,112,144] ]},
   
   "res3_0_sum"->TotalLayer[],
   "res3_0_sum_relu"->Ramp,
   
   "res3_1_branch2a"->{ConvolutionLayer[ 128, {1,1}, "Weights"->impW["res3_1_branch2a_w"], "Biases"->ConstantArray[0,128] ],
   ConstantTimesLayer[ "Scaling"->impM["res3_1_branch2a_bn_s",128,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_1_branch2a_bn_b",128,112,144] ],
   Ramp},
   "res3_1_branch2b"->{ConvolutionLayer[ 128, {3,3}, "Weights"->impW["res3_1_branch2b_w"], "Biases"->ConstantArray[0,128], "PaddingSize"->1 ],
   ConstantTimesLayer[ "Scaling"->impM["res3_1_branch2b_bn_s",128,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_1_branch2b_bn_b",128,112,144] ],
   Ramp},
   "res3_1_branch2c_bn"->{ConvolutionLayer[ 512, {1,1}, "Weights"->impW["res3_1_branch2c_w"], "Biases"->ConstantArray[0,512] ],
      ConstantTimesLayer[ "Scaling"->impM["res3_1_branch2c_bn_s",512,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_1_branch2c_bn_b",512,112,144] ]},
      
   "res3_1_sum"->TotalLayer[],
   "res3_1_sum_relu"->Ramp,
   
   "res3_2_branch2a"->{ConvolutionLayer[ 128, {1,1}, "Weights"->impW["res3_2_branch2a_w"], "Biases"->ConstantArray[0,128] ],
   ConstantTimesLayer[ "Scaling"->impM["res3_2_branch2a_bn_s",128,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_2_branch2a_bn_b",128,112,144] ],
   Ramp},
   "res3_2_branch2b"->{ConvolutionLayer[ 128, {3,3}, "Weights"->impW["res3_2_branch2b_w"], "Biases"->ConstantArray[0,128], "PaddingSize"->1 ],
   ConstantTimesLayer[ "Scaling"->impM["res3_2_branch2b_bn_s",128,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_2_branch2b_bn_b",128,112,144] ],
   Ramp},
   "res3_2_branch2c_bn"->{ConvolutionLayer[ 512, {1,1}, "Weights"->impW["res3_2_branch2c_w"], "Biases"->ConstantArray[0,512] ],
      ConstantTimesLayer[ "Scaling"->impM["res3_2_branch2c_bn_s",512,112,144] ], ConstantPlusLayer[ "Biases"->impM["res3_2_branch2c_bn_b",512,112,144] ]},
   
   "res3_2_sum"->TotalLayer[],
   "res3_2_sum_relu"->Ramp
*)

},{
   "conv1"->"pool1"->"res2_0_branch2a"->"res2_0_branch2b"->"res2_0_branch2c_bn"(*,
   "pool1"->"res2_0_branch1_bn",
   {"res2_0_branch2c_bn","res2_0_branch1_bn"}->"res2_0_branch2c_bn_sum"->"res2_0_branch2c_bn_sum_relu"
   ->"res2_1_branch2a"->"res2_1_branch2b"->"res2_1_branch2c_bn",
   {"res2_0_branch2c_bn_sum_relu","res2_1_branch2c_bn"}->"res2_1_branch2c_bn_sum"->"res2_1_branch2c_bn_sum_relu"->"res2_2_branch2a"->"res2_2_branch2b"->"res2_2_branch2c_bn",
   {"res2_1_branch2c_bn_sum_relu","res2_2_branch2c_bn"}->"res2_2_sum"->"res2_2_sum_relu"->"res3_0_branch2a"->"res3_0_branch2b"->"res3_0_branch2c_bn"->"res3_0_sum",
   "res2_2_sum_relu"->"res3_0_branch1"->"res3_0_sum"->"res3_0_sum_relu"->"res3_1_branch2a"->"res3_1_branch2b"->"res3_1_branch2c_bn",
   {"res3_0_sum_relu","res3_1_branch2c_bn"}->"res3_1_sum"->"res3_1_sum_relu"->"res3_2_branch2a"->"res3_2_branch2b"->"res3_2_branch2c_bn",
   {"res3_1_sum_relu","res3_2_branch2c_bn"}->"res3_2_sum"->"res3_2_sum_relu"*)
}];


(* Verification *)


dat=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","data"}];


ref=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","res2_0_branch2b"}];ref//Dimensions


my = Normal@ConvolutionLayer[ 512, {1,1}, "Weights"->impW["res3_1_branch2c_w"], "Biases"->ConstantArray[0,512] ]@Normal@NetTake[RetinaNet,{"conv1","res3_1_branch2b"}][ dat ];my//Dimensions


my = Normal@NetTake[RetinaNet,{"conv1","res3_2_sum_relu"}][ dat ];my//Dimensions


my = Normal@NetTake[RetinaNet,{"conv1","res2_0_branch2b"}][ dat ];my//Dimensions


diff = Abs[ref - my]; diff//Dimensions


Max@diff


Position[diff,Max[diff]]


my[[1,77,39,126]]


ref[[1,77,39,126]]
