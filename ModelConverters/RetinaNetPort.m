(* ::Package:: *)

impW[ hdfName_String ] := Import[ "/home/julian/detectron_mount/RetinaNetNew.hdf5", {"Datasets", hdfName} ];


impM[ hdfName_String, channels_, height_, width_ ] := Module[ {dat = Import[ "/home/julian/detectron_mount/RetinaNetNew.hdf5", {"Datasets", hdfName} ] }, Table[ dat[[c]], {c,1,channels}, {height}, {width} ] ];


BNConvolutionLayer[ outputChannels_Integer, kernelSize_List, stride_Integer, paddingSize_Integer, res_List, weightName_String, scalingName_String, biasesName_String ] :=
   NetChain[{ConvolutionLayer[ outputChannels, kernelSize, "Weights"->impW[weightName], "Biases"->ConstantArray[0,outputChannels], "Stride"->stride, "PaddingSize"->paddingSize ],
   ConstantTimesLayer[ "Scaling"->impM[scalingName,outputChannels,res[[1]],res[[2]]] ], ConstantPlusLayer[ "Biases"->impM[biasesName,outputChannels,res[[1]],res[[2]]] ]}
];


BNConvolutionLayer[ outputChannels_Integer, kernelSize_List, stride_Integer, paddingSize_Integer, res_List, rootName_String ] :=
   BNConvolutionLayer[ outputChannels, kernelSize, stride, paddingSize, res, rootName<>"_w", rootName<>"_bn_s", rootName<>"_bn_b" ];


RetinaNetMiniBlock[ rootName_, outputChannels_, stride_, dims_ ] := NetGraph[ {
   rootName<>"a"->{BNConvolutionLayer[ outputChannels/4, {1,1}, stride, 0, dims, rootName<>"a" ], Ramp},
   rootName<>"b"->{BNConvolutionLayer[ outputChannels/4, {3,3}, 1, 1, dims, rootName<>"b" ], Ramp},
   rootName<>"c_bn"->BNConvolutionLayer[ outputChannels, {1,1}, 1, 0, dims, rootName<>"c" ]
},{
   rootName<>"a"->rootName<>"b"->rootName<>"c_bn"}]


RetinaNet = NetGraph[{
   "conv1"->{BNConvolutionLayer[ 64, {7,7}, 2, 3, {448, 576}, "conv1_w", "res_conv1_bn_s","res_conv1_bn_b" ],Ramp},
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],
   
   "block1"->RetinaNetMiniBlock[ "res2_0_branch2", 256, 1, {224, 288} ],
   "res2_0_branch1"->BNConvolutionLayer[ 256, {1,1}, 1, 0, {224, 288}, "res2_0_branch1" ],
   "res2_0_branch2c_bn_sum"->TotalLayer[],
   "res2_0_branch2c_bn_sum_relu"->Ramp,
   
   "block2"->RetinaNetMiniBlock[ "res2_1_branch2", 256, 1, {224, 288} ],
   "res2_1_branch2c_bn_sum"->TotalLayer[],
   "res2_1_branch2c_bn_sum_relu"->Ramp,

   "block3"->RetinaNetMiniBlock[ "res2_2_branch2", 256, 1, {224, 288} ],
   "res2_2_sum"->TotalLayer[],
   "res2_2_sum_relu"->Ramp,
   
   "block4"->RetinaNetMiniBlock[ "res3_0_branch2", 512, 2, {112, 144} ],
   "res3_0_branch1"->BNConvolutionLayer[ 512, {1,1}, 2, 0, {112, 144}, "res3_0_branch1" ],
   "res3_0_branch2c_bn_sum"->TotalLayer[],
   "res3_0_branch2c_bn_sum_relu"->Ramp,
   
   "block5"->RetinaNetMiniBlock[ "res3_1_branch2", 512, 1, {112, 144} ],
   "res3_1_branch2c_bn_sum"->TotalLayer[],
   "res3_1_branch2c_bn_sum_relu"->Ramp,
   
   "block6"->RetinaNetMiniBlock[ "res3_2_branch2", 512, 1, {112, 144} ],
   "res3_2_branch2c_bn_sum"->TotalLayer[],
   "res3_2_branch2c_bn_sum_relu"->Ramp,

   "block7"->RetinaNetMiniBlock[ "res3_3_branch2", 512, 1, {112, 144} ],
   "res3_3_sum"->TotalLayer[],
   "res3_3_sum_relu"->Ramp
},{
   "conv1"->"pool1"->{"block1","res2_0_branch1"},
   {"block1","res2_0_branch1"}->"res2_0_branch2c_bn_sum"->"res2_0_branch2c_bn_sum_relu",
   
   "res2_0_branch2c_bn_sum_relu"->{"block2"},   
   {"block2","res2_0_branch2c_bn_sum_relu"}->"res2_1_branch2c_bn_sum"->"res2_1_branch2c_bn_sum_relu",
   
   "res2_1_branch2c_bn_sum_relu"->{"block3"},   
   {"block3","res2_1_branch2c_bn_sum_relu"}->"res2_2_sum"->"res2_2_sum_relu",
   
   "res2_2_sum_relu"->{"block4","res3_0_branch1"},
   {"block4","res3_0_branch1"}->"res3_0_branch2c_bn_sum"->"res3_0_branch2c_bn_sum_relu",
   
   "res3_0_branch2c_bn_sum_relu"->{"block5"},   
   {"block5","res3_0_branch2c_bn_sum_relu"}->"res3_1_branch2c_bn_sum"->"res3_1_branch2c_bn_sum_relu",

   "res3_1_branch2c_bn_sum_relu"->{"block6"},   
   {"block6","res3_1_branch2c_bn_sum_relu"}->"res3_2_branch2c_bn_sum"->"res3_2_branch2c_bn_sum_relu",
   
   "res3_2_branch2c_bn_sum_relu"->{"block7"},   
   {"block7","res3_2_branch2c_bn_sum_relu"}->"res3_3_sum"->"res3_3_sum_relu"
}];


(* Verification *)


dat=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","data"}];


ref=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","res3_3_sum"}];ref//Dimensions


my = Normal@ConvolutionLayer[ 512, {1,1}, "Weights"->impW["res3_1_branch2c_w"], "Biases"->ConstantArray[0,512] ]@Normal@NetTake[RetinaNet,{"conv1","res3_1_branch2b"}][ dat ];my//Dimensions


my = Normal@NetTake[RetinaNet,{"conv1","res3_3_sum_relu"}][ dat ];my//Dimensions


my = Normal@NetTake[NetFlatten@RetinaNet,{"conv1","res3_2_branch2c_bn_sum_relu"}][ dat ];my//Dimensions


diff = Abs[ref - my]; diff//Dimensions


Max@diff


Position[diff,Max[diff]]


my[[1,19,33,265]]


ref[[1,19,33,265]]
