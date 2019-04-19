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


RetinaNetBlock[ rootName_, outputChannels_, stride_, dims_ ] := NetGraph[{
   "branch2"->RetinaNetMiniBlock[ rootName<>"_branch2", outputChannels, stride, dims ],
   "sum"->TotalLayer[],
   "relu"->Ramp},
   {{NetPort["Input"],"branch2"}->"sum"->"relu"}]


RetinaNetBlock[ rootName_, outputChannels_, stride_, dims_, branch1_ ] := NetGraph[{
   "branch1"->branch1,
   "branch2"->RetinaNetMiniBlock[ rootName<>"_branch2", outputChannels, stride, dims ],
   "sum"->TotalLayer[],
   "relu"->Ramp},
   {{"branch1","branch2"}->"sum"->"relu"}]


net1 = NetGraph[{
   "conv1"->{BNConvolutionLayer[ 64, {7,7}, 2, 3, {448, 576}, "conv1_w", "res_conv1_bn_s","res_conv1_bn_b" ],Ramp},
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],
   
   "res2_0"->RetinaNetBlock[ "res2_0", 256, 1, {224,288}, BNConvolutionLayer[ 256, {1,1}, 1, 0, {224, 288}, "res2_0_branch1" ] ],
   "res2_1"->RetinaNetBlock[ "res2_1", 256, 1, {224,288} ],
   "res2_2"->RetinaNetBlock[ "res2_2", 256, 1, {224,288} ],

   "res3_0"->RetinaNetBlock[ "res3_0", 512, 2, {112,144}, BNConvolutionLayer[ 512, {1,1}, 2, 0, {112, 144}, "res3_0_branch1" ] ],
   "res3_1"->RetinaNetBlock[ "res3_1", 512, 1, {112,144} ],
   "res3_2"->RetinaNetBlock[ "res3_2", 512, 1, {112,144} ],
   "res3_3"->RetinaNetBlock[ "res3_3", 512, 1, {112,144} ],

   "res4_0"->RetinaNetBlock[ "res4_0", 1024, 2, {56,72}, BNConvolutionLayer[ 1024, {1,1}, 2, 0, {56,72}, "res4_0_branch1" ] ],
   "res4_1"->RetinaNetBlock[ "res4_1", 1024, 1, {56,72} ],
   "res4_2"->RetinaNetBlock[ "res4_2", 1024, 1, {56,72} ],
   "res4_3"->RetinaNetBlock[ "res4_3", 1024, 1, {56,72} ],
   "res4_4"->RetinaNetBlock[ "res4_4", 1024, 1, {56,72} ],
   "res4_5"->RetinaNetBlock[ "res4_5", 1024, 1, {56,72} ],
   "res4_6"->RetinaNetBlock[ "res4_6", 1024, 1, {56,72} ],
   "res4_7"->RetinaNetBlock[ "res4_7", 1024, 1, {56,72} ],
   "res4_8"->RetinaNetBlock[ "res4_8", 1024, 1, {56,72} ],
   "res4_9"->RetinaNetBlock[ "res4_9", 1024, 1, {56,72} ],
   "res4_10"->RetinaNetBlock[ "res4_10", 1024, 1, {56,72} ],
   "res4_11"->RetinaNetBlock[ "res4_11", 1024, 1, {56,72} ],
   "res4_12"->RetinaNetBlock[ "res4_12", 1024, 1, {56,72} ],
   "res4_13"->RetinaNetBlock[ "res4_13", 1024, 1, {56,72} ],
   "res4_14"->RetinaNetBlock[ "res4_14", 1024, 1, {56,72} ],
   "res4_15"->RetinaNetBlock[ "res4_15", 1024, 1, {56,72} ],
   "res4_16"->RetinaNetBlock[ "res4_16", 1024, 1, {56,72} ],
   "res4_17"->RetinaNetBlock[ "res4_17", 1024, 1, {56,72} ],
   "res4_18"->RetinaNetBlock[ "res4_18", 1024, 1, {56,72} ],
   "res4_19"->RetinaNetBlock[ "res4_19", 1024, 1, {56,72} ],
   "res4_20"->RetinaNetBlock[ "res4_20", 1024, 1, {56,72} ],
   "res4_21"->RetinaNetBlock[ "res4_21", 1024, 1, {56,72} ],
   "res4_22"->RetinaNetBlock[ "res4_22", 1024, 1, {56,72} ]
},
{
   "conv1"->"pool1"->
   "res2_0"->"res2_1"->"res2_2"->
   "res3_0"->"res3_1"->"res3_2"->"res3_3"->
   "res4_0"->"res4_1"->"res4_2"->"res4_3"->
   "res4_4"->"res4_5"->"res4_6"->"res4_7"->
   "res4_8"->"res4_9"->"res4_10"->"res4_11"->
   "res4_12"->"res4_13"->"res4_14"->"res4_15"->
   "res4_16"->"res4_17"->"res4_18"->"res4_19"->
   "res4_20"->"res4_21"->"res4_22"
}];


net2 = NetGraph[{
   "res5_0"->RetinaNetBlock[ "res5_0", 2048, 2, {28,36}, BNConvolutionLayer[ 2048, {1,1}, 2, 0, {28,36}, "res5_0_branch1" ] ],
   "res5_1"->RetinaNetBlock[ "res5_1", 2048, 1, {28,36} ],
   "res5_2"->RetinaNetBlock[ "res5_2", 2048, 1, {28,36} ]
},
{
   "res5_0"->"res5_1"->"res5_2"
}];


dat=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","data"}];


ref=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","res5_2_sum"}];ref//Dimensions


t1=Normal@net1@dat;t1//Dimensions


t2=Normal@net2@t1;t2//Dimensions


diff = Abs[ref - t2]; diff//Dimensions


Max@diff


Position[diff,Max[diff]]


t2[[1,1902,28,9]]


ref[[1,1902,28,9]]
