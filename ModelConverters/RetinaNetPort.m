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


RetinaNet = NetGraph[{
   "conv1"->{BNConvolutionLayer[ 64, {7,7}, 2, 3, {448, 576}, "conv1_w", "res_conv1_bn_s","res_conv1_bn_b" ],Ramp},
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],
   
   "res2_0_branch2c_bn_sum_relu"->RetinaNetBlock[ "res2_0", 256, 1, {224,288}, BNConvolutionLayer[ 256, {1,1}, 1, 0, {224, 288}, "res2_0_branch1" ] ],
   "res2_1_branch2c_bn_sum_relu"->RetinaNetBlock[ "res2_1", 256, 1, {224,288} ],
   "res2_2_sum_relu"->RetinaNetBlock[ "res2_2", 256, 1, {224,288} ],

   "res3_0_branch2c_bn_sum_relu"->RetinaNetBlock[ "res3_0", 512, 2, {112,144}, BNConvolutionLayer[ 512, {1,1}, 2, 0, {112, 144}, "res3_0_branch1" ] ],
   "res3_1_branch2c_bn_sum_relu"->RetinaNetBlock[ "res3_1", 512, 1, {112,144} ],
   "res3_2_branch2c_bn_sum_relu"->RetinaNetBlock[ "res3_2", 512, 1, {112,144} ],
   "res3_3_sum_relu"->RetinaNetBlock[ "res3_3", 512, 1, {112,144} ]

},{
   "conv1"->"pool1"->
   "res2_0_branch2c_bn_sum_relu"->"res2_1_branch2c_bn_sum_relu"->"res2_2_sum_relu"->
   "res3_0_branch2c_bn_sum_relu"->"res3_1_branch2c_bn_sum_relu"->"res3_2_branch2c_bn_sum_relu"->"res3_3_sum_relu"
}];


(* Verification *)


dat=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","data"}];


ref=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","res3_3_sum"}];ref//Dimensions


my = Normal@NetTake[RetinaNet,{"conv1","res3_3_sum_relu"}][ dat ];my//Dimensions


diff = Abs[ref - my]; diff//Dimensions


Max@diff


Position[diff,Max[diff]]


my[[1,19,33,265]]


ref[[1,19,33,265]]
