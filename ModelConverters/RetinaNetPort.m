(* ::Package:: *)

(* Warning not complete, in progress, being checked in to track changes *)


impW[ hdfName_String ] := If[hdfName=="conv1_w",
   Reverse/@Import[ "/home/julian/detectron_mount/RetinaNetNew.hdf5", {"Datasets", hdfName} ],
   Import[ "/home/julian/detectron_mount/RetinaNetNew.hdf5", {"Datasets", hdfName} ]
   ];


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
   "mult"->ElementwiseLayer[#*255.&],
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
   "mult"->"conv1"->"pool1"->
   "res2_0"->"res2_1"->"res2_2"->
   "res3_0"->"res3_1"->"res3_2"->"res3_3"->
   "res4_0"->"res4_1"->"res4_2"->"res4_3"->
   "res4_4"->"res4_5"->"res4_6"->"res4_7"->
   "res4_8"->"res4_9"->"res4_10"->"res4_11"->
   "res4_12"->"res4_13"->"res4_14"->"res4_15"->
   "res4_16"->"res4_17"->"res4_18"->"res4_19"->
   "res4_20"->"res4_21"->"res4_22"->NetPort["res4_22_sum"],
   "res3_3"->NetPort["res3_3_sum"]
}];


net2 = NetGraph[{
   "res5_0"->RetinaNetBlock[ "res5_0", 2048, 2, {28,36}, BNConvolutionLayer[ 2048, {1,1}, 2, 0, {28,36}, "res5_0_branch1" ] ],
   "res5_1"->RetinaNetBlock[ "res5_1", 2048, 1, {28,36} ],
   "res5_2"->RetinaNetBlock[ "res5_2", 2048, 1, {28,36} ]
},
{
   "res5_0"->"res5_1"->"res5_2"
}];


net3 = (* input fpn_inner_res3_3_sum *)
   NetGraph[{
      "fpn_inner_res3_3_sum_lateral"->ConvolutionLayer[ 256, {1,1}, "Weights"->impW["fpn_inner_res3_3_sum_lateral_w"], "Biases"->impW["fpn_inner_res3_3_sum_lateral_b"] ],
      "fpn_inner_res4_22_sum_lateral"->ConvolutionLayer[ 256, {1,1}, "Weights"->impW["fpn_inner_res4_22_sum_lateral_w"], "Biases"->impW["fpn_inner_res4_22_sum_lateral_b"] ],
      "fpn_inner_res5_2_sum"->ConvolutionLayer[ 256, {1,1}, "Weights"->impW["fpn_inner_res5_2_sum_w"], "Biases"->impW["fpn_inner_res5_2_sum_b"] ],
      "fpn_inner_res4_22_sum"->TotalLayer[],
      "fpn_inner_res4_22_sum_topdown"->ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"],
      "fpn_inner_res3_3_sum_topdown"->ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"],
      "fpn_inner_res3_3_sum"->TotalLayer[],
      "fpn_res3_3_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["fpn_res3_3_sum_w"], "Biases"->impW["fpn_res3_3_sum_b"]  ],
      "fpn_res4_22_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["fpn_res4_22_sum_w"], "Biases"->impW["fpn_res4_22_sum_b"]  ],
      "fpn_res5_2_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["fpn_res5_2_sum_w"], "Biases"->impW["fpn_res5_2_sum_b"]  ],
      "fpn_6"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["fpn_6_w"], "Biases"->impW["fpn_6_b"]  ],
      "fpn_7"->{Ramp,ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["fpn_7_w"], "Biases"->impW["fpn_7_b"]  ]},
      
      "retnet_cls_conv_n0_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n0_fpn3_w"], "Biases"->impW["retnet_cls_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n1_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n1_fpn3_w"], "Biases"->impW["retnet_cls_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n2_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n2_fpn3_w"], "Biases"->impW["retnet_cls_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n3_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n3_fpn3_w"], "Biases"->impW["retnet_cls_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_cls_pred_fpn3"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_pred_fpn3_w"], "Biases"->impW["retnet_cls_pred_fpn3_b"]  ],
      "retnet_cls_pred_prob3"->LogisticSigmoid,
      "retnet_bbox_conv_n0_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n0_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n1_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n1_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n2_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n2_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n3_fpn3"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n3_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_bbox_pred_fpn3"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_pred_fpn3_w"], "Biases"->impW["retnet_bbox_pred_fpn3_b"]  ],
            
      "retnet_cls_conv_n0_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n0_fpn3_w"], "Biases"->impW["retnet_cls_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n1_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n1_fpn3_w"], "Biases"->impW["retnet_cls_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n2_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n2_fpn3_w"], "Biases"->impW["retnet_cls_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n3_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n3_fpn3_w"], "Biases"->impW["retnet_cls_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_cls_pred_fpn4"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_pred_fpn3_w"], "Biases"->impW["retnet_cls_pred_fpn3_b"]  ],
      "retnet_cls_pred_prob4"->LogisticSigmoid,
      "retnet_bbox_conv_n0_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n0_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n1_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n1_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n2_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n2_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n3_fpn4"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n3_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_bbox_pred_fpn4"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_pred_fpn3_w"], "Biases"->impW["retnet_bbox_pred_fpn3_b"]  ],
      
      
      "retnet_cls_conv_n0_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n0_fpn3_w"], "Biases"->impW["retnet_cls_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n1_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n1_fpn3_w"], "Biases"->impW["retnet_cls_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n2_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n2_fpn3_w"], "Biases"->impW["retnet_cls_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n3_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n3_fpn3_w"], "Biases"->impW["retnet_cls_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_cls_pred_fpn5"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_pred_fpn3_w"], "Biases"->impW["retnet_cls_pred_fpn3_b"]  ],
      "retnet_cls_pred_prob5"->LogisticSigmoid,
      "retnet_bbox_conv_n0_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n0_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n1_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n1_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n2_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n2_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n3_fpn5"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n3_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_bbox_pred_fpn5"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_pred_fpn3_w"], "Biases"->impW["retnet_bbox_pred_fpn3_b"]  ],

            
      "retnet_cls_conv_n0_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n0_fpn3_w"], "Biases"->impW["retnet_cls_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n1_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n1_fpn3_w"], "Biases"->impW["retnet_cls_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n2_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n2_fpn3_w"], "Biases"->impW["retnet_cls_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n3_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n3_fpn3_w"], "Biases"->impW["retnet_cls_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_cls_pred_fpn6"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_pred_fpn3_w"], "Biases"->impW["retnet_cls_pred_fpn3_b"]  ],
      "retnet_cls_pred_prob6"->LogisticSigmoid,
      "retnet_bbox_conv_n0_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n0_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n1_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n1_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n2_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n2_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n3_fpn6"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n3_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_bbox_pred_fpn6"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_pred_fpn3_w"], "Biases"->impW["retnet_bbox_pred_fpn3_b"]  ],
      
      
      "retnet_cls_conv_n0_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n0_fpn3_w"], "Biases"->impW["retnet_cls_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n1_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n1_fpn3_w"], "Biases"->impW["retnet_cls_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n2_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n2_fpn3_w"], "Biases"->impW["retnet_cls_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n3_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n3_fpn3_w"], "Biases"->impW["retnet_cls_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_cls_pred_fpn7"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_pred_fpn3_w"], "Biases"->impW["retnet_cls_pred_fpn3_b"]  ],
      "retnet_cls_pred_prob7"->LogisticSigmoid,
      "retnet_bbox_conv_n0_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n0_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n1_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n1_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n2_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n2_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n3_fpn7"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n3_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_bbox_pred_fpn7"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_pred_fpn3_w"], "Biases"->impW["retnet_bbox_pred_fpn3_b"]  ]
      
      },{
      
      NetPort["res5_2_sum"]->"fpn_inner_res5_2_sum"->"fpn_inner_res4_22_sum_topdown",
      NetPort["res3_3_sum"]->"fpn_inner_res3_3_sum_lateral",
      NetPort["res4_22_sum"]->"fpn_inner_res4_22_sum_lateral",
      {"fpn_inner_res4_22_sum_topdown","fpn_inner_res4_22_sum_lateral"}->"fpn_inner_res4_22_sum"->"fpn_inner_res3_3_sum_topdown",

      {"fpn_inner_res3_3_sum_topdown","fpn_inner_res3_3_sum_lateral"}->"fpn_inner_res3_3_sum"->"fpn_res3_3_sum"->
      "retnet_cls_conv_n0_fpn3"->"retnet_cls_conv_n1_fpn3"->"retnet_cls_conv_n2_fpn3"->"retnet_cls_conv_n3_fpn3"->
      "retnet_cls_pred_fpn3"->"retnet_cls_pred_prob3"->NetPort["ClassProb3"],
      "fpn_res3_3_sum"->"retnet_bbox_conv_n0_fpn3"->"retnet_bbox_conv_n1_fpn3"->"retnet_bbox_conv_n2_fpn3"->"retnet_bbox_conv_n3_fpn3"->
      "retnet_bbox_pred_fpn3"->NetPort["Boxes3"],
      "fpn_inner_res4_22_sum"->"fpn_res4_22_sum"->"retnet_cls_conv_n0_fpn4"->"retnet_cls_conv_n1_fpn4"->"retnet_cls_conv_n2_fpn4"->"retnet_cls_conv_n3_fpn4"->
      "retnet_cls_pred_fpn4"->"retnet_cls_pred_prob4"->NetPort["ClassProb4"],
      "fpn_res4_22_sum"->"retnet_bbox_conv_n0_fpn4"->"retnet_bbox_conv_n1_fpn4"->"retnet_bbox_conv_n2_fpn4"->"retnet_bbox_conv_n3_fpn4"->
      "retnet_bbox_pred_fpn4"->NetPort["Boxes4"],
      "fpn_inner_res5_2_sum"->"fpn_res5_2_sum"->"retnet_cls_conv_n0_fpn5"->"retnet_cls_conv_n1_fpn5"->"retnet_cls_conv_n2_fpn5"->"retnet_cls_conv_n3_fpn5"->
      "retnet_cls_pred_fpn5"->"retnet_cls_pred_prob5"->NetPort["ClassProb5"],
      "fpn_res5_2_sum"->"retnet_bbox_conv_n0_fpn5"->"retnet_bbox_conv_n1_fpn5"->"retnet_bbox_conv_n2_fpn5"->"retnet_bbox_conv_n3_fpn5"->
      "retnet_bbox_pred_fpn5"->NetPort["Boxes5"],
      NetPort["res5_2_sum"]->"fpn_6"->"retnet_cls_conv_n0_fpn6"->"retnet_cls_conv_n1_fpn6"->"retnet_cls_conv_n2_fpn6"->"retnet_cls_conv_n3_fpn6"->
      "retnet_cls_pred_fpn6"->"retnet_cls_pred_prob6"->NetPort["ClassProb6"],
      "fpn_6"->"retnet_bbox_conv_n0_fpn6"->"retnet_bbox_conv_n1_fpn6"->"retnet_bbox_conv_n2_fpn6"->"retnet_bbox_conv_n3_fpn6"->
      "retnet_bbox_pred_fpn6"->NetPort["Boxes6"],
      "fpn_6"->"fpn_7"->"retnet_cls_conv_n0_fpn7"->"retnet_cls_conv_n1_fpn7"->"retnet_cls_conv_n2_fpn7"->"retnet_cls_conv_n3_fpn7"->
      "retnet_cls_pred_fpn7"->"retnet_cls_pred_prob7"->NetPort["ClassProb7"],
      "fpn_7"->"retnet_bbox_conv_n0_fpn7"->"retnet_bbox_conv_n1_fpn7"->"retnet_bbox_conv_n2_fpn7"->"retnet_bbox_conv_n3_fpn7"->
      "retnet_bbox_pred_fpn7"->NetPort["Boxes7"]
}];


ConcatNet = NetGraph[{
   "class1"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{112,144,9,80}],FlattenLayer[2]},
   "class2"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{56,72,9,80}],FlattenLayer[2]},
   "class3"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,80}],FlattenLayer[2]},
   "class4"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,80}],FlattenLayer[2]},
   "class5"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,80}],FlattenLayer[2]},
   "catenate1"->CatenateLayer[],
   "locs1"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{112,144,9,4}],FlattenLayer[2]},
   "locs2"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{56,72,9,4}],FlattenLayer[2]},
   "locs3"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,4}],FlattenLayer[2]},
   "locs4"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,4}],FlattenLayer[2]},
   "locs5"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,4}],FlattenLayer[2]},
   "catenate2"->CatenateLayer[]
   },{
   NetPort["ClassProb3"]->"class1",
   NetPort["ClassProb4"]->"class2",
   NetPort["ClassProb5"]->"class3",
   NetPort["ClassProb6"]->"class4",
   NetPort["ClassProb7"]->"class5",
   NetPort["Boxes3"]->"locs1",
   NetPort["Boxes4"]->"locs2",
   NetPort["Boxes5"]->"locs3",
   NetPort["Boxes6"]->"locs4",
   NetPort["Boxes7"]->"locs5",
   {"class1","class2","class3","class4","class5"}->"catenate1"->NetPort["ClassProb"],
   {"locs1","locs2","locs3","locs4","locs5"}->"catenate2"->NetPort["Locs"]
}];


anch = Import[ "/home/julian/detectron_mount/RetinaNetNew.hdf5", {"Datasets", "anchors"} ];


levels={{112,144},{56,72},{28,36},{28,36},{28,36}};
strides = {8,16,32,32,32};
biasesx = Flatten[Table[(x-1)*strides[[l]]+Mean[anch[[l,a,{1,3}]]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
biasesy = Flatten[Table[(y-1)*strides[[l]]+Mean[anch[[l,a,{2,4}]]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
scalesw = Flatten[Table[1+anch[[l,a,3]]-anch[[l,a,1]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
scalesh = Flatten[Table[1+anch[[l,a,4]]-anch[[l,a,2]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];


LocsToBoxesNet = NetGraph[ { (*input is in format {Y*X*A}*4*)
   "cx"->{PartLayer[{All,1}],ConstantTimesLayer["Scaling"->scalesw],ConstantPlusLayer["Biases"->biasesx]},
   "cy"->{PartLayer[{All,2}],ConstantTimesLayer["Scaling"->scalesh],ConstantPlusLayer["Biases"->biasesy]},
   "width"->{PartLayer[{All,3}],ElementwiseLayer[Exp],ConstantTimesLayer["Scaling"->scalesw]},
   "height"->{PartLayer[{All,4}],ElementwiseLayer[Exp],ConstantTimesLayer["Scaling"->scalesh]},
   "minx"->ThreadingLayer[#1-#2/2&],
   "miny"->ThreadingLayer[896+1-(#1+#2/2)&],
   "maxx"->ThreadingLayer[#1+#2/2&],
   "maxy"->ThreadingLayer[896+1-(#1-#2/2)&],
   "cat"->CatenateLayer[],"reshape"->ReshapeLayer[ {4, 208656} ], "transpose"->TransposeLayer[], "reshapePoint"->ReshapeLayer[ {208656, 2, 2 } ] }, {
   {"cx","width"}->"minx",{"cx","width"}->"maxx",{"cy","height"}->"miny",{"cy","height"}->"maxy",
   {"minx","miny","maxx","maxy"}->"cat"->"reshape"->"transpose"->"reshapePoint"->NetPort["Boxes"]}];


dat=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","data"}];


ref=Import["/home/julian/detectron_mount/RetinaNetNew.hdf5",{"Datasets","res5_2_sum"}];


(* Warning need to check BGR vs RGB reversing and any pixel remapping *)
RetinaNet = NetGraph[ {
   "n1"->net1,
   "n2"->net2,
   "n3"->net3,
   "concat"->ConcatNet,"boxes"->LocsToBoxesNet},
{NetPort["n1","res3_3_sum"]->NetPort["n3","res3_3_sum"],
 NetPort["n1","res4_22_sum"]->{"n2",NetPort["n3","res4_22_sum"]},
 "n2"->NetPort["n3","res5_2_sum"],
 NetPort["n3","ClassProb3"]->NetPort["concat","ClassProb3"],
 NetPort["n3","ClassProb4"]->NetPort["concat","ClassProb4"],
 NetPort["n3","ClassProb5"]->NetPort["concat","ClassProb5"],
 NetPort["n3","ClassProb6"]->NetPort["concat","ClassProb6"],
 NetPort["n3","ClassProb7"]->NetPort["concat","ClassProb7"],
  NetPort["n3","Boxes3"]->NetPort["concat","Boxes3"],
  NetPort["n3","Boxes4"]->NetPort["concat","Boxes4"],
  NetPort["n3","Boxes5"]->NetPort["concat","Boxes5"],
  NetPort["n3","Boxes6"]->NetPort["concat","Boxes6"],
  NetPort["n3","Boxes7"]->NetPort["concat","Boxes7"],
  NetPort["concat","Locs"]->"boxes"
 },
 "Input"->NetEncoder[{"Image",{1152,896},"ColorSpace"->"RGB"}]];


Options[ CZDetectObjects ] = Join[{
   TargetDevice->"CPU",
   Threshold->.6,
   NMSIntersectionOverUnionThreshold->.45 (* This is the Wei Liu default setting for this implementation *)
}, Options[ CZNonMaxSuppressionPerClass ] ];
CZDetectObjects[ img_Image, opts:OptionsPattern[] ] :=
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ img, {1152, 896}, "Fit" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (k=(RetinaNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[{1152,896},"Fit"]@img);


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


CZCOCOClasses = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
"zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
"fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
"cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"};


(* Private Implementation Code *)


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[Normal@netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[Normal@netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[CZCOCOClasses,detections[[All,2;;2]]],
      Extract[Normal@netOutput["ClassProb"], detections ]
   }]
];
