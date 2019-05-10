(* ::Package:: *)

(* Warning not complete, in progress, being checked in to track changes *)


hdfFile = "~/Google Drive/RetinaNetNew.hdf5";


impW[ hdfName_String ] := If[hdfName=="conv1_w",
   256.*Reverse/@Import[ hdfFile, {"Datasets", hdfName} ],
   Import[ hdfFile, {"Datasets", hdfName} ]
   ];


impM[ hdfName_String ] := Import[ hdfFile, {"Datasets", hdfName} ];


BNConvolutionLayer[ outputChannels_Integer, kernelSize_List, stride_Integer, paddingSize_Integer, res_List, weightName_String, scalingName_String, biasesName_String ] :=
   NetChain[{ConvolutionLayer[ outputChannels, kernelSize, "Weights"->impW[weightName], "Biases"->ConstantArray[0,outputChannels], "Stride"->stride, "PaddingSize"->paddingSize ],
   BatchNormalizationLayer[ "Epsilon"->.001, "MovingMean"->ConstantArray[0, outputChannels ], "MovingVariance"->ConstantArray[ 1.-10^-3, outputChannels ], "Scaling"->impM[ scalingName ], "Biases"->impM[biasesName ] ]
}];


BNConvolutionLayer[ outputChannels_Integer, kernelSize_List, stride_Integer, paddingSize_Integer, res_List, rootName_String ] :=
   BNConvolutionLayer[ outputChannels, kernelSize, stride, paddingSize, res, rootName<>"_w", rootName<>"_bn_s", rootName<>"_bn_b" ];


RetinaNetBranch2[ rootName_, outputChannels_, stride_, dims_ ] := NetGraph[ {
   rootName<>"a"->{BNConvolutionLayer[ outputChannels/4, {1,1}, stride, 0, dims, rootName<>"a" ], Ramp},
   rootName<>"b"->{BNConvolutionLayer[ outputChannels/4, {3,3}, 1, 1, dims, rootName<>"b" ], Ramp},
   rootName<>"c_bn"->BNConvolutionLayer[ outputChannels, {1,1}, 1, 0, dims, rootName<>"c" ]
},{
   rootName<>"a"->rootName<>"b"->rootName<>"c_bn"}]


RetinaNetBlock[ rootName_, outputChannels_, stride_, dims_, batchNormBranch1_Symbol: False ] := NetGraph[{
   "branch1"->If [batchNormBranch1, BNConvolutionLayer[ outputChannels, {1,1}, stride, 0, dims, rootName<>"_branch1" ], ElementwiseLayer[#*1.0&] ],
   "branch2"->RetinaNetBranch2[ rootName<>"_branch2", outputChannels, stride, dims ],
   "sum"->TotalLayer[],
   "relu"->Ramp},
   {{"branch1","branch2"}->"sum"->"relu"}]


ConvNet = NetGraph[{
   "conv1"->{BNConvolutionLayer[ 64, {7,7}, 2, 3, {448, 576}, "conv1_w", "res_conv1_bn_s","res_conv1_bn_b" ],Ramp},
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],
   
   "res2_0"->RetinaNetBlock[ "res2_0", 256, 1, {224,288}, True ],
   "res2_1"->RetinaNetBlock[ "res2_1", 256, 1, {224,288} ],
   "res2_2"->RetinaNetBlock[ "res2_2", 256, 1, {224,288} ],

   "res3_0"->RetinaNetBlock[ "res3_0", 512, 2, {112,144}, True ],
   "res3_1"->RetinaNetBlock[ "res3_1", 512, 1, {112,144} ],
   "res3_2"->RetinaNetBlock[ "res3_2", 512, 1, {112,144} ],
   "res3_3"->RetinaNetBlock[ "res3_3", 512, 1, {112,144} ],

   "res4_0"->RetinaNetBlock[ "res4_0", 1024, 2, {56,72}, True ],
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
   "res4_22"->RetinaNetBlock[ "res4_22", 1024, 1, {56,72} ],
   
   "res5_0"->RetinaNetBlock[ "res5_0", 2048, 2, {28,36}, True ],
   "res5_1"->RetinaNetBlock[ "res5_1", 2048, 1, {28,36} ],
   "res5_2"->RetinaNetBlock[ "res5_2", 2048, 1, {28,36} ]
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
   "res4_20"->"res4_21"->"res4_22"->
   "res5_0"->"res5_1"->"res5_2",
   "res3_3"->NetPort["res3_3_sum"],
   "res4_22"->NetPort["res4_22_sum"],
   "res5_2"->NetPort["res5_2_sum"]
}];


MultiBoxDecoder[ layerName_String ] := NetGraph[{
   "retnet_cls_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n0_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_cls_conv_n0_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_cls_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n1_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_cls_conv_n1_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_cls_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n2_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_cls_conv_n2_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_cls_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n3_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_cls_conv_n3_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_cls_pred_fpn"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_pred_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_cls_pred_fpn"<>layerName<>"_b"]  ],
   "retnet_cls_pred_prob"->LogisticSigmoid,
   "retnet_bbox_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n0_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_bbox_conv_n0_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_bbox_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n1_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_bbox_conv_n1_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_bbox_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n2_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_bbox_conv_n2_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_bbox_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n3_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_bbox_conv_n3_fpn"<>layerName<>"_b"]  ],Ramp},
   "retnet_bbox_pred_fpn"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_pred_fpn"<>layerName<>"_w"], "Biases"->impW["retnet_bbox_pred_fpn"<>layerName<>"_b"]]},{
   "retnet_cls_conv_n0_fpn"->"retnet_cls_conv_n1_fpn"->"retnet_cls_conv_n2_fpn"->"retnet_cls_conv_n3_fpn"->
   "retnet_cls_pred_fpn"->"retnet_cls_pred_prob"->NetPort["ClassProb"],
   "retnet_bbox_conv_n0_fpn"->"retnet_bbox_conv_n1_fpn"->"retnet_bbox_conv_n2_fpn"->"retnet_bbox_conv_n3_fpn"->
   "retnet_bbox_pred_fpn"->NetPort["Boxes"]}];


MultiBoxDecoderNet = NetGraph[{
   "retnet_cls_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n0_fpn3_w"], "Biases"->impW["retnet_cls_conv_n0_fpn3_b"]  ],Ramp},
   "retnet_cls_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n1_fpn3_w"], "Biases"->impW["retnet_cls_conv_n1_fpn3_b"]  ],Ramp},
   "retnet_cls_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n2_fpn3_w"], "Biases"->impW["retnet_cls_conv_n2_fpn3_b"]  ],Ramp},
   "retnet_cls_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_conv_n3_fpn3_w"], "Biases"->impW["retnet_cls_conv_n3_fpn3_b"]  ],Ramp},
   "retnet_cls_pred_fpn"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_cls_pred_fpn3_w"], "Biases"->impW["retnet_cls_pred_fpn3_b"]  ],
   "retnet_cls_pred_prob"->LogisticSigmoid,
   "retnet_bbox_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n0_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n0_fpn3_b"]  ],Ramp},
   "retnet_bbox_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n1_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n1_fpn3_b"]  ],Ramp},
   "retnet_bbox_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n2_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n2_fpn3_b"]  ],Ramp},
   "retnet_bbox_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_conv_n3_fpn3_w"], "Biases"->impW["retnet_bbox_conv_n3_fpn3_b"]  ],Ramp},
   "retnet_bbox_pred_fpn"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->impW["retnet_bbox_pred_fpn3_w"], "Biases"->impW["retnet_bbox_pred_fpn3_b"]]},{
   "retnet_cls_conv_n0_fpn"->"retnet_cls_conv_n1_fpn"->"retnet_cls_conv_n2_fpn"->"retnet_cls_conv_n3_fpn"->
   "retnet_cls_pred_fpn"->"retnet_cls_pred_prob"->NetPort["ClassProb"],
   "retnet_bbox_conv_n0_fpn"->"retnet_bbox_conv_n1_fpn"->"retnet_bbox_conv_n2_fpn"->"retnet_bbox_conv_n3_fpn"->
   "retnet_bbox_pred_fpn"->NetPort["Boxes"]}];


DecoderNet = (* input fpn_inner_res3_3_sum *)
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
      "fpn_6"->ConvolutionLayer[ 256, {3,3}, "Stride"->2, "PaddingSize"->1, "Weights"->impW["fpn_6_w"], "Biases"->impW["fpn_6_b"]  ],
      "fpn_7"->{Ramp,ConvolutionLayer[ 256, {3,3}, "Stride"->2, "PaddingSize"->1, "Weights"->impW["fpn_7_w"], "Biases"->impW["fpn_7_b"]  ]},
      
      "multibox3"->MultiBoxDecoderNet,
      "multibox4"->MultiBoxDecoderNet,
      "multibox5"->MultiBoxDecoderNet,
      "multibox6"->MultiBoxDecoderNet,
      "multibox7"->MultiBoxDecoderNet
      },{
      
      NetPort["res5_2_sum"]->"fpn_inner_res5_2_sum"->"fpn_inner_res4_22_sum_topdown",
      NetPort["res3_3_sum"]->"fpn_inner_res3_3_sum_lateral",
      NetPort["res4_22_sum"]->"fpn_inner_res4_22_sum_lateral",
      {"fpn_inner_res4_22_sum_topdown","fpn_inner_res4_22_sum_lateral"}->"fpn_inner_res4_22_sum"->"fpn_inner_res3_3_sum_topdown",

      {"fpn_inner_res3_3_sum_topdown","fpn_inner_res3_3_sum_lateral"}->"fpn_inner_res3_3_sum"->"fpn_res3_3_sum"->"multibox3",
      "fpn_inner_res4_22_sum"->"fpn_res4_22_sum"->"multibox4",
      "fpn_inner_res5_2_sum"->"fpn_res5_2_sum"->"multibox5",
      NetPort["res5_2_sum"]->"fpn_6"->"multibox6",
      "fpn_6"->"fpn_7"->"multibox7",
      NetPort["multibox3","ClassProb"]->NetPort["ClassProb3"],
      NetPort["multibox3","Boxes"]->NetPort["Boxes3"],
      NetPort["multibox4","ClassProb"]->NetPort["ClassProb4"],
      NetPort["multibox4","Boxes"]->NetPort["Boxes4"],
      NetPort["multibox5","ClassProb"]->NetPort["ClassProb5"],
      NetPort["multibox5","Boxes"]->NetPort["Boxes5"],
      NetPort["multibox6","ClassProb"]->NetPort["ClassProb6"],
      NetPort["multibox6","Boxes"]->NetPort["Boxes6"],
      NetPort["multibox7","ClassProb"]->NetPort["ClassProb7"],
      NetPort["multibox7","Boxes"]->NetPort["Boxes7"]
}];


ConcatNet = NetGraph[{
   "class1"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{112,144,9,80}],FlattenLayer[2]},
   "class2"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{56,72,9,80}],FlattenLayer[2]},
   "class3"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,80}],FlattenLayer[2]},
   "class4"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{14,18,9,80}],FlattenLayer[2]},
   "class5"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{7,9,9,80}],FlattenLayer[2]},
   "catenate1"->CatenateLayer[],
   "locs1"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{112,144,9,4}],FlattenLayer[2]},
   "locs2"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{56,72,9,4}],FlattenLayer[2]},
   "locs3"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{28,36,9,4}],FlattenLayer[2]},
   "locs4"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{14,18,9,4}],FlattenLayer[2]},
   "locs5"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{7,9,9,4}],FlattenLayer[2]},
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


anch = Import[ hdfFile, {"Datasets", "anchors"} ];


levels={{112,144},{56,72},{28,36},{14,18},{7,9}};
strides = {8,16,32,64,128};
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
   "cat"->CatenateLayer[],"reshape"->ReshapeLayer[ {4, 193347} ], "transpose"->TransposeLayer[], "reshapePoint"->ReshapeLayer[ {193347, 2, 2 } ] }, {
   {"cx","width"}->"minx",{"cx","width"}->"maxx",{"cy","height"}->"miny",{"cy","height"}->"maxy",
   {"minx","miny","maxx","maxy"}->"cat"->"reshape"->"transpose"->"reshapePoint"->NetPort["Boxes"]}];


(* Warning need to check BGR vs RGB reversing and any pixel remapping *)
RetinaNet = NetGraph[ {
   "ConvNet"->ConvNet,
   "DecoderNet"->DecoderNet,
   "concat"->ConcatNet,"boxes"->LocsToBoxesNet},
{
   NetPort["ConvNet","res3_3_sum"]->NetPort["DecoderNet","res3_3_sum"],
   NetPort["ConvNet","res4_22_sum"]->NetPort["DecoderNet","res4_22_sum"],
 NetPort["ConvNet","res5_2_sum"]->NetPort["DecoderNet","res5_2_sum"],
 NetPort["DecoderNet","ClassProb3"]->NetPort["concat","ClassProb3"],
 NetPort["DecoderNet","ClassProb4"]->NetPort["concat","ClassProb4"],
 NetPort["DecoderNet","ClassProb5"]->NetPort["concat","ClassProb5"],
 NetPort["DecoderNet","ClassProb6"]->NetPort["concat","ClassProb6"],
 NetPort["DecoderNet","ClassProb7"]->NetPort["concat","ClassProb7"],
  NetPort["DecoderNet","Boxes3"]->NetPort["concat","Boxes3"],
  NetPort["DecoderNet","Boxes4"]->NetPort["concat","Boxes4"],
  NetPort["DecoderNet","Boxes5"]->NetPort["concat","Boxes5"],
  NetPort["DecoderNet","Boxes6"]->NetPort["concat","Boxes6"],
  NetPort["DecoderNet","Boxes7"]->NetPort["concat","Boxes7"],
  NetPort["concat","Locs"]->"boxes"
 },
 "Input"->NetEncoder[{"Image",{1152,896},"ColorSpace"->"RGB","MeanImage"->{102.9801, 115.9465, 122.7717}/256.}]];
