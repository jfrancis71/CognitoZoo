(* ::Package:: *)

(* Warning not complete, in progress, being checked in to track changes *)


hdfFile = "~/Google Drive/RetinaNetNew1.hdf5";


imp[ hdfName_String ] := Import[ hdfFile, {"Datasets", hdfName} ];


BNConvolutionLayer[ outputChannels_Integer, kernelSize_List, stride_Integer, paddingSize_Integer, res_List, rootName_String ] :=
   NetChain[{ConvolutionLayer[ outputChannels, kernelSize, "Weights"->imp[rootName<>"_w"], "Biases"->ConstantArray[0,outputChannels], "Stride"->stride, "PaddingSize"->paddingSize ],
   BatchNormalizationLayer[ "Epsilon"->.001, "MovingMean"->ConstantArray[0, outputChannels ], "MovingVariance"->ConstantArray[ 1.-10^-3, outputChannels ], "Scaling"->imp[ rootName<>"_bn_s" ], "Biases"->imp[ rootName<>"_bn_b" ] ]
}];


RetinaNetBranch2[ rootName_, outputChannels_, stride_, dims_ ] := NetChain[ {
   rootName<>"a"->{BNConvolutionLayer[ outputChannels/4, {1,1}, stride, 0, dims, rootName<>"a" ], Ramp},
   rootName<>"b"->{BNConvolutionLayer[ outputChannels/4, {3,3}, 1, 1, dims, rootName<>"b" ], Ramp},
   rootName<>"c_bn"->BNConvolutionLayer[ outputChannels, {1,1}, 1, 0, dims, rootName<>"c" ]
}];


RetinaNetBlock[ rootName_, outputChannels_, stride_, dims_, batchNormBranch1_Symbol: False ] := NetGraph[{
   If [batchNormBranch1, "branch1"->BNConvolutionLayer[ outputChannels, {1,1}, stride, 0, dims, rootName<>"_branch1" ], Nothing ],
   "branch2"->RetinaNetBranch2[ rootName<>"_branch2", outputChannels, stride, dims ],
   "sum"->TotalLayer[],
   "relu"->Ramp},
   {{If[batchNormBranch1,"branch1",NetPort["Input"]],"branch2"}->"sum"->"relu"}]


ResidualNetBlock[ rootName_, repeats_, channels_, initStride_, dims_ ] := NetChain[Prepend[
   rootName<>"_0"->RetinaNetBlock[ rootName<>"_0", channels, initStride, dims, True ]][
   Table[ rootName<>"_"<>ToString[k]->RetinaNetBlock[ rootName<>"_"<>ToString[k], channels, 1, dims ], {k, repeats} ]
]]


ResBackboneNet = NetGraph[{
   "conv1"->{BNConvolutionLayer[ 64, {7,7}, 2, 3, {448, 576}, "conv1" ], Ramp},
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],   
   "res2"->ResidualNetBlock[ "res2", 2, 256, 1, {224,288} ],
   "res3"->ResidualNetBlock[ "res3", 3, 512, 2, {112,144} ],
   "res4"->ResidualNetBlock[ "res4", 22, 1024, 2, {56,72} ],
   "res5"->ResidualNetBlock[ "res5", 2, 2048, 2, {28,36} ]
},{
   "conv1"->"pool1"->"res2"->"res3"->"res4"->"res5",
   "res3"->NetPort["res3_3_sum"],
   "res4"->NetPort["res4_22_sum"],
   "res5"->NetPort["res5_2_sum"]
}];


FPNNet = NetGraph[{
      "fpn_inner_res3_3_sum_lateral"->ConvolutionLayer[ 256, {1,1}, "Weights"->imp["fpn_inner_res3_3_sum_lateral_w"], "Biases"->imp["fpn_inner_res3_3_sum_lateral_b"] ],
      "fpn_inner_res4_22_sum_lateral"->ConvolutionLayer[ 256, {1,1}, "Weights"->imp["fpn_inner_res4_22_sum_lateral_w"], "Biases"->imp["fpn_inner_res4_22_sum_lateral_b"] ],
      "fpn_inner_res5_2_sum"->ConvolutionLayer[ 256, {1,1}, "Weights"->imp["fpn_inner_res5_2_sum_w"], "Biases"->imp["fpn_inner_res5_2_sum_b"] ],
      "fpn_inner_res4_22_sum"->TotalLayer[],
      "fpn_inner_res4_22_sum_topdown"->ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"],
      "fpn_inner_res3_3_sum_topdown"->ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"],
      "fpn_inner_res3_3_sum"->TotalLayer[],
      "fpn_res3_3_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["fpn_res3_3_sum_w"], "Biases"->imp["fpn_res3_3_sum_b"]  ],
      "fpn_res4_22_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["fpn_res4_22_sum_w"], "Biases"->imp["fpn_res4_22_sum_b"]  ],
      "fpn_res5_2_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["fpn_res5_2_sum_w"], "Biases"->imp["fpn_res5_2_sum_b"]  ],
      "fpn_6"->ConvolutionLayer[ 256, {3,3}, "Stride"->2, "PaddingSize"->1, "Weights"->imp["fpn_6_w"], "Biases"->imp["fpn_6_b"]  ],
      "fpn_7"->{Ramp,ConvolutionLayer[ 256, {3,3}, "Stride"->2, "PaddingSize"->1, "Weights"->imp["fpn_7_w"], "Biases"->imp["fpn_7_b"]  ]}},{
      
      NetPort["res5_2_sum"]->{"fpn_inner_res5_2_sum","fpn_6"},
      NetPort["res4_22_sum"]->"fpn_inner_res4_22_sum_lateral",
      NetPort["res3_3_sum"]->"fpn_inner_res3_3_sum_lateral",
      
      {"fpn_inner_res4_22_sum_topdown","fpn_inner_res4_22_sum_lateral"}->"fpn_inner_res4_22_sum",
      {"fpn_inner_res3_3_sum_topdown","fpn_inner_res3_3_sum_lateral"}->"fpn_inner_res3_3_sum"->"fpn_res3_3_sum"->NetPort["multibox3"],
      "fpn_inner_res5_2_sum"->"fpn_inner_res4_22_sum_topdown",
      "fpn_6"->"fpn_7"->NetPort["multibox7"],
      "fpn_6"->NetPort["multibox6"],
      "fpn_inner_res5_2_sum"->"fpn_res5_2_sum"->NetPort["multibox5"],
      "fpn_inner_res4_22_sum"->"fpn_inner_res3_3_sum_topdown",
      "fpn_inner_res4_22_sum"->"fpn_res4_22_sum"->NetPort["multibox4"]}];


MultiBoxDecoderNet = NetGraph[{
   "ClassDecoder"->{
      "retnet_cls_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_cls_conv_n0_fpn3_w"], "Biases"->imp["retnet_cls_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_cls_conv_n1_fpn3_w"], "Biases"->imp["retnet_cls_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_cls_conv_n2_fpn3_w"], "Biases"->imp["retnet_cls_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_cls_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_cls_conv_n3_fpn3_w"], "Biases"->imp["retnet_cls_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_cls_pred_fpn"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_cls_pred_fpn3_w"], "Biases"->imp["retnet_cls_pred_fpn3_b"]  ],
      "retnet_cls_pred_prob"->LogisticSigmoid,
      "flatten"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{Inherited,Inherited,9,80}],FlattenLayer[2]}},
   "BoxesDecoder"->{
      "retnet_bbox_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_bbox_conv_n0_fpn3_w"], "Biases"->imp["retnet_bbox_conv_n0_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_bbox_conv_n1_fpn3_w"], "Biases"->imp["retnet_bbox_conv_n1_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_bbox_conv_n2_fpn3_w"], "Biases"->imp["retnet_bbox_conv_n2_fpn3_b"]  ],Ramp},
      "retnet_bbox_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_bbox_conv_n3_fpn3_w"], "Biases"->imp["retnet_bbox_conv_n3_fpn3_b"]  ],Ramp},
      "retnet_bbox_pred_fpn"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1, "Weights"->imp["retnet_bbox_pred_fpn3_w"], "Biases"->imp["retnet_bbox_pred_fpn3_b"]],
      "flatten"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{Inherited,Inherited,9,4}],FlattenLayer[2]}}
      },{
   "ClassDecoder"->NetPort["ClassProb"],
   "BoxesDecoder"->NetPort["Boxes"]}];


DecoderNet =
   NetGraph[{
      "multibox3"->MultiBoxDecoderNet,
      "multibox4"->MultiBoxDecoderNet,
      "multibox5"->MultiBoxDecoderNet,
      "multibox6"->MultiBoxDecoderNet,
      "multibox7"->MultiBoxDecoderNet,
      "catenate1"->CatenateLayer[],
      "catenate2"->CatenateLayer[]
},{
      NetPort["multibox3"]->"multibox3",
      NetPort["multibox4"]->"multibox4",
      NetPort["multibox5"]->"multibox5",
      NetPort["multibox6"]->"multibox6",
      NetPort["multibox7"]->"multibox7",
      {NetPort["multibox3","ClassProb"],NetPort["multibox4","ClassProb"],NetPort["multibox5","ClassProb"],NetPort["multibox6","ClassProb"],NetPort["multibox7","ClassProb"]}->"catenate1"->NetPort["ClassProb"],
   {NetPort["multibox3","Boxes"],NetPort["multibox4","Boxes"],NetPort["multibox5","Boxes"],NetPort["multibox6","Boxes"],NetPort["multibox7","Boxes"]}->"catenate2"->NetPort["Locs"]
}];


anch = Import[ hdfFile, {"Datasets", "anchors"} ];


levels={{112,144},{56,72},{28,36},{14,18},{7,9}};
strides = {8,16,32,64,128};
biasesx = Flatten[Table[(x-1)*strides[[l]]+Mean[anch[[l,a,{1,3}]]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
biasesy = Flatten[Table[(y-1)*strides[[l]]+Mean[anch[[l,a,{2,4}]]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
scalesw = Flatten[Table[1+anch[[l,a,3]]-anch[[l,a,1]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
scalesh = Flatten[Table[1+anch[[l,a,4]]-anch[[l,a,2]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];


BoxTransformationNet = NetGraph[ { (*input is in format {Y*X*A}*4*)
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


RetinaNet = NetGraph[ {
   "ResBackboneNet"->ResBackboneNet,
   "FPNNet"->FPNNet,
   "DecoderNet"->DecoderNet,
   "BoxTransformation"->BoxTransformationNet},
{
   NetPort["ResBackboneNet","res3_3_sum"]->NetPort["FPNNet","res3_3_sum"],
   NetPort["ResBackboneNet","res4_22_sum"]->NetPort["FPNNet","res4_22_sum"],
   NetPort["ResBackboneNet","res5_2_sum"]->NetPort["FPNNet","res5_2_sum"],
   NetPort["FPNNet","multibox3"]->NetPort["DecoderNet","multibox3"],
   NetPort["FPNNet","multibox4"]->NetPort["DecoderNet","multibox4"],
   NetPort["FPNNet","multibox5"]->NetPort["DecoderNet","multibox5"],
   NetPort["FPNNet","multibox6"]->NetPort["DecoderNet","multibox6"],
   NetPort["FPNNet","multibox7"]->NetPort["DecoderNet","multibox7"],
   NetPort["DecoderNet","Locs"]->"BoxTransformation"

   },
 "Input"->NetEncoder[{"Image",{1152,896},"ColorSpace"->"RGB","MeanImage"->{102.9801, 115.9465, 122.7717}/256.}]];
