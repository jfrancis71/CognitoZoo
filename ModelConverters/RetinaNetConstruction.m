(* ::Package:: *)

BNConvolutionLayer[ outputChannels_Integer, kernelSize_List, stride_Integer, paddingSize_Integer ] :=
   NetChain[{ConvolutionLayer[ outputChannels, kernelSize, "Stride"->stride, "PaddingSize"->paddingSize ],
   BatchNormalizationLayer[]
}];


RetinaNetBranch2[ outputChannels_, stride_ ] := NetChain[ {
   "a"->{BNConvolutionLayer[ outputChannels/4, {1,1}, stride, 0 ], Ramp},
   "b"->{BNConvolutionLayer[ outputChannels/4, {3,3}, 1, 1 ], Ramp},
   "c"->BNConvolutionLayer[ outputChannels, {1,1}, 1, 0 ]
}];


RetinaNetBlock[ outputChannels_, stride_, batchNormBranch1_Symbol: False ] := NetGraph[{
   If [batchNormBranch1, "branch1"->BNConvolutionLayer[ outputChannels, {1,1}, stride, 0 ], Nothing ],
   "branch2"->RetinaNetBranch2[ outputChannels, stride ],
   "sum"->TotalLayer[],
   "relu"->Ramp},
   {{If[batchNormBranch1,"branch1",NetPort["Input"]],"branch2"}->"sum"->"relu"}]


ResidualNetBlock[ repeats_, channels_, initStride_] := NetChain[Prepend[
   ToString[0]->RetinaNetBlock[ channels, initStride, True ]][
   Table[ ToString[k]->RetinaNetBlock[ channels, 1 ], {k, repeats} ]
]]


ResBackboneNet = NetGraph[{
   "conv1"->{BNConvolutionLayer[ 64, {7,7}, 2, 3 ], Ramp},
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],   
   "res2"->ResidualNetBlock[ 2, 256, 1 ],
   "res3"->ResidualNetBlock[ 3, 512, 2 ],
   "res4"->ResidualNetBlock[ 22, 1024, 2 ],
   "res5"->ResidualNetBlock[ 2, 2048, 2 ]
},{
   "conv1"->"pool1"->"res2"->"res3"->"res4"->"res5",
   "res3"->NetPort["res3_3_sum"],
   "res4"->NetPort["res4_22_sum"],
   "res5"->NetPort["res5_2_sum"]
}];


FPNNet = NetGraph[{
      "inner_res3_3_sum_lateral"->ConvolutionLayer[ 256, {1,1} ],
      "inner_res4_22_sum_lateral"->ConvolutionLayer[ 256, {1,1} ],
      "inner_res5_2_sum"->ConvolutionLayer[ 256, {1,1} ],
      "inner_res4_22_sum"->TotalLayer[],
      "inner_res4_22_sum_topdown"->ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"],
      "inner_res3_3_sum_topdown"->ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"],
      "inner_res3_3_sum"->TotalLayer[],
      "res3_3_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],
      "res4_22_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],
      "res5_2_sum"->ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1  ],
      "6"->ConvolutionLayer[ 256, {3,3}, "Stride"->2, "PaddingSize"->1 ],
      "7"->{Ramp,ConvolutionLayer[ 256, {3,3}, "Stride"->2, "PaddingSize"->1  ]}},{
      
      NetPort["res5_2_sum"]->{"inner_res5_2_sum","6"},
      NetPort["res4_22_sum"]->"inner_res4_22_sum_lateral",
      NetPort["res3_3_sum"]->"inner_res3_3_sum_lateral",
      
      {"inner_res4_22_sum_topdown","inner_res4_22_sum_lateral"}->"inner_res4_22_sum",
      {"inner_res3_3_sum_topdown","inner_res3_3_sum_lateral"}->"inner_res3_3_sum"->"res3_3_sum"->NetPort["multibox3"],
      "inner_res5_2_sum"->"inner_res4_22_sum_topdown",
      "6"->"7"->NetPort["multibox7"],
      "6"->NetPort["multibox6"],
      "inner_res5_2_sum"->"res5_2_sum"->NetPort["multibox5"],
      "inner_res4_22_sum"->"inner_res3_3_sum_topdown",
      "inner_res4_22_sum"->"res4_22_sum"->NetPort["multibox4"]}];


MultiBoxDecoderNet = NetGraph[{
   "ClassDecoder"->{
      "retnet_cls_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_cls_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_cls_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_cls_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_cls_pred_fpn"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1 ],
      "retnet_cls_pred_prob"->LogisticSigmoid,
      "flatten"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{Inherited,Inherited,9,80}],FlattenLayer[2]}},
   "BoxesDecoder"->{
      "retnet_bbox_conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_bbox_conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_bbox_conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_bbox_conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "retnet_bbox_pred_fpn"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1 ],
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


BoxTransformationNet = NetGraph[ { (*input is in format {Y*X*A}*4*)
   "cx"->{PartLayer[{All,1}],ConstantTimesLayer[],ConstantPlusLayer[]},
   "cy"->{PartLayer[{All,2}],ConstantTimesLayer[],ConstantPlusLayer[]},
   "width"->{PartLayer[{All,3}],ElementwiseLayer[Exp],ConstantTimesLayer[]},
   "height"->{PartLayer[{All,4}],ElementwiseLayer[Exp],ConstantTimesLayer[]},
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
 "Input"->NetEncoder[{"Image",{1152,896},"ColorSpace"->"RGB","MeanImage"->Reverse@{102.9801, 115.9465, 122.7717}/256.}]];
