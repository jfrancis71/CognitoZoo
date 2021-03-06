(* ::Package:: *)

(*

Facebook Detectron model: R-101-FPN LRN 2
https: https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md

@misc{Detectron2018,
  author =       {Ross Girshick and Ilija Radosavovic and Georgia Gkioxari and
                  Piotr Doll\'{a}r and Kaiming He},
  title =        {Detectron},
  howpublished = {\url{https://github.com/facebookresearch/detectron}},
  year =         {2018}
}
*)


ResBranch2[ outputChannels_, stride_ ] := NetChain[{
   "conv1a"->ConvolutionLayer[ outputChannels/4, {1,1}, "Stride"->stride, "PaddingSize"->0 ],
   "bn1a"-> BatchNormalizationLayer[],
   "relu1a"-> Ramp,
   "conv1b"->ConvolutionLayer[outputChannels/4, {3,3}, "Stride"->1, "PaddingSize"->1],
   "bn1b"-> BatchNormalizationLayer[],
   "relu1b"-> Ramp,
   "conv1c"->ConvolutionLayer[outputChannels, {1,1}, "Stride"->1, "PaddingSize"->0],
   "bn1c"-> BatchNormalizationLayer[]}
];


ResBlock[ outputChannels_, stride_, identityShortCut_Symbol: True ] := NetGraph[{
   If [identityShortCut,
      Nothing,
      "branch1"->{
         ConvolutionLayer[ outputChannels, {1,1} , "Stride"->stride, "PaddingSize"->-0 ],
         BatchNormalizationLayer[]} ],
   "branch2"->ResBranch2[ outputChannels, stride ],
   "sum"->TotalLayer[],
   "relu"->Ramp
},{
   {If[identityShortCut,NetPort["Input"],"branch1"],"branch2"}->"sum"->"relu"
}];


ResChain[ resBlockRootName_, repeats_, channels_, initStride_] := Prepend[
   resBlockRootName<>"_"<>ToString[0]->ResBlock[ channels, initStride, False ]][
   Table[ resBlockRootName<>"_"<>ToString[k]->ResBlock[ channels, 1 ], {k, repeats} ]
];


ResBackboneNet = NetGraph[Flatten@{
   "conv1"->{
      ConvolutionLayer[ 64, {7,7} , "Stride"->2, "PaddingSize"->3 ],
      BatchNormalizationLayer[],
      Ramp },
   "pool1"->PoolingLayer[ {3,3}, "Stride"->2, "PaddingSize"->1 ],   
   ResChain[ "res2", 2, 256, 1 ],
   ResChain[ "res3", 3, 512, 2 ],
   ResChain[ "res4", 22, 1024, 2 ],
   ResChain[ "res5", 2, 2048, 2 ]
},{
   "conv1"->"pool1"->"res2_0","res2_2"->"res3_0","res3_3"->"res4_0","res4_22"->"res5_0",
   Table["res2_"<>ToString[i-1]-> "res2_"<>ToString[i],{i,1,2}],
   Table["res3_"<>ToString[i-1]-> "res3_"<>ToString[i],{i,1,3}],
   Table["res4_"<>ToString[i-1]-> "res4_"<>ToString[i],{i,1,22}],
   Table["res5_"<>ToString[i-1]-> "res5_"<>ToString[i],{i,1,2}],
   "res3_3"->NetPort["res3_3_sum"],
   "res4_22"->NetPort["res4_22_sum"],
   "res5_2"->NetPort["res5_2_sum"]
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
   "7"->{Ramp,ConvolutionLayer[ 256, {3,3}, "Stride"->2, "PaddingSize"->1  ]}
},{      
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
   "inner_res4_22_sum"->"res4_22_sum"->NetPort["multibox4"]
}];


MultiBoxDecoderNet = NetGraph[{
   "ClassDecoder"->{
      "conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "pred_fpn"->ConvolutionLayer[ 720, {3,3}, "PaddingSize"->1 ],
      "pred_prob"->LogisticSigmoid,
      "flatten"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{Inherited,Inherited,9,80}],FlattenLayer[2]}},
   "BoxesDecoder"->{
      "conv_n0_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "conv_n1_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "conv_n2_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "conv_n3_fpn"->{ConvolutionLayer[ 256, {3,3}, "PaddingSize"->1 ],Ramp},
      "pred_fpn"->ConvolutionLayer[ 36, {3,3}, "PaddingSize"->1 ],
      "flatten"->{TransposeLayer[{1->2,2->3}],ReshapeLayer[{Inherited,Inherited,9,4}],FlattenLayer[2]}}
},{
   "ClassDecoder"->NetPort["ClassProb"],
   "BoxesDecoder"->NetPort["Locs"]
}];


DecoderNet = NetGraph[{
   "multibox3"->MultiBoxDecoderNet,
   "multibox4"->MultiBoxDecoderNet,
   "multibox5"->MultiBoxDecoderNet,
   "multibox6"->MultiBoxDecoderNet,
   "multibox7"->MultiBoxDecoderNet,
   "ClassProb"->CatenateLayer[],
    "Locs"->CatenateLayer[]
},{
   NetPort["multibox3"]->"multibox3",
   NetPort["multibox4"]->"multibox4",
   NetPort["multibox5"]->"multibox5",
   NetPort["multibox6"]->"multibox6",
   NetPort["multibox7"]->"multibox7",
   {NetPort["multibox3","ClassProb"],NetPort["multibox4","ClassProb"],NetPort["multibox5","ClassProb"],NetPort["multibox6","ClassProb"],NetPort["multibox7","ClassProb"]}->"ClassProb"->NetPort["ClassProb"],
   {NetPort["multibox3","Locs"],NetPort["multibox4","Locs"],NetPort["multibox5","Locs"],NetPort["multibox6","Locs"],NetPort["multibox7","Locs"]}->"Locs"->NetPort["Locs"]
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
   "boxes"->CatenateLayer[],"reshape"->ReshapeLayer[ {4, 193347} ], "transpose"->TransposeLayer[], "reshapePoint"->ReshapeLayer[ {193347, 2, 2 } ] }, {
   {"cx","width"}->"minx",{"cx","width"}->"maxx",{"cy","height"}->"miny",{"cy","height"}->"maxy",
   {"minx","miny","maxx","maxy"}->"boxes"->"reshape"->"transpose"->"reshapePoint"->NetPort["Boxes"]}];


RetinaNet = NetGraph[ {
   "ResBackbone"->ResBackboneNet,
   "FPN"->FPNNet,
   "Decoder"->DecoderNet,
   "BoxTransformation"->BoxTransformationNet
},{
   NetPort["ResBackbone","res3_3_sum"]->NetPort["FPN","res3_3_sum"],
   NetPort["ResBackbone","res4_22_sum"]->NetPort["FPN","res4_22_sum"],
   NetPort["ResBackbone","res5_2_sum"]->NetPort["FPN","res5_2_sum"],
   NetPort["FPN","multibox3"]->NetPort["Decoder","multibox3"],
   NetPort["FPN","multibox4"]->NetPort["Decoder","multibox4"],
   NetPort["FPN","multibox5"]->NetPort["Decoder","multibox5"],
   NetPort["FPN","multibox6"]->NetPort["Decoder","multibox6"],
   NetPort["FPN","multibox7"]->NetPort["Decoder","multibox7"],
   NetPort["Decoder","Locs"]->"BoxTransformation"
   },
   "Input"->NetEncoder[{"Image",{1152,896},"ColorSpace"->"RGB","MeanImage"->Reverse@{102.9801, 115.9465, 122.7717}/256.}]
 ];
