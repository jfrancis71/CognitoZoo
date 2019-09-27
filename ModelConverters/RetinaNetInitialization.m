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


(*
   hdfFile has been converted from:
   Facebook Detectron model: R-101-FPN LRN 2
   https: https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md
   License: Apache License 2.0
*)


hdfFile = "~/Google Drive/Personal/Computer Science/WolframSubmissions/RetinaNet/RetinaNetR101FPNLR2.hdf5";


imp[ hdfName_String ] := Import[ hdfFile, {"Datasets", hdfName} ];


ConvLayerRules[ rootName_, biases_Symbol: False ] := {
   {"Weights"}->imp[rootName<>"_w"], {"Biases"}->If[ biases, imp[rootName<>"_b"], ConstantArray[0,Length@imp[rootName<>"_w"]]]};
BNLayerRules[ rootName_, outputChannels_ ] := {
   {"Epsilon"}->.001, {"MovingMean"}->ConstantArray[0, outputChannels ], {"MovingVariance"}->ConstantArray[ 1.-10^-3, outputChannels ],
   {"Scaling"}->imp[ rootName<>"_bn_s" ], {"Biases"}->imp[ rootName<>"_bn_b" ] };


BNConvLayerRules[ rootName_, outputChannels_ ] := Flatten@{
   InsertLevels[ ConvLayerRules[ rootName ], { 1 } ],
   InsertLevels[ BNLayerRules[ rootName, outputChannels ], { 2 } ]
};


InsertLevels[ rules_, levels_ ] := Map[ Join[levels,#[[1]]]->#[[2]]&, rules ];


ResidualBranch2Rules[ rootName_, outputChannels_ ] := Flatten@{
   InsertLevels[ ConvLayerRules[ rootName<>"a" ], { "conv1a" } ],
   InsertLevels[ BNLayerRules[ rootName<>"a", outputChannels/4 ], { "bn1a" } ],
   InsertLevels[ ConvLayerRules[ rootName<>"b" ], { "conv1b" } ],
   InsertLevels[ BNLayerRules[ rootName<>"b", outputChannels/4 ], { "bn1b" } ],
   InsertLevels[ ConvLayerRules[ rootName<>"c" ], { "conv1c" } ],
   InsertLevels[ BNLayerRules[ rootName<>"c", outputChannels ], { "bn1c" } ]
};


ResidualBlockRules[ rootName_, outputChannels_, batchNormBranch1_Symbol: False  ] := Join[
   InsertLevels[ ResidualBranch2Rules[ rootName<>"_branch2", outputChannels ], { "branch2" } ],
   If[ batchNormBranch1, InsertLevels[ BNConvLayerRules[ rootName<>"_branch1", outputChannels ], { "branch1" } ], {} ]
];


ResidualNetBlockRules[ rootName_, repeats_, outputChannels_ ] := Flatten[Prepend[
   InsertLevels[ ResidualBlockRules[ rootName<>"_0", outputChannels, True ], { rootName<>"_0" } ]][
   Table[ InsertLevels[ ResidualBlockRules[ rootName<>"_"<>ToString[k], outputChannels ], { rootName<>"_"<>ToString[k] } ], {k, repeats} ]],1];


ResBackboneNetRules = Join[
   InsertLevels[ BNConvLayerRules[ "conv1", 64 ], { "conv1" } ],
   ResidualNetBlockRules[ "res2", 2, 256 ],
   ResidualNetBlockRules[ "res3", 3, 512 ],
   ResidualNetBlockRules[ "res4", 22, 1024 ],
   ResidualNetBlockRules[ "res5", 2, 2048 ]
];


FPNNetRules = Join[
   InsertLevels[ ConvLayerRules[ "fpn_inner_res3_3_sum_lateral", True ], { "inner_res3_3_sum_lateral" } ],
   InsertLevels[ ConvLayerRules[ "fpn_inner_res4_22_sum_lateral", True ], { "inner_res4_22_sum_lateral" } ],
   InsertLevels[ ConvLayerRules[ "fpn_inner_res5_2_sum", True ], { "inner_res5_2_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_res3_3_sum", True ], { "res3_3_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_res4_22_sum", True ], { "res4_22_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_res5_2_sum", True ], { "res5_2_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_6", True ], { "6" } ],
   InsertLevels[ ConvLayerRules[ "fpn_7", True ], {"7", 2} ]
];


MultiBoxDecoderNetRules = Join[
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n0_fpn3", True ], {"ClassDecoder", "conv_n0_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n1_fpn3", True ], {"ClassDecoder", "conv_n1_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n2_fpn3", True ], {"ClassDecoder", "conv_n2_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n3_fpn3", True ], {"ClassDecoder", "conv_n3_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_pred_fpn3", True ], { "ClassDecoder", "pred_fpn"} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n0_fpn3", True ], {"BoxesDecoder", "conv_n0_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n1_fpn3", True ], {"BoxesDecoder", "conv_n1_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n2_fpn3", True ], {"BoxesDecoder", "conv_n2_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n3_fpn3", True ], {"BoxesDecoder", "conv_n3_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_pred_fpn3", True ], {"BoxesDecoder", "pred_fpn"} ]
];


DecoderNetRules = Join[
   InsertLevels[ MultiBoxDecoderNetRules, { "multibox3" } ],
   InsertLevels[ MultiBoxDecoderNetRules, { "multibox4" } ],
   InsertLevels[ MultiBoxDecoderNetRules, { "multibox5" } ],
   InsertLevels[ MultiBoxDecoderNetRules, { "multibox6" } ],
   InsertLevels[ MultiBoxDecoderNetRules, { "multibox7" } ]
];


anch = Import[ hdfFile, {"Datasets", "anchors"} ];


levels = {{112,144},{56,72},{28,36},{14,18},{7,9}};
strides = {8,16,32,64,128};
biasesx = Flatten[Table[(x-1)*strides[[l]]+Mean[anch[[l,a,{1,3}]]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
biasesy = Flatten[Table[(y-1)*strides[[l]]+Mean[anch[[l,a,{2,4}]]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
scalesw = Flatten[Table[1+anch[[l,a,3]]-anch[[l,a,1]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];
scalesh = Flatten[Table[1+anch[[l,a,4]]-anch[[l,a,2]],{l,1,5},{y,1,levels[[l,1]]},{x,1,levels[[l,2]]},{a,1,9}]];


BoxTransformationNetRules = {
   { "cx", 2, "Scaling" } -> scalesw,
   { "cx", 3, "Biases" } -> biasesx,
   { "cy", 2, "Scaling" } -> scalesh,
   { "cy", 3, "Biases" } -> biasesy,
   { "width", 3, "Scaling" } -> scalesw,
   { "height", 3, "Scaling" } -> scalesh
};


RetinaNetRules = Join[
   InsertLevels[ ResBackboneNetRules, { "ResBackbone" } ],
   InsertLevels[ FPNNetRules, { "FPN" } ],
   InsertLevels[ DecoderNetRules, { "Decoder" } ],
   InsertLevels[ BoxTransformationNetRules, { "BoxTransformation" } ]
];


RetinaNetInit = NetReplacePart[ RetinaNet, RetinaNetRules ];
