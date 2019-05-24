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


hdfFile = "~/Google Drive/RetinaNetNew1.hdf5";


imp[ hdfName_String ] := Import[ hdfFile, {"Datasets", hdfName} ];


ConvLayerRules[ rootName_, outputChannels_, biases_Symbol: False ] := {
   {"Weights"}->imp[rootName<>"_w"], {"Biases"}->If[ biases, imp[rootName<>"_b"], ConstantArray[0,Length@imp[rootName<>"_w"]]]};


BNConvLayerRules[ rootName_, outputChannels_ ] := {
   {1,"Weights"}->imp[rootName<>"_w"], {1,"Biases"}->ConstantArray[0,Length@imp[rootName<>"_w"]],
   {2,"Epsilon"}->.001, {2,"MovingMean"}->ConstantArray[0, outputChannels ], {2,"MovingVariance"}->ConstantArray[ 1.-10^-3, outputChannels ], {2,"Scaling"}->imp[ rootName<>"_bn_s" ], {2,"Biases"}->imp[ rootName<>"_bn_b" ]
};


InsertLevels[ rules_, levels_ ] := Map[ Join[levels,#[[1]]]->#[[2]]&, rules ];


ResidualBranch2Rules[ rootName_, outputChannels_ ] := Join[
   InsertLevels[ BNConvLayerRules[ rootName<>"a", outputChannels/4 ], { "a", 1 } ],
   InsertLevels[ BNConvLayerRules[ rootName<>"b", outputChannels/4 ], { "b", 1 } ],
   InsertLevels[ BNConvLayerRules[ rootName<>"c", outputChannels ], { "c" } ]
];


ResidualBlockRules[ rootName_, outputChannels_, batchNormBranch1_Symbol: False  ] := Join[
   InsertLevels[ ResidualBranch2Rules[ rootName<>"_branch2", outputChannels ], { "branch2" } ],
   If[ batchNormBranch1, InsertLevels[ BNConvLayerRules[ rootName<>"_branch1", outputChannels ], { "branch1" } ], {} ]
];


ResidualNetBlockRules[ rootName_, repeats_, outputChannels_ ] := Flatten[Prepend[
   InsertLevels[ ResidualBlockRules[ rootName<>"_0", outputChannels, True ], { "0" } ]][
   Table[ InsertLevels[ ResidualBlockRules[ rootName<>"_"<>ToString[k], outputChannels ], { ToString[k] } ], {k, repeats} ]],1];


ResBackboneNetRules = Join[
   InsertLevels[ BNConvLayerRules[ "conv1", 64 ], { "conv1", 1} ],
   InsertLevels[ ResidualNetBlockRules[ "res2", 2, 256 ], { "res2" } ],
   InsertLevels[ ResidualNetBlockRules[ "res3", 3, 512 ], { "res3" } ],
   InsertLevels[ ResidualNetBlockRules[ "res4", 22, 1024 ], { "res4" } ],
   InsertLevels[ ResidualNetBlockRules[ "res5", 2, 2048 ], { "res5" } ]   
];


FPNNetRules = Join[
   InsertLevels[ ConvLayerRules[ "fpn_inner_res3_3_sum_lateral", 256, True ], { "inner_res3_3_sum_lateral" } ],
   InsertLevels[ ConvLayerRules[ "fpn_inner_res4_22_sum_lateral", 256, True ], { "inner_res4_22_sum_lateral" } ],
   InsertLevels[ ConvLayerRules[ "fpn_inner_res5_2_sum", 256, True ], { "inner_res5_2_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_res3_3_sum", 256, True ], { "res3_3_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_res4_22_sum", 256, True ], { "res4_22_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_res5_2_sum", 256, True ], { "res5_2_sum" } ],
   InsertLevels[ ConvLayerRules[ "fpn_6", 256, True ], { "6" } ],
   InsertLevels[ ConvLayerRules[ "fpn_7", 256, True ], {"7", 2} ]
];


MultiBoxDecoderNetRules = Join[
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n0_fpn3", 256, True ], {"ClassDecoder", "conv_n0_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n1_fpn3", 256, True ], {"ClassDecoder", "conv_n1_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n2_fpn3", 256, True ], {"ClassDecoder", "conv_n2_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_conv_n3_fpn3", 256, True ], {"ClassDecoder", "conv_n3_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_cls_pred_fpn3", 256, True ], { "ClassDecoder", "pred_fpn"} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n0_fpn3", 256, True ], {"BoxesDecoder", "conv_n0_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n1_fpn3", 256, True ], {"BoxesDecoder", "conv_n1_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n2_fpn3", 256, True ], {"BoxesDecoder", "conv_n2_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_conv_n3_fpn3", 256, True ], {"BoxesDecoder", "conv_n3_fpn",1} ],
   InsertLevels[ ConvLayerRules[ "retnet_bbox_pred_fpn3", 256, True ], {"BoxesDecoder", "pred_fpn"} ]
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
