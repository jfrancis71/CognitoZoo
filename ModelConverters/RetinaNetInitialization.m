(* ::Package:: *)

hdfFile = "~/Google Drive/RetinaNetNew1.hdf5";


imp[ hdfName_String ] := Import[ hdfFile, {"Datasets", hdfName} ];


ConvolutionLayerRules[ rootName_, outputChannels_, biases_Symbol: False ] := {
   {"Weights"}->imp[rootName<>"_w"], {"Biases"}->If[ biases, imp[rootName<>"_b"], ConstantArray[0,Length@imp[rootName<>"_w"]]]};


BNConvolutionLayerRules[ rootName_, outputChannels_ ] := {
   {1,"Weights"}->imp[rootName<>"_w"], {1,"Biases"}->ConstantArray[0,Length@imp[rootName<>"_w"]],
   {2,"Epsilon"}->.001, {2,"MovingMean"}->ConstantArray[0, outputChannels ], {2,"MovingVariance"}->ConstantArray[ 1.-10^-3, outputChannels ], {2,"Scaling"}->imp[ rootName<>"_bn_s" ], {2,"Biases"}->imp[ rootName<>"_bn_b" ]
};


InsertLevel[ rules_, level_ ] := Map[ Prepend[level][#[[1]]]->#[[2]]&, rules ];


InsertLevels[ rules_, levels_ ] := Map[ Join[levels,#[[1]]]->#[[2]]&, rules ];


RetinaNetBranch2Rules[ rootName_, outputChannels_ ] := Join[
   InsertLevels[ BNConvolutionLayerRules[ rootName<>"a", outputChannels/4 ], { "a", 1 } ],
   InsertLevels[ BNConvolutionLayerRules[ rootName<>"b", outputChannels/4 ], { "b", 1 } ],
   InsertLevel[ BNConvolutionLayerRules[ rootName<>"c", outputChannels ], "c" ]
];


RetinaNetBlockRules[ rootName_, outputChannels_, batchNormBranch1_Symbol: False  ] := Join[
   InsertLevel[ RetinaNetBranch2Rules[ rootName<>"_branch2", outputChannels ], "branch2" ],
   If[ batchNormBranch1, InsertLevel[ BNConvolutionLayerRules[ rootName<>"_branch1", outputChannels ], "branch1" ], {} ]
];


ResidualNetBlockRules[ rootName_, repeats_, outputChannels_ ] := Flatten[Prepend[
   InsertLevel[ RetinaNetBlockRules[ rootName<>"_0", outputChannels, True ], "0" ]][
   Table[ InsertLevel[ RetinaNetBlockRules[ rootName<>"_"<>ToString[k], outputChannels ], ToString[k] ], {k, repeats} ]],1];


ResBackboneNetRules = Join[
   InsertLevels[ BNConvolutionLayerRules[ "conv1", 64 ], {"conv1",1} ],
   InsertLevel[ ResidualNetBlockRules[ "res2", 2, 256 ], "res2" ],
   InsertLevel[ ResidualNetBlockRules[ "res3", 3, 512 ], "res3" ],
   InsertLevel[ ResidualNetBlockRules[ "res4", 22, 1024 ], "res4" ],
   InsertLevel[ ResidualNetBlockRules[ "res5", 2, 2048 ], "res5" ]   
];


FPNNetRules = Join[
   InsertLevel[ ConvolutionLayerRules[ "fpn_inner_res3_3_sum_lateral", 256, True ], "inner_res3_3_sum_lateral" ],
   InsertLevel[ ConvolutionLayerRules[ "fpn_inner_res4_22_sum_lateral", 256, True ], "inner_res4_22_sum_lateral" ],
   InsertLevel[ ConvolutionLayerRules[ "fpn_inner_res5_2_sum", 256, True ], "inner_res5_2_sum" ],
   InsertLevel[ ConvolutionLayerRules[ "fpn_res3_3_sum", 256, True ], "res3_3_sum" ],
   InsertLevel[ ConvolutionLayerRules[ "fpn_res4_22_sum", 256, True ], "res4_22_sum" ],
   InsertLevel[ ConvolutionLayerRules[ "fpn_res5_2_sum", 256, True ], "res5_2_sum" ],
   InsertLevel[ ConvolutionLayerRules[ "fpn_6", 256, True ], "6" ],
   InsertLevels[ ConvolutionLayerRules[ "fpn_7", 256, True ], {"7", 2} ]
];


MultiBoxDecoderNetRules = Join[
   InsertLevels[ ConvolutionLayerRules[ "retnet_cls_conv_n0_fpn3", 256, True ], {"ClassDecoder", "retnet_cls_conv_n0_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_cls_conv_n1_fpn3", 256, True ], {"ClassDecoder", "retnet_cls_conv_n1_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_cls_conv_n2_fpn3", 256, True ], {"ClassDecoder", "retnet_cls_conv_n2_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_cls_conv_n3_fpn3", 256, True ], {"ClassDecoder", "retnet_cls_conv_n3_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_cls_pred_fpn3", 256, True ], {"ClassDecoder", "retnet_cls_pred_fpn"} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_bbox_conv_n0_fpn3", 256, True ], {"BoxesDecoder", "retnet_bbox_conv_n0_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_bbox_conv_n1_fpn3", 256, True ], {"BoxesDecoder", "retnet_bbox_conv_n1_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_bbox_conv_n2_fpn3", 256, True ], {"BoxesDecoder", "retnet_bbox_conv_n2_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_bbox_conv_n3_fpn3", 256, True ], {"BoxesDecoder", "retnet_bbox_conv_n3_fpn",1} ],
   InsertLevels[ ConvolutionLayerRules[ "retnet_bbox_pred_fpn3", 256, True ], {"BoxesDecoder", "retnet_bbox_pred_fpn"} ]
];


DecoderNetRules = Join[
   InsertLevel[ MultiBoxDecoderNetRules, "multibox3" ],
   InsertLevel[ MultiBoxDecoderNetRules, "multibox4" ],
   InsertLevel[ MultiBoxDecoderNetRules, "multibox5" ],
   InsertLevel[ MultiBoxDecoderNetRules, "multibox6" ],
   InsertLevel[ MultiBoxDecoderNetRules, "multibox7" ]
];


anch = Import[ hdfFile, {"Datasets", "anchors"} ];


levels={{112,144},{56,72},{28,36},{14,18},{7,9}};
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
   InsertLevel[ ResBackboneNetRules, "ResBackboneNet" ],
   InsertLevel[ FPNNetRules, "FPNNet" ],
   InsertLevel[ DecoderNetRules, "DecoderNet" ],
   InsertLevel[ BoxTransformationNetRules, "BoxTransformation" ]
];


RetinaNetInit = NetReplacePart[ RetinaNet, RetinaNetRules ];
